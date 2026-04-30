from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import cache, db
from app.cache import get_json, set_json, smiles_key
from app.config import settings
from app.data_store import read_geometry
from app.inference_client import ModelClient
from app.models import RamanRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await cache.init_cache(settings.redis_cluster_nodes)
    app.state.model_client = ModelClient()
    yield
    await db.close_db()
    await cache.close_cache()
    await app.state.model_client.close()


app = FastAPI(title="DetaNet API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> str:
    return Path("/app/static/index.html").read_text()


@app.get("/datasets")
async def list_datasets() -> dict:
    rows = await db.fetch("SELECT name, description, source_uri FROM datasets ORDER BY name")
    return {"datasets": [dict(row) for row in rows]}


@app.get("/datapoints")
async def list_datapoints(
    dataset: str = Query(...),
    smiles: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    if smiles:
        rows = await db.fetch(
            """
            SELECT dataset, molecule_id, smiles, n_atoms, environment
            FROM molecules WHERE dataset=$1 AND smiles=$2
            ORDER BY molecule_id LIMIT $3 OFFSET $4
            """,
            dataset, smiles, limit, offset,
        )
    else:
        rows = await db.fetch(
            """
            SELECT dataset, molecule_id, smiles, n_atoms, environment
            FROM molecules WHERE dataset=$1
            ORDER BY molecule_id LIMIT $2 OFFSET $3
            """,
            dataset, limit, offset,
        )
    return {"datapoints": [dict(row) for row in rows]}


@app.get("/datapoints/{dataset}/{molecule_id}")
async def get_datapoint(dataset: str, molecule_id: int, include_geometry: bool = False) -> dict:
    cache_key = f"dp:{dataset}:{molecule_id}:{int(include_geometry)}"
    cached = await get_json(cache_key)
    if cached:
        return cached

    row = await db.fetchrow(
        """
        SELECT dataset, molecule_id, qm9_id, smiles, n_atoms, environment, source_path, source_row
        FROM molecules WHERE dataset=$1 AND molecule_id=$2
        """,
        dataset, molecule_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    payload = dict(row)
    if include_geometry:
        pos, z = read_geometry(settings.parquet_dir, row["source_path"], row["source_row"])
        payload["pos"] = pos
        payload["z"] = z

    await set_json(cache_key, payload, settings.cache_ttl_seconds)
    return payload


async def _resolve_smiles(request: RamanRequest) -> str:
    if request.smiles is not None:
        return request.smiles
    row = await db.fetchrow(
        "SELECT smiles FROM molecules WHERE dataset=$1 AND molecule_id=$2",
        request.dataset, request.molecule_id,
    )
    if not row or not row["smiles"]:
        raise HTTPException(status_code=404, detail="Molecule not found or has no SMILES")
    return row["smiles"]


@app.post("/predict/raman")
async def predict_raman(request: RamanRequest) -> dict:
    smiles = await _resolve_smiles(request)

    # Check DB-scoped key first (fast path for repeated dataset+id queries)
    db_key = (
        f"pred:raman:{request.dataset}:{request.molecule_id}"
        if request.dataset and request.molecule_id
        else None
    )
    s_key = smiles_key(smiles)

    for key in filter(None, [db_key, s_key]):
        hit = await get_json(key)
        if hit:
            return hit

    try:
        result = await app.state.model_client.post("/predict/raman", {"smiles": smiles})
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Model service unavailable") from exc

    # Write to both keys so either lookup path hits cache next time
    for key in filter(None, [db_key, s_key]):
        await set_json(key, result, settings.cache_ttl_seconds)

    return result
