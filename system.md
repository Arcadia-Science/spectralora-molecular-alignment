# System Architecture

This document describes the current system design and data flow for the DetaNet-backed inference stack.

## Overview
The system is split into two FastAPI services:
- **API service** (`apps/api`): public API for dataset browsing and inference orchestration.
- **Model service** (`apps/model`): runs the DetaNet model code and returns predictions.

Supporting services:
- **Postgres**: dataset registry and molecule metadata lookup.
- **Redis**: response cache for datapoints and predictions.
- **Parquet storage**: geometry data stored as Parquet shards under `data/processed/`.

## High-Level Diagram
```
Client
  |
  v
API (FastAPI)  <---->  Redis (cache)
  |  \
  |   \--> Postgres (dataset registry)
  |
  v
Model (FastAPI, DetaNet)
  |
  v
Model code + weights (capsule-3259363/code)
```

## Components

### API service (`apps/api`)
- Handles dataset browsing:
  - `GET /datasets`
  - `GET /datapoints?dataset=...`
  - `GET /datapoints/{dataset}/{molecule_id}?include_geometry=true`
- Handles inference:
  - `/predict/charge`, `/predict/vib`, `/predict/raman`, `/predict/uv`, `/predict/nmr`
  - `/predict/nmr/aggregate`
- Reads geometry from Parquet via `app.data_store.read_geometry`.
- Caches datapoints and inference responses in Redis (TTL from `CACHE_TTL_SECONDS`).
- Calls the model service via HTTP (`MODEL_URL`).

### Model service (`apps/model`)
- Loads the DetaNet model code + weights from `capsule-3259363/code`.
- Exposes inference endpoints for charge, vib/raman, uv, and nmr.
- Returns:
  - `raman`: x-axis, normalized y, and `png_base64`.
  - `nmr`: raw `sc` / `sh` values; aggregation happens via `/predict/nmr/aggregate`.

### Postgres (`db/init/001_init.sql`)
- `datasets` table: dataset metadata.
- `molecules` table (partitioned): indexed lookup of molecule metadata and Parquet row location.

### Redis
- Simple key/value cache for:
  - datapoint payloads (with optional geometry)
  - predictions keyed by dataset + molecule_id

### Parquet storage
- Geometry data stored as Parquet shards in `data/processed/`.
- Each molecule row stores `source_path` and `source_row` for random access.

## Data Flow

### Dataset browsing
1. Client requests `/datapoints`.
2. API queries Postgres for rows (by dataset and optional SMILES).
3. Response returned to client.

### Geometry lookup
1. Client requests `/datapoints/{dataset}/{molecule_id}?include_geometry=true`.
2. API fetches row metadata from Postgres.
3. API reads geometry from Parquet shard and returns it.
4. Response is cached in Redis.

### Inference
1. Client calls `/predict/*`.
2. API resolves geometry:
   - Inline `pos` + `z` if provided, or
   - Look up Parquet geometry via Postgres metadata.
3. API checks Redis cache; if miss, calls the model service.
4. Model service runs DetaNet inference and returns results.
5. API caches the response and returns it to the client.

## Ingestion Pipeline
1. `scripts/ingest/build_parquet.py` converts CSV zips into Parquet shards.
2. `scripts/ingest/load_db.py` loads dataset metadata into Postgres.

## Deployment
`docker-compose.yml` builds and runs:
- `postgres` (port 5432)
- `redis` (port 6379)
- `model` (port 8001)
- `api` (port 8000)

## Testing
- Unit tests live under `apps/api/tests`.
- Integration tests (marked `integration`) exercise the running API and require:
  - `docker compose up --build`
  - dataset loaded into Postgres

Example:
```
INTEGRATION_BASE_URL=http://localhost:8000 INTEGRATION_DATASET=ext_val .venv-311/bin/python -m pytest -m integration -q
```

## Notes / Constraints
- The model service runs on Python 3.8 to match the DetaNet codebase.
- `predict/vib` and `predict/raman` require gradients; model code depends on PyTorch autograd.
