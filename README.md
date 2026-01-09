# hp-proteins-ml

Scaffold for a DetaNet-backed inference API + model service, with Postgres for dataset registry and Redis caching.

## Layout
- `capsule-3259363/`: original model code + weights + datasets
- `apps/api/`: async FastAPI for dataset browsing and inference
- `apps/model/`: Python 3.8 FastAPI model service
- `db/init/`: Postgres schema + partitions
- `scripts/ingest/`: build Parquet shards and load DB
- `data/processed/`: Parquet shards (ignored by git)

## Quickstart
1) Build Parquet shards (from the CSV zips):
```bash
python scripts/ingest/build_parquet.py --dataset qm9s --zip capsule-3259363/data/qm9s_csv.zip --out data/processed --shard-size 1000
python scripts/ingest/build_parquet.py --dataset ext_val --zip capsule-3259363/data/ext_val.zip --out data/processed --shard-size 1000
python scripts/ingest/build_parquet.py --dataset ext_val_env --zip capsule-3259363/data/ext_val_env.zip --out data/processed --shard-size 1000
```

2) Start services:
```bash
docker-compose up --build
```

3) Load registry rows into Postgres:
```bash
python scripts/ingest/load_db.py --db-url postgresql://detanet:detanet@localhost:5432/detanet --dataset qm9s --parquet-dir data/processed
python scripts/ingest/load_db.py --db-url postgresql://detanet:detanet@localhost:5432/detanet --dataset ext_val --parquet-dir data/processed
python scripts/ingest/load_db.py --db-url postgresql://detanet:detanet@localhost:5432/detanet --dataset ext_val_env --parquet-dir data/processed
```

4) (Optional) Run integration tests against the live services:
```bash
INTEGRATION_BASE_URL=http://localhost:8000 INTEGRATION_DATASET=ext_val .venv-311/bin/python -m pytest -m integration -q
```

## API Examples
List datasets:
```bash
curl "http://localhost:8000/datasets"
```

List datapoints (exact SMILES match):
```bash
curl "http://localhost:8000/datapoints?dataset=qm9s&smiles=C"
```

Fetch geometry for a datapoint:
```bash
curl "http://localhost:8000/datapoints/ext_val/1?include_geometry=true"
```

Raman inference by dataset ID:
```bash
curl -X POST "http://localhost:8000/predict/raman" \
  -H "Content-Type: application/json" \
  -d '{"dataset":"qm9s","molecule_id":1}'
```

Raman inference with inline geometry:
```bash
curl -X POST "http://localhost:8000/predict/raman" \
  -H "Content-Type: application/json" \
  -d '{"pos":[[0,0,0],[0,0,1]],"z":[6,1]}'
```

Vibrational/raman inference using dataset geometry (recommended):
```bash
python - <<'PY'
import json
from urllib.request import Request, urlopen

base = "http://localhost:8000"
dp = json.load(urlopen(f"{base}/datapoints/ext_val/1?include_geometry=true"))
payload = {"pos": dp["pos"], "z": dp["z"]}

for path in ("/predict/vib", "/predict/raman"):
    req = Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        data = json.load(resp)
        print(path, list(data.keys()))
PY
```

Charge/UV inference with inline geometry:
```bash
curl -X POST "http://localhost:8000/predict/charge" \
  -H "Content-Type: application/json" \
  -d '{"pos":[[0,0,0],[0,0,1]],"z":[6,1]}'

curl -X POST "http://localhost:8000/predict/uv" \
  -H "Content-Type: application/json" \
  -d '{"pos":[[0,0,0],[0,0,1]],"z":[6,1]}'
```

NMR aggregation (after `/predict/nmr`):
```bash
curl -X POST "http://localhost:8000/predict/nmr/aggregate" \
  -H "Content-Type: application/json" \
  -d '{"sc":[1.0,2.0],"sh":[3.0],"indexc":[0,1],"indexh":[0]}'
```

## Notebook
Open `inference.ipynb` and run the cells to call the API from a notebook. You can override
the target host with:
```bash
export INFERENCE_BASE_URL=http://localhost:8000
export INFERENCE_DATASET=ext_val
```

## Notes
- The model service uses Python 3.8 and loads code/weights from `capsule-3259363/code`.
- Raman endpoint returns `x`, normalized `y`, and a `png_base64` image.
- NMR aggregation is a separate endpoint; the base NMR endpoint returns raw `sc`/`sh`.


### Install

```
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```