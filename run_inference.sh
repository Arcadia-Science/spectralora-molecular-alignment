#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="inference-docker-compose.yml"
VENV_PYTHON=".venv/bin/python"

mkdir -p data/processed

# ---------------------------------------------------------------------------
# 1. Build + start infrastructure — everything EXCEPT the API
#    (API needs Redis cluster ready before its lifespan connects)
# ---------------------------------------------------------------------------
echo "[1/7] Building and starting infrastructure (model x3, Redis nodes, nginx)..."
docker compose -f "$COMPOSE_FILE" up --build --scale model=3 -d \
  postgres redis-1 redis-2 redis-3 model model-nginx frontend

# ---------------------------------------------------------------------------
# 2. Wait for Postgres
# ---------------------------------------------------------------------------
echo "[2/7] Waiting for Postgres..."
until docker compose -f "$COMPOSE_FILE" exec -T postgres \
    psql -U detanet -d detanet -c "SELECT 1" >/dev/null 2>&1; do
  sleep 2; printf "."
done
echo " ready."

# ---------------------------------------------------------------------------
# 3. Build parquet shards (host-side, skipped if present)
# ---------------------------------------------------------------------------
if compgen -G "data/processed/*.parquet" >/dev/null 2>&1; then
  echo "[3/7] Parquet shards already exist, skipping."
else
  echo "[3/7] Building parquet shards..."
  "$VENV_PYTHON" scripts/ingest/build_parquet.py \
    --dataset qm9s \
    --zip capsule-3259363/data/qm9s_csv.zip \
    --out data/processed \
    --shard-size 1000
fi

# ---------------------------------------------------------------------------
# 4. Initialise Redis Cluster (idempotent)
# ---------------------------------------------------------------------------
echo "[4/7] Setting up Redis Cluster..."
for node in redis-1 redis-2 redis-3; do
  until docker compose -f "$COMPOSE_FILE" exec -T "$node" \
      redis-cli ping >/dev/null 2>&1; do
    sleep 1; printf "."
  done
done

if docker compose -f "$COMPOSE_FILE" exec -T redis-1 \
    redis-cli cluster info 2>/dev/null | grep -q "cluster_state:ok"; then
  echo " already initialised."
else
  docker compose -f "$COMPOSE_FILE" exec -T redis-1 \
    redis-cli --cluster create \
      redis-1:6379 redis-2:6379 redis-3:6379 \
      --cluster-replicas 0 --cluster-yes
  until docker compose -f "$COMPOSE_FILE" exec -T redis-1 \
      redis-cli cluster info 2>/dev/null | grep -q "cluster_state:ok"; do
    sleep 2; printf "."
  done
  echo " ready."
fi

# ---------------------------------------------------------------------------
# 5. Start API (cluster is ready, lifespan will connect cleanly)
# ---------------------------------------------------------------------------
echo "[5/7] Starting API..."
docker compose -f "$COMPOSE_FILE" up -d --no-deps api
until curl -sf http://localhost:8000/healthz >/dev/null 2>&1; do
  sleep 2; printf "."
done
echo " ready."

# ---------------------------------------------------------------------------
# 6. Seed Postgres (runs inside API container on Docker network)
# ---------------------------------------------------------------------------
echo "[6/7] Loading molecules into Postgres..."
docker compose -f "$COMPOSE_FILE" exec -T api \
  python /app/scripts/ingest/load_db.py \
  --db-url postgresql://detanet:detanet@postgres:5432/detanet \
  --dataset qm9s \
  --parquet-dir /data/processed
echo "      Done."

# ---------------------------------------------------------------------------
# 7. Wait for model service (weight loading across 3 replicas)
# ---------------------------------------------------------------------------
echo "[7/7] Waiting for model service (loading weights — takes a few minutes)..."
until docker compose -f "$COMPOSE_FILE" exec -T model-nginx \
    wget -q -O- http://localhost/healthz >/dev/null 2>&1; do
  sleep 5; printf "."
done
echo ""

echo ""
echo "All services ready."
echo ""
echo "  http://localhost:3000  — frontend"
echo ""
echo "  curl -s -X POST http://localhost:8000/predict/raman \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"smiles\":\"c1ccccc1\"}'"
