#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT/OVERRIDE THESE ======
KEY="${KEY:-~/.ssh/your-key.pem}"
HOST="${HOST:-ec2-user@your-ec2-host}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-/fsx/repos/hp-proteins-ml}"
PY="${PY:-/home/ec2-user/miniforge3/envs/hp/bin/python}"
# ================================

echo ">> Using HOST=$HOST"
echo ">> Ray dashboard tunnel: ssh -i ${KEY} -N -L 8265:127.0.0.1:8265 ${HOST}"
echo ">> TensorBoard tunnel: ssh -i ${KEY} -N -L 6006:127.0.0.1:6006 ${HOST}"

echo ">> Sync code changes to FSx repo"
rsync -avz -e "ssh -i ${KEY}" \
  "${LOCAL_REPO}/train/train_tune.py" \
  "${HOST}:${REMOTE_REPO}/train/train_tune.py"
rsync -avz -e "ssh -i ${KEY}" \
  "${LOCAL_REPO}/train/train_detanet.py" \
  "${HOST}:${REMOTE_REPO}/train/train_detanet.py"
rsync -avz -e "ssh -i ${KEY}" \
  "${LOCAL_REPO}/capsule-3259363/code/detanet_model/detanet.py" \
  "${HOST}:${REMOTE_REPO}/capsule-3259363/code/detanet_model/detanet.py"
rsync -avz -e "ssh -i ${KEY}" \
  "${LOCAL_REPO}/capsule-3259363/code/detanet_model/modules/radial_basis.py" \
  "${HOST}:${REMOTE_REPO}/capsule-3259363/code/detanet_model/modules/radial_basis.py"

echo ">> Run remote Ray Tune"
ssh -i "${KEY}" "${HOST}" \
  "PY='$PY' REMOTE_REPO='$REMOTE_REPO' bash -s" <<'REMOTE'
set -euo pipefail

PY="$PY"
REPO="$REMOTE_REPO"
RAY_CLI="$(dirname "$PY")/ray"

echo ">> Python path: $PY"
ls -l "$PY" || true

if [ ! -x "$PY" ]; then
  echo "ERROR: Python env not found at $PY"
  exit 1
fi

"$PY" - <<'PY'
import ray
print("Ray version:", ray.__version__)
PY

# Ensure pytorch-optimizer (torch_optimizer) is available.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch_optimizer") else 1)
PY
then
  echo ">> Installing pytorch-optimizer (required for pt_shampoo)"
  "$PY" -m pip install --quiet pytorch-optimizer
fi

# Ensure PEFT (AdaLoRA) is available.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("peft") else 1)
PY
then
  echo ">> Installing peft (required for AdaLoRA)"
  "$PY" -m pip install --quiet peft
fi

if [ -x "$RAY_CLI" ]; then
  "$RAY_CLI" stop || true
  "$RAY_CLI" start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats
  if ss -lntp | grep -q ':8265 '; then
    echo ">> Ray dashboard listening on 0.0.0.0:8265"
  else
    echo ">> Ray dashboard NOT listening on 8265"
    echo ">> Installing ray[default] (dashboard) and restarting..."
    "$PY" -m pip install --quiet 'ray[default]'
    "$RAY_CLI" stop || true
    "$RAY_CLI" start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats
    if ss -lntp | grep -q ':8265 '; then
      echo ">> Ray dashboard listening on 0.0.0.0:8265"
    fi
  fi
else
  echo ">> Ray CLI not found; relying on Ray auto-init (no dashboard)."
fi

# Build shard list
find /fsx/processed_all -name 'shard_*.pt' > /tmp/all_shards.txt

# Base args (JSON list)
cat > /tmp/base_args.json <<'JSON'
[
  "--task","Hij",
  "--shard-list","/tmp/all_shards.txt",
  "--checkpoint","/fsx/repos/hp-proteins-ml/capsule-3259363/code/trained_param/qm9spectra/Hij.pth",
  "--no-checkpoint-strict",
  "--checkpoint-relax-embeddings",
  "--checkpoint-relax-mismatch",
  "--split","train",
  "--split-key","number",
  "--split-train","0.8","--split-val","0.1",
  "--epochs","5",
  "--eval-every","1",
  "--amp",
  "--grad-clip","1.0",
  "--use-elora",
  "--use-adalora",
  "--adapter-freeze-base",
  "--no-use-impute-mask",
  "--skip-nonfinite",
  "--skip-bad-batches",
  "--normalize","dataset",
  "--norm-cache","/fsx/model_registry/norm_cache_hij.json",
  "--exclude-keys","mol_key,subset,source,smile,field_source,field_generated,field_imputed,field_confidence,conformer_id",
  "--num-workers","4",
  "--tensorboard"
]
JSON

# Param space (pt_shampoo only)
cat > /tmp/param_space.json <<'JSON'
{
  "lr": {"type":"loguniform","min":1e-5,"max":3e-4},
  "batch_size": {"type":"choice","values":[16,24,32]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[8,16]},
  "adalora_alpha": {"type":"choice","values":[16,32]}
}
JSON

WANDB_MODE=offline WANDB_SILENT=true \
"$PY" "$REPO/train/train_tune.py" \
  --registry-dir /fsx/model_registry \
  --run-prefix tune-hij \
  --param-space-file /tmp/param_space.json \
  --base-args "$(tr -d '\n' </tmp/base_args.json)" \
  --num-samples 8 \
  --max-concurrent 4 \
  --cpus-per-trial 8 \
  --gpus-per-trial 2 \
  --scheduler asha \
  --max-t 5 \
  --report-interval 60 \
  --best-copy \
  --best-dir best

echo ">> Metrics live under: /fsx/model_registry/${RUN_PREFIX:-tune-hij}-*"
echo ">> TensorBoard: tensorboard --logdir /fsx/model_registry --bind_all --port 6006"
REMOTE
