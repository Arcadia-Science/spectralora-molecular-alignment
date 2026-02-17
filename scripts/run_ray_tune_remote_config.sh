#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT/OVERRIDE THESE ======
KEY="${KEY:-~/.ssh/your-key.pem}"
HOST="${HOST:-ec2-user@your-instance.compute.amazonaws.com}"
LOCAL_REPO="${LOCAL_REPO:-$PWD}"
REMOTE_REPO="${REMOTE_REPO:-/fsx/repos/hp-proteins-ml}"
PY="${PY:-/home/ec2-user/miniforge3/envs/hp/bin/python}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/codex_known_hosts}"
RUN_PREFIX="${RUN_PREFIX:-tune-hij}"
TASK="${TASK:-Hij}"
CHECKPOINT="${CHECKPOINT:-}"
NORM_CACHE="${NORM_CACHE:-}"
DETACH="${DETACH:-1}"
START_TB="${START_TB:-1}"
INSTALL_PROFILING="${INSTALL_PROFILING:-1}"
CLEAN_TB="${CLEAN_TB:-0}"
SMOKE="${SMOKE:-0}"
USER_TRAIN_USE_DISTRIBUTED="${TRAIN_USE_DISTRIBUTED:-}"
TRAIN_USE_DISTRIBUTED="${TRAIN_USE_DISTRIBUTED:-1}"
LOG_TRAIN_PREDS="${LOG_TRAIN_PREDS:-0}"
SHARD_ROOTS="${SHARD_ROOTS:-/fsx/processed_all}"
STAGE_SHARDS_PER_DATASET="${STAGE_SHARDS_PER_DATASET:-0}"
STAGE_TOTAL_SHARDS="${STAGE_TOTAL_SHARDS:-0}"
STAGE_SEED="${STAGE_SEED:-123}"
MIN_SAMPLES="${MIN_SAMPLES:-10}"
SPLIT_KEY="${SPLIT_KEY:-mol_key}"
SPLIT_METHOD="${SPLIT_METHOD:-hash}"
SCAFFOLD_GROUP_KEY="${SCAFFOLD_GROUP_KEY:-mol_key}"
SCAFFOLD_SMILES_KEY="${SCAFFOLD_SMILES_KEY:-smile}"
SCAFFOLD_INCLUDE_CHIRALITY="${SCAFFOLD_INCLUDE_CHIRALITY:-0}"
SCAFFOLD_FALLBACK="${SCAFFOLD_FALLBACK:-molecule}"
SPLIT_TRAIN="${SPLIT_TRAIN:-0.7}"
SPLIT_VAL="${SPLIT_VAL:-0.1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
STEP_EVAL_MAX_BATCHES="${STEP_EVAL_MAX_BATCHES:-0}"
STEP_EVAL_INCLUDE_TEST="${STEP_EVAL_INCLUDE_TEST:-0}"
EPOCH_EVAL_MAX_BATCHES="${EPOCH_EVAL_MAX_BATCHES:-0}"
# ================================

if [ "$TASK" != "Hij" ] && [ "$RUN_PREFIX" = "tune-hij" ]; then
  RUN_PREFIX="tune-${TASK}"
fi

echo ">> Using HOST=$HOST"
echo ">> Ray dashboard tunnel: ssh ${SSH_OPTS} -i ${KEY} -N -L 8265:127.0.0.1:8265 ${HOST}"
echo ">> TensorBoard tunnel: ssh ${SSH_OPTS} -i ${KEY} -N -L 6006:127.0.0.1:6006 ${HOST}"

echo ">> Sync code changes to FSx repo"
rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
  "${LOCAL_REPO}/train/train_tune.py" \
  "${HOST}:${REMOTE_REPO}/train/train_tune.py"
rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
  "${LOCAL_REPO}/train/train_lightning.py" \
  "${HOST}:${REMOTE_REPO}/train/train_lightning.py"
rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
  "${LOCAL_REPO}/train/train_detanet.py" \
  "${HOST}:${REMOTE_REPO}/train/train_detanet.py"
rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
  "${LOCAL_REPO}/capsule-3259363/code/detanet_model/detanet.py" \
  "${HOST}:${REMOTE_REPO}/capsule-3259363/code/detanet_model/detanet.py"
rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
  "${LOCAL_REPO}/capsule-3259363/code/detanet_model/modules/radial_basis.py" \
  "${HOST}:${REMOTE_REPO}/capsule-3259363/code/detanet_model/modules/radial_basis.py"
if [ -f "${LOCAL_REPO}/scripts/summarize_tune_metrics.py" ]; then
  rsync -avz -e "ssh ${SSH_OPTS} -i ${KEY}" \
    "${LOCAL_REPO}/scripts/summarize_tune_metrics.py" \
    "${HOST}:${REMOTE_REPO}/scripts/summarize_tune_metrics.py"
fi

echo ">> Run remote Ray Tune"
ssh ${SSH_OPTS} -i "${KEY}" "${HOST}" \
  "PY='$PY' REMOTE_REPO='$REMOTE_REPO' RUN_PREFIX='$RUN_PREFIX' TASK='${TASK:-}' CHECKPOINT='${CHECKPOINT:-}' NORM_CACHE='${NORM_CACHE:-}' DETACH='$DETACH' START_TB='$START_TB' INSTALL_PROFILING='$INSTALL_PROFILING' CLEAN_TB='$CLEAN_TB' SMOKE='$SMOKE' SMOKE_SHARDS='${SMOKE_SHARDS:-}' SMOKE_ITEMS='${SMOKE_ITEMS:-}' SMOKE_ITEMS_PER_SHARD='${SMOKE_ITEMS_PER_SHARD:-}' SMOKE_SPLIT_KEY='${SMOKE_SPLIT_KEY:-}' SMOKE_SPLIT_TRAIN='${SMOKE_SPLIT_TRAIN:-}' SMOKE_SPLIT_VAL='${SMOKE_SPLIT_VAL:-}' SMOKE_SPLIT_SEED='${SMOKE_SPLIT_SEED:-}' SPLIT_KEY='${SPLIT_KEY:-}' SPLIT_METHOD='${SPLIT_METHOD:-}' SCAFFOLD_GROUP_KEY='${SCAFFOLD_GROUP_KEY:-}' SCAFFOLD_SMILES_KEY='${SCAFFOLD_SMILES_KEY:-}' SCAFFOLD_INCLUDE_CHIRALITY='${SCAFFOLD_INCLUDE_CHIRALITY:-}' SCAFFOLD_FALLBACK='${SCAFFOLD_FALLBACK:-}' SPLIT_TRAIN='${SPLIT_TRAIN:-}' SPLIT_VAL='${SPLIT_VAL:-}' LOG_PREDS_MAX='${LOG_PREDS_MAX:-}' TRAIN_USE_DISTRIBUTED='${TRAIN_USE_DISTRIBUTED:-}' LOG_TRAIN_PREDS='${LOG_TRAIN_PREDS:-}' EPOCHS='${EPOCHS:-}' EVAL_EVERY='${EVAL_EVERY:-}' EVAL_EVERY_STEPS='${EVAL_EVERY_STEPS:-}' STEP_EVAL_MAX_BATCHES='${STEP_EVAL_MAX_BATCHES:-}' STEP_EVAL_INCLUDE_TEST='${STEP_EVAL_INCLUDE_TEST:-}' EPOCH_EVAL_MAX_BATCHES='${EPOCH_EVAL_MAX_BATCHES:-}' MAX_T='${MAX_T:-}' STAGE_SHARDS='${STAGE_SHARDS:-}' STAGE_DIR='${STAGE_DIR:-}' STAGE_LIMIT_GB='${STAGE_LIMIT_GB:-}' SHARD_ROOTS='${SHARD_ROOTS:-}' STAGE_SHARDS_PER_DATASET='${STAGE_SHARDS_PER_DATASET:-}' STAGE_TOTAL_SHARDS='${STAGE_TOTAL_SHARDS:-}' STAGE_SEED='${STAGE_SEED:-}' MIN_SAMPLES='${MIN_SAMPLES:-}' NUM_WORKERS='${NUM_WORKERS:-}' LOADER_TIMEOUT='${LOADER_TIMEOUT:-}' PREFETCH_FACTOR='${PREFETCH_FACTOR:-}' PERSISTENT_WORKERS='${PERSISTENT_WORKERS:-}' LOG_EVERY='${LOG_EVERY:-}' DDP_TIMEOUT='${DDP_TIMEOUT:-}' GPUS_PER_TRIAL='${GPUS_PER_TRIAL:-}' MAX_CONCURRENT='${MAX_CONCURRENT:-}' NUM_SAMPLES='${NUM_SAMPLES:-}' FIXED_LR='${FIXED_LR:-}' FIXED_BATCH_SIZE='${FIXED_BATCH_SIZE:-}' FIXED_ADALORA_R='${FIXED_ADALORA_R:-}' FIXED_ADALORA_ALPHA='${FIXED_ADALORA_ALPHA:-}' bash -s" <<'REMOTE'
set -euo pipefail

PY="$PY"
REPO="$REMOTE_REPO"
RUN_PREFIX="${RUN_PREFIX:-tune-hij}"
TASK="${TASK:-Hij}"
CHECKPOINT="${CHECKPOINT:-/fsx/repos/hp-proteins-ml/capsule-3259363/code/trained_param/qm9spectra/${TASK}.pth}"
NORM_CACHE="${NORM_CACHE:-}"
DETACH="${DETACH:-0}"
START_TB="${START_TB:-1}"
INSTALL_PROFILING="${INSTALL_PROFILING:-1}"
CLEAN_TB="${CLEAN_TB:-0}"
SMOKE="${SMOKE:-0}"
SMOKE_SHARDS="${SMOKE_SHARDS:-8}"
SMOKE_ITEMS="${SMOKE_ITEMS:-0}"
SMOKE_ITEMS_PER_SHARD="${SMOKE_ITEMS_PER_SHARD:-0}"
SMOKE_SPLIT_KEY="${SMOKE_SPLIT_KEY:-smoke_key}"
SMOKE_SPLIT_TRAIN="${SMOKE_SPLIT_TRAIN:-0.6}"
SMOKE_SPLIT_VAL="${SMOKE_SPLIT_VAL:-0.0}"
SMOKE_SPLIT_SEED="${SMOKE_SPLIT_SEED:-123}"
SMOKE_TASK="${SMOKE_TASK:-$TASK}"
TRAIN_USE_DISTRIBUTED="${TRAIN_USE_DISTRIBUTED:-1}"
USER_TRAIN_USE_DISTRIBUTED="${USER_TRAIN_USE_DISTRIBUTED:-}"
LOG_TRAIN_PREDS="${LOG_TRAIN_PREDS:-0}"
STAGE_SHARDS="${STAGE_SHARDS:-0}"
STAGE_DIR="${STAGE_DIR:-/home/ec2-user/shards}"
STAGE_LIMIT_GB="${STAGE_LIMIT_GB:-100}"
SHARD_ROOTS="${SHARD_ROOTS:-/fsx/processed_all}"
STAGE_SHARDS_PER_DATASET="${STAGE_SHARDS_PER_DATASET:-0}"
STAGE_TOTAL_SHARDS="${STAGE_TOTAL_SHARDS:-0}"
STAGE_SEED="${STAGE_SEED:-123}"
MIN_SAMPLES="${MIN_SAMPLES:-10}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_SHARDS="${PREFLIGHT_SHARDS:-5}"
PREFLIGHT_FULL_SHARDS="${PREFLIGHT_FULL_SHARDS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOADER_TIMEOUT="${LOADER_TIMEOUT:-120}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
LOG_PREDS_MAX="${LOG_PREDS_MAX:-5}"
EPOCHS="${EPOCHS:-5}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
STEP_EVAL_MAX_BATCHES="${STEP_EVAL_MAX_BATCHES:-0}"
STEP_EVAL_INCLUDE_TEST="${STEP_EVAL_INCLUDE_TEST:-0}"
EPOCH_EVAL_MAX_BATCHES="${EPOCH_EVAL_MAX_BATCHES:-0}"
MAX_T="${MAX_T:-5}"
SCHEDULER="${SCHEDULER:-asha}"
FIXED_LR="${FIXED_LR:-}"
FIXED_BATCH_SIZE="${FIXED_BATCH_SIZE:-}"
FIXED_ADALORA_R="${FIXED_ADALORA_R:-}"
FIXED_ADALORA_ALPHA="${FIXED_ADALORA_ALPHA:-}"
PARAM_SPACE_LOCKED=0
RAY_CLI="$(dirname "$PY")/ray"

if [ "$SMOKE" = "1" ] && [ "${SMOKE_ITEMS_PER_SHARD}" -gt 0 ] && [ "${SMOKE_SPLIT_VAL}" = "0.0" ]; then
  SMOKE_SPLIT_VAL="0.2"
fi

echo ">> Training task: ${TASK}"
echo ">> Checkpoint: ${CHECKPOINT}"

if [ "${PERSISTENT_WORKERS}" = "1" ]; then
  PERSISTENT_WORKERS_FLAG="--persistent-workers"
else
  PERSISTENT_WORKERS_FLAG="--no-persistent-workers"
fi

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

# Ensure Ray Tune deps are available (ray[tune] + pyarrow).
if ! "$PY" - <<'PY' 2>/dev/null
from ray import tune  # noqa: F401
import pyarrow  # noqa: F401
PY
then
  echo ">> Installing Ray Tune dependencies (ray[tune], pyarrow)"
  "$PY" -m pip install --quiet 'ray[tune]' pyarrow
fi
if ! "$PY" - <<'PY' 2>/dev/null
from ray import tune  # noqa: F401
import pyarrow  # noqa: F401
PY
then
  echo "ERROR: Ray Tune dependencies missing (ray[tune]/pyarrow)."
  exit 1
fi

# Ensure pytorch-optimizer (pytorch_optimizer / torch_optimizer) is available.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
has_p = importlib.util.find_spec("pytorch_optimizer") is not None
has_t = importlib.util.find_spec("torch_optimizer") is not None
raise SystemExit(0 if (has_p or has_t) else 1)
PY
then
  echo ">> Installing pytorch-optimizer / torch_optimizer"
  "$PY" -m pip install --quiet pytorch-optimizer || true
  "$PY" -m pip install --quiet torch_optimizer || true
fi
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
has_p = importlib.util.find_spec("pytorch_optimizer") is not None
has_t = importlib.util.find_spec("torch_optimizer") is not None
raise SystemExit(0 if (has_p or has_t) else 1)
PY
then
  echo "ERROR: pytorch-optimizer not available in env."
  exit 1
fi
"$PY" - <<'PY'
import importlib.util
print("pytorch_optimizer OK" if importlib.util.find_spec("pytorch_optimizer") else "torch_optimizer OK")
PY

# Ensure PEFT (AdaLoRA) is available.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("peft") else 1)
PY
then
  echo ">> Installing peft (required for AdaLoRA)"
  "$PY" -m pip install --quiet peft
fi
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("peft") else 1)
PY
then
  echo "ERROR: peft not available in env."
  exit 1
fi
"$PY" - <<'PY'
import peft  # noqa: F401
print("peft OK")
PY

# Ensure ELoRA vendored e3nn dependency is available.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("opt_einsum_fx") else 1)
PY
then
  echo ">> Installing opt-einsum-fx (required by vendored ELoRA/e3nn)"
  "$PY" -m pip install --quiet opt-einsum-fx
fi
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("opt_einsum_fx") else 1)
PY
then
  echo "ERROR: opt-einsum-fx not available in env."
  exit 1
fi

if [ "$INSTALL_PROFILING" = "1" ]; then
  echo ">> Installing profiling tools (py-spy, memray)"
  "$PY" -m pip install --quiet py-spy memray || true
fi

if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("lightning") else 1)
PY
then
  echo ">> Installing Lightning + torchmetrics"
  "$PY" -m pip install --quiet lightning torchmetrics
fi

if [ "$CLEAN_TB" = "1" ]; then
  echo ">> Cleaning old TensorBoard event files"
  find /fsx/model_registry -path '*/tensorboard/*' -name 'events.out.tfevents*' -delete || true
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

# Start TensorBoard server if requested.
if [ "$START_TB" = "1" ]; then
  if ! ss -lntp | grep -q ':6006 '; then
    echo ">> Starting TensorBoard on 0.0.0.0:6006"
    "$PY" -m pip install --quiet tensorboard || true
    nohup tensorboard --logdir /fsx/model_registry --bind_all --port 6006 \
      > /fsx/model_registry/tensorboard.log 2>&1 &
  else
    echo ">> TensorBoard already listening on 6006"
  fi
fi

# Build shard list across dataset roots.
export SHARD_ROOTS
export STAGE_SHARDS_PER_DATASET
export STAGE_TOTAL_SHARDS
export STAGE_SEED
"$PY" - <<'PY'
import json
import os
import random
from pathlib import Path

roots = [r.strip() for r in os.environ.get("SHARD_ROOTS", "").split(",") if r.strip()]
per_dataset = int(os.environ.get("STAGE_SHARDS_PER_DATASET", "0"))
total_limit = int(os.environ.get("STAGE_TOTAL_SHARDS", "0"))
seed = int(os.environ.get("STAGE_SEED", "123"))
random.seed(seed)

paths = []
for root in roots:
    root_path = Path(root)
    if not root_path.exists():
        continue
    manifests = list(root_path.rglob("manifest.jsonl"))
    if manifests:
        for manifest in manifests:
            entries = []
            for line in manifest.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                shard = entry.get("shard")
                if not shard:
                    continue
                shard_path = (manifest.parent / shard).resolve()
                entries.append(str(shard_path))
            if per_dataset > 0 and len(entries) > per_dataset:
                random.shuffle(entries)
                entries = entries[:per_dataset]
            paths.extend(entries)
    else:
        entries = [str(p) for p in root_path.rglob("shard_*.pt")]
        if per_dataset > 0 and len(entries) > per_dataset:
            random.shuffle(entries)
            entries = entries[:per_dataset]
        paths.extend(entries)

if total_limit > 0 and len(paths) > total_limit:
    random.shuffle(paths)
    paths = paths[:total_limit]

out = "/tmp/all_shards.txt"
with open(out, "w", encoding="utf-8") as f:
    for p in sorted(set(paths)):
        f.write(p + "\n")

print(f"Wrote {len(set(paths))} shards to {out}")
PY
SHARD_LIST="/tmp/all_shards.txt"
if [ "$SMOKE" = "1" ]; then
  head -n "$SMOKE_SHARDS" /tmp/all_shards.txt > /tmp/smoke_shards.txt
  SHARD_LIST="/tmp/smoke_shards.txt"
  echo ">> SMOKE mode: using $SMOKE_SHARDS shards from $SHARD_LIST"
fi

if [ "$STAGE_SHARDS" = "1" ]; then
  echo ">> Staging shards to ${STAGE_DIR} (limit ${STAGE_LIMIT_GB}GB)"
  mkdir -p "$STAGE_DIR"
  export SRC_LIST="$SHARD_LIST"
  export STAGE_DIR
  export STAGE_LIMIT_GB
  "$PY" - <<'PY'
import os
import random

src_list = os.environ["SRC_LIST"]
stage_dir = os.environ["STAGE_DIR"]
limit_gb = int(os.environ.get("STAGE_LIMIT_GB", "100"))
limit = limit_gb * (1024**3)

paths = [p.strip() for p in open(src_list) if p.strip()]
random.shuffle(paths)
total = 0
selected = []
for p in paths:
    try:
        size = os.path.getsize(p)
    except OSError:
        continue
    if total + size > limit:
        continue
    selected.append(p)
    total += size

out = "/tmp/stage_shards.txt"
with open(out, "w") as f:
    for p in selected:
        f.write(p + "\n")
print(f"Selected {len(selected)} shards, {total/1e9:.2f} GB")
PY
  rsync -a --files-from=/tmp/stage_shards.txt / "$STAGE_DIR"/
  "$PY" - <<'PY'
import os

stage_dir = os.environ["STAGE_DIR"]
in_path = "/tmp/stage_shards.txt"
out_path = "/tmp/staged_shards.txt"
with open(in_path) as f:
    paths = [line.strip() for line in f if line.strip()]
with open(out_path, "w") as f:
    for p in paths:
        staged = os.path.join(stage_dir, p.lstrip("/"))
        f.write(staged + "\n")
print(f"Wrote staged list: {out_path}")
PY
  SHARD_LIST="/tmp/staged_shards.txt"
  echo ">> Using staged shard list: $SHARD_LIST"
fi

SPLIT_KEY="${SPLIT_KEY:-mol_key}"
SPLIT_METHOD="${SPLIT_METHOD:-hash}"
SCAFFOLD_GROUP_KEY="${SCAFFOLD_GROUP_KEY:-mol_key}"
SCAFFOLD_SMILES_KEY="${SCAFFOLD_SMILES_KEY:-smile}"
SCAFFOLD_INCLUDE_CHIRALITY="${SCAFFOLD_INCLUDE_CHIRALITY:-0}"
SCAFFOLD_FALLBACK="${SCAFFOLD_FALLBACK:-molecule}"
SPLIT_TRAIN="${SPLIT_TRAIN:-0.7}"
SPLIT_VAL="${SPLIT_VAL:-0.1}"
LOG_PREDS="0"

if [ "$PREFLIGHT" = "1" ]; then
  export SHARD_LIST
  export TASK
  export SPLIT_KEY
  export SPLIT_METHOD
  export SCAFFOLD_GROUP_KEY
  export SCAFFOLD_SMILES_KEY
  export SCAFFOLD_INCLUDE_CHIRALITY
  export SCAFFOLD_FALLBACK
  export SPLIT_TRAIN
  export SPLIT_VAL
  export SPLIT_SEED
  export PREFLIGHT_SHARDS
  export PREFLIGHT_FULL_SHARDS
  export MIN_SAMPLES
  "$PY" - <<'PY'
import os
import random
from pathlib import Path

import torch
from train.train_detanet import _resolve_split_token, _split_label

shard_list = os.environ["SHARD_LIST"]
split_key = os.environ["SPLIT_KEY"]
split_method = os.environ.get("SPLIT_METHOD", "hash")
scaffold_group_key = os.environ.get("SCAFFOLD_GROUP_KEY", "mol_key")
scaffold_smiles_key = os.environ.get("SCAFFOLD_SMILES_KEY", "smile")
scaffold_include_chirality = os.environ.get("SCAFFOLD_INCLUDE_CHIRALITY", "0") in ("1", "true", "True")
scaffold_fallback = os.environ.get("SCAFFOLD_FALLBACK", "molecule")
split_train = float(os.environ["SPLIT_TRAIN"])
split_val = float(os.environ["SPLIT_VAL"])
split_seed = int(os.environ.get("SPLIT_SEED", "123"))
pref_shards = int(os.environ.get("PREFLIGHT_SHARDS", "5"))
full_limit = int(os.environ.get("PREFLIGHT_FULL_SHARDS", "20"))
min_samples = int(os.environ.get("MIN_SAMPLES", "10"))

paths = [p.strip() for p in Path(shard_list).read_text().splitlines() if p.strip()]
if not paths:
    raise SystemExit("Preflight failed: shard list is empty.")

if len(paths) <= full_limit:
    sample_paths = paths
    sampled = False
else:
    random.seed(split_seed)
    sample_paths = random.sample(paths, k=min(pref_shards, len(paths)))
    sampled = True

counts = {"total": 0, "finite": 0, "train": 0, "val": 0, "test": 0}
task = os.environ.get("TASK", "Hij")
split_cache = {}
for path in sample_paths:
    data = torch.load(path, map_location="cpu", weights_only=False)
    for item in data:
        counts["total"] += 1
        target = getattr(item, task, None)
        if target is None:
            continue
        pos = getattr(item, "pos", None)
        if pos is None:
            continue
        if torch.is_tensor(target) and not torch.isfinite(target).all().item():
            continue
        if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
            continue
        counts["finite"] += 1
        token = _resolve_split_token(
            item,
            split_method=split_method,
            split_key=split_key,
            scaffold_group_key=scaffold_group_key,
            scaffold_smiles_key=scaffold_smiles_key,
            scaffold_include_chirality=scaffold_include_chirality,
            scaffold_fallback=scaffold_fallback,
            split_cache=split_cache,
        )
        if token is None:
            continue
        label = _split_label(token, split_seed, split_train, split_val)
        counts[label] += 1

prefix = "Sampled" if sampled else "Full"
print(f">> PREFLIGHT ({prefix} {len(sample_paths)}/{len(paths)} shards)")
print(f">> counts: total={counts['total']} finite={counts['finite']} train={counts['train']} val={counts['val']} test={counts['test']}")
if counts["train"] < min_samples or counts["val"] < min_samples or counts["test"] < min_samples:
    print(f">> WARNING: sample split below min_samples={min_samples}.")
PY
fi

if [ -n "$FIXED_LR" ] || [ -n "$FIXED_BATCH_SIZE" ] || [ -n "$FIXED_ADALORA_R" ] || [ -n "$FIXED_ADALORA_ALPHA" ]; then
  lr_val="${FIXED_LR:-1e-4}"
  bs_val="${FIXED_BATCH_SIZE:-16}"
  r_val="${FIXED_ADALORA_R:-8}"
  alpha_val="${FIXED_ADALORA_ALPHA:-16}"
  cat > /tmp/param_space.json <<JSON
{
  "lr": {"type":"choice","values":[${lr_val}]},
  "batch_size": {"type":"choice","values":[${bs_val}]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[${r_val}]},
  "adalora_alpha": {"type":"choice","values":[${alpha_val}]}
}
JSON
  PARAM_SPACE_LOCKED=1
elif [ "$SMOKE" = "1" ] && [ "${SMOKE_ITEMS}" -gt 0 ]; then
  echo ">> SMOKE_ITEMS=${SMOKE_ITEMS}: building tiny shard for smoke test"
  export SHARD_LIST
  export SMOKE_ITEMS
  export SMOKE_SPLIT_KEY
  export SMOKE_SPLIT_TRAIN
  export SMOKE_SPLIT_VAL
  export SMOKE_SPLIT_SEED
  export SMOKE_TASK
  "$PY" - <<'PY'
import hashlib
import math
import os
from pathlib import Path

import torch

shard_list = os.environ["SHARD_LIST"]
items = int(os.environ["SMOKE_ITEMS"])
split_key = os.environ["SMOKE_SPLIT_KEY"]
split_train = float(os.environ["SMOKE_SPLIT_TRAIN"])
split_val = float(os.environ["SMOKE_SPLIT_VAL"])
seed = int(os.environ["SMOKE_SPLIT_SEED"])
task = os.environ.get("SMOKE_TASK", "Hij")

paths = [p.strip() for p in Path(shard_list).read_text().splitlines() if p.strip()]
if not paths:
    raise SystemExit("SMOKE_ITEMS requested but shard list is empty.")

def is_finite_item(item) -> bool:
    target = getattr(item, task, None)
    if target is None:
        return False
    pos = getattr(item, "pos", None)
    if pos is None:
        return False
    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
        return False
    if not torch.is_tensor(target) and isinstance(target, (float, int)) and not math.isfinite(target):
        return False
    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
        return False
    for key in item.keys():
        val = item[key]
        if torch.is_tensor(val) and val.dtype.is_floating_point:
            if not torch.isfinite(val).all().item():
                return False
    return True

subset = []
src_used = None
for src in paths:
    data = torch.load(src, map_location="cpu", weights_only=False)
    for item in data:
        if is_finite_item(item):
            subset.append(item)
            if len(subset) >= items:
                src_used = src
                break
    if len(subset) >= items:
        break

if len(subset) < items:
    raise SystemExit(f"Only collected {len(subset)} finite items from shards; need {items}.")

def split_label(key: str) -> str:
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    bucket = int(digest, 16) % 1000
    train_cutoff = int(split_train * 1000)
    val_cutoff = int((split_train + split_val) * 1000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"

train_needed = max(1, int(round(split_train * items)))
val_needed = int(round(split_val * items))
test_needed = items - train_needed - val_needed
labels = ["train"] * train_needed + ["val"] * val_needed + ["test"] * test_needed

def find_key(label: str, start: int = 0) -> str:
    i = start
    while True:
        key = f"smoke_{label}_{i}"
        if split_label(key) == label:
            return key
        i += 1

counts = {"train": 0, "val": 0, "test": 0}
key_index = {"train": 0, "val": 0, "test": 0}
for item, label in zip(subset, labels):
    key = find_key(label, key_index[label])
    key_index[label] += 1
    setattr(item, split_key, key)
    counts[label] += 1

out_path = "/tmp/smoke_subset.pt"
torch.save(subset, out_path)
Path("/tmp/smoke_shards.txt").write_text(out_path + "\n")
print(f"Prepared {out_path} from {src_used}")
print("Split counts:", counts)
PY
  SHARD_LIST="/tmp/smoke_shards.txt"
  SPLIT_KEY="$SMOKE_SPLIT_KEY"
  SPLIT_TRAIN="$SMOKE_SPLIT_TRAIN"
  SPLIT_VAL="$SMOKE_SPLIT_VAL"
  LOG_PREDS="1"
  LOG_TRAIN_PREDS="1"
  if [ -z "$USER_TRAIN_USE_DISTRIBUTED" ]; then
    TRAIN_USE_DISTRIBUTED="0"
  fi
  if [ "$EPOCHS" = "5" ]; then
    EPOCHS=10
  fi
  if [ "$LOG_EVERY" = "10" ]; then
    LOG_EVERY=1
  fi
  echo ">> SMOKE_ITEMS: using shard list $SHARD_LIST split_key=$SPLIT_KEY train=$SPLIT_TRAIN val=$SPLIT_VAL"
fi

# Param space (pt_shampoo only)
if [ "$PARAM_SPACE_LOCKED" = "1" ]; then
  :
elif [ "$SMOKE" = "1" ] && [ "${SMOKE_ITEMS_PER_SHARD}" -gt 0 ]; then
  echo ">> SMOKE_ITEMS_PER_SHARD=${SMOKE_ITEMS_PER_SHARD}: building tiny shards per input shard"
  export SHARD_LIST
  export SMOKE_ITEMS_PER_SHARD
  export SMOKE_SPLIT_KEY
  export SMOKE_SPLIT_TRAIN
  export SMOKE_SPLIT_VAL
  export SMOKE_SPLIT_SEED
  export SMOKE_TASK
  "$PY" - <<'PY'
import hashlib
import math
import os
from pathlib import Path

import torch

shard_list = os.environ["SHARD_LIST"]
items_per_shard = int(os.environ["SMOKE_ITEMS_PER_SHARD"])
split_key = os.environ["SMOKE_SPLIT_KEY"]
split_train = float(os.environ["SMOKE_SPLIT_TRAIN"])
split_val = float(os.environ["SMOKE_SPLIT_VAL"])
seed = int(os.environ["SMOKE_SPLIT_SEED"])
task = os.environ.get("SMOKE_TASK", "Hij")

paths = [p.strip() for p in Path(shard_list).read_text().splitlines() if p.strip()]
if not paths:
    raise SystemExit("SMOKE_ITEMS_PER_SHARD requested but shard list is empty.")

def is_finite_item(item) -> bool:
    target = getattr(item, task, None)
    if target is None:
        return False
    pos = getattr(item, "pos", None)
    if pos is None:
        return False
    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
        return False
    if not torch.is_tensor(target) and isinstance(target, (float, int)) and not math.isfinite(target):
        return False
    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
        return False
    for key in item.keys():
        val = item[key]
        if torch.is_tensor(val) and val.dtype.is_floating_point:
            if not torch.isfinite(val).all().item():
                return False
    return True

def split_label(key: str) -> str:
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    bucket = int(digest, 16) % 1000
    train_cutoff = int(split_train * 1000)
    val_cutoff = int((split_train + split_val) * 1000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"

def find_key(label: str, start: int = 0) -> str:
    i = start
    while True:
        key = f"smoke_{label}_{i}"
        if split_label(key) == label:
            return key
        i += 1

out_paths = []
overall_counts = {"train": 0, "val": 0, "test": 0}
for shard_idx, src in enumerate(paths):
    data = torch.load(src, map_location="cpu", weights_only=False)
    subset = []
    for item in data:
        if is_finite_item(item):
            subset.append(item)
            if len(subset) >= items_per_shard:
                break
    if len(subset) < items_per_shard:
        raise SystemExit(f"Shard {src} only had {len(subset)} finite items; need {items_per_shard}.")

    train_needed = max(1, int(round(split_train * items_per_shard)))
    val_needed = int(round(split_val * items_per_shard))
    test_needed = items_per_shard - train_needed - val_needed
    labels = ["train"] * train_needed + ["val"] * val_needed + ["test"] * test_needed

    counts = {"train": 0, "val": 0, "test": 0}
    key_index = {"train": 0, "val": 0, "test": 0}
    for item, label in zip(subset, labels):
        key = find_key(label, key_index[label])
        key_index[label] += 1
        setattr(item, split_key, key)
        counts[label] += 1
        overall_counts[label] += 1

    out_path = f"/tmp/smoke_subset_{shard_idx}.pt"
    torch.save(subset, out_path)
    out_paths.append(out_path)
    print(f"Prepared {out_path} from {src} split={counts}")

Path("/tmp/smoke_shards.txt").write_text("\n".join(out_paths) + "\n")
print("Overall split counts:", overall_counts)
PY
  SHARD_LIST="/tmp/smoke_shards.txt"
  SPLIT_KEY="$SMOKE_SPLIT_KEY"
  SPLIT_TRAIN="$SMOKE_SPLIT_TRAIN"
  SPLIT_VAL="$SMOKE_SPLIT_VAL"
  LOG_PREDS="1"
  LOG_TRAIN_PREDS="1"
  if [ "$LOG_EVERY" = "10" ]; then
    LOG_EVERY=1
  fi
  echo ">> SMOKE_ITEMS_PER_SHARD: using shard list $SHARD_LIST split_key=$SPLIT_KEY train=$SPLIT_TRAIN val=$SPLIT_VAL"
  cat > /tmp/param_space.json <<'JSON'
{
  "lr": {"type":"choice","values":[1e-4]},
  "batch_size": {"type":"choice","values":[1]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[32]},
  "adalora_alpha": {"type":"choice","values":[64]}
}
JSON
elif [ "$SMOKE" = "1" ] && [ "${SMOKE_ITEMS}" -gt 0 ]; then
  cat > /tmp/param_space.json <<'JSON'
{
  "lr": {"type":"choice","values":[1e-4]},
  "batch_size": {"type":"choice","values":[1]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[8]},
  "adalora_alpha": {"type":"choice","values":[16]}
}
JSON
else
  cat > /tmp/param_space.json <<'JSON'
{
  "lr": {"type":"loguniform","min":1e-5,"max":3e-4},
  "batch_size": {"type":"choice","values":[16,24,32]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[8,16]},
  "adalora_alpha": {"type":"choice","values":[16,32]}
}
JSON
fi

USER_GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-12}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

if [ "$NUM_SAMPLES" -le 1 ] && [ -z "${SCHEDULER_OVERRIDE:-}" ]; then
  SCHEDULER="none"
fi

if [ "$SMOKE" = "1" ]; then
  if [ "${SMOKE_ITEMS}" -gt 0 ]; then
    if [ -z "$USER_GPUS_PER_TRIAL" ]; then
      GPUS_PER_TRIAL=1
    fi
    MAX_CONCURRENT=1
    NUM_SAMPLES=1
  else
    GPUS_PER_TRIAL=8
    MAX_CONCURRENT=1
    NUM_SAMPLES=1
  fi
fi

# Base args (JSON list) -- build after SMOKE updates and GPU sizing
export SHARD_LIST
export TASK
export CHECKPOINT
export NORM_CACHE
export SPLIT_KEY
export SPLIT_METHOD
export SCAFFOLD_GROUP_KEY
export SCAFFOLD_SMILES_KEY
export SCAFFOLD_INCLUDE_CHIRALITY
export SCAFFOLD_FALLBACK
export SPLIT_TRAIN
export SPLIT_VAL
export EPOCHS
export EVAL_EVERY
export EVAL_EVERY_STEPS
export STEP_EVAL_MAX_BATCHES
export STEP_EVAL_INCLUDE_TEST
export EPOCH_EVAL_MAX_BATCHES
export LOG_EVERY
export DDP_TIMEOUT
export NUM_WORKERS
export LOADER_TIMEOUT
export PREFETCH_FACTOR
export PERSISTENT_WORKERS_FLAG
export LOG_PREDS
export LOG_PREDS_MAX
export LOG_TRAIN_PREDS
export TRAIN_USE_DISTRIBUTED
export MIN_SAMPLES
export GPUS_PER_TRIAL

"$PY" - <<'PY'
import json
import os

args = [
    "--task",
    os.environ["TASK"],
    "--shard-list",
    os.environ["SHARD_LIST"],
    "--checkpoint",
    os.environ["CHECKPOINT"],
    "--no-checkpoint-strict",
    "--checkpoint-relax-embeddings",
    "--checkpoint-relax-mismatch",
    "--split-key",
    os.environ["SPLIT_KEY"],
    "--split-method",
    os.environ.get("SPLIT_METHOD", "hash"),
    "--scaffold-group-key",
    os.environ.get("SCAFFOLD_GROUP_KEY", "mol_key"),
    "--scaffold-smiles-key",
    os.environ.get("SCAFFOLD_SMILES_KEY", "smile"),
    "--scaffold-fallback",
    os.environ.get("SCAFFOLD_FALLBACK", "molecule"),
    "--split-train",
    os.environ["SPLIT_TRAIN"],
    "--split-val",
    os.environ["SPLIT_VAL"],
    "--epochs",
    os.environ["EPOCHS"],
    "--eval-every",
    os.environ["EVAL_EVERY"],
    "--eval-every-steps",
    os.environ.get("EVAL_EVERY_STEPS", "0"),
    "--step-eval-max-batches",
    os.environ.get("STEP_EVAL_MAX_BATCHES", "0"),
    "--epoch-eval-max-batches",
    os.environ.get("EPOCH_EVAL_MAX_BATCHES", "0"),
    "--log-every",
    os.environ["LOG_EVERY"],
    "--amp",
    "--grad-clip",
    "1.0",
    "--ddp-find-unused-parameters",
    "--ddp-timeout",
    os.environ["DDP_TIMEOUT"],
    "--use-elora",
    "--use-adalora",
    "--adapter-freeze-base",
    "--no-use-impute-mask",
    "--skip-nonfinite",
    "--skip-bad-batches",
    "--normalize",
    "dataset",
    "--exclude-keys",
    "mol_key,subset,source,smile,field_source,field_generated,field_imputed,field_confidence,conformer_id",
    "--num-workers",
    os.environ["NUM_WORKERS"],
    "--loader-timeout",
    os.environ["LOADER_TIMEOUT"],
    "--prefetch-factor",
    os.environ["PREFETCH_FACTOR"],
    os.environ["PERSISTENT_WORKERS_FLAG"],
    "--tensorboard",
]

if os.environ.get("LOG_PREDS") == "1":
    args += ["--log-preds", "--log-preds-max", os.environ.get("LOG_PREDS_MAX", "5")]
if os.environ.get("LOG_TRAIN_PREDS") == "1":
    args += ["--log-train-preds"]
if os.environ.get("STEP_EVAL_INCLUDE_TEST", "0") in ("1", "true", "True"):
    args += ["--step-eval-include-test"]
if os.environ.get("SCAFFOLD_INCLUDE_CHIRALITY", "0") in ("1", "true", "True"):
    args += ["--scaffold-include-chirality"]
norm_cache = os.environ.get("NORM_CACHE", "")
if not norm_cache:
    task_name = os.environ.get("TASK", "task")
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in task_name)
    norm_cache = f"/fsx/model_registry/norm_cache_{safe}.json"
args += ["--norm-cache", norm_cache]
if os.environ.get("TRAIN_USE_DISTRIBUTED", "1") in ("0", "false", "False"):
    args += ["--no-train-use-distributed"]

with open("/tmp/base_args.json", "w", encoding="utf-8") as f:
    json.dump(args, f)
PY

normalize_bool_env() {
  local name="$1"
  local default="$2"
  local val="${!name-}"
  case "$val" in
    "" ) val="$default" ;;
    0|1 ) ;;
    * )
      echo ">> Warning: ${name}=${val} is invalid; using ${default}."
      val="$default"
      ;;
  esac
  export "${name}=${val}"
}

normalize_bool_env NCCL_ASYNC_ERROR_HANDLING 1
normalize_bool_env NCCL_BLOCKING_WAIT 1
normalize_bool_env TORCH_NCCL_BLOCKING_WAIT 1
normalize_bool_env TORCH_NCCL_ASYNC_ERROR_HANDLING 1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

RUN_CMD=(
  "$PY" "$REPO/train/train_tune.py"
  --registry-dir /fsx/model_registry
  --run-prefix "$RUN_PREFIX"
  --param-space-file /tmp/param_space.json
  --base-args "$(tr -d '\n' </tmp/base_args.json)"
  --num-samples "$NUM_SAMPLES"
  --local-dir /fsx/model_registry/ray_results
  --max-concurrent "$MAX_CONCURRENT"
  --cpus-per-trial "$CPUS_PER_TRIAL"
  --gpus-per-trial "$GPUS_PER_TRIAL"
  --scheduler "$SCHEDULER"
  --max-t "$MAX_T"
)

if [ "$DETACH" = "1" ]; then
  LOG="/fsx/model_registry/${RUN_PREFIX}-nohup.log"
  nohup env WANDB_MODE=offline WANDB_SILENT=true \
    "${RUN_CMD[@]}" > "$LOG" 2>&1 &
  echo ">> Detached PID $! log=$LOG"
  exit 0
fi

WANDB_MODE=offline WANDB_SILENT=true \
"${RUN_CMD[@]}"

echo ">> Metrics live under: /fsx/model_registry/${RUN_PREFIX}-*"
echo ">> TensorBoard: tensorboard --logdir /fsx/model_registry --bind_all --port 6006"
REMOTE
