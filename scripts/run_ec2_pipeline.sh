#!/usr/bin/env bash
set -euo pipefail

# Configurable paths (override via env vars)
REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv2/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-/mnt/data/Datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/processed_all}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/checkpoints/DPA2-Drug-v1.pth}"
DEEPMD_DIPOLE_MODEL="${DEEPMD_DIPOLE_MODEL:-}"
DEEPMD_POLAR_MODEL="${DEEPMD_POLAR_MODEL:-}"
DEEPMD_HEAD="${DEEPMD_HEAD:-}"
DEEPMD_TYPE_MAP="${DEEPMD_TYPE_MAP:-}"
LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/run.log}"

# Parallelism controls
WORKERS="${WORKERS:-10}"
OMP_THREADS="${OMP_THREADS:-8}"
MKL_THREADS="${MKL_THREADS:-8}"
OPENBLAS_THREADS="${OPENBLAS_THREADS:-8}"
TORCH_THREADS="${TORCH_THREADS:-8}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
DP_INTRA_THREADS="${DP_INTRA_THREADS:-8}"
DP_INTER_THREADS="${DP_INTER_THREADS:-8}"

# Pipeline controls
SHARD_SIZE="${SHARD_SIZE:-256}"
SAVE_DEVICE="${SAVE_DEVICE:-cpu}"
NO_PSI4="${NO_PSI4:-1}"
JOB_INDEX="${JOB_INDEX:-}"
JOB_COUNT="${JOB_COUNT:-}"
DP_BACKEND="${DP_BACKEND:-pytorch}"

mkdir -p "$OUTPUT_ROOT"

cmd=(
  "$PYTHON_BIN" "$REPO_DIR/data-gen-pipeline/process_datasets.py"
  --datasets-root "$DATASETS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --deepmd-pot-model "$CHECKPOINT_PATH"
  --workers "$WORKERS"
  --omp-threads "$OMP_THREADS"
  --mkl-threads "$MKL_THREADS"
  --openblas-threads "$OPENBLAS_THREADS"
  --torch-threads "$TORCH_THREADS"
  --torch-interop-threads "$TORCH_INTEROP_THREADS"
  --dp-intra-threads "$DP_INTRA_THREADS"
  --dp-inter-threads "$DP_INTER_THREADS"
  --shard-size "$SHARD_SIZE"
  --save-device "$SAVE_DEVICE"
  --allow-missing-dipole
  --allow-missing-polar
  --allow-missing-hyperpolar
)

if [[ -n "$DEEPMD_DIPOLE_MODEL" ]]; then
  cmd+=(--deepmd-dipole-model "$DEEPMD_DIPOLE_MODEL")
fi
if [[ -n "$DEEPMD_POLAR_MODEL" ]]; then
  cmd+=(--deepmd-polar-model "$DEEPMD_POLAR_MODEL")
fi
if [[ -n "$DEEPMD_HEAD" ]]; then
  cmd+=(--deepmd-head "$DEEPMD_HEAD")
fi
if [[ -n "$DEEPMD_TYPE_MAP" ]]; then
  cmd+=(--deepmd-type-map "$DEEPMD_TYPE_MAP")
fi
if [[ -n "$JOB_INDEX" && -n "$JOB_COUNT" ]]; then
  cmd+=(--job-index "$JOB_INDEX" --job-count "$JOB_COUNT")
fi

if [[ "$NO_PSI4" == "1" ]]; then
  cmd+=(--no-psi4)
fi

echo "Launching pipeline:"
printf '  %q' "${cmd[@]}"
echo

nohup env PYTHONUNBUFFERED=1 DP_BACKEND="$DP_BACKEND" "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$OUTPUT_ROOT/pipeline.pid"
echo "PID: $(cat "$OUTPUT_ROOT/pipeline.pid")"
echo "Log: $LOG_FILE"
