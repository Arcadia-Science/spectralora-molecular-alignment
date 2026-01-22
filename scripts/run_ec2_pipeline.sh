#!/usr/bin/env bash
set -euo pipefail

# Configurable paths (override via env vars)
REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv2/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-/mnt/data/Datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/processed_all}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/checkpoints/DPA2-Drug-v1.pth}"
LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/run.log}"

# Parallelism controls
WORKERS="${WORKERS:-10}"
OMP_THREADS="${OMP_THREADS:-8}"
MKL_THREADS="${MKL_THREADS:-8}"
OPENBLAS_THREADS="${OPENBLAS_THREADS:-8}"
TORCH_THREADS="${TORCH_THREADS:-8}"
DP_INTRA_THREADS="${DP_INTRA_THREADS:-8}"
DP_INTER_THREADS="${DP_INTER_THREADS:-8}"

# Pipeline controls
SHARD_SIZE="${SHARD_SIZE:-256}"
SAVE_DEVICE="${SAVE_DEVICE:-cpu}"
NO_PSI4="${NO_PSI4:-1}"

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
  --dp-intra-threads "$DP_INTRA_THREADS"
  --dp-inter-threads "$DP_INTER_THREADS"
  --shard-size "$SHARD_SIZE"
  --save-device "$SAVE_DEVICE"
  --allow-missing-dipole
  --allow-missing-polar
  --allow-missing-hyperpolar
)

if [[ "$NO_PSI4" == "1" ]]; then
  cmd+=(--no-psi4)
fi

echo "Launching pipeline:"
printf '  %q' "${cmd[@]}"
echo

nohup env PYTHONUNBUFFERED=1 "${cmd[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$OUTPUT_ROOT/pipeline.pid"
echo "PID: $(cat "$OUTPUT_ROOT/pipeline.pid")"
echo "Log: $LOG_FILE"
