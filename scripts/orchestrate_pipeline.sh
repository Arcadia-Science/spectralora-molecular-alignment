#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scripts/ec2_env.sh}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

export REPO_DIR PYTHON_BIN DATASETS_ROOT OUTPUT_ROOT CHECKPOINT_PATH
export WORKERS OMP_THREADS MKL_THREADS OPENBLAS_THREADS TORCH_THREADS DP_INTRA_THREADS DP_INTER_THREADS
export SHARD_SIZE SAVE_DEVICE NO_PSI4
export JOB_INDEX JOB_COUNT

export LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/run.log}"

mkdir -p "$OUTPUT_ROOT"

scripts/mount_fsx.sh
scripts/setup_cloudwatch_agent.sh "$ENV_FILE"

NO_PSI4="${NO_PSI4:-1}"
SAVE_DEVICE="${SAVE_DEVICE:-cpu}"

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
  --job-index "$JOB_INDEX"
  --job-count "$JOB_COUNT"
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
