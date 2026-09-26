#!/usr/bin/env bash
set -euo pipefail
PYTHON=$1
DRIVER=$2
CONFIG=$3
STAGE=$4
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ "$STAGE" == diagnostics || "$STAGE" == panel_a ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
else
  unset CUBLAS_WORKSPACE_CONFIG
fi
exec "$PYTHON" -u "$DRIVER" --config "$CONFIG" --stage "$STAGE" --execute
