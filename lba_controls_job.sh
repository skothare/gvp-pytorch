#!/bin/bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: lba_controls_job.sh STAGE RUN_ROOT' >&2; exit 2; }
STAGE=$1
RUN_ROOT=$2
case "$STAGE" in
  train|smoke_gvp) ENV_NAME=gvp ;;
  smoke|cache_cpu|cache_random|audit|report) ENV_NAME=estats ;;
  *) echo "Unsupported stage: $STAGE" >&2; exit 2 ;;
esac
GVP_ROOT=$(pwd -P)
source "${EP20_CONDA_ROOT:-$GVP_ROOT/../miniconda3}/bin/activate" "$ENV_NAME"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR
TMPDIR=$(mktemp -d /tmp/lba-controls.XXXXXXXX)
if [[ "$STAGE" == cache_cpu || "$STAGE" == audit || "$STAGE" == report ]]; then
  export CUDA_VISIBLE_DEVICES=''
fi
exec python -u lba_controls.py "$STAGE" --run-root "$RUN_ROOT"
