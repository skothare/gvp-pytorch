#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONNOUSERSITE=1
exec "${LBA_DRIVER_PYTHON:-python}" "$HERE/pipeline.py" "$@"
