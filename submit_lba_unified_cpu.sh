#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
RUN_ROOT=${1:?absolute run root}
CONDA_ROOT=${2:?conda root}
source "$CONDA_ROOT/bin/activate" estats
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u "$RUN_ROOT/code/unified_lba_workflow.py" report --run-root "$RUN_ROOT"
