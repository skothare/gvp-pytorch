#!/bin/bash
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --constraint=L40
#SBATCH --exclude=g021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -euo pipefail
STAGE=${1:?stage}
RUN_ROOT=${2:?absolute run root}
CONDA_ROOT=${3:?conda root}
source "$CONDA_ROOT/bin/activate" gvp
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u "$RUN_ROOT/code/unified_lba_workflow.py" "$STAGE" --run-root "$RUN_ROOT"
