#!/bin/bash
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=g021
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the gvp-pytorch repository}"
exec bash ./ep20_job.sh "$@"
