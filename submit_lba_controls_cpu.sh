#!/bin/bash
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the gvp-pytorch repository}"
exec bash ./lba_controls_job.sh "$@"
