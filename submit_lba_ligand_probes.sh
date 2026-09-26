#!/bin/bash
#SBATCH --job-name=lba_ligand_ridge
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/lba_ligand_%A_%a.out
#SBATCH --error=logs/lba_ligand_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from gvp-pytorch}"
[[ $# == 3 ]] || { echo "Usage: SCRIPT LIGAND_DATA RESULTS_ROOT MODE" >&2; exit 2; }
case "$3" in morgan|maccs|maccs_morgan) ;; *) exit 2 ;; esac
case "${SLURM_ARRAY_TASK_ID:?Submit as an array}" in
  0) EXTRA=() ;;
  1) EXTRA=(--with-ep) ;;
  *) exit 2 ;;
esac
GVP_ROOT=$(pwd -P)
source "${EP20_CONDA_ROOT:-$GVP_ROOT/../miniconda3}/bin/activate" estats
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u lba_ligand_probes.py run --data-root "$1" --results-root "$2" --mode "$3" "${EXTRA[@]}"
