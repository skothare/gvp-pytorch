#!/bin/bash
#SBATCH --job-name=gvp_lba_baseline
#SBATCH --partition=koes_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40"
#SBATCH --time=02:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/baseline_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/baseline_%j.err

mkdir -p /net/galaxy/home/koes/skothare/gvp-pytorch/logs

# --- scratch space (matches your existing scripts) ---
mkdir -p /scr/$USER/job_tmp
export TMPDIR=/scr/$USER/job_tmp

# --- environment (exact pattern from your submit_atom3d_eval.sh) ---
source /net/galaxy/home/koes/skothare/miniconda3/bin/activate gvp
export PYTHONNOUSERSITE=1
export LBA_DATA="/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/data/atom3D/split-by-sequence-identity-30"

# --- run ---
cd /net/galaxy/home/koes/skothare/gvp-pytorch

echo "========================================"
echo "GVP LBA Baseline — Phase 0 green run"
echo "1 epoch only — just verifying pipeline"
echo "========================================"

python -u run_atom3d.py LBA \
    --lba-split 30 \
    --epochs 1 \
    --batch 8 \
    --num-workers 4

echo "========================================"
echo "Done. Check test metric above."
echo "========================================"
