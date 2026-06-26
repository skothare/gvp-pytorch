#!/bin/bash
# =============================================================================
# submit_gvp_lba_baseline.sh
#
# Train GVP-GNN on ATOM3D LBA (30% sequence identity split) from scratch.
# Three independent seeds — establishes mean ± std baseline for V3 study.
#
# This is the CONTROL GROUP. Lock in results before any V3 injection.
# Do not retrain after V3 injection begins.
#
# Output per seed:
#   GVP_ROOT/models/LBA_seed{N}_best.pt    — best val-loss checkpoint
#   GVP_ROOT/models/LBA_seed{N}_last.pt    — last epoch checkpoint (resume)
#   GVP_ROOT/logs/gvp_lba_baseline_%j.out  — training log + test metrics
#
# Test metrics reported (via regression_metrics from estats):
#   RMSE, Pearson R, Spearman R, R²
#
# Reference (ATOM3D paper Table 8, GNN, LBA-30%):
#   RMSE: 1.601 ± 0.048
#
# Usage (location-agnostic — can sbatch from any directory):
#   sbatch /net/galaxy/home/koes/skothare/gvp-pytorch/submit_gvp_lba_baseline.sh
# =============================================================================

#SBATCH --job-name=gvp_lba_baseline
#SBATCH --partition=koes_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40"
#SBATCH --time=12:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/gvp_lba_baseline_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/gvp_lba_baseline_%j.err

# --------------------------------------------------------------------------
# Absolute paths — script works correctly regardless of where sbatch is called
# --------------------------------------------------------------------------
GVP_ROOT="/net/galaxy/home/koes/skothare/gvp-pytorch"
LBA_DATA="/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/data/atom3D/split-by-sequence-identity-30"

mkdir -p "${GVP_ROOT}/logs"
mkdir -p "${GVP_ROOT}/models"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
source /net/galaxy/home/koes/skothare/miniconda3/bin/activate gvp

mkdir -p /scr/$USER/job_tmp
export TMPDIR=/scr/$USER/job_tmp
export PYTHONNOUSERSITE=1
export LBA_DATA="${LBA_DATA}"

cd "${GVP_ROOT}"

echo "========================================================"
echo "GVP-GNN LBA Baseline — 3 seeds"
echo "Dataset : PDBBind v2019 refined, 30% sequence identity"
echo "Metrics : RMSE, Pearson R, Spearman R, R²"
echo "Epochs  : 50  |  Batch: 8  |  LR: 1e-4"
echo "Reference: ATOM3D GNN RMSE 1.601 ± 0.048 (Table 8)"
echo "Working dir: $(pwd)"
echo "========================================================"

# --------------------------------------------------------------------------
# Seed loop — three independent runs back to back in one job
# --------------------------------------------------------------------------
for SEED in 42 123 7; do

    echo ""
    echo "###############################################################"
    echo "SEED ${SEED} — training"
    echo "###############################################################"

    python -u run_atom3d.py LBA \
        --lba-split 30 \
        --epochs 50 \
        --batch 8 \
        --num-workers 4 \
        --lr 1e-4 \
        --seed $SEED

    BEST_CKPT="${GVP_ROOT}/models/LBA_seed${SEED}_best.pt"

    if [ ! -f "$BEST_CKPT" ]; then
        echo "ERROR: $BEST_CKPT not found. Training may have failed for seed $SEED."
        echo "Skipping test evaluation for this seed."
        continue
    fi

    echo ""
    echo "###############################################################"
    echo "SEED ${SEED} — test evaluation"
    echo "###############################################################"

    python -u run_atom3d.py LBA \
        --lba-split 30 \
        --batch 8 \
        --num-workers 4 \
        --seed $SEED \
        --test "$BEST_CKPT"

    echo "Seed ${SEED} complete."
    echo ""

done

echo "========================================================"
echo "All 3 seeds complete."
echo ""
echo "Extract all test metrics with:"
echo "  grep -A4 'LBA Test Metrics' ${GVP_ROOT}/logs/gvp_lba_baseline_*.out"
echo ""
echo "Checkpoints saved to:"
echo "  ${GVP_ROOT}/models/"
ls -lh "${GVP_ROOT}/models/" 2>/dev/null
echo "========================================================"