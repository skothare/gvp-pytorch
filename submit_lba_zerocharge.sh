#!/bin/bash
#SBATCH --job-name=gvp_lba_zerocharge
#SBATCH --partition=koes_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40"
#SBATCH --time=12:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/zero_charge/gvp_lba_zerocharge_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/zero_charge/gvp_lba_zerocharge_%j.err

GVP_ROOT="/net/galaxy/home/koes/skothare/gvp-pytorch"
ARM="zero_charge"
CKPT_DIR="${GVP_ROOT}/models/${ARM}"
V3_CACHE="/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/v3_lba_precomputed_cache/charge_zero"

mkdir -p "${GVP_ROOT}/logs/${ARM}" "${CKPT_DIR}"
mkdir -p /scr/$USER/job_tmp; export TMPDIR=/scr/$USER/job_tmp
source /net/galaxy/home/koes/skothare/miniconda3/bin/activate gvp
export PYTHONNOUSERSITE=1
cd "${GVP_ROOT}"

echo "GVP LBA ${ARM} — 3 seeds — V3 cache ${V3_CACHE} — working dir $(pwd)"

for SEED in 42 123 7; do
    echo "############ SEED ${SEED} — ${ARM} train ############"
    python -u run_atom3d.py LBA --lba-split 30 --epochs 50 --batch 8 \
        --num-workers 4 --lr 1e-4 --seed $SEED --v3-cache "$V3_CACHE"

    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_best.pt" "${CKPT_DIR}/" 2>/dev/null
    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_last.pt" "${CKPT_DIR}/" 2>/dev/null

    BEST="${CKPT_DIR}/LBA_seed${SEED}_best.pt"
    if [ -f "$BEST" ]; then
        echo "############ SEED ${SEED} — ${ARM} test ############"
        python -u run_atom3d.py LBA --lba-split 30 --batch 8 --num-workers 4 \
            --seed $SEED --test "$BEST" --v3-cache "$V3_CACHE"
    else
        echo "ERROR: $BEST not found; training may have failed for seed $SEED."
    fi
done

echo "############ ${ARM} COMPLETE ############"
echo "Metrics: grep -A5 'LBA Test Metrics' ${GVP_ROOT}/logs/${ARM}/*.out"