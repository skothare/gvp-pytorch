#!/bin/bash
#SBATCH --job-name=gvp_lba_realcharge
#SBATCH --partition=koes_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40"
#SBATCH --exclude=g021
#SBATCH --time=12:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/real_charge/gvp_lba_realcharge_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/real_charge/gvp_lba_realcharge_%j.err

# =============================================================================
# Option C (real_charge) GVP arm. Structurally IDENTICAL to
# submit_lba_zerocharge.sh -- only V3_CACHE and ARM differ. Same seeds, epochs,
# batch, lr, num-workers as Option A, so the A-vs-C contrast isolates exactly
# one variable: the charge/radius streams entering V3's frozen encoder.
# =============================================================================

GVP_ROOT="/net/galaxy/home/koes/skothare/gvp-pytorch"
ARM="real_charge"
CKPT_DIR="${GVP_ROOT}/models/${ARM}"
V3_CACHE="/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/v3_lba_precomputed_cache/charge_real"

mkdir -p "${GVP_ROOT}/logs/${ARM}" "${CKPT_DIR}"
mkdir -p /scr/$USER/job_tmp; export TMPDIR=/scr/$USER/job_tmp
source /net/galaxy/home/koes/skothare/miniconda3/bin/activate gvp
export PYTHONNOUSERSITE=1
cd "${GVP_ROOT}"

# --- preflight: fail before burning queue time on a bad cache or dead GPU ---
python -c "import torch; torch.zeros(1).cuda(); print('GPU OK:', torch.cuda.get_device_name(0))" || exit 1
for S in train val test; do
    N=$(find "$V3_CACHE/$S" -name '*.pt' 2>/dev/null | wc -l)
    echo "cache $S: $N .pt files"
    [ "$N" -eq 0 ] && { echo "ERROR: empty cache at $V3_CACHE/$S"; exit 1; }
done

echo "GVP LBA ${ARM} — 3 seeds — V3 cache ${V3_CACHE} — working dir $(pwd)"

for SEED in 42 123 7; do
    # Clear any stale FLAT checkpoint from a previous arm/run before training.
    # train() writes to models/LBA_seed${SEED}_best.pt, and the mv below is
    # unconditional -- so if this seed's training crashed, a leftover file from
    # an earlier arm would be moved into real_charge/ and tested as if it were
    # Option C's result. Silent, and exactly the failure class this project
    # has spent weeks eliminating.
    rm -f "${GVP_ROOT}/models/LBA_seed${SEED}_best.pt" \
          "${GVP_ROOT}/models/LBA_seed${SEED}_last.pt"

    echo "############ SEED ${SEED} — ${ARM} train ############"
    python -u run_atom3d.py LBA --lba-split 30 --epochs 50 --batch 8 \
        --num-workers 4 --lr 1e-4 --seed $SEED --v3-cache "$V3_CACHE"

    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_best.pt" "${CKPT_DIR}/" 2>/dev/null
    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_last.pt" "${CKPT_DIR}/" 2>/dev/null

    BEST="${CKPT_DIR}/LBA_seed${SEED}_best.pt"
    if [ -f "$BEST" ]; then
        echo "############ SEED ${SEED} — ${ARM} test ############"
        # --v3-cache MUST stay here: without it the model rebuilds at
        # node-scalar width 9 instead of 137 and the checkpoint will not load.
        python -u run_atom3d.py LBA --lba-split 30 --batch 8 --num-workers 4 \
            --seed $SEED --test "$BEST" --v3-cache "$V3_CACHE"
    else
        echo "ERROR: $BEST not found; training may have failed for seed $SEED."
    fi
done

echo "############ ${ARM} COMPLETE ############"
echo "Metrics: grep -A5 'LBA Test Metrics' ${GVP_ROOT}/logs/${ARM}/*.out"