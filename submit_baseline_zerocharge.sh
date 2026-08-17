#!/bin/bash
#SBATCH --job-name=lba_both_arms
#SBATCH --partition=dept_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/both_arms_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/both_arms_%j.err

GVP_ROOT="/net/galaxy/home/koes/skothare/gvp-pytorch"
V3_CACHE="/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/v3_lba_precomputed_cache/charge_zero"

mkdir -p "${GVP_ROOT}/logs" "${GVP_ROOT}/models/baseline" "${GVP_ROOT}/models/zero_charge"
mkdir -p /scr/$USER/job_tmp; export TMPDIR=/scr/$USER/job_tmp
source /net/galaxy/home/koes/skothare/miniconda3/bin/activate gvp
export PYTHONNOUSERSITE=1
cd "${GVP_ROOT}"

run_one () {   # $1=seed  $2=arm  $3=extra flags
    local SEED=$1 ARM=$2 EXTRA=$3
    echo "######## SEED ${SEED} — ${ARM} train ########"
    python -u run_atom3d.py LBA --lba-split 30 --epochs 50 --batch 8 \
        --num-workers 4 --lr 1e-4 --seed $SEED $EXTRA
    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_best.pt" "${GVP_ROOT}/models/${ARM}/" 2>/dev/null
    mv -f "${GVP_ROOT}/models/LBA_seed${SEED}_last.pt" "${GVP_ROOT}/models/${ARM}/" 2>/dev/null
    local BEST="${GVP_ROOT}/models/${ARM}/LBA_seed${SEED}_best.pt"
    if [ -f "$BEST" ]; then
        echo "######## SEED ${SEED} — ${ARM} test ########"
        python -u run_atom3d.py LBA --lba-split 30 --batch 8 --num-workers 4 \
            --seed $SEED --test "$BEST" $EXTRA
    else
        echo "ERROR: $BEST missing; ${ARM} seed ${SEED} failed."
    fi
}

for SEED in 42 123 7; do
    run_one $SEED baseline     ""
    run_one $SEED zero_charge  "--v3-cache $V3_CACHE"
done

echo "######## ALL RUNS COMPLETE ########"