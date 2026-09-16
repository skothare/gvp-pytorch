#!/bin/bash
#SBATCH --job-name=gvp_v8_comparison
#SBATCH --partition=koes_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --exclude=g021
#SBATCH --time=48:00:00
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v8_comparison_%j.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v8_comparison_%j.err
set -euo pipefail
ROOT=/net/galaxy/home/koes/skothare
GVP="$ROOT/gvp-pytorch"
EVAL="$ROOT/Estats/dl_model/evaluate_proteinshake_V2"
CACHE="${V8_CACHE_ROOT:-$EVAL/v8_lba_precomputed_cache}"
PHASE="${EXPERIMENT_PHASE:-production}"
case "$PHASE" in
    production) EPOCHS=50; SEEDS=(42 123 7) ;;
    rehearsal) EPOCHS=1; SEEDS=(42) ;;
    *) echo 'EXPERIMENT_PHASE must be production or rehearsal' >&2; exit 2 ;;
esac
RUN_ID="${RUN_ID:-v8_${PHASE}_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local}}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'RUN_ID must contain only letters, numbers, underscores, or hyphens' >&2
    exit 2
fi
MODEL_ROOT="$GVP/models/$RUN_ID"
LOG_ROOT="$GVP/logs/$RUN_ID"
mkdir -p "$GVP/models" "$GVP/logs"
# mkdir without -p deliberately refuses to reuse any previous run.
mkdir "$MODEL_ROOT" "$LOG_ROOT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="/scr/$USER/v8_${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
source "$ROOT/miniconda3/bin/activate" "${V8_ENV:-estats}"
python -u "$EVAL/check_v8_lba_cache.py" --cache-root "$CACHE" \
    --report "$LOG_ROOT/cache_audit.json"
source "$ROOT/miniconda3/bin/activate" gvp
cd "$GVP"
command -v rg >/dev/null
python -c 'import torch; torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0))'
python summarize_lba_v8.py --initialize --log-root "$LOG_ROOT" \
    --model-root "$MODEL_ROOT" --cache-root "$CACHE" --phase "$PHASE"
for ARM in v8_real v8_zero baseline; do
    mkdir "$MODEL_ROOT/$ARM" "$LOG_ROOT/$ARM"
    EXTRA=()
    case "$ARM" in
        v8_real) EXTRA=(--v8-cache "$CACHE/charge_real") ;;
        v8_zero) EXTRA=(--v8-cache "$CACHE/charge_zero") ;;
    esac
    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_ROOT/$ARM/seed${SEED}.log"
        COMMON=(LBA --lba-split 30 --batch 8 --num-workers 4 --lr 1e-4 \
                --seed "$SEED" --models-dir "$MODEL_ROOT/$ARM" "${EXTRA[@]}")
        COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --epochs "$EPOCHS" --train-time 0 --val-time 0)
        printf 'COMMAND: %q ' "${COMMAND[@]}" >> "$LOG"
        printf '\n' >> "$LOG"
        "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
        if rg -q 'Skipped batch due to OOM|CUDA out of memory' "$LOG"; then
            echo "Rejected OOM run: $LOG" >&2; exit 1
        fi
        BEST="$MODEL_ROOT/$ARM/LBA_seed${SEED}_best.pt"
        test -s "$BEST"
        printf '\nTRAIN_OK\n' >> "$LOG"
        COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --test "$BEST")
        printf 'COMMAND: %q ' "${COMMAND[@]}" >> "$LOG"
        printf '\n' >> "$LOG"
        "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
        printf '\nTEST_OK\n' >> "$LOG"
    done
done
python summarize_lba_v8.py --log-root "$LOG_ROOT"
