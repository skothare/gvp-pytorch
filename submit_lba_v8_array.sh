#!/bin/bash
#SBATCH --job-name=gvp_v8_array
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=g021
#SBATCH --time=12:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v8_train_%A_%a.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v8_train_%A_%a.err
set -euo pipefail

# Slurm executes a spool copy: use the submission working directory, not $0.
GVP="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$GVP"
ROOT="$(dirname "$GVP")"
: "${RUN_ID:?Export a shared RUN_ID initialized with --array}"
: "${SLURM_ARRAY_TASK_ID:?Submit an array task}"
: "${SLURM_JOB_ID:?Missing Slurm job ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-8]$ ]] || { echo 'Task index must be 0 through 8' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 2; }
[[ "${SLURM_RESTART_COUNT:-0}" =~ ^[0-9]+$ ]] || exit 2
ARMS=(v8_real v8_real v8_real v8_zero v8_zero v8_zero baseline baseline baseline)
SEEDS=(42 123 7 42 123 7 42 123 7)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
LOG_ROOT="$GVP/logs/$RUN_ID"

source "$ROOT/miniconda3/bin/activate" gvp
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CONFIG="$(python - "$LOG_ROOT" <<'PY'
import sys
from summarize_lba_v8 import array_config
info = array_config(sys.argv[1])
for key in ('phase', 'cache_root', 'model_root', 'epochs'):
    print(info[key])
PY
)"
mapfile -t SETTINGS <<< "$CONFIG"
PHASE="${SETTINGS[0]}"
CACHE="${SETTINGS[1]}"
MODEL_ROOT="${SETTINGS[2]}"
EPOCHS="${SETTINGS[3]}"
if [[ "$PHASE" == rehearsal && "$SEED" != 42 ]]; then
    echo 'Rehearsal requires --array=0,3,6%2' >&2; exit 2
fi
if [[ "${1:-}" == --describe ]]; then
    echo "task=$SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED phase=$PHASE epochs=$EPOCHS cache=$CACHE"
    exit 0
fi
[[ $# == 0 ]] || { echo 'Only --describe is supported' >&2; exit 2; }
mkdir -p "$LOG_ROOT/locks"
exec 9>"$LOG_ROOT/locks/${ARM}_${SEED}.lock"
flock -n 9 || { echo 'Another writer owns this arm/seed' >&2; exit 1; }
for OUTPUT in "$LOG_ROOT/$ARM/seed$SEED.complete.json" "$LOG_ROOT/$ARM/seed$SEED.log" \
    "$MODEL_ROOT/$ARM/LBA_seed${SEED}_best.pt" "$MODEL_ROOT/$ARM/LBA_seed${SEED}_last.pt"; do
    [[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 1; }
done
ATTEMPT="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_r${SLURM_RESTART_COUNT:-0}"
ATTEMPT_LOG="$LOG_ROOT/attempts/${ARM}_${SEED}/$ATTEMPT"
ATTEMPT_MODEL="$MODEL_ROOT/attempts/${ARM}_${SEED}/$ATTEMPT"
mkdir -p "$(dirname "$ATTEMPT_LOG")" "$(dirname "$ATTEMPT_MODEL")"
mkdir "$ATTEMPT_LOG" "$ATTEMPT_MODEL"
# PyTorch workers create AF_UNIX sockets below tempfile.gettempdir().
# Deep experiment paths exceed Linux's socket-address limit. Keep scratch
# node-local and short; retain experiment logs/checkpoints in the attempt tree.
export TMPDIR
TMPDIR="$(mktemp -d /tmp/gvp-v8.XXXXXXXX)"
LOG="$ATTEMPT_LOG/run.log"
GPU="$(python -c 'import torch; torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0))')"
echo "Task $SLURM_ARRAY_TASK_ID: $ARM seed=$SEED GPU=$GPU phase=$PHASE"
export ARRAY_LOG_ROOT="$LOG_ROOT" ARRAY_ATTEMPT_LOG="$ATTEMPT_LOG" ARRAY_ATTEMPT_MODEL="$ATTEMPT_MODEL"
export ARRAY_ARM="$ARM" ARRAY_SEED="$SEED" ARRAY_GPU="$GPU"
python - <<'PY'
import json, os, platform, sys, time
from pathlib import Path
record = dict(job_id=os.environ['SLURM_JOB_ID'], array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
              task_id=int(os.environ['SLURM_ARRAY_TASK_ID']), node=os.environ.get('SLURMD_NODENAME', platform.node()),
              gpu=os.environ['ARRAY_GPU'], python=sys.executable, started_at=time.time())
Path(os.environ['ARRAY_ATTEMPT_LOG'], 'attempt.json').write_text(json.dumps(record, indent=2) + '\n')
PY
EXTRA=()
case "$ARM" in
    v8_real) EXTRA=(--v8-cache "$CACHE/charge_real") ;;
    v8_zero) EXTRA=(--v8-cache "$CACHE/charge_zero") ;;
esac
COMMON=(LBA --lba-split 30 --batch 8 --num-workers 4 --lr 1e-4 --seed "$SEED"
        --models-dir "$ATTEMPT_MODEL" "${EXTRA[@]}")
COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --epochs "$EPOCHS" --train-time 0 --val-time 0)
printf 'COMMAND: ' >> "$LOG"
printf '%q ' "${COMMAND[@]}" >> "$LOG"
printf '\n' >> "$LOG"
START=$SECONDS
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
export ARRAY_TRAIN_SECONDS=$((SECONDS - START))
if grep -Eq 'Skipped batch due to OOM|CUDA out of memory|Traceback \(most recent call last\)' "$LOG"; then
    echo "Rejected failed/OOM training: $LOG" >&2; exit 1
fi
BEST="$ATTEMPT_MODEL/LBA_seed${SEED}_best.pt"
test -s "$BEST"
printf '\nTRAIN_OK\n' >> "$LOG"
COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --test "$BEST")
printf 'COMMAND: ' >> "$LOG"
printf '%q ' "${COMMAND[@]}" >> "$LOG"
printf '\n' >> "$LOG"
START=$SECONDS
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
export ARRAY_TEST_SECONDS=$((SECONDS - START))
printf '\nTEST_OK\n' >> "$LOG"
python - <<'PY'
import json, os
from pathlib import Path
from summarize_lba_v8 import publish_array_task
attempt = Path(os.environ['ARRAY_ATTEMPT_LOG'])
record = json.loads((attempt / 'attempt.json').read_text())
record.update(train_seconds=int(os.environ['ARRAY_TRAIN_SECONDS']),
              test_seconds=int(os.environ['ARRAY_TEST_SECONDS']))
publish_array_task(os.environ['ARRAY_LOG_ROOT'], os.environ['ARRAY_ARM'], int(os.environ['ARRAY_SEED']),
                   attempt, os.environ['ARRAY_ATTEMPT_MODEL'], record)
PY
echo "TASK_COMPLETE arm=$ARM seed=$SEED"
