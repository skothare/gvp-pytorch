#!/bin/bash
#SBATCH --job-name=gvp_v9
#SBATCH --partition=koes_gpu,dept_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=g021
#SBATCH --time=12:00:00
#SBATCH --array=0-19%2
#SBATCH --output=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v9_train_%A_%a.out
#SBATCH --error=/net/galaxy/home/koes/skothare/gvp-pytorch/logs/v9_train_%A_%a.err
set -euo pipefail
ROOT=/net/galaxy/home/koes/skothare
cd "$ROOT/gvp-pytorch"
source "$ROOT/miniconda3/bin/activate" gvp
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${RUN_ROOT:?Export initialized absolute RUN_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task index}"
CONFIG=$(python lba_v9_workflow.py describe --run-root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID")
mapfile -t SETTINGS <<< "$CONFIG"
ARM=${SETTINGS[0]}; SEED=${SETTINGS[1]}; EPOCHS=${SETTINGS[2]}; CACHE=${SETTINGS[3]}
if [[ "${1:-}" == --describe ]]; then echo "$ARM seed=$SEED epochs=$EPOCHS cache=$CACHE"; exit 0; fi
[[ $# == 0 ]] || exit 2
mkdir -p "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/${ARM}_${SEED}.lock"
flock -n 9 || { echo 'Task already running' >&2; exit 1; }
[[ ! -e "$RUN_ROOT/completed/${ARM}_${SEED}.json" ]] || { echo 'Task already complete' >&2; exit 1; }
ATTEMPT="$RUN_ROOT/attempts/${ARM}_${SEED}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_r${SLURM_RESTART_COUNT:-0}"
mkdir -p "$(dirname "$ATTEMPT")"
mkdir "$ATTEMPT"
mkdir "$ATTEMPT/models"
export TMPDIR
TMPDIR=$(mktemp -d /tmp/gvp-v9.XXXXXXXX)
GPU=$(python -c 'import torch; torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0))')
export V9_ATTEMPT="$ATTEMPT" V9_GPU="$GPU"
python - <<'PY'
import os,json,time,platform
from pathlib import Path
Path(os.environ['V9_ATTEMPT'],'attempt.json').write_text(json.dumps(dict(
 job=os.environ['SLURM_JOB_ID'],array=os.environ.get('SLURM_ARRAY_JOB_ID'),
 task=os.environ['SLURM_ARRAY_TASK_ID'],node=platform.node(),gpu=os.environ['V9_GPU'],started=time.time())))
PY
EXTRA=()
if [[ "$ARM" == v9_real ]]; then EXTRA=(--v9-decoder-cache "$CACHE"); fi
COMMON=(LBA --lba-split 30 --batch 8 --num-workers 4 --lr 1e-4 --seed "$SEED" --models-dir "$ATTEMPT/models" "${EXTRA[@]}")
LOG="$ATTEMPT/run.log"
COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --epochs "$EPOCHS" --train-time 0 --val-time 0)
printf 'COMMAND: %q ' "${COMMAND[@]}" >> "$LOG"; printf '\n' >> "$LOG"
START=$SECONDS
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
if grep -Eq 'Skipped batch due to OOM|CUDA out of memory|Traceback \(most recent call last\)' "$LOG"; then exit 1; fi
BEST="$ATTEMPT/models/LBA_seed${SEED}_best.pt"
test -s "$BEST"
printf '\nTRAIN_OK\n' >> "$LOG"
export V9_TRAIN_SECONDS=$((SECONDS-START))
COMMAND=(python -u run_atom3d.py "${COMMON[@]}" --test "$BEST" --predictions-file "$ATTEMPT/predictions.csv")
printf 'COMMAND: %q ' "${COMMAND[@]}" >> "$LOG"; printf '\n' >> "$LOG"
START=$SECONDS
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
printf '\nTEST_OK\n' >> "$LOG"
export V9_TEST_SECONDS=$((SECONDS-START))
python - <<'PY'
import os,json,time
from pathlib import Path
p=Path(os.environ['V9_ATTEMPT'],'attempt.json'); d=json.loads(p.read_text())
d.update(train_seconds=int(os.environ['V9_TRAIN_SECONDS']),test_seconds=int(os.environ['V9_TEST_SECONDS']),finished=time.time())
p.write_text(json.dumps(d,indent=2))
PY
python lba_v9_workflow.py finish --run-root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID" --attempt "$ATTEMPT"
echo "TASK_COMPLETE arm=$ARM seed=$SEED"
