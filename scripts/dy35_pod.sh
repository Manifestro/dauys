#!/usr/bin/env bash
# DY-35 t on one g7e machine: wait for the utterance pass -> checklist items 1, 2, 5 (3, 4 run earlier) -> training
# only if every item passes. STEPS (default 12257) and STOP_AT (unix time, default 0) set the run length.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy35
U=/opt/dlami/nvme/utt
STEPS=${STEPS:-12257}; STOP_AT=${STOP_AT:-0}
EXP="DY-35: мысль о всей следующей реплике — перед её началом модель узнаёт суть реплики среди других реплик того же человека, без текста"
echo "DY35 commit $DY21_COMMIT steps $STEPS stop_at $STOP_AT"
until [[ $(aws s3 ls $S3/utt/ | grep -c '\.npz$') -ge 123 ]]; do sleep 30; done
aws s3 sync $S3/utt/ $U/ --only-show-errors
echo "utt files: $(ls $U/*.npz | wc -l)  $(date -u +%H:%M)"
while pgrep -f "[d]y35_check.py --utt /nonexist" >/dev/null; do sleep 20; done  # items 3, 4 write the same json
python "$repo/scripts/dy35_check.py" --utt $U --out /data/t/check --only 1,2,5 || { echo "CHECKLIST CRASHED"; exit 1; }
aws s3 cp /data/t/check/checklist.md $S3/checklist.md --only-show-errors
aws s3 cp /data/t/check/checklist.json $S3/checklist.json --only-show-errors
python -c "import json,sys; R=json.load(open('/data/t/check/checklist.json')); sys.exit(0 if len(R)==5 and all(v.get('pass') for v in R.values()) else 1)" \
  || { echo "CHECKLIST FAILED: training not started"; exit 1; }
echo "=== training start $(date -u +%H:%M)"
python "$repo/scripts/dy35_train.py" --utt $U --out /data/t/run --cfg 1024,24,16,4096,2048 --batch 32 --micro 16 \
  --lr 3e-4 --warmup 1000 --val-every 500 --steps $STEPS --seed 1 --stop-at $STOP_AT --name dy35-t --experiment "$EXP"
echo "=== training end $(date -u +%H:%M)"
