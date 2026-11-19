#!/usr/bin/env bash
# DY-32 track G: our 422M UnitLM (dy21_lm, from scratch) on a released-GSLM unit set.
#   CH=km100 DEDUP=0 STEPS=n STOP="2026-11-19 07:15" bash scripts/dy32g_train.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
CH=${CH:-km100}; DEDUP=${DEDUP:-0}; STEPS=${STEPS:-12000}; CTX=${CTX:-2048}
K=$([[ $CH == km100 ]] && echo 100 || echo 500)
U=/opt/dlami/nvme/units-$CH
mkdir -p $U
aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/g/units-$CH/ $U/ --only-show-errors || exit 1
echo "units: $(ls $U/*.npz | wc -l) shards"
EXTRA=""; [[ $DEDUP == 1 ]] && EXTRA="--dedup"
python "$repo/scripts/dy21_train.py" --units $U --out /data/run-$CH --steps $STEPS --k $K $EXTRA \
  --cfg 1024,24,16,4096,$CTX --batch 64 --micro ${MICRO:-16} --val-every 500 --save-every 500 \
  --stop-at $(date -d "${STOP:-2026-11-19 07:15}Z" +%s) ${TRACK:-} \
  --experiment "DY-32: канал, через который слова выживают, без текста — три идеи и повтор GSLM параллельными ветками" \
  --name dy32-g
