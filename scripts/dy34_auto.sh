#!/usr/bin/env bash
# DY-34 track M: wait for the four dev folders (100 wavs each), run the dev selection, then wait for the chosen
# test folders b-g<w> (50) and b-g<w>-x (150) and run the test round.   nohup bash scripts/dy34_auto.sh &
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
G=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy34/gen
count() { aws s3 ls $G/$1/ 2>/dev/null | grep -c '\.wav$'; }
until [[ $(count b-g1-dev) -ge 100 && $(count b-g2-dev) -ge 100 && $(count b-g4-dev) -ge 100 && $(count b-g8-dev) -ge 100 ]]; do sleep 60; done
echo "dev folders complete $(date -u +%H:%M)"; sleep 30
python "$repo/scripts/dy34_eval.py" dev || { echo "DEV FAILED"; exit 1; }
w=$(python -c "import json; print(json.load(open('/data/dy34/eval/chosen_w.json'))['w'])")
echo "chosen w=$w"
until [[ $(count b-g$w) -ge 50 && $(count b-g$w-x) -ge 150 ]]; do sleep 60; done
echo "test folders complete $(date -u +%H:%M)"; sleep 30
bash "$repo/scripts/dy34_round.sh" $w final
echo "DY34 AUTO DONE"
