#!/usr/bin/env bash
# DY-33 track M: one portable evaluation round on any machine (run scripts/dy33_setup.sh first).
# Evaluates gen/s-t10, gen/s-ra08 (+ -x twins, + -early) against model b (b-t10 from DY-32, b-ra08 from DY-33) and
# anchor-real, with the union controls, run-length statistics and a small listening set (prompts 00/01/04/05).
#   bash scripts/dy33_round.sh <round> [hf] [folders]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
export DY_S3=$B/dy33 DY_HF=dy33 DY_TABLE=eval/channel
R=${1:?round}; HF=${2:-}
F=${3:-b-t10,b-ra08,s-t10,s-ra08,s-t10-early,s-ra08-early}
W=/data/dy32/a
mkdir -p $W/gen
for x in b-t10 b-t10-x; do aws s3 sync $B/dy32/gen/$x/ $W/gen/$x/ --only-show-errors; done
aws s3 sync $B/dy33/eval/cache/judge_cache/ $W/judge_cache/ --only-show-errors
python "$repo/scripts/dy32_eval.py" --round $R ${HF:+--hf} --only "$F" --extra anchor-real=$W/anchor_real \
  || { echo "EVAL FAILED once, retry in 60 s"; sleep 60; python "$repo/scripts/dy32_eval.py" --round $R ${HF:+--hf} --only "$F" \
       --extra anchor-real=$W/anchor_real || echo "EVAL FAILED"; }
python "$repo/scripts/dy33_runs.py" --round $R --folders "$F" || echo "RUNS FAILED"
python "$repo/scripts/dy32_listen.py" --round $R ${HF:+--hf} --only "$F" --ids 00,01,04,05 --no-resynth || echo "LISTEN FAILED"
for f in ref.json eval_base.json eval_base_x.json eval_controls.json runs_real.json; do
  [[ -s $W/$f ]] && aws s3 cp $W/$f $B/dy33/eval/cache/$f --only-show-errors
done
aws s3 sync $W/judge_cache/ $B/dy33/eval/cache/judge_cache/ --only-show-errors
if [[ -n $HF ]]; then
  python - <<EOF || echo "HF FAILED"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); r = "manifestro/dauys-runs"
assert api.repo_info(r).private
api.upload_folder(folder_path="$W/eval/$R", path_in_repo="dy33/eval/$R", repo_id=r)
print("HF dy33/eval/$R")
EOF
fi
echo "ROUND $R DONE $(date -u +%H:%M) UTC"
