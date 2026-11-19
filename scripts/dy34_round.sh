#!/usr/bin/env bash
# DY-34 track M: test evaluation of the guided folder b-g<w> (+ -x) next to unguided b-t10 (+ -x) and anchor-real;
# union controls (dy32_eval), late Dself (dy34_eval late), listening set prompts 00/01/04/05.
#   bash scripts/dy34_round.sh <w> [round]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
export DY_S3=$B/dy34 DY_HF=dy34 DY_TABLE=eval/channel
w=${1:?w}; R=${2:-final}
F=b-g$w,b-t10
W=/data/dy32/a
for x in b-t10 b-t10-x; do aws s3 sync $B/dy32/gen/$x/ $W/gen/$x/ --only-show-errors; done
python "$repo/scripts/dy32_eval.py" --round $R --hf --only "$F" --extra anchor-real=$W/anchor_real || { echo "EVAL FAILED"; exit 1; }
python "$repo/scripts/dy34_eval.py" late --round $R --folders "$F" || echo "LATE FAILED"
[[ -s /data/dy34/eval/listen_ids.json ]] || python "$repo/scripts/dy34_listen.py" select --out /data/dy34/eval/listen_ids.json
python "$repo/scripts/dy34_listen.py" build --ids /data/dy34/eval/listen_ids.json --folders b-t10,b-g$w \
  --labels "unguided,guided w=$w" --out $W/eval/$R/listen && aws s3 sync $W/eval/$R/listen/ $DY_S3/eval/$R/listen/ --only-show-errors \
  || echo "LISTEN FAILED"
python - <<EOF || echo "HF FAILED"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); r = "manifestro/dauys-runs"
assert api.repo_info(r).private
api.upload_folder(folder_path="$W/eval/$R", path_in_repo="dy34/eval/$R", repo_id=r)
api.upload_folder(folder_path="/data/dy34/eval", path_in_repo="dy34/eval", repo_id=r)
print("HF dy34/eval")
EOF
echo "DY34 ROUND $R DONE $(date -u +%H:%M) UTC"
