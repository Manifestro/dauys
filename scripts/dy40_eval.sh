#!/usr/bin/env bash
# DY-40 final evaluation (DY-33's pipeline, run scripts/dy33_setup.sh first): union controls (dy32_eval), late Dself
# (dy34_eval), copy measure + paired preference and the summary with the pre-registered threshold (dy33_extras),
# listening set for the fixed prompt ids; S3 dy40/eval/<round>/ and HF dy40/eval/<round>/.
#   bash scripts/dy40_eval.sh [round]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
[[ -f /etc/airon.env ]] && { set -a; . /etc/airon.env; set +a; }
V=${VENV:-$HOME/venv}
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys
export PATH=$V/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
DD=${DY_DIR:-dy40}
export DY_S3=$B/$DD DY_HF=$DD DY_TITLE=${DY_TITLE:-DY-40}
export DY_DECLARED=${DY_DECLARED:-'{"S1 (temperature 1.0)": ["m-t10", "b-t10"], "S2 (repeat-aware sharpening)": ["m-ra08", "b-ra08"]}'}
export DY_MODELS="${DY_MODELS_PREFIX:-Model m = DY-40} depth LM on Mimi 8 x 2048 (12.5 Hz, acoustic codes one frame after code 0, loss weight 10:1); model b = DY-32 track B (WavLM L6 K2000, 50 Hz);"
R=${1:-final}
W=/data/dy32/a
E=$W/eval/$R
IDS="47,60,115,118,133,150,153,188"
aws s3 sync $B/dy33/eval/cache/judge_cache/ $W/judge_cache/ --only-show-errors
F=${DY_FOLDERS:-m-t10,m-ra08,b-t10,b-ra08}
P=${DY_PAIRS:-m-t10:b-t10,m-ra08:b-ra08}
python "$repo/scripts/dy32_eval.py" --round $R --only "$F" --extra anchor-real=$W/anchor_real || { echo "EVAL FAILED"; exit 1; }
mv $E/summary.json $E/channel_summary.json; mv $E/summary.md $E/channel_summary.md
python "$repo/scripts/dy34_eval.py" late --round $R --folders "$F" --s3 $B/$DD || echo "LATE FAILED"
python "$repo/scripts/dy33_extras.py" extras --round $R --folders "$F,anchor-real" --pairs "$P" || echo "EXTRAS FAILED"
python "$repo/scripts/dy33_extras.py" summary --round $R || echo "SUMMARY FAILED"
[[ -s /data/dy34/eval/listen_ids.json ]] || python "$repo/scripts/dy34_listen.py" select --out /data/dy34/eval/listen_ids.json
python "$repo/scripts/dy34_listen.py" build --ids /data/dy34/eval/listen_ids.json --folders ${DY_LISTEN:-b-t10,m-t10,m-ra08} --out $E/listen || echo "LISTEN FAILED"
aws s3 sync $W/judge_cache/ $B/dy33/eval/cache/judge_cache/ --only-show-errors
aws s3 sync $E/ $B/$DD/eval/$R/ --only-show-errors
python - <<EOF || echo "HF FAILED"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); r = "manifestro/dauys-runs"
assert api.repo_info(r).private
api.upload_folder(folder_path="$E", path_in_repo="$DD/eval/$R", repo_id=r)
print("HF $DD/eval/$R")
EOF
cat $E/summary.md 2>/dev/null | head -30
echo "${DY_TITLE:-DY-40} EVAL $R DONE $(date -u +%H:%M) UTC"
