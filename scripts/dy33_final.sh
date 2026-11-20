#!/usr/bin/env bash
# DY-33 track M, final evaluation (run scripts/dy33_setup.sh first): union controls (dy32_eval), late Dself (dy34_eval),
# copy measure + paired preference (dy33_extras), combined summary, listening set for the fixed prompt ids;
# S3 dy33/eval/<round>/ and HF dy33/eval/<round>/.
#   bash scripts/dy33_final.sh [round]        (s-g4 is included when s-g4 and s-g4-x are complete in S3)
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
V=${VENV:-$HOME/venv}
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys
export PATH=$V/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
export DY_S3=$B/dy33 DY_HF=dy33
R=${1:-final}
W=/data/dy32/a
E=$W/eval/$R
IDS="47,60,115,118,133,150,153,188"
mkdir -p $W/gen
for x in b-t10 b-t10-x; do aws s3 sync $B/dy32/gen/$x/ $W/gen/$x/ --only-show-errors; done
aws s3 sync $B/dy33/eval/cache/judge_cache/ $W/judge_cache/ --only-show-errors
G4=""
if [[ $(aws s3 ls $B/dy33/gen/s-g4/ | grep -c '\.wav') -ge 50 && $(aws s3 ls $B/dy33/gen/s-g4-x/ | grep -c '\.wav') -ge 150 ]]; then G4=s-g4; fi
F=s-t10,s-ra08,s-early-t10,s-early-ra08,b-t10,b-ra08${G4:+,$G4}
UF=s-t10,s-ra08,b-t10,b-ra08${G4:+,$G4}
P=s-t10:b-t10,s-ra08:b-ra08${G4:+,s-g4:s-t10}
echo "folders $F; pairs $P"
python "$repo/scripts/dy32_eval.py" --round $R --only "$F" --extra anchor-real=$W/anchor_real || { echo "EVAL FAILED"; exit 1; }
mv $E/summary.json $E/channel_summary.json; mv $E/summary.md $E/channel_summary.md
python "$repo/scripts/dy34_eval.py" late --round $R --folders "$UF" --s3 $B/dy33 || echo "LATE FAILED"
python "$repo/scripts/dy33_extras.py" extras --round $R --folders "$UF,anchor-real" --pairs "$P" || echo "EXTRAS FAILED"
python "$repo/scripts/dy33_extras.py" summary --round $R || echo "SUMMARY FAILED"
[[ -s /data/dy34/eval/listen_ids.json ]] || python "$repo/scripts/dy34_listen.py" select --out /data/dy34/eval/listen_ids.json
python - <<EOF
import json; from pathlib import Path
r = json.loads(Path("/data/dy34/eval/listen_ids.json").read_text())
ids = [int(x) for x in "$IDS".split(",")]
assert sorted(r["ids"]) == sorted(ids), ("listening ids differ from the fixed set", r["ids"], ids)
EOF
[[ $? == 0 ]] && python "$repo/scripts/dy34_listen.py" build --ids /data/dy34/eval/listen_ids.json \
  --folders b-t10,s-t10${G4:+,$G4} --out $E/listen && python "$repo/scripts/dy33_extras.py" mark_listen --round $R || echo "LISTEN FAILED"
for f in ref.json eval_base.json eval_base_x.json eval_controls.json; do
  [[ -s $W/$f ]] && aws s3 cp $W/$f $B/dy33/eval/cache/$f --only-show-errors
done
aws s3 sync $W/judge_cache/ $B/dy33/eval/cache/judge_cache/ --only-show-errors
aws s3 sync $E/ $B/dy33/eval/$R/ --only-show-errors
python - <<EOF || echo "HF FAILED"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); r = "manifestro/dauys-runs"
assert api.repo_info(r).private
api.upload_folder(folder_path="$E", path_in_repo="dy33/eval/$R", repo_id=r)
print("HF dy33/eval/$R")
EOF
echo "DY33 FINAL $R DONE $(date -u +%H:%M) UTC"
