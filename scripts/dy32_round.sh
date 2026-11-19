#!/usr/bin/env bash
# DY-32 track A, one round: measure new resynth/<name>/ folders of other tracks (channel table), evaluate every
# complete gen/<name>/ folder, build the listening set.   bash scripts/dy32_round.sh early|final [hf]
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
R=${1:-early}; HF=${2:-}
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32
X=/data/dy32/a/ext_resynth
mkdir -p $X
aws s3 sync $S3/resynth/ $X/ --only-show-errors
dirs=(); for d in $X/*/; do [[ -d $d ]] && dirs+=("trk-$(basename $d)=$d"); done
if (( ${#dirs[@]} )); then
  python "$repo/scripts/dy32_channel.py" measure --dirs "${dirs[@]}" || echo "RESYNTH MEASURE FAILED"
fi
python "$repo/scripts/dy32_eval.py" --round $R ${HF:+--hf} --extra anchor-real=/data/dy32/a/anchor_real \
  || { echo "EVAL FAILED once (retry in 60 s; GPU is shared)"; sleep 60; \
       python "$repo/scripts/dy32_eval.py" --round $R ${HF:+--hf} --extra anchor-real=/data/dy32/a/anchor_real || echo "EVAL FAILED"; }
python "$repo/scripts/dy32_listen.py" --round $R ${HF:+--hf} || echo "LISTEN FAILED"
echo "ROUND $R DONE $(date -u +%H:%M) UTC"
