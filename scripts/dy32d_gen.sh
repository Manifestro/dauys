#!/usr/bin/env bash
# DY-32 track D: continue a prompt pack from a snapshot of best.pt in both heard modes and upload to S3.
#   bash scripts/dy32d_gen.sh SUFFIX [TEMP TOP_P [PROMPTS_DIR]]
#   bash scripts/dy32d_gen.sh -early            -> gen/d-cont-early/, gen/d-disc-early/ (0.8 / 0.95)
#   bash scripts/dy32d_gen.sh -t06 0.6 0.9      -> gen/d-cont-t06/, gen/d-disc-t06/
# SNAP=dir picks the snapshot folder (default $W/snap<SUFFIX>); KEEP_SNAP=1 reuses an existing snapshot.
# Both modes use the same seeds, so cont and disc continuations are paired.
set -uo pipefail
SUF=${1:-}; TEMP=${2:-0.8}; TOPP=${3:-0.95}
source "$(dirname "$0")/dy32d_env.sh"
PR=${4:-$W/prompts}
CB=$W/acb/km-corpus-K2000.npy
SNAP=${SNAP:-$W/snap${SUF}}; mkdir -p $SNAP
[[ ${KEEP_SNAP:-0} == 1 && -s $SNAP/best.pt ]] || cp $W/run/best.pt $SNAP/best.pt
python "$repo/scripts/dy32d.py" gen --run $SNAP --centroids $CB --metric cos --prompts $PR --out $W/gen --suffix="$SUF" \
  --temp $TEMP --top-p $TOPP \
  --unit-desc "track A's whole-corpus k-means (s3 dy32/a/centroids/km-corpus-K2000.npy), 50 Hz, cosine, no dedup" || exit 1
for m in cont disc; do aws s3 sync $W/gen/d-$m$SUF/ $S3D/gen/d-$m$SUF/ --only-show-errors || exit 1; done
echo "GEN UPLOADED $SUF"
