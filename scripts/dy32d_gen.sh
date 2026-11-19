#!/usr/bin/env bash
# DY-32 track D: continue the prompt pack from a snapshot of best.pt in both heard modes and upload to S3.
#   bash scripts/dy32d_gen.sh -early     -> gen/d-cont-early/, gen/d-disc-early/
#   bash scripts/dy32d_gen.sh ""         -> gen/d-cont/, gen/d-disc/
set -uo pipefail
SUF=${1:-}
source "$(dirname "$0")/dy32d_env.sh"
CB=$W/acb/km-corpus-K2000.npy
SNAP=$W/snap${SUF}; mkdir -p $SNAP
[[ ${KEEP_SNAP:-0} == 1 && -s $SNAP/best.pt ]] || cp $W/run/best.pt $SNAP/best.pt
python "$repo/scripts/dy32d.py" gen --run $SNAP --centroids $CB --metric cos --prompts $W/prompts --out $W/gen --suffix="$SUF" \
  --unit-desc "track A's whole-corpus k-means (s3 dy32/a/centroids/km-corpus-K2000.npy), 50 Hz, cosine, no dedup" || exit 1
for m in cont disc; do aws s3 sync $W/gen/d-$m$SUF/ $S3D/gen/d-$m$SUF/ --only-show-errors || exit 1; done
echo "GEN UPLOADED $SUF"
