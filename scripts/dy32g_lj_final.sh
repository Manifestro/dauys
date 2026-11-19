#!/usr/bin/env bash
# DY-32 track G one-voice twin, after the contract round (dy32g_lj_after.sh): two more samplings of the final best.pt
# (t 1.0 / top-p 1.0), then the extension prompt pack at t 1.0 / top-p 1.0. Track B's generator.
set -uo pipefail
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$HOME/dauys-b/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32
CK=/data/g/run_blj/best.pt
until grep -q "FINAL GEN DONE" /data/g/after.log; do sleep 30; done
run() {  # $1 variant, $2 pack, $3 temp, $4 top-p
  rm -rf /data/g/gen_$1
  python $HOME/dauys-b/scripts/dy32b_gen.py --ckpt $CK --cb /data/g/cb/km-lj-K2000.npy --pack $2 --out /data/g/gen_$1 \
    --temp $3 --top-p $4 --note "final best.pt, variant $1" 2>&1 | grep -v Warn | tail -n 1
  aws s3 sync /data/g/gen_$1/ $S3/gen/$1/ --only-show-errors && echo "UPLOADED $1 $(date -u +%H:%M)"
}
run g-lj-t10 /data/pack 1.0 1.0
mkdir -p /data/pack_x
aws s3 sync $S3/prompts_x/ /data/pack_x/ --only-show-errors
if ls /data/pack_x/prompt_*.wav > /dev/null 2>&1; then
  run g-lj-t10-x /data/pack_x 1.0 1.0
else
  echo "NO EXTENSION PACK"
fi
echo "FINAL EXTRAS DONE $(date -u +%H:%M)"
