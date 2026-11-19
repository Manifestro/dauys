#!/usr/bin/env bash
# DY-36 on one GPU machine, after G0 chose dictionary $DICT: checklist -> training (only if every item passes) ->
# C3 -> C1/C2 + listening set; everything to S3 outputs/dauys/dy36/.
#   DICT=A bash scripts/dy36_pod.sh
set -uo pipefail
DICT=${DICT:?}
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy36
R=/data/run.sh
W=/data/dy36
up() { aws s3 cp "$1" "$S3/$2" --only-show-errors; }
if [[ ${SKIP_CHECK:-0} != 1 ]]; then
  $R python scripts/dy36_train.py check --dict $W/dict --name $DICT --out $W/check_$DICT || { echo "CHECKLIST CRASHED"; exit 1; }
fi
up $W/check_$DICT/checklist.md checklist.md
up $W/check_$DICT/checklist.json checklist.json
python3 -c "import json,sys; R=json.load(open('$W/check_$DICT/checklist.json')); sys.exit(0 if all(v['pass'] for v in R.values() if isinstance(v, dict)) else 1)" \
  || { echo "CHECKLIST FAILED: no training"; exit 1; }
echo "=== training $(date -u +%H:%M)"
$R python scripts/dy36_train.py train --dict $W/dict --name $DICT --out $W/run_$DICT ${TRAIN_ARGS:-} || { echo "TRAIN CRASHED"; exit 1; }
for f in best.pt val_history.json metadata.json DONE; do up $W/run_$DICT/$f model/$DICT/$f; done
echo "=== C3 $(date -u +%H:%M)"
$R python scripts/dy36_eval.py c3 --dict $W/dict --name $DICT --ckpt $W/run_$DICT/best.pt --out $W/eval_$DICT
up $W/eval_$DICT/c3.json eval/c3.json
echo "=== C1/C2 $(date -u +%H:%M)"
$R python scripts/dy36_eval.py --dict $W/dict --name $DICT --ckpt $W/run_$DICT/best.pt --out $W/eval_$DICT
aws s3 sync $W/eval_$DICT/ $S3/eval/ --only-show-errors
echo "=== DY36 POD DONE $(date -u +%H:%M)"
