#!/usr/bin/env bash
# DY-36 d after training: dictionary D for every utterance -> S3 dy36/dict/D/, checks (ii)/(iii), judge-free measure.
#   CKPT=/data/dy36/d/run/best.pt bash scripts/dy36_d_after.sh
set -uo pipefail
CKPT=${CKPT:-/data/dy36/d/run/best.pt}
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy36
R=/data/run.sh
W=/data/dy36
for f in best.pt last.pt val_history.json metadata.json; do aws s3 cp $W/d/run/$f $S3/d/run/$f --only-show-errors; done
$R python scripts/dy36_d_train.py dict --dict $W/dict --ckpt $CKPT --name D || { echo "DICT FAILED"; exit 1; }
aws s3 cp $W/dict/dict_D.npz $S3/dict/D/dict_D.npz --only-show-errors
aws s3 cp $W/dict/dict_Draw.npz $S3/dict/D/dict_Draw.npz --only-show-errors
echo "=== DICT D IN S3 $(date -u +%H:%M)"
$R python scripts/dy36_d_train.py checks --dict $W/dict --ckpt $CKPT --out $W/d/checks
aws s3 sync $W/d/checks/ $S3/d/checks/ --only-show-errors
$R python scripts/dy36_d_measure.py --dict $W/dict --dicts D,A,C2500 --out $W/d/measure
aws s3 sync $W/d/measure/ $S3/d/measure/ --only-show-errors
echo "=== D AFTER DONE $(date -u +%H:%M)"
