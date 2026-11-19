#!/usr/bin/env bash
# DY-38 on one GPU machine: split -> features -> reference words (text only for measuring) -> segmentation -> G1 ->
# training-free vectors -> checklist -> dry run -> training -> dictionary -> G2/G3 + listening; results to S3.
#   bash scripts/dy38_pod.sh            (FROM=<stage> to resume at a stage)
set -uo pipefail
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy38
R=/data/run.sh
W=/data/dy38
PY="$R python scripts/dy38_words.py"
FROM=${FROM:-select}
mkdir -p $W
up() { aws s3 cp "$1" "$S3/$2" --only-show-errors; }
stages=(select feats ref seg g1 base check dry train final)
go=0
for st in "${stages[@]}"; do
  [[ $st == "$FROM" ]] && go=1
  [[ $go == 0 ]] && continue
  echo "=== $st $(date -u +%H:%M)"
  case $st in
    select)
      [[ -f /data/dy36/dict/table.npz ]] || aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy36/dict/table.npz /data/dy36/dict/table.npz --only-show-errors
      $PY select || exit 1; up $W/split.npz split.npz ;;
    feats)
      for sp in train dev test; do $PY feats --split $sp --batch_sec ${BATCH_SEC:-600} || { echo "FEATS $sp FAILED"; exit 1; }; done
      up $W/pca.npz pca.npz ;;
    ref)
      for sp in dev test; do $PY ref --split $sp || { echo "REF $sp FAILED"; exit 1; }; up $W/$sp/ref.json $sp/ref.json; done ;;
    seg) $PY seg || exit 1; up $W/seg_params.json seg_params.json ;;
    g1)
      $PY g1 || exit 1; up $W/g1.json g1.json
      python3 -c "import json,sys; sys.exit(0 if json.load(open('$W/g1.json'))['decision']['G1_pass'] else 1)" \
        || { echo "G1 FAILED: stop"; exit 1; } ;;
    base) $PY base || exit 1 ;;
    check)
      $PY check || { echo "CHECKLIST CRASHED"; exit 1; }
      up $W/checklist.json checklist.json
      python3 -c "import json,sys; sys.exit(0 if json.load(open('$W/checklist.json'))['all_pass'] else 1)" \
        || { echo "CHECKLIST FAILED: no training"; exit 1; } ;;
    dry)
      $PY train --name dry --rounds 1 --steps 200 --eval_every 100 --dry || { echo "DRY TRAIN FAILED"; exit 1; }
      $PY dict --name dry && $PY eval --name dry || { echo "DRY EVAL FAILED"; exit 1; }
      up $W/dry/eval.json dry/eval.json ;;
    train)
      $PY train --name run || { echo "TRAIN CRASHED"; exit 1; }
      for f in best.pt last.pt val_history.json metadata.json DONE; do up $W/run/$f run/$f; done ;;
    final)
      $PY dict --name run || { echo "DICT FAILED"; exit 1; }
      $PY eval --name run || { echo "EVAL FAILED"; exit 1; }
      up $W/run/dict.npz run/dict.npz
      up $W/run/eval.json run/eval.json
      up $W/run/test_tokens.npz run/test_tokens.npz
      aws s3 sync $W/run/listen/ $S3/run/listen/ --only-show-errors
      for sp in train dev test; do for f in seg_W.npy seg_S.npy seg_U.npy rows.npy off.npy; do up $W/$sp/$f $sp/$f; done; done ;;
  esac
done
echo "=== DY38 POD DONE $(date -u +%H:%M)"
