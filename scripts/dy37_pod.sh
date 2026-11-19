#!/usr/bin/env bash
# DY-37 t on machine t: training on the given dictionary, then (unless AFTER=0) the evaluation: generations on the
# reply-moment pack first (uploaded at once), then P2a and P1, then uploads of checkpoint, metadata, logs and results.
#   DICT=/opt/dlami/nvme/dict37/dict_X.npz STOP_AT=<unix> nohup bash scripts/dy37_pod.sh > ~/logs/pod37.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy37
U=/opt/dlami/nvme/utt
TABLE=${TABLE:-/opt/dlami/nvme/dict37/table.npz}
KEY=${KEY:-thought}
STEPS=${STEPS:-12257}; STOP_AT=${STOP_AT:-0}; RUN=${RUN:-/data/t/run37}
EXP="DY-37: замысел и речь, перезапуск — участник разговора говорит по выбранному замыслу; замыслы из словаря, который ловит содержание; без текста"
echo "DY37 commit $DY21_COMMIT dict $DICT key $KEY steps $STEPS stop_at $STOP_AT $(date -u +%H:%M)"
echo "=== training start $(date -u +%H:%M)"
python "$repo/scripts/dy37_train.py" --utt $U --dict "$DICT" --table $TABLE --dict-key $KEY --out $RUN \
  --cfg 1024,24,16,4096,2048 --batch 32 --micro 16 --lr 3e-4 --warmup 1000 --val-every 500 --steps $STEPS --seed 1 \
  --stop-at $STOP_AT --name dy37-t --experiment "$EXP"
echo "=== training end $(date -u +%H:%M)"
aws s3 cp $RUN/metadata.json $S3/run/metadata.json --only-show-errors
aws s3 cp $RUN/val_history.json $S3/run/val_history.json --only-show-errors
[[ ${AFTER:-1} == 1 ]] || exit 0
E="python $repo/scripts/dy37_eval.py"
R=/data/t/results37; mkdir -p $R
CK=$RUN/best.pt
echo "=== gen $(date -u +%H:%M)"
$E gen --ckpt $CK --pack /data/t/reply_pack --out /data/t/gen37 --batch 50 || echo "GEN FAILED"
aws s3 sync /data/t/gen37/ $S3/gen/ --only-show-errors && echo "GEN IN S3 $(date -u +%H:%M)"
echo "=== p2a $(date -u +%H:%M)"
$E p2a --ckpt $CK --n 1200 --batch 25 --out $R/p2a.json || echo "P2A FAILED"
echo "=== p1 $(date -u +%H:%M)"
$E p1 --ckpt $CK --n 1200 --batch 50 --out $R/p1.json || echo "P1 FAILED"
aws s3 sync $R/ $S3/eval/ --only-show-errors
echo "=== upload $(date -u +%H:%M)"
for f in best.pt last.pt metadata.json val_history.json; do [[ -f $RUN/$f ]] && aws s3 cp $RUN/$f $S3/t/$f --only-show-errors; done
aws s3 cp ~/logs/pod37.log $S3/t/train.log --only-show-errors
echo "=== done $(date -u +%H:%M)"
