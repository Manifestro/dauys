#!/usr/bin/env bash
# DY-32 track G, one-voice twin of track B: early round at EARLY (UTC) from best.pt, final round after training,
# both with track B's generator (dy32-b at 5ff2c63 in ~/dauys-b) and the LJ K2000 codebook; checkpoints/logs to HF.
set -uo pipefail
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$HOME/dauys-b/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
set -a; . /etc/airon.env; set +a
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32
R=/data/g/run_blj
gen() {  # $1 out dir, $2 note, $3 checkpoint
  python $HOME/dauys-b/scripts/dy32b_gen.py --ckpt $3 --cb /data/g/cb/km-lj-K2000.npy --pack /data/pack \
    --out $1 --note "$2" 2>&1 | grep -v Warn | tail -n 3
}
step() { python -c "import torch; print(torch.load('$1', map_location='cpu', weights_only=False)['step'])"; }
until [[ $(date -u +%H%M) > ${EARLY:-0425} ]]; do sleep 60; done
if [[ -s $R/best.pt ]]; then
  cp $R/best.pt /data/g/early.pt; n=$(step /data/g/early.pt); rm -rf /data/g/gen_early
  gen /data/g/gen_early "early round, best.pt at step $n (training still running)" /data/g/early.pt
  aws s3 sync /data/g/gen_early/ $S3/gen/g-lj-early/ --only-show-errors
  aws s3 sync /data/g/gen_early/ $S3/gen/g-lj/step$n/ --only-show-errors
  echo "EARLY DONE step $n $(date -u +%H:%M)"
fi
until grep -q "TRAIN DONE\|Traceback" /data/g/train_lj.log; do sleep 60; done
grep -q "TRAIN DONE" /data/g/train_lj.log || { echo "TRAIN FAILED"; tail -n 5 /data/g/train_lj.log; }
n=$(step $R/best.pt); rm -rf /data/g/gen_final
gen /data/g/gen_final "final, best.pt by validation CE at step $n" $R/best.pt
aws s3 sync /data/g/gen_final/ $S3/gen/g-lj/ --only-show-errors
echo "FINAL GEN DONE step $n $(date -u +%H:%M)"
python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private
for src, dst in [("/data/g/run_blj/best.pt", "dy32/g/lj/best.pt"), ("/data/g/run_blj/metadata.json", "dy32/g/lj/metadata.json"),
                 ("/data/g/train_lj.log", "dy32/g/lj/train.log"), ("/data/g/gen_final/info.json", "dy32/g/lj/gen_info.json"),
                 ("/data/g/gen_early/info.json", "dy32/g/lj/gen_early_info.json"), ("/data/channel.log", "dy32/g/gslm/channel.log"),
                 ("/data/released.log", "dy32/g/gslm/released.log"), ("/data/g/g-released/info.json", "dy32/g/gslm/released_info.json"),
                 ("/data/g/g-released/transcripts.json", "dy32/g/gslm/released_transcripts.json")]:
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo)
print("HF UPLOADED dy32/g/")
EOF
echo "ALL DONE $(date -u +%H:%M)"
