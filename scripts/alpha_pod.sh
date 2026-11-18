#!/usr/bin/env bash
# DY-24 alphabet quantiser on an Airon AWS GPU instance: venv (virtualenv, no apt) -> ZeroSpeech dev from S3 ->
# pair features from the TTS alphabet data -> train -> eval (gate Q) -> upload HF dy24/.
#   K=256 FILES=8 [W=/data/alpha_x FEAT=microsoft/wavlm-large:12 WAIT_CUTS=1] nohup bash ~/dauys/scripts/alpha_pod.sh > ~/alpha.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
sudo mkdir -p /data && sudo chown "$USER" /data
W=${W:-/data/alpha}; mkdir -p $W/marks
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }
if ! done_ setup; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba sylber==0.1.4 accelerate airon "datasets==3.6.0"
  ~/venv/bin/python -c "import torch, sylber, numba; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))" || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=262144
if ! done_ zr; then
  mkdir -p /data/bench
  aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/bench/zr_dev.tar - --only-show-errors | tar -C /data/bench -xf - || exit 1
  mark zr
fi
if ! done_ feats; then
  python "$repo/scripts/alphabet.py" feats --files ${FILES:-8} --feat ${FEAT:-sylber} ${FEATS_ARGS:-} --out $W/pairs.pt > $W/feats.log 2>&1 || { echo "FEATS FAILED"; tail -20 $W/feats.log; exit 1; }
  tail -2 $W/feats.log
  mark feats
fi
for k in ${K:-256}; do
  if ! done_ train_$k; then
    python "$repo/scripts/alphabet.py" train --pairs $W/pairs.pt --out $W/run$k --k $k --steps ${STEPS:-6000} > $W/train$k.log 2>&1 || { echo "TRAIN $k FAILED"; tail -20 $W/train$k.log; exit 1; }
    tail -2 $W/train$k.log
    mark train_$k
  fi
  if ! done_ eval_$k; then
    for sm in ${SMOOTH:-0}; do
      python "$repo/scripts/alphabet.py" eval --run $W/run$k --zr /data/bench/zr --out $W/eval${k}_s$sm.json --smooth $sm > $W/eval${k}_s$sm.log 2>&1 || { echo "EVAL $k FAILED"; tail -20 $W/eval${k}_s$sm.log; exit 1; }
      grep -E "^EVAL" $W/eval${k}_s$sm.log
    done
    mark eval_$k
  fi
done
if ! done_ words; then
  until [[ -z "${WAIT_CUTS:-}" || -s /data/alpha/words_pick.pkl ]]; do sleep 30; done  # cut words made by another stage
  runs=""; for k in ${K:-256}; do runs="$runs $W/run$k"; done
  python "$repo/scripts/alphabet.py" words --runs $runs --out $W/words.json > $W/words.log 2>&1 || { echo "WORDS FAILED"; tail -20 $W/words.log; exit 1; }
  grep -E "^WORDS" $W/words.log
  mark words
fi
W=$W python - <<'EOF2'
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.repo_info("manifestro/dauys-runs").private
tag = os.environ.get("TAG", "dy24")
w = os.environ["W"]
for f in glob.glob(f"{w}/*.json") + glob.glob(f"{w}/*.log") + glob.glob(f"{w}/run*/alpha.pt"):
    api.upload_file(path_or_fileobj=f, path_in_repo=f"{tag}/" + os.path.relpath(f, w), repo_id="manifestro/dauys-runs")
print("UPLOADED", tag)
EOF2
echo "ALPHA POD DONE"
