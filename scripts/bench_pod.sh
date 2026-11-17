#!/usr/bin/env bash
# Open spoken-LM benchmarks (sWUGGY, sBLIMP, sSC/tSC) for the DY-21 and DY-22 checkpoints.
#   nohup bash ~/dauys/scripts/bench_pod.sh > ~/bench.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/bench
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export BENCH_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks /data/dy14
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }

if ! done_ setup; then
  sudo apt-get update -qq > /dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv libsndfile1 ffmpeg > /dev/null 2>&1
  python3 -m venv ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    airon soundfile pyyaml "datasets==3.6.0" zerospeech-benchmarks jiwer accelerate sylber==0.1.4 speech-articulatory-coding
  ~/venv/bin/python -c "import torch, sylber, sparc, jiwer; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__)" || exit 1
  curl -sfL -o $W/sylber_resynthesis.yaml https://raw.githubusercontent.com/Berkeley-Speech-Group/sylber/main/sylber_configs/sylber_resynthesis.yaml || exit 1
  [[ -d ~/ke-zek ]] || git clone -q "https://$GITHUB_API@github.com/Manifestro/kezek-research.git" ~/ke-zek
  git -C ~/ke-zek checkout -q $KZ_COMMIT || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH
SPLIT=$HOME/ke-zek/experiments/K-R1/split_round1.json




if ! done_ data; then
  mkdir -p /data/dy14 /data/dy21/run /data/dy22/run
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - centroids.pt || exit 1
  python - <<'EOF2' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
for d in ["dy21", "dy22"]:
    for f in ["run/best.pt", "run/metadata.json"]:
        shutil.copy(hf_hub_download("manifestro/dauys-runs", f"{d}/{f}", token=os.environ["HF_TOKEN"]), f"/data/{d}/{f}")
EOF2
  mkdir -p $W/zr && cd $W/zr
  (zrc datasets:pull sLM21-dataset 2>&1 | tail -3) || true
  ZR=$(dirname "$(dirname "$(find / -path '*lexical/dev/gold.csv' 2>/dev/null | head -1)")")
  [[ -n "$ZR" && -d "$ZR/lexical" ]] || { echo "ZR DATA MISSING"; exit 1; }
  echo "$ZR" > $W/zr_path
  ls "$ZR"; ls "$ZR/lexical/dev" | head -3; head -3 "$ZR/lexical/dev/gold.csv" "$ZR/syntactic/dev/gold.csv"
  cd ~
  mark data
fi
ZR=$(cat $W/zr_path)
A="--zr $ZR --centroids /data/dy14/centroids.pt --dy21 /data/dy21/run/best.pt --dy22 /data/dy22/run/best.pt"
if ! done_ smoke; then
  python "$repo/scripts/bench_slm.py" $A --out $W/smoke.json --pairs 20 --sc-splits tsc_bm || { echo "SMOKE FAILED"; exit 1; }
  mark smoke
fi
if ! done_ bench; then
  python "$repo/scripts/bench_slm.py" $A --out $W/out.json > $W/bench.log 2>&1 || { echo "BENCH FAILED"; tail -20 $W/bench.log; exit 1; }
  grep -vE "^\[" $W/bench.log | tail -12
  mark bench
fi
python - <<'EOF2'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
if api.repo_info("manifestro/dauys-runs").private:
    api.upload_file(path_or_fileobj="/data/bench/out.json", path_in_repo="bench/dy21_dy22.json", repo_id="manifestro/dauys-runs")
    print("UPLOADED bench/")
EOF2
echo "BENCH POD DONE"
