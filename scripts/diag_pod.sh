#!/usr/bin/env bash
# Unit diagnostics (ABX on sWUGGY voices) for DY-14 units vs Sylber / HuBERT features.
#   nohup bash ~/dauys/scripts/diag_pod.sh > ~/diag.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/diag
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
    airon soundfile pyyaml "datasets==3.6.0" numba jiwer accelerate sylber==0.1.4 speech-articulatory-coding
  ~/venv/bin/python -c "import torch, sylber, sparc, jiwer; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__)" || exit 1
  curl -sfL -o $W/sylber_resynthesis.yaml https://raw.githubusercontent.com/Berkeley-Speech-Group/sylber/main/sylber_configs/sylber_resynthesis.yaml || exit 1
  [[ -d ~/ke-zek ]] || git clone -q "https://$GITHUB_API@github.com/Manifestro/kezek-research.git" ~/ke-zek
  git -C ~/ke-zek checkout -q $KZ_COMMIT || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH
SPLIT=$HOME/ke-zek/experiments/K-R1/split_round1.json




if ! done_ data; then
  mkdir -p /data/dy14 /data/bench/zr
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - centroids.pt || exit 1
  command -v aria2c >/dev/null || sudo apt-get install -y -qq aria2 >/dev/null 2>&1
  aria2c -q -x 16 -s 16 -k 64M -d /data/bench -o sLM21.zip https://download.zerospeech.com/datasets/sLM21.dataset.zip || exit 1
  unzip -q /data/bench/sLM21.zip "*lexical/dev/*" -d /data/bench/zr || exit 1
  rm -f /data/bench/sLM21.zip
  [[ -f /data/bench/zr/lexical/dev/gold.csv ]] || { echo "ZR DATA MISSING"; exit 1; }
  mark data
fi
if ! done_ smoke; then
  python "$repo/scripts/units_diag.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --out $W/smoke.json --ids 5 --fit-ids 20 || { echo "SMOKE FAILED"; exit 1; }
  mark smoke
fi
python "$repo/scripts/units_diag.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --out $W/out.json > $W/diag.log 2>&1 || { echo "DIAG FAILED"; tail -20 $W/diag.log; exit 1; }
grep -E "SUMMARY|abx" $W/diag.log
echo "DIAG POD DONE"
