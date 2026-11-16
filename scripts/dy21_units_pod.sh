#!/usr/bin/env bash
# DY-21 data on a fresh Airon AWS instance (g7e.4xlarge): DuplexChat en-16k (~20 000 h of podcast audio, one mono
# stream) to DY-14 units, shard by shard, into S3 outputs/dauys/dc-mono-units/. Clone at the recorded commit, then
#   nohup bash ~/dauys/scripts/dy21_units_pod.sh > ~/dy21u.log 2>&1 &
# setup -> DY-14 centroids -> dry (one shard, checked) -> 8 workers on the GPU, work dirs on the local NVMe -> DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy21u
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
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
    airon soundfile pyyaml jiwer accelerate sylber==0.1.4 speech-articulatory-coding
  ~/venv/bin/python -c "import torch, sylber, sparc, jiwer; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__)" || exit 1
  curl -sfL -o $W/sylber_resynthesis.yaml https://raw.githubusercontent.com/Berkeley-Speech-Group/sylber/main/sylber_configs/sylber_resynthesis.yaml || exit 1
  [[ -d ~/ke-zek ]] || git clone -q "https://$GITHUB_API@github.com/Manifestro/kezek-research.git" ~/ke-zek
  git -C ~/ke-zek checkout -q $KZ_COMMIT || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH
SPLIT=$HOME/ke-zek/experiments/K-R1/split_round1.json



if ! done_ data; then
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - centroids.pt || exit 1
  ls -l /data/dy14/centroids.pt || exit 1
  mark data
fi
NV=/opt/dlami/nvme; mountpoint -q $NV && sudo chown "$USER" $NV || NV=/data
if ! done_ dry; then
  python "$repo/scripts/dy21_mono_units.py" --centroids /data/dy14/centroids.pt --part 0/1 --work $NV/dry \
    --dst s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dc-mono-units-dry --max-shards 1 || exit 1
  python - <<'EOF' || { echo "DRY CHECK FAILED"; exit 1; }
import subprocess, numpy as np
B = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dc-mono-units-dry"
name = subprocess.run(f"aws s3 ls {B}/", shell=True, capture_output=True, text=True).stdout.split()[-1]
subprocess.run(f"aws s3 cp {B}/{name} /tmp/dry.npz --only-show-errors", shell=True, check=True)
z = np.load("/tmp/dry.npz")
tok = z["tok"]
unit = tok < 8192
act = tok != 8193
rate = unit.sum() / max(act.sum() / 10, 1e-6)
print("dry:", len(z["index"]), "pieces,", len(tok) / 36000, "h; units/s speech", round(float(rate), 2), "; speech share", round(float(act.mean()), 3))
assert 3 <= rate <= 7 and act.mean() > 0.5
EOF
  mark dry
fi
if ! done_ units; then
  # NPODS instances share the shards: instance POD runs parts POD*8 .. POD*8+7 of NPODS*8
  NPODS=${NPODS:-1}; POD=${POD:-0}
  for k in 0 1 2 3 4 5 6 7; do
    python "$repo/scripts/dy21_mono_units.py" --centroids /data/dy14/centroids.pt --part $((POD * 8 + k))/$((NPODS * 8)) \
      --sub ${SUB:-0/1} --work $NV/w$k > $W/units_$k.log 2>&1 &
  done
  wait
  for k in 0 1 2 3 4 5 6 7; do grep -q "MONO UNITS DONE" $W/units_$k.log || { echo "part $k failed"; tail -5 $W/units_$k.log; exit 1; }; done
  mark units
fi
echo "DY21 UNITS DONE"
