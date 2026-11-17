#!/usr/bin/env bash
# DY-21b on an Airon AWS instance: sampling settings and the timing check on the saved DY-21 checkpoint.
#   nohup bash ~/dauys/scripts/dy21b_pod.sh > ~/dy21b.log 2>&1 &
# setup -> checkpoint (HF dy21/), held-out units only, unit means, reference voice -> smoke (4 prompts) -> full ->
# upload to private HF manifestro/dauys-runs dy21b/ -> DY21B DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy21
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
mkdir -p $W/marks_b /data/dy14
mark() { touch "$W/marks_b/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks_b/$1" ]]; }

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




U=/data/units
if ! done_ data; then
  mkdir -p $U $W/run
  python - <<'EOF2' || exit 1
import os
from huggingface_hub import hf_hub_download
for f in ["run/best.pt", "run/metadata.json"]:
    p = hf_hub_download("manifestro/dauys-runs", f"dy21/{f}", token=os.environ["HF_TOKEN"])
    os.system(f"cp {p} /data/dy21/{f}")
EOF2
  for n in $(python -c "import json; print(' '.join(json.load(open('$W/run/metadata.json'))['val_files_names']))"); do
    aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dc-mono-units/$n.npz $U/ --only-show-errors || exit 1
  done
  ls $U/*.npz | wc -l
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - unit_means.pt || exit 1
  python - <<EOF2 || exit 1
import json, subprocess, numpy as np, soundfile as sf
held = sorted(json.load(open("$W/run/metadata.json"))["val_files_names"])
src = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k/" + held[0]
subprocess.run(f"aws s3 cp {src}/index.npy /tmp/ref_index.npy --only-show-errors", shell=True, check=True)
idx = np.load("/tmp/ref_index.npy")
st, n = max(idx, key=lambda r: r[1])
subprocess.run(f"aws s3 cp {src}/segs.bin /tmp/ref_segs.bin --only-show-errors", shell=True, check=True)
x = np.fromfile("/tmp/ref_segs.bin", dtype=np.int16)[st + 60 * 16000 : st + 70 * 16000].astype(np.float32) / 32768
sf.write("$W/ref.wav", x, 16000)
print("ref voice from held-out shard", held[0])
EOF2
  mark data
fi
A="--units $U --run $W/run --means /data/dy14/unit_means.pt --config $W/sylber_resynthesis.yaml --ref $W/ref.wav"
if ! done_ smoke; then
  python "$repo/scripts/dy21_sampling.py" $A --out $W/smoke --n 4 || { echo "SMOKE FAILED"; exit 1; }
  mark smoke
fi
if ! done_ full; then
  python "$repo/scripts/dy21_sampling.py" $A --out $W/b > $W/b.log 2>&1 || { echo "FULL FAILED"; tail -20 $W/b.log; exit 1; }
  grep SUMMARY $W/b.log
  mark full
fi
if ! done_ upload; then
  python - <<'EOF2' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
api.upload_file(path_or_fileobj="/data/dy21/b/eval.json", path_in_repo="dy21b/eval.json", repo_id=repo)
api.upload_file(path_or_fileobj="/data/dy21/b.log", path_in_repo="dy21b/b.log", repo_id=repo)
api.upload_folder(folder_path="/data/dy21/b/listen", path_in_repo="dy21b/listen", repo_id=repo)
print("UPLOADED dy21b/")
EOF2
  mark upload
fi
echo "DY21B DONE"
