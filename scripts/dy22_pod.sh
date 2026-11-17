#!/usr/bin/env bash
# DY-22 on an Airon AWS instance (g7e): live speech — grid LM with timing, pauses and prosody, warm-started from DY-21.
#   nohup bash ~/dauys/scripts/dy22_pod.sh > ~/dy22.log 2>&1 &
# setup -> units + DY-21 checkpoint + means + ref voice -> checklist -> dry -> train -> eval (gate L) -> upload HF dy22/ -> DY22 DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy22
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY22_COMMIT=$(git -C "$repo" rev-parse HEAD)
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




NV=/opt/dlami/nvme; mountpoint -q $NV && sudo chown "$USER" $NV || NV=/data
U=$NV/units
if ! done_ data; then
  aws configure set default.s3.max_concurrent_requests 64
  mkdir -p $U /data/dy14 /data/dy21/run
  aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dc-mono-units/ $U/ --only-show-errors || exit 1
  ls $U/*.npz | wc -l
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - unit_means.pt || exit 1
  python - <<'EOF2' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
for f in ["run/best.pt", "run/metadata.json"]:
    shutil.copy(hf_hub_download("manifestro/dauys-runs", f"dy21/{f}", token=os.environ["HF_TOKEN"]), f"/data/dy21/{f}")
EOF2
  python - <<EOF2 || exit 1
import json, subprocess, numpy as np, soundfile as sf
held = sorted(json.load(open("/data/dy21/run/metadata.json"))["val_files_names"])
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
INIT=/data/dy21/run/best.pt
E="--means /data/dy14/unit_means.pt --config $W/sylber_resynthesis.yaml --ref $W/ref.wav"
if ! done_ checklist; then
  python "$repo/scripts/dy22_checklist.py" --units $U --init $INIT || { echo "CHECKLIST FAILED"; exit 1; }
  mark checklist
fi
if ! done_ dry; then
  D=$W/dry; mkdir -p $D
  python "$repo/scripts/dy22_train.py" --units $U --out $D/run --steps 100 --val-every 50 --small --limit 6 --tracking off || { echo "DRY FAILED"; exit 1; }
  python "$repo/scripts/dy22_eval.py" --units $U --run $D/run $E --out $D/eval --n 4 || { echo "DRY FAILED"; exit 1; }
  mark dry
fi
if ! done_ train; then
  python "$repo/scripts/dy22_train.py" --units $U --init $INIT --out $W/run --steps ${STEPS:-8000} > $W/train.log 2>&1 || { echo "TRAIN FAILED"; tail -5 $W/train.log; exit 1; }
  mark train
fi
if ! done_ eval; then
  python "$repo/scripts/dy22_eval.py" --units $U --run $W/run $E --out $W/eval > $W/eval.log 2>&1 || { echo "EVAL FAILED"; tail -20 $W/eval.log; exit 1; }
  grep SUMMARY $W/eval.log
  mark eval
fi
if ! done_ upload; then
  python - <<'EOF2' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
W = "/data/dy22"
for f in ["eval/eval.json", "eval.log", "train.log", "run/best.pt", "run/metadata.json"]:
    api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy22/{f}", repo_id=repo)
api.upload_folder(folder_path=f"{W}/eval/listen", path_in_repo="dy22/listen", repo_id=repo)
print("UPLOADED dy22/")
EOF2
  mark upload
fi
echo "DY22 DONE"
