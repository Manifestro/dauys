#!/usr/bin/env bash
# DY-21 training on an Airon AWS instance (g7e.4xlarge) once S3 outputs/dauys/dc-mono-units/ is complete:
#   nohup bash ~/dauys/scripts/dy21_pod.sh > ~/dy21.log 2>&1 &
# setup -> units (S3, to local NVMe), DY-14 unit means, a held-out reference voice -> checklist -> dry -> train ->
# eval (gate W, listening files) -> upload to private HF manifestro/dauys-runs dy21/ -> DY21 DONE.
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
  mkdir -p $U /data/dy14
  aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dc-mono-units/ $U/ --only-show-errors || exit 1
  ls $U/*.npz | wc -l
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - unit_means.pt centroids.pt || exit 1
  curl -sfL -o $W/sylber_resynthesis.yaml https://raw.githubusercontent.com/Berkeley-Speech-Group/sylber/main/sylber_configs/sylber_resynthesis.yaml || exit 1
  python - <<EOF || exit 1
import hashlib, subprocess, numpy as np, soundfile as sf
from pathlib import Path
names = sorted(p.stem for p in Path("$U").glob("*.npz"))
held = [n for n in names if int(hashlib.sha1(n.encode()).hexdigest(), 16) % 50 == 0]
src = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k/" + held[0]
subprocess.run(f"aws s3 cp {src}/index.npy /tmp/ref_index.npy --only-show-errors", shell=True, check=True)
idx = np.load("/tmp/ref_index.npy")
st, n = max(idx, key=lambda r: r[1])
subprocess.run(f"aws s3 cp {src}/segs.bin /tmp/ref_segs.bin --only-show-errors", shell=True, check=True)
x = np.fromfile("/tmp/ref_segs.bin", dtype=np.int16)[st + 60 * 16000 : st + 70 * 16000].astype(np.float32) / 32768
sf.write("$W/ref.wav", x, 16000)
print("ref voice from held-out shard", held[0], "piece at", st, "files", len(names), "held out", len(held))
EOF
  mark data
fi
if ! done_ checklist; then
  python "$repo/scripts/dy21_checklist.py" --units $U || { echo "CHECKLIST FAILED"; exit 1; }
  mark checklist
fi
if ! done_ dry; then
  D=$W/dry; mkdir -p $D
  python "$repo/scripts/dy21_train.py" --units $U --out $D/run --steps 200 --val-every 100 --small --limit 6 --tracking off || exit 1
  python "$repo/scripts/dy21_eval.py" --units $U --run $D/run --means /data/dy14/unit_means.pt --config $W/sylber_resynthesis.yaml \
    --ref $W/ref.wav --out $D/eval --n 4 || exit 1
  mark dry
fi
if ! done_ train; then
  python "$repo/scripts/dy21_train.py" --units $U --out $W/run --steps ${STEPS:-16000} > $W/train.log 2>&1 || { echo "train failed"; tail -5 $W/train.log; exit 1; }
  mark train
fi
if ! done_ eval; then
  python "$repo/scripts/dy21_eval.py" --units $U --run $W/run --means /data/dy14/unit_means.pt --config $W/sylber_resynthesis.yaml \
    --ref $W/ref.wav --out $W/eval || exit 1
  mark eval
fi
if ! done_ upload; then
  python - <<'EOF' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
W = "/data/dy21"
for f in ["eval/eval.json", "train.log", "run/best.pt", "run/metadata.json"]:
    api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy21/{f}", repo_id=repo)
api.upload_folder(folder_path=f"{W}/eval/listen", path_in_repo="dy21/listen", repo_id=repo)
print("UPLOADED dy21/")
EOF
  mark upload
fi
echo "DY21 DONE"
