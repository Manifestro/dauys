#!/usr/bin/env bash
# DY-16 on a fresh Airon AWS instance (g7e.4xlarge). Clone this repository at the recorded commit to ~/dauys, then
#   nohup bash ~/dauys/scripts/dy16_pod.sh > ~/dy16.log 2>&1 &
# setup -> data (Open Yap from the ke-zek S3 ready set; DY-14 anchors, centroids, checkpoints from private HF) ->
# features + grid for all parts (+ unit means; the grid goes to private S3 for later experiments) -> moments ->
# checklist -> dry run -> two fine-tune seeds -> generation + judge for both -> voice (gate V, listening files) ->
# upload JSON and listening files to private HF manifestro/dauys-runs dy16/ -> DY16 DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy16
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY16_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks /data/dy14/runs/s0 /data/dy14/runs/s1
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
  S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/kezek-stage1-ready
  mkdir -p /data/oy16 /data/.got
  aws s3 cp "$S3/MANIFEST" /data/manifest_stage1.txt --only-show-errors || exit 1
  one() {
    local name=$1 want
    [[ -e /data/.got/$name ]] && return 0
    want=$(awk -v n="$name" '$2 == n {print $1}' /data/manifest_stage1.txt)
    aws s3 cp "$S3/$name" - --only-show-errors | tee >(sha256sum | cut -d" " -f1 > "/tmp/$name.sum") | tar -C /data/oy16 -xf -
    sleep 3
    [[ "$(cat "/tmp/$name.sum")" == "$want" ]] || { echo "checksum mismatch $name"; return 1; }
    touch "/data/.got/$name"; echo "got $name"
  }
  export -f one; export S3
  awk '$2 ~ /^oy16-/ {print $2}' /data/manifest_stage1.txt | xargs -P 6 -n 1 bash -c 'one "$0"' || { echo "download failed"; exit 1; }
  for d in /data/oy16/conv_conv_*; do ln -sfn "$d" "/data/oy16/${d#/data/oy16/conv_}"; done
  python - <<'EOF' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
for f, dst in [("dy14/test_anchors.json", "/data/dy14/test_anchors.json"), ("dy14/centroids.pt", "/data/dy14/centroids.pt"),
               ("dy14/s0/best.pt", "/data/dy14/runs/s0/best.pt"), ("dy14/s1/best.pt", "/data/dy14/runs/s1/best.pt")]:
    shutil.copy(hf_hub_download("manifestro/dauys-runs", f, token=os.environ["HF_TOKEN"]), dst)
print("HF artifacts ok")
EOF
  mark data
fi

if ! done_ grid; then
  python "$repo/scripts/dy14_features.py" --root /data/oy16 --split "$SPLIT" --out /data/dy14/seg --workers 14 || exit 1
  python "$repo/scripts/dy14_grid.py" --seg /data/dy14/seg --split "$SPLIT" --out /data/dy14 || exit 1
  (cd /data/dy14 && tar cf - grid stats.json unit_means.pt centroids.pt) | aws s3 cp - "$S3OUT/dy14-grid.tar" --only-show-errors \
    && echo "grid saved to $S3OUT/dy14-grid.tar" || echo "WARNING: grid not saved to S3"
  mark grid
fi

if ! done_ moments; then
  python - <<EOF || exit 1
import json, sys
sys.path.insert(0, "$repo/scripts")
from dy16_generate import moments
from dauys.dy14_data import load_convs
names = json.load(open("/data/dy14/test_anchors.json"))["names"]
items = moments(load_convs("/data/dy14/grid", names, 8192), 200)
json.dump(items, open("$W/moments.json", "w"))
print("moments", len(items))
EOF
  sha256sum $W/moments.json
  mark moments
fi

if ! done_ checklist; then
  python "$repo/scripts/dy16_checklist.py" --moments $W/moments.json --anchors /data/dy14/test_anchors.json --split "$SPLIT" \
    --init /data/dy14/runs/s1/best.pt --judge /data/dy14/runs/s0/best.pt || { echo "CHECKLIST FAILED"; exit 1; }
  mark checklist
fi

if ! done_ dry; then
  D=$W/dry; mkdir -p $D
  python "$repo/scripts/dy16_train.py" --grid /data/dy14/grid --split "$SPLIT" --init /data/dy14/runs/s1/best.pt \
    --out $D/run --seed 0 --steps 50 --val-every 50 --limit 6 --tracking off || exit 1
  python -c "import json; json.dump(json.load(open('$W/moments.json'))[:10], open('$D/moments.json', 'w'))"
  python "$repo/scripts/dy16_generate.py" --grid /data/dy14/grid --anchors /data/dy14/test_anchors.json --model $D/run/best.pt \
    --judge /data/dy14/runs/s0/best.pt --moments $D/moments.json --out $D/gen.json || exit 1
  python "$repo/scripts/dy16_voice.py" --oy /data/oy16 --anchors /data/dy14/test_anchors.json --moments $D/moments.json \
    --tokens $D/gen.tokens.npy --centroids /data/dy14/centroids.pt --means /data/dy14/unit_means.pt \
    --config $W/sylber_resynthesis.yaml --out $D/voice --n 3 || exit 1
  mark dry
fi

if ! done_ train; then
  for s in 0 1; do
    python "$repo/scripts/dy16_train.py" --grid /data/dy14/grid --split "$SPLIT" --init /data/dy14/runs/s1/best.pt \
      --out $W/runs/f$s --seed $s --workers 6 > $W/train_f$s.log 2>&1 &
  done
  wait
  for s in 0 1; do [[ -e $W/runs/f$s/DONE ]] || { echo "fine-tune $s failed"; tail -5 $W/train_f$s.log; exit 1; }; done
  mark train
fi
if ! done_ generate; then
  for s in 0 1; do
    python "$repo/scripts/dy16_generate.py" --grid /data/dy14/grid --anchors /data/dy14/test_anchors.json \
      --model $W/runs/f$s/best.pt --judge /data/dy14/runs/s0/best.pt --moments $W/moments.json --out $W/gen_f$s.json || exit 1
  done
  mark generate
fi
if ! done_ voice; then
  python "$repo/scripts/dy16_voice.py" --oy /data/oy16 --anchors /data/dy14/test_anchors.json --moments $W/moments.json \
    --tokens $W/gen_f0.tokens.npy --centroids /data/dy14/centroids.pt --means /data/dy14/unit_means.pt \
    --config $W/sylber_resynthesis.yaml --out $W/voice || exit 1
  mark voice
fi
if ! done_ upload; then
  python - <<'EOF' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
W = "/data/dy16"
for f in ["moments.json", "gen_f0.json", "gen_f1.json", "gen_f0.tokens.npy", "gen_f1.tokens.npy", "voice/voice.json",
          "train_f0.log", "train_f1.log", "runs/f0/best.pt", "runs/f1/best.pt", "runs/f0/metadata.json", "runs/f1/metadata.json"]:
    api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy16/{f}", repo_id=repo)
api.upload_folder(folder_path=f"{W}/voice/listen", path_in_repo="dy16/listen", repo_id=repo)
print("UPLOADED dy16/")
EOF
  mark upload
fi
echo "DY16 DONE"
