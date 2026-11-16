#!/usr/bin/env bash
# DY-20 on a fresh Airon AWS instance (g7e.4xlarge). Clone this repository at the recorded commit to ~/dauys, then
#   nohup bash ~/dauys/scripts/dy20_pod.sh > ~/dy20.log 2>&1 &
# setup -> DY-14 grid + centroids (private S3) and DY-14 test anchors (private HF) -> DuplexChat 1000 h separated by
# Kezek v1 (S3 outputs/duplexchat/kezek-v1-1000h) to the Open Yap layout -> Sylber features -> units with DY-14's
# centroids -> train = Open Yap train + DuplexChat, val/test = Open Yap -> checklist -> dry -> two seeds of the DY-14
# config from scratch -> test evaluation on DY-14's anchors -> upload JSON + checkpoints to private HF dy20/ -> DY20 DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy20
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY20_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks /data/dy14 /data/dcraw /data/dc
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
  aws configure set default.s3.max_concurrent_requests 64
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - || exit 1
  python - <<'EOF' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("manifestro/dauys-runs", "dy14/test_anchors.json", token=os.environ["HF_TOKEN"]), "/data/dy14/test_anchors.json")
print("artifacts ok")
EOF
  aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/kezek-v1-1000h/ /data/dcraw/ \
    --exclude "*act*" --exclude "CLAIM*" --exclude "DONE*" --only-show-errors || exit 1
  ( cd /data/dcraw && cat MANIFEST.* | grep -E "/(index\.npy|pieces\.jsonl)$" | sha256sum -c --quiet ) \
    || { echo "DC checksum mismatch"; exit 1; }
  cat /data/dcraw/MANIFEST.* > /tmp/dc_manifest.txt
  # the converted pieces go to the local NVMe when there is one (EBS reads at ~125 MB/s)
  NV=/opt/dlami/nvme  # the AMI mounts the instance NVMe here
  if mountpoint -q $NV; then sudo chown "$USER" $NV; rm -rf /data/dc; mkdir -p $NV/dc; ln -sfn $NV/dc /data/dc; fi
  python "$repo/scripts/dy20_dc_prepare.py" --src /data/dcraw --out /data/dc --manifest /tmp/dc_manifest.txt || exit 1
  python - <<EOF || exit 1
import json
split = json.load(open("$SPLIT"))
dc = json.load(open("/data/dc/dc_names.json"))["names"]
assert not set(dc) & set(split["train"] + split["val"] + split["test"])
json.dump({"dc": dc}, open("$W/split_dc.json", "w"))
json.dump({"train": split["train"] + dc, "val": split["val"], "test": split["test"]}, open("$W/split.json", "w"))
print("split: train", len(split["train"]), "+ dc", len(dc), "val", len(split["val"]), "test", len(split["test"]))
EOF
  mark data
fi

if ! done_ features; then
  python "$repo/scripts/dy14_features.py" --root /data/dc --split $W/split_dc.json --parts dc --out $W/seg --workers 16 --shards 8 || exit 1
  mark features
fi

if ! done_ grid; then
  mkdir -p $W/u && cp /data/dy14/centroids.pt $W/u/centroids.pt
  python "$repo/scripts/dy14_grid.py" --seg $W/seg --split $W/split_dc.json --parts dc --out $W/u || exit 1
  mkdir -p $W/grid
  for f in /data/dy14/grid/*.npz; do ln -sf "$f" $W/grid/; done
  for f in $W/u/grid/*.npz; do ln -sf "$f" $W/grid/; done
  ls $W/grid | wc -l
  mark grid
fi

if ! done_ checklist; then
  python - <<EOF || { echo "CHECKLIST FAILED"; exit 1; }
import json, torch, numpy as np
split = json.load(open("$W/split.json"))
anchors = json.load(open("/data/dy14/test_anchors.json"))
ok = set(anchors["names"]) <= set(split["test"]) and not set(split["test"]) & set(split["train"]) and not set(split["val"]) & set(split["train"])
print("CHECK leak", "PASS" if ok else "FAIL")
a = torch.load("$W/u/centroids.pt"); b = torch.load("/data/dy14/centroids.pt")
same = torch.equal(a, b)
print("CHECK same_units", "PASS" if same else "FAIL")
st = json.load(open("$W/u/stats.json"))["dc"]
print("DC stats", json.dumps(st))
sane = 3 <= st["units_per_s_speech"] <= 7
print("CHECK dc_units_rate", "PASS" if sane else "FAIL")
assert ok and same and sane
EOF
  mark checklist
fi

if ! done_ dry; then
  D=$W/dry; mkdir -p $D
  DAUYS_TAG=dy20dry python "$repo/scripts/dy14_train.py" --grid $W/grid --split $W/split.json --out $D/run --seed 0 --steps 100 \
    --val-every 100 --limit 6 --tracking off || exit 1
  mark dry
fi

export DAUYS_EXPERIMENT="DY-20: те же 109M с нуля на Open Yap + 1000 ч DuplexChat (Kezek v1) — растёт ли слух от данных" DAUYS_TAG=dy20
if ! done_ train; then
  for s in 0 1; do
    python "$repo/scripts/dy14_train.py" --grid $W/grid --split $W/split.json --out $W/runs/s$s --seed $s --steps 20000 --stop-at 8000 \
      > $W/train_s$s.log 2>&1 &
  done
  wait
  for s in 0 1; do [[ -e $W/runs/s$s/DONE || -e $W/runs/s$s/STOPPED ]] || { echo "seed $s did not finish"; tail -5 $W/train_s$s.log; exit 1; }; done
  mark train
fi
if ! done_ eval; then
  python "$repo/scripts/dy14_eval.py" --grid $W/grid --split $W/split.json --anchors /data/dy14/test_anchors.json \
    --runs $W/runs/s0,$W/runs/s1 --out $W/result.json || exit 1
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
W = "/data/dy20"
for f in ["result.json", "u/stats.json", "train_s0.log", "train_s1.log", "runs/s0/best.pt", "runs/s1/best.pt",
          "runs/s0/metadata.json", "runs/s1/metadata.json"]:
    if os.path.exists(f"{W}/{f}"):
        api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy20/{f}", repo_id=repo)
print("UPLOADED dy20/")
EOF
  mark upload
fi
echo "DY20 DONE"
