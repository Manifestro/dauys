#!/usr/bin/env bash
# DY-15 on a fresh Airon AWS instance (g7e.4xlarge). Clone this repository at the recorded commit to ~/dauys, then
#   nohup bash ~/dauys/scripts/dy15_pod.sh > ~/dy15.log 2>&1 &
# setup -> data (Open Yap test tracks from the ke-zek S3 ready set; DY-14 anchors, centroids and checkpoints from the
# private HF repo) -> DY-14 test grid -> checks (channels, anchor hash, known answer, 20-anchor dry run) -> Moshi on
# 2000 onset anchors -> DY-14 on the same anchors -> upload JSON to HF manifestro/dauys-runs dy15/ -> DY15 DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy15
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
ANCH_SHA=79dd25fff1a203b0340f87ff931791efae432f98ca86b0559f90ed248921b688
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks /data/dy14/runs/s0 /data/dy14/runs/s1
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }

if ! done_ setup; then
  sudo apt-get update -qq > /dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv > /dev/null 2>&1
  python3 -m venv ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  echo "torch==2.7.1" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors airon "moshi>=0.2.5,<0.3"
  ~/venv/bin/python -c "import torch, moshi; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__, 'moshi', moshi.__version__)" || exit 1
  [[ -d ~/ke-zek ]] || git clone -q "https://$GITHUB_API@github.com/Manifestro/kezek-research.git" ~/ke-zek
  git -C ~/ke-zek checkout -q $KZ_COMMIT || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH
SPLIT=$HOME/ke-zek/experiments/K-R1/split_round1.json

if ! done_ data; then
  S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/kezek-stage1-ready
  mkdir -p /data/oy16
  aws s3 cp "$S3/MANIFEST" /data/manifest_stage1.txt --only-show-errors || exit 1
  want=$(awk '$2 == "oy16-test.tar" {print $1}' /data/manifest_stage1.txt)
  aws s3 cp "$S3/oy16-test.tar" - --only-show-errors | tee >(sha256sum | cut -d" " -f1 > /tmp/test.sum) | tar -C /data/oy16 -xf -
  sleep 3
  [[ "$(cat /tmp/test.sum)" == "$want" ]] || { echo "checksum mismatch oy16-test.tar"; exit 1; }
  for d in /data/oy16/conv_conv_*; do ln -sfn "$d" "/data/oy16/${d#/data/oy16/conv_}"; done
  python - <<'EOF' || exit 1
import os
from huggingface_hub import hf_hub_download
for f, dst in [("dy14/test_anchors.json", "/data/dy14/test_anchors.json"), ("dy14/centroids.pt", "/data/dy14/centroids.pt"),
               ("dy14/s0/best.pt", "/data/dy14/runs/s0/best.pt"), ("dy14/s1/best.pt", "/data/dy14/runs/s1/best.pt")]:
    p = hf_hub_download("manifestro/dauys-runs", f, token=os.environ["HF_TOKEN"])
    os.system(f"cp {p} {dst}")
print("HF artifacts ok")
EOF
  mark data
fi

if ! done_ grid; then
  python "$repo/scripts/dy14_features.py" --root /data/oy16 --split "$SPLIT" --out /data/dy14/seg --parts test || exit 1
  python "$repo/scripts/dy14_grid.py" --seg /data/dy14/seg --split "$SPLIT" --out /data/dy14 --parts test || exit 1
  mark grid
fi

if ! done_ checks; then
  python - <<EOF || { echo "CHECK anchors FAIL"; exit 1; }
import hashlib, json
a = json.load(open("/data/dy14/test_anchors.json"))
blob = json.dumps({"names": a["names"], "items": a["items"]}, sort_keys=True).encode()
assert hashlib.sha256(blob).hexdigest() == "$ANCH_SHA", "anchor hash differs from DY-14"
split = json.load(open("$SPLIT"))
assert set(a["names"]) <= set(split["test"])
print("CHECK anchors PASS", len(a["items"]))
EOF
  python "$repo/scripts/dy15_moshi.py" --anchors /data/dy14/test_anchors.json --oy /data/oy16 --out $W --known 50 \
    --tracking off | tee $W/known.log || exit 1
  python - <<'EOF' || { echo "CHECK known_answer FAIL"; exit 1; }
import json
r = json.load(open("/data/dy15/known_result.json"))["summary"]
print("CHECK known_answer", "PASS" if r["acc_onset"] > 0.5 else "FAIL", round(r["acc_onset"], 3))
assert r["acc_onset"] > 0.5
EOF
  python "$repo/scripts/dy15_moshi.py" --anchors /data/dy14/test_anchors.json --oy /data/oy16 --out $W/dry --n 20 \
    --tracking off || { echo "CHECK dry FAIL"; exit 1; }
  mark checks
fi

if ! done_ moshi; then
  python "$repo/scripts/dy15_moshi.py" --anchors /data/dy14/test_anchors.json --oy /data/oy16 --out $W --n 2000 || exit 1
  mark moshi
fi
if ! done_ dy14; then
  python "$repo/scripts/dy15_dy14.py" --grid /data/dy14/grid --anchors /data/dy14/test_anchors.json --items $W/items.json \
    --runs /data/dy14/runs/s0,/data/dy14/runs/s1 --out $W/dy14_subset.json || exit 1
  mark dy14
fi
if ! done_ upload; then
  python - <<'EOF' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
for f in ["moshi_result.json", "dy14_subset.json", "items.json", "known_result.json"]:
    api.upload_file(path_or_fileobj=f"/data/dy15/{f}", path_in_repo=f"dy15/{f}", repo_id=repo)
print("UPLOADED dy15/")
EOF
  mark upload
fi
echo "DY15 DONE"
