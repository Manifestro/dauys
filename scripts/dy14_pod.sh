#!/usr/bin/env bash
# DY-14 on a fresh Airon AWS instance (g7e.4xlarge, RTX PRO 6000 Blackwell). Clone this repository at the commit
# recorded in the experiment to ~/dauys, then:
#   nohup bash ~/dauys/scripts/dy14_pod.sh > ~/dy14.log 2>&1 &
# Stages, each skipped when its marker exists (run the same command again to resume):
#   setup -> data (Open Yap 16 kHz tracks from the ke-zek S3 ready set, SHA-256 checked) -> checklist (stops on FAIL)
#   -> dry (features + grid + 200 steps on 6 conversations per part) -> features -> grid -> test anchors (hash printed)
#   -> two seeds in parallel -> test evaluation -> upload (checkpoints and JSON only, never audio or features, to the
#   private HF repository manifestro/dauys-runs, dy14/) -> DONE. Features and units stay on the instance and S3 only.
# STAGE=checklist stops after the checklist and the dry run (for a gate-only instance).
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy14
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
STAGE=${STAGE:-all}
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY14_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }

if ! done_ setup; then
  sudo apt-get update -qq > /dev/null 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv > /dev/null 2>&1
  python3 -m venv ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  ~/venv/bin/pip install -q "numpy<2.1" transformers==4.46.3 huggingface_hub==0.25.2 safetensors airon
  ~/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__)" || exit 1
  [[ -d ~/ke-zek ]] || git clone -q "https://$GITHUB_API@github.com/Manifestro/kezek-research.git" ~/ke-zek
  git -C ~/ke-zek checkout -q $KZ_COMMIT || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH
SPLIT=$HOME/ke-zek/experiments/K-R1/split_round1.json

if ! done_ data; then
  S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/kezek-stage1-ready
  mkdir -p /data/oy16 /data/.got
  aws s3 cp "$S3/MANIFEST" /data/manifest_stage1.txt --only-show-errors || { echo "no MANIFEST"; exit 1; }
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
  python - <<EOF || exit 1
import json, pathlib
s = json.load(open("$SPLIT"))
root = pathlib.Path("/data/oy16")
miss = {p: [c for c in v if not (root / c / "a.npy").exists()] for p, v in s.items()}
print({p: (len(s[p]), len(m)) for p, m in miss.items()})
assert not any(miss.values()), "conversations missing"
EOF
  mark data
fi

if ! done_ checklist; then
  python "$repo/scripts/dy14_checklist.py" --split "$SPLIT" --root /data/oy16 || { echo "CHECKLIST FAILED: not training"; exit 1; }
  mark checklist
fi

if ! done_ dry; then
  D=$W/dry
  python "$repo/scripts/dy14_features.py" --root /data/oy16 --split "$SPLIT" --out $D/seg --limit 6 || exit 1
  python "$repo/scripts/dy14_grid.py" --seg $D/seg --split "$SPLIT" --out $D --k 256 --sample 200000 --limit 6 || exit 1
  python "$repo/scripts/dy14_eval.py" --grid $D/grid --split "$SPLIT" --anchors $D/anchors.json --build-anchors --k 256 --limit 6 || exit 1
  python "$repo/scripts/dy14_train.py" --grid $D/grid --split "$SPLIT" --out $D/run --seed 0 --steps 200 --val-every 100 \
    --k 256 --limit 6 --tracking off || exit 1
  python "$repo/scripts/dy14_eval.py" --grid $D/grid --split "$SPLIT" --anchors $D/anchors.json --runs $D/run \
    --stats $D/stats.json --k 256 --limit 6 || exit 1
  mark dry
fi
[[ $STAGE == checklist ]] && { echo "STAGE checklist: stopping after the checks"; exit 0; }

if ! done_ features; then
  python "$repo/scripts/dy14_features.py" --root /data/oy16 --split "$SPLIT" --out $W/seg --workers 14 || exit 1
  mark features
fi
if ! done_ grid; then
  python "$repo/scripts/dy14_grid.py" --seg $W/seg --split "$SPLIT" --out $W || exit 1
  mark grid
fi
if ! done_ anchors; then
  python "$repo/scripts/dy14_eval.py" --grid $W/grid --split "$SPLIT" --anchors $W/test_anchors.json --build-anchors || exit 1
  python "$repo/scripts/dy14_checklist.py" --split "$SPLIT" --anchors $W/test_anchors.json > $W/leak_anchors.log 2>&1
  grep "CHECK leak" $W/leak_anchors.log
  grep -q "CHECK leak_anchors PASS" $W/leak_anchors.log || { echo "anchor leak check failed"; exit 1; }
  mark anchors
fi
if ! done_ train; then
  for s in 0 1; do
    o=$((1 - s))
    python "$repo/scripts/dy14_train.py" --grid $W/grid --split "$SPLIT" --out $W/runs/s$s --seed $s \
      --peer $W/runs/s$o/status_5000.json > $W/train_s$s.log 2>&1 &
  done
  wait
  for s in 0 1; do
    [[ -e $W/runs/s$s/DONE || -e $W/runs/s$s/STOPPED ]] || { echo "seed $s did not finish"; tail -5 $W/train_s$s.log; exit 1; }
  done
  mark train
fi
if ! done_ eval; then
  python "$repo/scripts/dy14_eval.py" --grid $W/grid --split "$SPLIT" --anchors $W/test_anchors.json \
    --runs $W/runs/s0,$W/runs/s1 --stats $W/stats.json --out $W/result.json || exit 1
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
W = "/data/dy14"
for f in ["result.json", "stats.json", "test_anchors.json", "centroids.pt", "train_s0.log", "train_s1.log"]:
    api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy14/{f}", repo_id=repo)
for s in ("s0", "s1"):
    for f in ("best.pt", "metadata.json", "val_anchors.json"):
        if os.path.exists(f"{W}/runs/{s}/{f}"):
            api.upload_file(path_or_fileobj=f"{W}/runs/{s}/{f}", path_in_repo=f"dy14/{s}/{f}", repo_id=repo)
print("UPLOADED dy14/")
EOF
  mark upload
fi
echo "DY14 DONE"
