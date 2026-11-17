#!/usr/bin/env bash
# DY-23 on an Airon AWS instance (g7e): grid LM over voice-invariant product codes, warm start from DY-22.
#   nohup bash ~/dauys/scripts/dy23_pod.sh > ~/dy23.log 2>&1 &
# setup -> data (dy23-units, DY-22 checkpoint, coder, ZeroSpeech dev) -> checklist -> dry -> train -> bench (gate M) -> upload HF dy23/ -> DY23 DONE.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy23
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY23_COMMIT=$(git -C "$repo" rev-parse HEAD)
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




NV=/opt/dlami/nvme; mountpoint -q $NV && sudo chown "$USER" $NV || NV=/data
U=$NV/units23
if ! done_ data; then
  aws configure set default.s3.max_concurrent_requests 64
  mkdir -p $U /data/dy14 /data/dy22/run /data/bench/zr
  n=$(aws s3 ls s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy23-units/ | grep -c "\.npz$"); echo "dy23-units shards: $n"
  [[ $n -ge 1222 ]] || { echo "DY23 UNITS NOT COMPLETE ($n)"; exit 1; }
  aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy23-units/ $U/ --only-show-errors || exit 1
  aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy23/vinv_inv1.0.pt $W/coder.pt --only-show-errors || exit 1
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - centroids.pt || exit 1
  python - <<'EOF2' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
for f in ["run/best.pt", "run/metadata.json"]:
    shutil.copy(hf_hub_download("manifestro/dauys-runs", f"dy22/{f}", token=os.environ["HF_TOKEN"]), f"/data/dy22/{f}")
EOF2
  if [[ ! -f /data/bench/zr/syntactic/dev/gold.csv ]]; then
    command -v aria2c >/dev/null || sudo apt-get install -y -qq aria2 >/dev/null 2>&1
    aria2c -q -x 16 -s 16 -k 64M -d /data/bench -o sLM21.zip https://download.zerospeech.com/datasets/sLM21.dataset.zip || exit 1
    unzip -q /data/bench/sLM21.zip "*lexical/dev/*" "*syntactic/dev/*" -d /data/bench/zr || exit 1
    rm -f /data/bench/sLM21.zip
  fi
  mark data
fi
INIT=/data/dy22/run/best.pt
if ! done_ checklist; then
  python - <<EOF2 || { echo "CHECKLIST FAILED"; exit 1; }
import sys, json, numpy as np, torch
sys.path.insert(0, "$repo/scripts")
from pathlib import Path
from dy23_train import load_grid
from dauys.dy21_lm import LMConfig, make_target
from dauys.dy23_lm import CodeLM, from_dy22, loss, UNIT
files = sorted(Path("$U").glob("*.npz"))[:4]
ev, cd, pb = load_grid(files)
u = ev == UNIT
info = {"frames": len(ev), "unit_share": float(u.mean()), "codes_nonzero": float((cd[u].sum(1) > 0).mean()),
        "distinct_g0": int(len(np.unique(cd[u][:, 0]))), "max_ev": int(ev.max())}
print("CHECK data", json.dumps(info))
assert 0.25 < info["unit_share"] < 0.55 and info["codes_nonzero"] > 0.95 and info["distinct_g0"] > 200 and info["max_ev"] <= 3
ck = torch.load("$INIT", map_location="cpu", weights_only=False)
m = from_dy22(ck["model"], LMConfig(vocab=4))
same = all(torch.equal(m.state_dict()[k], v) for k, v in ck["model"].items() if k.startswith("blocks."))
print("CHECK warm_start", same); assert same
dev = torch.device("cuda")
s = CodeLM(LMConfig(vocab=4, d=256, layers=4, heads=4, ffn=1024, ctx=256)).to(dev)
t = make_target(s)
e = torch.from_numpy(ev[:257].astype(np.int64))[None].repeat(4, 1).to(dev)
c = torch.from_numpy(cd[:257].astype(np.int64))[None].repeat(4, 1, 1).to(dev)
p = torch.from_numpy(pb[:257].astype(np.int64))[None].repeat(4, 1, 1).to(dev)
opt = torch.optim.AdamW(s.parameters(), lr=1e-3); first = None
for _ in range(300):
    tot, part = loss(s, t, e, c, p); first = first or part
    opt.zero_grad(); tot.backward(); opt.step()
print("CHECK one_batch", json.dumps({k: [round(first[k], 3), round(part[k], 3)] for k in ("ce_ev", "ce_code")}))
assert part["ce_code"] < 0.3 * first["ce_code"] and part["ce_ev"] < 0.5 * first["ce_ev"]
print("CHECKLIST PASS")
EOF2
  mark checklist
fi
if ! done_ dry; then
  python "$repo/scripts/dy23_train.py" --units $U --out $W/dry/run --steps 100 --val-every 50 --small --limit 6 --tracking off || { echo "DRY FAILED"; exit 1; }
  python "$repo/scripts/bench_dy23.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --coder $W/coder.pt \
    --run $W/dry/run/best.pt --out $W/dry/bench.json --pairs 20 --tasks swuggy || { echo "DRY FAILED"; exit 1; }
  mark dry
fi
if ! done_ train; then
  python "$repo/scripts/dy23_train.py" --units $U --init $INIT --out $W/run --steps ${STEPS:-8000} > $W/train.log 2>&1 || { echo "TRAIN FAILED"; tail -5 $W/train.log; exit 1; }
  mark train
fi
if ! done_ bench; then
  python "$repo/scripts/bench_dy23.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --coder $W/coder.pt \
    --run $W/run/best.pt --out $W/bench.json > $W/bench.log 2>&1 || { echo "BENCH FAILED"; tail -20 $W/bench.log; exit 1; }
  grep -E "^(swuggy|sblimp|tsc|ssc)" $W/bench.log
  mark bench
fi
if ! done_ upload; then
  python - <<'EOF2' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
for f in ["bench.json", "train.log", "run/best.pt", "run/metadata.json"]:
    api.upload_file(path_or_fileobj=f"/data/dy23/{f}", path_in_repo=f"dy23/{f}", repo_id=repo)
print("UPLOADED dy23/")
EOF2
  mark upload
fi
echo "DY23 DONE"
