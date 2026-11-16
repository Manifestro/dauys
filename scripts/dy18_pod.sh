#!/usr/bin/env bash
# DY-18 on a fresh Airon AWS instance (g7e.4xlarge). Clone this repository at the recorded commit to ~/dauys, then
#   nohup bash ~/dauys/scripts/dy18_pod.sh > ~/dy18.log 2>&1 &
# setup -> grid (DY-16's, from private S3) + DY-14 judge, DY-16 models, DY-16 moments and DY-17 control moments from
# private HF -> checklist (leak, oracle onset = exact expectation) -> dry run -> threshold calibration on val for f0 and f1
# -> gated generation on test with that threshold + judge -> listening files only if f0 starts in >= 20% of moments
# -> upload to private HF manifestro/dauys-runs dy18/ -> DY18 DONE. No training.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
W=/data/dy18
KZ_COMMIT=8be51925ecb872cf4894deafee716a317eb93516
S3OUT=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy14-grid
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
GITHUB_API=${GITHUB_API:-${GH_TOKEN:-}}  # the AWS instances name the GitHub token GH_TOKEN
AIRON_URL=${AIRON_URL%/v1}  # AWS gives the URL with /v1; the airon client adds /v1 itself
for v in GITHUB_API HF_TOKEN AIRON_API_KEY AIRON_URL; do
  [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }
done
export DY18_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH=$repo/src OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W/marks /data/dy14/runs/s0 /data/dy16/runs/f0 /data/dy16/runs/f1
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

audio() {
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
  mark audio
}

if ! done_ data; then
  aws s3 cp "$S3OUT/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - || exit 1
  python - <<'EOF' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
for f, dst in [("dy14/test_anchors.json", "/data/dy14/test_anchors.json"), ("dy14/s0/best.pt", "/data/dy14/runs/s0/best.pt"),
               ("dy16/runs/f0/best.pt", "/data/dy16/runs/f0/best.pt"), ("dy16/runs/f1/best.pt", "/data/dy16/runs/f1/best.pt"),
               ("dy16/moments.json", "/data/dy18/moments.json"), ("dy17/control.json", "/data/dy18/control.json")]:
    shutil.copy(hf_hub_download("manifestro/dauys-runs", f, token=os.environ["HF_TOKEN"]), dst)
print("artifacts ok")
EOF
  sha256sum $W/moments.json $W/control.json
  mark data
fi
AUDIO_PID=
done_ audio || { audio > $W/audio.log 2>&1 & AUDIO_PID=$!; }
G="--grid /data/dy14/grid --anchors /data/dy14/test_anchors.json --moments $W/moments.json --control $W/control.json"

if ! done_ checklist; then
  python - <<EOF || { echo "CHECKLIST FAILED"; exit 1; }
import json
names = json.load(open("/data/dy14/test_anchors.json"))["names"]
split = json.load(open("$SPLIT"))
test, val = set(split["test"]), set(split["val"])
ok = not (test & val) and all(names[it["conv"]] in test for f in ["moments", "control"] for it in json.load(open(f"$W/{f}.json")))
print("CHECK leak", "PASS" if ok else "FAIL")
assert ok
EOF
  python "$repo/scripts/dy17_generate.py" $G --model /data/dy16/runs/f0/best.pt --oracle --out $W/oracle.json \
    || { echo "CHECKLIST FAILED"; exit 1; }
  python - <<EOF || { echo "CHECKLIST FAILED"; exit 1; }
import json, numpy as np
from dauys.dy14_data import load_convs
r = json.load(open("$W/oracle.json"))["timing_form"]
convs = load_convs("/data/dy14/grid", json.load(open("/data/dy14/test_anchors.json"))["names"], 8192)
def exp(items):
    return float(np.mean([any(convs[it["conv"]].active[it["ch"], it["t"] + g + 1 : it["t"] + g + 11].mean() > 0.5 for g in range(30)) for it in items]))
e = exp(json.load(open("$W/moments.json"))), exp(json.load(open("$W/control.json")))
ok = abs(r["gen"]["started_3s"] - e[0]) < 1e-9 and abs(r["gen_control"]["started_3s"] - e[1]) < 1e-9
print("CHECK oracle", "PASS" if ok else "FAIL", r["gen"]["started_3s"], r["gen_control"]["started_3s"], "expected", e)
assert ok
EOF
  mark checklist
fi

if ! done_ dry; then
  D=$W/dry; mkdir -p $D
  python "$repo/scripts/dy18_calibrate.py" --grid /data/dy14/grid --split "$SPLIT" --model /data/dy16/runs/f0/best.pt \
    --out $D/thr.json --n 20 || exit 1
  python -c "import json; json.dump(json.load(open('$W/moments.json'))[:10], open('$D/m.json', 'w')); json.dump(json.load(open('$W/control.json'))[:10], open('$D/c.json', 'w'))"
  python "$repo/scripts/dy17_generate.py" --grid /data/dy14/grid --anchors /data/dy14/test_anchors.json --moments $D/m.json \
    --control $D/c.json --model /data/dy16/runs/f0/best.pt --judge /data/dy14/runs/s0/best.pt --threshold-file $D/thr.json \
    --out $D/gen.json || exit 1
  mark dry
fi

if ! done_ calibrate; then
  for s in 0 1; do
    python "$repo/scripts/dy18_calibrate.py" --grid /data/dy14/grid --split "$SPLIT" --model /data/dy16/runs/f$s/best.pt \
      --out $W/thr_f$s.json || exit 1
  done
  mark calibrate
fi

if ! done_ generate; then
  for s in 0 1; do
    python "$repo/scripts/dy17_generate.py" $G --model /data/dy16/runs/f$s/best.pt --judge /data/dy14/runs/s0/best.pt \
      --threshold-file $W/thr_f$s.json --out $W/gen_f$s.json || exit 1
  done
  mark generate
fi

FILES="control.json oracle.json thr_f0.json thr_f1.json gen_f0.json gen_f1.json gen_f0.tokens.npy gen_f1.tokens.npy"
if python -c "import json, sys; sys.exit(0 if json.load(open('$W/gen_f0.json'))['timing_form']['gen']['started_3s'] >= 0.2 else 1)"; then
  [[ -n "$AUDIO_PID" ]] && wait "$AUDIO_PID"
  done_ audio || { echo "AUDIO FAILED"; tail -5 $W/audio.log; exit 1; }
  if ! done_ voice; then
    python "$repo/scripts/dy16_voice.py" --oy /data/oy16 --anchors /data/dy14/test_anchors.json --moments $W/moments.json \
      --tokens $W/gen_f0.tokens.npy --centroids /data/dy14/centroids.pt --means /data/dy14/unit_means.pt \
      --config $W/sylber_resynthesis.yaml --out $W/voice || exit 1
    mark voice
  fi
  FILES="$FILES voice/voice.json"
else
  echo "LISTENING SKIPPED: f0 starts in < 20% of moments"
  [[ -n "$AUDIO_PID" ]] && kill "$AUDIO_PID" 2>/dev/null
fi
if ! done_ upload; then
  FILES="$FILES" python - <<'EOF' || exit 1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
if not api.repo_info(repo).private:
    raise SystemExit(f"UPLOAD SKIPPED: {repo} is not private")
W = "/data/dy18"
for f in os.environ["FILES"].split():
    api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f"dy18/{f}", repo_id=repo)
if os.path.isdir(f"{W}/voice/listen"):
    api.upload_folder(folder_path=f"{W}/voice/listen", path_in_repo="dy18/listen", repo_id=repo)
print("UPLOADED dy18/")
EOF
  mark upload
fi
echo "DY18 DONE"
