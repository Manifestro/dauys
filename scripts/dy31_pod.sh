#!/usr/bin/env bash
# DY-31 on Airon AWS GPU instances: DY-30 with letters instead of phonemes (facebook/wav2vec2-large-960h-lv60-self,
# word delimiter kept), so the model's continuation is read directly as text.
#   1 setup -> 2 cheap gates on one held-out shard (token known answer, symbols gate S, judge gate J; the big pass does
#   not start if they fail) -> 3 letters pass over all shards (this machine's POD share; S3 outputs/dauys/char-units/)
#   -> 4 training checklist (leak, known answer, one batch, dry run through the whole evaluation) -> 5 training ->
#   6 evaluation -> 7 upload HF dy31/.
# Main machine:     nohup bash ~/dauys/scripts/dy31_pod.sh > ~/dy31.log 2>&1 &
# Gates only:       GATES_ONLY=1 nohup bash ~/dauys/scripts/dy31_pod.sh > ~/dy31.log 2>&1 &
# Several machines: NPODS=n POD=k on each (POD=0 is the main one; helpers add PASS_ONLY=1 and need the gates passed
#                   on POD=0 first). The shard parts are numbered from GPUs x WPG, so EVERY machine must have the
#                   same GPU count and the same WPG, or shards are skipped or done twice.
# Tokens already in S3 (pass done elsewhere): SKIP_PASS=1. STEPS overrides the 30000-step cap.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
for v in HF_TOKEN AIRON_API_KEY AIRON_URL; do [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }; done
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy31; mkdir -p $W/marks $W/logs
NV=/opt/dlami/nvme; mountpoint -q $NV && sudo chown "$USER" $NV || NV=/data
U=$NV/char-units
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
SRC=$S3/duplexchat/en-16k
S3P=$S3/dauys/char-units
S3R=$S3/dauys/dy31
EXP="DY-31: буквы на ~19 тыс. ч — появляется ли смысл в продолжении при лучших возможных символах (422M, рецепт DY-21)"
POD=${POD:-0}; NPODS=${NPODS:-1}; WPG=${WPG:-3}
GATES_ONLY=${GATES_ONLY:-0}; PASS_ONLY=${PASS_ONLY:-0}; SKIP_PASS=${SKIP_PASS:-0}; PASS_WAIT_H=${PASS_WAIT_H:-24}
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S) UTC"; }
done_() { [[ -e "$W/marks/$1" ]]; }
echo "DY31 POD=$POD NPODS=$NPODS WPG=$WPG GATES_ONLY=$GATES_ONLY PASS_ONLY=$PASS_ONLY SKIP_PASS=$SKIP_PASS commit $DY21_COMMIT"

# 1 setup
if ! done_ setup; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba sylber==0.1.4 accelerate airon "datasets==3.6.0" jiwer onnxruntime
  ~/venv/bin/python -c "import torch, sylber, airon, jiwer, onnxruntime; assert torch.cuda.is_available(); print('GPU', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))" || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2
NG=$(nvidia-smi -L | wc -l)

# 2 cheap gates (POD=0 runs them; other machines only accept a passed gates.json from S3)
if ! done_ gates; then
  mkdir -p $W/gates
  if [[ $POD == 0 ]]; then
    GS=${GATE_SHARD:-$(python -c "
import subprocess, sys; sys.path.insert(0, '$repo/scripts'); from dy21_train import held_out
out = subprocess.run('aws s3 ls $SRC/', shell=True, capture_output=True, text=True).stdout
print(next(s for s in sorted(l.split()[-1].rstrip('/') for l in out.splitlines() if l.strip().startswith('PRE')) if held_out(s)))")}
    echo "gate shard: $GS"
    python "$repo/scripts/dy31_units.py" known --shard $GS --work $NV/known --out $W/known_tokens.json > $W/logs/known.log 2>&1 \
      || { echo "TOKEN KNOWN ANSWER FAILED"; tail -5 $W/logs/known.log; exit 1; }
    grep -E "^(KNOWN|SAMPLE)" $W/logs/known.log
    python "$repo/scripts/dy31_units.py" run --shards $GS --work $NV/gate_units > $W/logs/gate_units.log 2>&1 \
      || { echo "GATE SHARD TOKENS FAILED"; tail -5 $W/logs/gate_units.log; exit 1; }
    aws s3 cp $S3P/$GS.npz $W/gate_tokens.npz --only-show-errors || exit 1
    python "$repo/scripts/dy31_eval.py" gates --shard $GS --tokens $W/gate_tokens.npz --out $W/gates > $W/logs/gates.log 2>&1
    rc=$?
    grep -E "^(S |GATE|GATES)" $W/logs/gates.log
    [[ $rc == 0 ]] || { echo "GATES FAILED (see above); the letters pass does not start"; tail -5 $W/logs/gates.log; exit 1; }
    aws s3 cp $W/gates/gates.json $S3R/gates.json --only-show-errors
    aws s3 cp $W/known_tokens.json $S3R/known_tokens.json --only-show-errors
  else
    aws s3 cp $S3R/gates.json $W/gates/gates.json --only-show-errors || { echo "no gates.json in S3: run the gates on POD=0 first"; exit 1; }
    python -c "import json, sys; sys.exit(0 if json.load(open('$W/gates/gates.json'))['pass'] else 1)" || { echo "gates did not pass; not starting"; exit 1; }
  fi
  mark gates
fi
[[ $GATES_ONLY == 1 ]] && { echo "DY31 GATES DONE"; exit 0; }

# 3 letters pass: this machine takes parts POD*NW .. POD*NW+NW-1 of NPODS*NW (NW = GPUs x WPG workers)
if ! done_ pass; then
  if [[ $SKIP_PASS != 1 ]] && ! done_ pass_share; then
    NW=$((NG * WPG))
    for ((k = 0; k < NW; k++)); do
      CUDA_VISIBLE_DEVICES=$((k % NG)) python "$repo/scripts/dy31_units.py" run --part $((POD * NW + k))/$((NPODS * NW)) \
        --work $NV/w$k > $W/logs/pass_$k.log 2>&1 &
    done
    wait
    for ((k = 0; k < NW; k++)); do grep -q "CHAR UNITS DONE" $W/logs/pass_$k.log || { echo "pass worker $k failed (rerun resumes)"; tail -5 $W/logs/pass_$k.log; exit 1; }; done
    grep -h x_realtime_total $W/logs/pass_*.log | tail -$NW | sed 's/^/last /'
    mark pass_share
  fi
  [[ $PASS_ONLY == 1 ]] && { echo "DY31 PASS SHARE DONE"; exit 0; }
  total=$(aws s3 ls $SRC/ | grep -c " PRE ")
  t0=$(date +%s)
  while true; do
    have=$(aws s3 ls $S3P/ | grep -c '\.npz$')
    (( have >= total )) && break
    (( $(date +%s) - t0 > PASS_WAIT_H * 3600 )) && { echo "PASS INCOMPLETE: $have of $total shards in S3"; exit 1; }
    echo "waiting for other machines: $have of $total shards $(date -u +%H:%M) UTC"; sleep 300
  done
  mkdir -p $U
  aws configure set default.s3.max_concurrent_requests 64
  aws s3 sync $S3P/ $U/ --only-show-errors || exit 1
  echo "tokens: $(ls $U/*.npz | wc -l) shards, $(du -sh $U | cut -f1)"
  mark pass
fi

# 4 training checklist
if ! done_ checklist; then
  python - <<EOF || { echo "CHECKLIST (data) FAILED"; exit 1; }
import subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "$repo/scripts")
from dy21_train import held_out
out = subprocess.run("aws s3 ls $SRC/", shell=True, capture_output=True, text=True).stdout
src = sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))
have = sorted(p.stem for p in Path("$U").glob("*.npz"))
missing = sorted(set(src) - set(have))
tr, va = [s for s in have if not held_out(s)], [s for s in have if held_out(s)]
tok = sum(len(np.load(f"$U/{s}.npz")["index"]) for s in va)
print("DATA", {"source_shards": len(src), "token_shards": len(have), "missing": missing[:5], "train": len(tr), "held_out": len(va),
               "held_out_pieces": tok, "train_held_overlap": len(set(tr) & set(va))})
sys.exit(0 if not missing and va and not set(tr) & set(va) else 1)
EOF
  python "$repo/scripts/dy21_checklist.py" --units $U > $W/logs/checklist.log 2>&1; rc=$?
  cat $W/logs/checklist.log
  [[ $rc == 0 ]] || { echo "CHECKLIST FAILED"; exit 1; }
  D=$W/dry; rm -rf $D; mkdir -p $D/data
  h=$(python -c "import sys, glob, os; sys.path.insert(0, '$repo/scripts'); from dy21_train import held_out; fs=sorted(glob.glob('$U/*.npz')); print(next(f for f in fs if held_out(os.path.basename(f)[:-4])), next(f for f in fs if not held_out(os.path.basename(f)[:-4])))")
  for f in $h; do ln -sf $f $D/data/; done
  python "$repo/scripts/dy21_train.py" --units $D/data --out $D/run --steps 200 --val-every 100 --small --tracking off > $W/logs/dry_train.log 2>&1 \
    || { echo "DRY TRAIN FAILED"; tail -5 $W/logs/dry_train.log; exit 1; }
  python "$repo/scripts/dy31_eval.py" eval --units $D/data --run $D/run --n 4 --out $D/eval > $W/logs/dry_eval.log 2>&1 \
    || { echo "DRY EVAL FAILED"; tail -20 $W/logs/dry_eval.log; exit 1; }
  grep -E "^(TEXTLESS|JUDGE|VERDICT)" $W/logs/dry_eval.log | sed 's/^/dry /'
  mark checklist
fi

# 5 training
if ! done_ train; then
  python "$repo/scripts/dy21_train.py" --units $U --out $W/run --steps ${STEPS:-30000} --val-every 1000 --patience 3 \
    --experiment "$EXP" --name dy31-s0 > $W/train.log 2>&1 || { echo "TRAIN FAILED"; tail -5 $W/train.log; exit 1; }
  [[ -s $W/run/best.pt ]] || { echo "TRAIN FAILED: no best.pt"; exit 1; }
  grep -aE "early_stop|TRAIN DONE" $W/train.log | tail -1
  mark train
fi

# 6 evaluation
if ! done_ eval; then
  python "$repo/scripts/dy31_eval.py" eval --units $U --run $W/run --out $W/eval > $W/logs/eval.log 2>&1 \
    || { echo "EVAL FAILED"; tail -20 $W/logs/eval.log; exit 1; }
  grep -E "^(TEXTLESS|JUDGE|VERDICT)" $W/logs/eval.log
  mark eval
fi

# 7 upload
if ! done_ upload; then
  python - <<'EOF' || { echo "UPLOAD FAILED"; exit 1; }
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private, "not private"
w = "/data/dy31"
fs = ["eval/eval.json", "eval/pairs.md", "gates/gates.json", "known_tokens.json", "train.log",
      "run/best.pt", "run/metadata.json", "dry/eval/eval.json"] + [os.path.relpath(f, w) for f in glob.glob(f"{w}/logs/*.log")]
for f in fs:
    if os.path.exists(f"{w}/{f}"):
        api.upload_file(path_or_fileobj=f"{w}/{f}", path_in_repo=f"dy31/{f}", repo_id=repo)
print("UPLOADED dy31/")
EOF
  mark upload
fi
echo "DY31 POD DONE"
