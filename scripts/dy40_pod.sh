#!/usr/bin/env bash
# DY-40 on Airon AWS GPU instances. Roles (env ROLE):
#   main  setup -> Mimi pass (part POD/NPODS) -> checklist (layout, known answer) -> one-batch overfit -> wait for every
#         shard's units in S3 -> leak check -> dry run -> training -> continuations (prompts + prompts_x, t10 + ra08)
#         -> heard-content test -> HF upload dy40/.
#   pass  setup -> Mimi pass (part POD/NPODS) -> exit.
#   eval  setup -> evaluation environment (dy33_setup.sh) -> model b reference folders into dy40/gen -> wait for the
#         four m-* folders in S3 -> final evaluation (dy32_eval union, late Dself, extras, summary) -> S3/HF dy40/eval/final.
# Start detached:  ssh ... 'ROLE=main POD=0 NPODS=4 nohup bash ~/dauys/scripts/dy40_pod.sh > ~/dy40.log 2>&1 < /dev/null & disown'
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}; export AIRON_URL
for v in HF_TOKEN AIRON_API_KEY AIRON_URL; do [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }; done
export DY_COMMIT=$(git -C "$repo" rev-parse HEAD)
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy40; mkdir -p $W/marks $W/logs
NV=/opt/dlami/nvme; mountpoint -q $NV && sudo chown "$USER" $NV || NV=/data
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
S3=$B/dauys/dy40
ROLE=${ROLE:-main}; POD=${POD:-0}; NPODS=${NPODS:-1}; WPG=${WPG:-2}; NTRAIN=${NTRAIN:-600}
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S) UTC"; }
done_() { [[ -e "$W/marks/$1" ]]; }
echo "DY40 ROLE=$ROLE POD=$POD NPODS=$NPODS WPG=$WPG commit $DY_COMMIT $(date -u +%H:%M) UTC"

if ! done_ setup; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba accelerate airon "datasets==3.6.0" jiwer onnxruntime
  ~/venv/bin/python -c "import torch, airon, jiwer; from transformers import MimiModel; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))" || exit 1
  ~/venv/bin/airon doctor || echo "airon doctor failed (tracking may be offline)"
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src:$repo/scripts PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2

do_pass() {
  if ! done_ pass; then
    pids=()
    for w in $(seq 0 $((WPG - 1))); do
      part=$((POD * WPG + w))/$((NPODS * WPG))
      python "$repo/scripts/dy40_mimi.py" pass --part $part --n-train $NTRAIN --out $NV/units --work $NV/w$w > $W/logs/pass_$w.log 2>&1 &
      pids+=($!)
    done
    ok=1; for p in "${pids[@]}"; do wait $p || ok=0; done
    grep -h "PASS DONE" $W/logs/pass_*.log
    [[ $ok == 1 ]] || { echo "PASS FAILED"; tail -5 $W/logs/pass_*.log; exit 1; }
    mark pass
  fi
}

if [[ $ROLE == pass ]]; then do_pass; echo "DY40 PASS ROLE DONE $(date -u +%H:%M) UTC"; exit 0; fi

if [[ $ROLE == eval ]]; then
  if ! done_ evalsetup; then bash "$repo/scripts/dy33_setup.sh" > $W/logs/eval_setup.log 2>&1 || { echo "EVAL SETUP FAILED"; exit 1; }; mark evalsetup; fi
  for x in b-t10 b-t10-x; do aws s3 sync $B/dauys/dy32/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  for x in b-ra08 b-ra08-x; do aws s3 sync $B/dauys/dy33/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  echo "waiting for m-t10, m-ra08 (50) and -x twins (150) $(date -u +%H:%M)"
  while :; do
    c=0
    for x in m-t10 m-ra08; do [[ $(aws s3 ls $S3/gen/$x/ | grep -c '\.wav') -ge 50 ]] && c=$((c+1)); done
    for x in m-t10-x m-ra08-x; do [[ $(aws s3 ls $S3/gen/$x/ | grep -c '\.wav') -ge 150 ]] && c=$((c+1)); done
    [[ $c == 4 ]] && break; sleep 120
  done
  bash "$repo/scripts/dy40_eval.sh" final > $W/logs/eval_final.log 2>&1; tail -40 $W/logs/eval_final.log
  echo "DY40 EVAL ROLE DONE $(date -u +%H:%M) UTC"; exit 0
fi

# ---- main
do_pass
if ! done_ check1; then
  python "$repo/scripts/dy40_checklist.py" --units $NV/units --only layout,known --out $W/checklist_1.json > $W/logs/check1.log 2>&1
  tail -3 $W/logs/check1.log; grep -q "CHECKLIST PASS" $W/logs/check1.log || { echo "CHECKLIST 1 FAILED"; cat $W/logs/check1.log | tail -40; exit 1; }
  aws s3 cp $W/checklist_1.json $S3/checklist_1.json --only-show-errors
  mark check1
fi
echo "waiting for all units in S3 $(date -u +%H:%M)"
NEED=$((NTRAIN + 2))
until [[ $(aws s3 ls $S3/units/ | grep -c '\.npz$') -ge $NEED ]]; do sleep 60; done
aws s3 sync $S3/units/ $NV/units/ --only-show-errors
if ! done_ check2; then
  python "$repo/scripts/dy40_checklist.py" --units $NV/units --only leak --out $W/checklist_2.json > $W/logs/check2.log 2>&1
  tail -3 $W/logs/check2.log; grep -q "CHECKLIST PASS" $W/logs/check2.log || { echo "CHECKLIST 2 (leak) FAILED"; exit 1; }
  python "$repo/scripts/dy40_train.py" --units $NV/units --out $W/overfit --overfit 300 --tracking off > $W/logs/overfit.log 2>&1
  grep -E "overfit_step\": (0|275|299)" $W/logs/overfit.log | cut -c1-200; grep -q "OVERFIT DONE" $W/logs/overfit.log || { echo "OVERFIT FAILED"; tail -20 $W/logs/overfit.log; exit 1; }
  python "$repo/scripts/dy40_train.py" --units $NV/units --out $W/dry --steps 30 --val-every 15 --snapshot-min 0 --tracking off > $W/logs/dry_train.log 2>&1 \
    && python "$repo/scripts/dy40_gen.py" gen --run $W/dry --out $W/dry_gen --variants "dry:t10" --limit 2 --new-steps 25 > $W/logs/dry_gen.log 2>&1 \
    || { echo "DRY RUN FAILED"; tail -20 $W/logs/dry_*.log; exit 1; }
  grep "params" $W/logs/dry_train.log | head -1 | cut -c1-400
  aws s3 cp $W/checklist_2.json $S3/checklist_2.json --only-show-errors
  aws s3 cp $W/logs/ $S3/logs/ --recursive --only-show-errors
  mark check2
fi
if ! done_ train; then
  python "$repo/scripts/dy40_train.py" --units $NV/units --out $W/run > $W/train.log 2>&1
  tail -3 $W/train.log; grep -q "TRAIN DONE" $W/train.log || { echo "TRAIN FAILED"; tail -30 $W/train.log; exit 1; }
  aws s3 cp $W/train.log $S3/logs/train.log --only-show-errors
  mark train
fi
if ! done_ gen; then
  python "$repo/scripts/dy40_gen.py" gen --run $W/run --out $W/gen --pack-src prompts --variants "m-t10:t10;m-ra08:ra08" --upload > $W/logs/gen.log 2>&1 \
    && python "$repo/scripts/dy40_gen.py" gen --run $W/run --out $W/gen-x --pack-src prompts_x --variants "m-t10-x:t10;m-ra08-x:ra08" --upload > $W/logs/gen_x.log 2>&1 \
    || { echo "GEN FAILED"; tail -20 $W/logs/gen*.log; exit 1; }
  mark gen
fi
if ! done_ heard; then
  mkdir -p $NV/held
  for s in la-00030 la-00071; do aws s3 sync $B/duplexchat/en-16k/$s/ $NV/held/$s/ --only-show-errors; done
  python "$repo/scripts/dy40_gen.py" heard --held $NV/held --ckpt $W/run/best.pt --out $W/heard_use.json > $W/logs/heard.log 2>&1 || echo "HEARD FAILED"
  aws s3 cp $W/heard_use.json $S3/heard_use.json --only-show-errors; aws s3 cp $W/heard_use.md $S3/heard_use.md --only-show-errors
  mark heard
fi
aws s3 cp $W/logs/ $S3/logs/ --recursive --only-show-errors
python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private, "not private"
w = "/data/dy40"
up = {}
for f in ["train.log", "run/best.pt", "run/snapshot.pt", "run/metadata.json", "heard_use.json", "heard_use.md",
          "checklist_1.json", "checklist_2.json"]:
    up[f"{w}/{f}"] = f
for f in glob.glob(f"{w}/logs/*.log") + glob.glob(f"{w}/gen*/*/info.json"):
    up[f] = os.path.relpath(f, w)
for src, dst in up.items():
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=f"dy40/{dst}", repo_id=repo)
print("UPLOADED HF dy40/")
PY
echo "DY40 MAIN DONE $(date -u +%H:%M) UTC"
