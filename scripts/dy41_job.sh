#!/usr/bin/env bash
# DY-41 stages for `airon launch` (airon.yaml at the repository root names them):
#   setup  venv with the pinned packages
#   data   Mimi pass over every en-16k training shard not yet in S3 (catalogue dauys-mimi-en16k; two workers),
#          then all codes, the two held-out shards' audio and the prompt packs onto the local disk
#   test   checklist: leak by shard names, layout (generation = training), known answer (Mimi on 50 real pieces),
#          one-batch overfit (300 steps)
#   train  scripts/dy41_train.py (extra arguments are passed on: the dry run adds --steps 60)
#   eval   continuations (prompts + prompts_x, t10 + ra08), heard-content test, DY-33/40 evaluation against DY-40's
#          model and model b, eval/... metrics into the run, HF dauys-runs/dy41/
#   bash scripts/dy41_job.sh <stage> [args]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
[[ -f /etc/airon.env ]] && { set -a; . /etc/airon.env; set +a; }
export AIRON_URL=${AIRON_URL%/v1}
NV=/opt/dlami/nvme; if mountpoint -q $NV 2>/dev/null; then sudo chown "$USER" $NV; else NV=$HOME/nvme; mkdir -p $NV; fi
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy41; mkdir -p $W/logs
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
S3=$B/dauys/dy41
export DY_OUT_S3=$S3 DY_COMMIT=$(git -C "$repo" rev-parse HEAD)
V=$HOME/venv
export PATH=$V/bin:$PATH PYTHONPATH=$repo/src:$repo/scripts PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
stage=${1:?stage}; shift || true

case $stage in
setup)
  if ! $V/bin/python -c "import torch, transformers, airon, jiwer" 2>/dev/null; then
    python3 -m pip install -q --user virtualenv
    python3 -m virtualenv -q $V
    $V/bin/pip install -q --upgrade pip
    $V/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
    $V/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
      soundfile pyyaml numba accelerate "airon>=0.4.0" "datasets==3.6.0" jiwer onnxruntime
  fi
  python -c "import torch; from transformers import MimiModel; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
  ;;
data)
  pids=()
  for w in 0 1; do
    python "$repo/scripts/dy40_mimi.py" pass --part $w/2 --n-train 100000 --out $NV/units --work $NV/w$w > $W/logs/pass_$w.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { tail -n 20 $W/logs/pass_*.log; exit 1; }; done
  grep -h "PASS DONE" $W/logs/pass_*.log
  aws s3 sync $B/dauys/dy40/units/ $NV/units/ --only-show-errors
  echo "units on disk: $(ls $NV/units/*.npz | wc -l)"
  for s in la-00030 la-00071; do aws s3 sync $B/duplexchat/en-16k/$s/ $NV/held/$s/ --only-show-errors; done
  for p in prompts prompts_x; do aws s3 sync $B/dauys/dy32/$p/ /data/dy32/$p/ --only-show-errors; done
  ;;
test)
  python "$repo/scripts/dy40_checklist.py" --units $NV/units --out $W/checklist.json
  rm -rf $NV/units_small && mkdir -p $NV/units_small  # overfit on a small subset: three training shards + one held-out
  for f in $(ls $NV/units/*.npz | grep -v -e la-00030 -e la-00071 | head -n 3) $NV/units/la-00030.npz; do ln -s $f $NV/units_small/; done
  python "$repo/scripts/dy40_train.py" --units $NV/units_small --out $W/overfit --overfit 300 --tracking off > $W/logs/overfit.log 2>&1
  grep -E "overfit_step\": (0|299)," $W/logs/overfit.log | cut -c1-160
  python - <<EOF
import json
last = [json.loads(l) for l in open("$W/logs/overfit.log") if l.startswith('{"overfit_step"')][-1]
assert last["ce"] < 2.0, f"one-batch overfit did not fall far enough: {last['ce']}"
print("overfit ok", last["ce"])
EOF
  aws s3 cp $W/checklist.json $S3/checklist.json --only-show-errors
  ;;
train)
  python "$repo/scripts/dy41_train.py" --units $NV/units --out $W/run "$@"
  ;;
eval)
  python "$repo/scripts/dy40_gen.py" gen --run $W/run --out $W/gen --pack-src prompts --variants "m-t10:t10;m-ra08:ra08" --upload
  python "$repo/scripts/dy40_gen.py" gen --run $W/run --out $W/gen-x --pack-src prompts_x --variants "m-t10-x:t10;m-ra08-x:ra08" --upload
  python "$repo/scripts/dy40_gen.py" heard --held $NV/held --ckpt $W/run/best.pt --out $W/heard_use.json
  aws s3 cp $W/heard_use.json $S3/heard_use.json --only-show-errors; aws s3 cp $W/heard_use.md $S3/heard_use.md --only-show-errors
  VENV=$V bash "$repo/scripts/dy33_setup.sh" > $W/logs/eval_setup.log 2>&1
  for x in b-t10 b-t10-x; do aws s3 sync $B/dauys/dy32/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  for x in b-ra08 b-ra08-x; do aws s3 sync $B/dauys/dy33/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  for x in t10 ra08 t10-x ra08-x; do
    v=${x%-x}; sfx=${x#$v}
    aws s3 sync $B/dauys/dy40/gen/m-$x/ $S3/gen/p40-$v$sfx/ --only-show-errors
  done
  mkdir -p /data/dy34/eval; [[ -s /data/dy34/eval/listen_ids.json ]] || cp /data/dy32/a/listen_ids.json /data/dy34/eval/listen_ids.json
  DY_DIR=dy41 DY_TITLE=DY-41 DY_MODELS_PREFIX="Model m = DY-41 (all ~19k h, until validation stops); p40 = DY-40 (7.2k h);" \
    DY_FOLDERS=m-t10,m-ra08,p40-t10,p40-ra08,b-t10,b-ra08 DY_PAIRS=m-t10:p40-t10,m-ra08:p40-ra08 DY_LISTEN=p40-t10,m-t10,m-ra08 \
    DY_DECLARED='{"S1 (temperature 1.0)": ["m-t10", "p40-t10"], "S2 (repeat-aware sharpening)": ["m-ra08", "p40-ra08"]}' \
    VENV=$V bash "$repo/scripts/dy40_eval.sh" final
  python - <<EOF
import json, airon
s = json.load(open("/data/dy32/a/eval/final/summary.json"))
t = s["table"]
m = {}
for n, r in t.items():
    m[f"eval/dself_{n}"] = r["Dself"]["D"]
    m[f"eval/words_whisper_{n}"] = r["words_median_whisper"]
    m[f"eval/invocab_{n}"] = r["in_vocab_first50"]
run = airon.init()
run.log(m)
run.finish()
print(json.dumps(m, indent=1))
EOF
  python - <<EOF
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private
w = "$W"
for f in ["run/best.pt", "run/metadata.json", "run/DONE", "heard_use.json", "heard_use.md", "checklist.json"] :
    if os.path.exists(f"{w}/{f}"):
        api.upload_file(path_or_fileobj=f"{w}/{f}", path_in_repo=f"dy41/{f}", repo_id=repo)
for f in glob.glob(f"{w}/logs/*.log") + glob.glob(f"{w}/gen*/*/info.json"):
    api.upload_file(path_or_fileobj=f, path_in_repo="dy41/" + os.path.relpath(f, w), repo_id=repo)
print("UPLOADED HF dy41/")
EOF
  ;;
*) echo "unknown stage $stage"; exit 2 ;;
esac
echo "DY41 $stage DONE $(date -u +%H:%M) UTC"
