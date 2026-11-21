#!/usr/bin/env bash
# DY-41 stages for `airon launch` (airon.yaml at the repository root names them):
#   setup  venv with the pinned packages
#   data   all codes (made beforehand by the split pass, airon-pass.yaml), checked complete; the two held-out
#          shards' audio and the prompt packs onto the local disk
#   pass   one machine's part of the split Mimi pass (airon-pass.yaml)
#   test   checklist: leak by shard names, layout (generation = training), known answer (Mimi on 50 real pieces),
#          one-batch overfit (600 steps; at 300 the warmup keeps it at CE 2.9, still falling)
#   train  scripts/dy41_train.py (extra arguments are passed on: the dry run adds --steps 60)
#   eval   continuations (prompts + prompts_x, t10 + ra08), heard-content test, DY-33/40 evaluation against DY-40's
#          model and model b, eval/... metrics into the run, HF dauys-runs/dy41/
#   bash scripts/dy41_job.sh <stage> [args]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# inside an Airon job the job's own key and address must win over the machine secrets (with the project key
# run.checkpoint() is refused: "This key is not a job key", and nothing is stored to resume from)
job_key=${AIRON_API_KEY:-}; job_url=${AIRON_URL:-}
[[ -f /etc/airon.env ]] && { set -a; . /etc/airon.env; set +a; }
if [[ -n ${AIRON_JOB:-} ]]; then
  [[ -n $job_key ]] && export AIRON_API_KEY=$job_key
  [[ -n $job_url ]] && export AIRON_URL=$job_url
fi
export AIRON_URL=${AIRON_URL%/v1}
NV=/opt/dlami/nvme; if mountpoint -q $NV 2>/dev/null; then sudo chown "$USER" $NV; else NV=$HOME/nvme; mkdir -p $NV; fi
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy41; mkdir -p $W/logs
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
S3=$B/dauys/dy41
export DY41_INIT=$S3/ckpt/state_periodic.pt  # DY-41/23's state: its checkpoints did not reach Airon's store
export DY_OUT_S3=$S3 DY_COMMIT=$(git -C "$repo" rev-parse HEAD)
V=$HOME/venv
export PATH=$V/bin:$PATH PYTHONPATH=$repo/src:$repo/scripts PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
stage=${1:?stage}; shift || true

case $stage in
setup)
  if ! $V/bin/python -c "import torch, transformers, airon, jiwer, boto3, sklearn" 2>/dev/null; then
    python3 -m pip install -q --user virtualenv
    python3 -m virtualenv -q $V
    $V/bin/pip install -q --upgrade pip
    $V/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
    $V/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
      soundfile pyyaml numba accelerate boto3 scikit-learn "airon>=0.4.0" "datasets==3.6.0" jiwer onnxruntime
  fi
  python -c "import torch; from transformers import MimiModel; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
  ;;
data)
  # the codes come from the split pass (airon-pass.yaml); here they are only fetched and checked complete
  aws s3 sync $B/dauys/dy40/units/ $NV/units/ --only-show-errors
  python -c "
from pathlib import Path
from dy40_mimi import shard_list
want = set(shard_list(100000)); have = {f.stem for f in Path('$NV/units').glob('*.npz')}
miss = sorted(want - have)
print('units on disk:', len(have), 'expected:', len(want), 'missing:', len(miss), miss[:10])
assert not miss, 'run the split Mimi pass first (airon launch airon-pass.yaml)'
"
  for s in la-00030 la-00071; do aws s3 sync $B/duplexchat/en-16k/$s/ $NV/held/$s/ --only-show-errors; done
  for p in prompts prompts_x; do aws s3 sync $B/dauys/dy32/$p/ /data/dy32/$p/ --only-show-errors; done
  ;;
pass)
  # one machine of the split Mimi pass (airon-pass.yaml, resources.shards): $1 workers (default 1; A10G is GPU-bound)
  k=${AIRON_SHARD:-0}; n=${AIRON_SHARDS:-1}; nw=${1:-1}; shift || true  # further arguments go to dy40_mimi.py pass
  (( $# )) || set -- --batch 2 --chunk-s 60  # A10G 24 GB: batch 8 x 120 s ran out of GPU memory
  pids=()
  for w in $(seq 0 $((nw-1))); do
    python "$repo/scripts/dy40_mimi.py" pass --part $((nw*k+w))/$((nw*n)) --n-train 100000 --out $NV/units --work $NV/w$w "$@" > $W/logs/pass_$w.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { tail -n 20 $W/logs/pass_*.log; exit 1; }; done
  grep -h "PASS DONE" $W/logs/pass_*.log
  ;;
mideval)
  # mid-run evaluation (airon-mideval.yaml): one checkpoint of the training job -> continuations of all 200 prompts
  # (t10, ra08) as m<tag>-*, Dself etc. by DY-33/40's pipeline against DY-40 (p40) and model b, round mid<tag>,
  # plus the listening set; S3 dy41/eval/mid<tag>/, HF dy41/eval/mid<tag>/ and dy41/listen-mid/
  uri=${1:?checkpoint uri}; tag=${2:?tag, e.g. 10k}
  mkdir -p $W/mid$tag
  python -c "import airon, shutil; shutil.copy(airon.fetch('$uri'), '$W/mid$tag/best.pt')"
  python "$repo/scripts/dy40_gen.py" gen --run $W/mid$tag --out $W/gen-mid$tag --pack-src prompts --variants "m$tag-t10:t10;m$tag-ra08:ra08" --batch 25 --upload
  python "$repo/scripts/dy40_gen.py" gen --run $W/mid$tag --out $W/gen-mid$tag-x --pack-src prompts_x --variants "m$tag-t10-x:t10;m$tag-ra08-x:ra08" --batch 25 --upload
  VENV=$V bash "$repo/scripts/dy33_setup.sh" > $W/logs/eval_setup.log 2>&1
  for x in b-t10 b-t10-x; do aws s3 sync $B/dauys/dy32/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  for x in b-ra08 b-ra08-x; do aws s3 sync $B/dauys/dy33/gen/$x/ $S3/gen/$x/ --only-show-errors; done
  for x in t10 ra08 t10-x ra08-x; do
    v=${x%-x}; sfx=${x#$v}
    aws s3 sync $B/dauys/dy40/gen/m-$x/ $S3/gen/p40-$v$sfx/ --only-show-errors
  done
  mkdir -p /data/dy34/eval; [[ -s /data/dy34/eval/listen_ids.json ]] || cp /data/dy32/a/listen_ids.json /data/dy34/eval/listen_ids.json
  DY_DIR=dy41 DY_TITLE="DY-41 mid $tag" DY_MODELS_PREFIX="Model m$tag = DY-41 at step $tag (all ~14k h usable); p40 = DY-40 (7.2k h);" \
    DY_FOLDERS=m$tag-t10,m$tag-ra08,p40-t10,p40-ra08,b-t10,b-ra08 DY_PAIRS=m$tag-t10:p40-t10,m$tag-ra08:p40-ra08 \
    DY_LISTEN=p40-t10,m$tag-t10,m$tag-ra08 \
    DY_DECLARED="{\"S1 (temperature 1.0)\": [\"m$tag-t10\", \"p40-t10\"], \"S2 (repeat-aware sharpening)\": [\"m$tag-ra08\", \"p40-ra08\"]}" \
    VENV=$V bash "$repo/scripts/dy40_eval.sh" mid$tag
  python "$repo/scripts/dy41_listen.py" --uri "$uri" --out $W/listen || echo "LISTEN FAILED"
  python - <<EOF
import json, airon
t = json.load(open("/data/dy32/a/eval/mid$tag/summary.json"))["table"]
m = {}
for n, r in t.items():
    print(n, "Dself", r["Dself"]["D"], "words", r["words_median_whisper"], "in_vocab", r["in_vocab_first50"])
    n = n.replace("-", "_")
    m[f"eval/dself_{n}"] = r["Dself"]["D"]
    m[f"eval/words_whisper_{n}"] = r["words_median_whisper"]
    m[f"eval/invocab_{n}"] = r["in_vocab_first50"]
run = airon.init()  # the job's run: the numbers are readable in Airon (get_run), not only on S3/HF
run.log(m)
EOF
  ;;
content)
  # does Mimi (and the DY-41 LM) carry utterance content? DY-36's yardstick (airon-content.yaml)
  python "$repo/scripts/dy41_mimi_content.py" --out $W/mc ${1:+--lm $1}
  ;;
teacher)
  # teachers trained with text (Whisper large-v3 encoder, wav2vec2-960h): content by DY-36's yardstick (airon-teacher.yaml)
  python "$repo/scripts/dy41_teacher_content.py" --out $W/tc
  ;;
listen)
  # mid-run listening (airon-listen.yaml): newest best checkpoint of the training job -> HF dy41/listen-mid/
  aws s3 sync $B/dauys/dy32/prompts/ /data/dy32/prompts/ --only-show-errors
  python "$repo/scripts/dy41_listen.py" --job ${1:-DY-41/24} --out $W/listen --uri ${2:-s3://airon-checkpoints-876108973178/dauys/DY-41/24/step-000007000-a1/best.pt}
  ;;
test)
  python "$repo/scripts/dy40_checklist.py" --units $NV/units --out $W/checklist.json
  rm -rf $NV/units_small && mkdir -p $NV/units_small  # overfit on a small subset: three training shards + one held-out
  for f in $(ls $NV/units/*.npz | grep -v -e la-00030 -e la-00071 | head -n 3) $NV/units/la-00030.npz; do ln -s $f $NV/units_small/; done
  python "$repo/scripts/dy40_train.py" --units $NV/units_small --out $W/overfit --overfit 600 --tracking off > $W/logs/overfit.log 2>&1
  grep -E "overfit_step\": (0|599)," $W/logs/overfit.log | cut -c1-160
  # the batch must be learned: best CE of the last 100 steps under half the first (bf16 runs differ: 1.01 and 2.79
  # at step 599 on the same batch, both from 7.77; a fixed 2.0 made the check a coin toss)
  python - <<EOF
import json
r = [json.loads(l) for l in open("$W/logs/overfit.log") if l.startswith('{"overfit_step"')]
first, best = r[0]["ce"], min(x["ce"] for x in r[-100:])
assert best < 0.5 * first, f"one-batch overfit did not fall far enough: {best} from {first}"
print("overfit ok", first, "->", best)
EOF
  aws s3 cp $W/checklist.json $S3/checklist.json --only-show-errors
  ;;
train)
  python "$repo/scripts/dy41_train.py" --units $NV/units --out $W/run --init-from $DY41_INIT "$@"
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
    n = n.replace("-", "_")
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
