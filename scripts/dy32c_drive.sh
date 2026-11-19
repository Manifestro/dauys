# DY-32 c driver on the machine: train -> early samples at EARLY -> final samples -> HF upload.
. ~/env.sh
cd ~/dauys
W=/data/dy32c
STEPS=${STEPS:-15000}
EARLY=${EARLY:-04:30}
mkdir -p $W/logs
python scripts/dy32c_train.py --units /opt/dlami/nvme/c-units --out $W/run --steps $STEPS --val-every 500 --stop-utc 07:10 \
  --name dy32-c > $W/train.log 2>&1 &
TP=$!
# early round
until [[ $(date -u +%H:%M) > "$EARLY" || $(date -u +%H:%M) == "$EARLY" ]]; do sleep 30; kill -0 $TP 2>/dev/null || break; done
if [[ -s $W/run/best.pt ]]; then
  python scripts/dy32c_gen.py --run $W/run --out $W/gen-early --s3 c-early > $W/logs/gen_early.log 2>&1
  S=$(python -c "import json; print(json.load(open('$W/gen-early/info.json'))['step'])")
  B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32
  aws s3 cp $W/gen-early/ $B/gen/c-early/step$S/ --recursive --exclude "*" --include "*.wav" --include "info.json" --only-show-errors
  echo "EARLY DONE step $S $(date -u +%H:%M)"
fi
wait $TP
echo "TRAIN EXIT $? $(date -u +%H:%M)"; tail -3 $W/train.log
python scripts/dy32c_gen.py --run $W/run --out $W/gen --s3 c > $W/logs/gen.log 2>&1 && echo "GEN DONE $(date -u +%H:%M)"
tail -2 $W/logs/gen.log
python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private, "not private"
w = "/data/dy32c"
fs = ["train.log", "run/best.pt", "run/metadata.json", "gen/info.json", "gen-early/info.json", "resynth/asr_rvq.json",
      "resynth/asr.json", "logs/gen.log", "logs/gen_early.log"]
for f in fs:
    if os.path.exists(f"{w}/{f}"):
        api.upload_file(path_or_fileobj=f"{w}/{f}", path_in_repo=f"dy32/c/{f}", repo_id=repo)
print("UPLOADED HF dy32/c/")
PY
echo "DRIVE DONE $(date -u +%H:%M)"
