# DY-33 s driver on the machine: train (max 5.5 h, stop by 23:30 UTC, patience 3) -> early round from the 90-min
# snapshot (first 50 prompts, S1/S2 + heard test) while training goes on -> final continuations -> heard test ->
# guided extra -> HF upload.
. ~/env.sh
cd ~/dauys
W=/data/dy33s
N=/opt/dlami/nvme
C=/data/codec/frozen.pt
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy33
mkdir -p $W/logs
STOP=$(python - <<'PY'
import datetime as dt
now = dt.datetime.now(dt.timezone.utc)
cap = now.replace(hour=23, minute=30, second=0, microsecond=0)
print(min(now + dt.timedelta(minutes=330), cap).strftime("%H:%M"))
PY
)
echo "TRAIN START $(date -u +%H:%M) stop-utc $STOP"
python scripts/dy32c_train.py --units $N/units --out $W/run --steps ${STEPS:-60000} --batch 32 --micro ${MICRO:-16} --lr 3e-4 \
  --val-every 500 --runs --rate 12.5 --patience 3 --snapshot-min 90 --stop-utc $STOP \
  --units-desc "kNN-VC WavLM-Large layer 6 (50 Hz) -> DY-33 slow codec B step 4000 (x4 down, RVQ 4 x 1024), 12.5 Hz; windows inside runs of contiguous audio of one episode" \
  --experiment "DY-33: шаг модели размером со слог — те же признаки, 12,5 шага в секунду по 4 кода, без текста" \
  --name dy33-s > $W/train.log 2>&1 &
TP=$!
# early round from the snapshot
until [[ -s $W/run/snapshot.pt ]]; do sleep 30; kill -0 $TP 2>/dev/null || break; done
if [[ -s $W/run/snapshot.pt ]]; then
  sleep 60  # let the copy finish
  python scripts/dy33s_gen.py --run $W/run --ckpt snapshot.pt --codec $C --out $W/gen-early --pack-src prompts \
    --variants "s-early-t10:t10;s-early-ra08:ra08" --upload > $W/logs/gen_early.log 2>&1
  python scripts/dy33s_heard_use.py --held $N/held --ckpt $W/run/snapshot.pt --codec $C --out $W/heard_use_early.json \
    > $W/logs/heard_early.log 2>&1
  aws s3 cp $W/heard_use_early.json $S3/s/heard_use_early.json --only-show-errors
  aws s3 cp $W/heard_use_early.md $S3/s/heard_use_early.md --only-show-errors
  echo "EARLY DONE $(date -u +%H:%M)"
fi
wait $TP
echo "TRAIN EXIT $? $(date -u +%H:%M)"; tail -3 $W/train.log
aws s3 cp $W/train.log $S3/s/logs/train.log --only-show-errors
python scripts/dy33s_gen.py --run $W/run --codec $C --out $W/gen --pack-src prompts \
  --variants "s-t10:t10;s-ra08:ra08" --upload > $W/logs/gen.log 2>&1 && echo "GEN 50 DONE $(date -u +%H:%M)"
python scripts/dy33s_gen.py --run $W/run --codec $C --out $W/gen-x --pack-src prompts_x \
  --variants "s-t10-x:t10;s-ra08-x:ra08" --upload > $W/logs/gen_x.log 2>&1 && echo "GEN X DONE $(date -u +%H:%M)"
python scripts/dy33s_heard_use.py --held $N/held --ckpt $W/run/best.pt --codec $C --out $W/heard_use.json > $W/logs/heard.log 2>&1
aws s3 cp $W/heard_use.json $S3/s/heard_use.json --only-show-errors
aws s3 cp $W/heard_use.md $S3/s/heard_use.md --only-show-errors
echo "HEARD DONE $(date -u +%H:%M)"
# guided extra (w = 0 check against s-t10 first, then w = 4)
python scripts/dy33s_gen.py --run $W/run --codec $C --out $W/gen-g0 --pack-src prompts --swaps $N/swap \
  --variants "s-g0-check:g0" --check-codes $W/gen/s-t10/gen_units.npz > $W/logs/gen_g0.log 2>&1
python scripts/dy33s_gen.py --run $W/run --codec $C --out $W/gen-g --pack-src prompts --swaps $N/swap \
  --variants "s-g4:g4" --upload > $W/logs/gen_g4.log 2>&1 && echo "GEN G4 DONE $(date -u +%H:%M)"
python scripts/dy33s_gen.py --run $W/run --codec $C --out $W/gen-gx --pack-src prompts_x --swaps $N/swap \
  --variants "s-g4-x:g4" --upload > $W/logs/gen_g4x.log 2>&1 && echo "GEN G4 X DONE $(date -u +%H:%M)"
aws s3 cp $W/logs/ $S3/s/logs/ --recursive --only-show-errors
python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private, "not private"
w = "/data/dy33s"
up = {"/data/codec/frozen.pt": "codec/codec_B_step4000.pt"}
for f in ["train.log", "run/best.pt", "run/snapshot.pt", "run/metadata.json", "heard_use.json", "heard_use.md",
          "heard_use_early.json", "heard_use_early.md"]:
    up[f"{w}/{f}"] = f
for f in glob.glob(f"{w}/logs/*.log") + glob.glob(f"{w}/gen*/*/info.json"):
    up[f] = os.path.relpath(f, w)
for src, dst in up.items():
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=f"dy33/s/{dst}", repo_id=repo)
print("UPLOADED HF dy33/s/")
PY
echo "DRIVE DONE $(date -u +%H:%M)"
