#!/usr/bin/env bash
# DY-35 t after training (part F) on machine t: generations first (the evaluation track needs them), then P2a, P1
# (head; model b and acoustic baselines were run on t2), G1, uploads to S3 dy35/t/ and HF dy35/t/, then the extra
# sampled-bank generations (t-samp).
#   WAIT_TRAIN=1 CKPT=/data/t/run/best.pt nohup bash scripts/dy35_after.sh > ~/logs/after.log 2>&1 &
# WAIT_TRAIN=1: start only once the trainer process has exited (finished, stopped or crashed).
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy35
CKPT=${CKPT:-/data/t/run/best.pt}
E="python $repo/scripts/dy35_eval.py"
R=/data/t/results; mkdir -p $R
step() { echo "=== $1 $(date -u +%H:%M)"; }
if [[ ${WAIT_TRAIN:-0} == 1 ]]; then
  echo "waiting for the trainer to exit $(date -u +%H:%M)"
  while pgrep -f "[d]y35_train.py --utt" >/dev/null; do sleep 30; done
  sleep 20
  echo "trainer gone; DONE file: $(cat /data/t/run/DONE 2>/dev/null || echo missing)"
fi
echo "after commit $(git -C "$repo" rev-parse HEAD) ckpt $CKPT"
step bank;  $E bank --ckpt $CKPT --bank /data/t/bank.pt || echo "BANK FAILED"
step gen;   $E gen --ckpt $CKPT --out /data/t/gen --batch 100 \
              --modes t-pred,t-oracle,t-swap,t-near,b-turn,t-t10,t-t10-x || echo "GEN FAILED"
aws s3 sync /data/t/gen/ $S3/gen/ --only-show-errors && echo "GEN IN S3 $(date -u +%H:%M) $(ls /data/t/gen | tr '\n' ' ')"
step p2a;   $E p2a --ckpt $CKPT --out $R/p2a.json --n 1200 --batch 50 || echo "P2A FAILED"
aws s3 sync $R/ $S3/eval/ --only-show-errors
step p1;    $E p1 --ckpt $CKPT --parts head --out $R/p1_head.json --n 1200 --batch 100 || echo "P1 FAILED"
aws s3 sync $R/ $S3/eval/ --only-show-errors
step g1;    $E g1 --ckpt $CKPT --out $R/g1.json --n 1000 --batch 50 || echo "G1 FAILED"
aws s3 sync $R/ $S3/eval/ --only-show-errors
step upload
up=/data/t/hf; mkdir -p $up
python - <<'P' || echo "PACK FAILED"
import os, shutil, torch
from pathlib import Path
run, up = Path("/data/t/run"), Path("/data/t/hf")
ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
torch.save({"enc": ck["enc"], "step": ck["step"]}, up / "gist_encoder.pt")
for f in ["best.pt", "metadata.json", "val_history.json"]:
    if (run / f).exists() and not (up / f).exists():
        os.link(run / f, up / f)
for f in Path("/data/t/results").glob("*.json"):
    shutil.copy(f, up / f.name)
for f in ["/data/t/check/checklist.json", "/data/t/check/checklist.md", os.path.expanduser("~/logs/pod.log")]:
    if os.path.exists(f):
        shutil.copy(f, up / Path(f).name)
print("PACKED", sorted((p.name, p.stat().st_size) for p in up.iterdir()), flush=True)
P
cp ~/logs/after.log $up/after.log 2>/dev/null
aws s3 sync $up/ $S3/t/ --only-show-errors && echo "S3 T UPLOADED $(date -u +%H:%M)" || echo "S3 T UPLOAD FAILED"
aws s3 cp /data/t/run/metadata.json $S3/run/metadata.json --only-show-errors
aws s3 cp /data/t/run/val_history.json $S3/run/val_history.json --only-show-errors
aws s3 cp ~/logs/pod.log $S3/run/train.log --only-show-errors
python - <<'P' 2>&1 | tail -3 | tee /data/t/hf_result.txt
import os
from huggingface_hub import HfApi
try:
    HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path="/data/t/hf", path_in_repo="dy35/t",
                                                      repo_id="manifestro/dauys-runs", repo_type="model")
    print("HF UPLOADED")
except Exception as e:
    print("HF UPLOAD REFUSED", type(e).__name__, str(e)[:300])
P
aws s3 cp /data/t/hf_result.txt $S3/t/hf_result.txt --only-show-errors
step samp;  $E gen --ckpt $CKPT --out /data/t/gen --batch 100 --modes t-samp || echo "SAMP FAILED"
aws s3 sync /data/t/gen/t-samp/ $S3/gen/t-samp/ --only-show-errors && echo "SAMP IN S3 $(date -u +%H:%M)"
step done
aws s3 cp ~/logs/after.log $S3/t/after.log --only-show-errors
