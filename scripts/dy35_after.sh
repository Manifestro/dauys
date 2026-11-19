#!/usr/bin/env bash
# DY-35 t after training (part F) on machine t: generations first (the evaluation track needs them), then P2a, P1
# (head; model b and acoustic baselines were run on t2), G1, uploads to S3 and HF dy35/t/.
#   CKPT=/data/t/run/best.pt nohup bash scripts/dy35_after.sh > ~/logs/after.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
set -a; . /etc/airon.env; set +a
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy35
CKPT=${CKPT:-/data/t/run/best.pt}
E="python $repo/scripts/dy35_eval.py"
R=/data/t/results; mkdir -p $R
step() { echo "=== $1 $(date -u +%H:%M)"; }
step bank;  $E bank --ckpt $CKPT --bank /data/t/bank.pt || echo "BANK FAILED"
step gen;   $E gen --ckpt $CKPT --out /data/t/gen --batch 100 || echo "GEN FAILED"
aws s3 sync /data/t/gen/ $S3/gen/ --only-show-errors && echo "GEN IN S3 $(date -u +%H:%M)"
step p2a;   $E p2a --ckpt $CKPT --out $R/p2a.json --n 1200 --batch 50 || echo "P2A FAILED"
step p1;    $E p1 --ckpt $CKPT --parts head --out $R/p1_head.json --n 1200 --batch 100 || echo "P1 FAILED"
step g1;    $E g1 --ckpt $CKPT --out $R/g1.json --n 1000 --batch 50 || echo "G1 FAILED"
aws s3 sync $R/ $S3/eval/ --only-show-errors
step upload
python - <<'P'
import os, shutil, torch
from pathlib import Path
from huggingface_hub import HfApi
run = Path("/data/t/run"); up = Path("/data/t/hf"); up.mkdir(exist_ok=True)
ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
torch.save({"enc": ck["enc"], "step": ck["step"]}, up / "gist_encoder.pt")
for f in ["best.pt", "metadata.json", "val_history.json"]:
    if (run / f).exists():
        os.link(run / f, up / f) if not (up / f).exists() else None
for f in Path("/data/t/results").glob("*.json"):
    shutil.copy(f, up / f.name)
for f in ["/data/t/check/checklist.json", "/data/t/check/checklist.md", os.path.expanduser("~/logs/pod.log"), os.path.expanduser("~/logs/after.log")]:
    if os.path.exists(f):
        shutil.copy(f, up / Path(f).name)
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path=str(up), path_in_repo="dy35/t", repo_id="manifestro/dauys-runs", repo_type="model")
print("HF UPLOADED", sorted((p.name, p.stat().st_size) for p in up.iterdir()))
P
aws s3 cp /data/t/run/metadata.json $S3/run/metadata.json --only-show-errors
aws s3 cp /data/t/run/val_history.json $S3/run/val_history.json --only-show-errors
aws s3 cp ~/logs/pod.log $S3/run/train.log --only-show-errors
step done
