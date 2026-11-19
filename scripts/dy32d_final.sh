#!/usr/bin/env bash
# DY-32 track D: after training ends, generate the final round (gen/d-cont, gen/d-disc) and upload checkpoints and
# logs to the private HF repo manifestro/dauys-runs under dy32/d/ and to S3 dy32/d/.
set -uo pipefail
source "$(dirname "$0")/dy32d_env.sh"
until [[ -e $W/run/DONE ]]; do pgrep -f "dy32d.py [t]rain" > /dev/null || { echo "TRAIN NOT RUNNING AND NO DONE"; break; }; sleep 30; done
bash "$repo/scripts/dy32d_gen.sh" "" || { echo "FINAL GEN FAILED"; exit 1; }
aws s3 sync $W/logs/ $S3D/d/logs/ --only-show-errors
aws s3 cp $W/run/metadata.json $S3D/d/metadata.json --only-show-errors
python - <<PY || { echo "HF UPLOAD FAILED"; exit 1; }
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "manifestro/dauys-runs"
assert api.repo_info(repo).private, "not private"
w = "$W"
fs = {"run/best.pt": "best.pt", "run/metadata.json": "metadata.json", "snap-early/best.pt": "best_early.pt",
      "gen/d-cont/info.json": "gen/d-cont/info.json", "gen/d-disc/info.json": "gen/d-disc/info.json",
      "gen/d-cont-early/info.json": "gen/d-cont-early/info.json", "gen/d-disc-early/info.json": "gen/d-disc-early/info.json"}
fs.update({os.path.relpath(f, w): os.path.relpath(f, w) for f in glob.glob(f"{w}/logs/*.log")})
for src, dst in fs.items():
    if os.path.exists(f"{w}/{src}"):
        api.upload_file(path_or_fileobj=f"{w}/{src}", path_in_repo=f"dy32/d/{dst}", repo_id=repo)
print("UPLOADED dy32/d/")
PY
echo "DY32D FINAL DONE"
