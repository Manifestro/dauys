#!/usr/bin/env bash
# DY-32 track D: environment for scripts/dy32d.py on the machine (source it). Secrets by reference only.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2
export S3D=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32
export W=/data/dy32d A=/opt/dlami/nvme/audio H=/opt/dlami/nvme/heldaudio
