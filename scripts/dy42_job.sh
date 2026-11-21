#!/usr/bin/env bash
# DY-42 (phrase vectoriser on all ~800k utterances). Each command first makes sure the pinned venv exists.
#   bash scripts/dy42_job.sh frames|texts [args]   airon-dy42-frames.yaml / airon-dy42-texts.yaml (type: run)
#   bash scripts/dy42_job.sh head [args]           airon-dy42-head.yaml
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
sudo mkdir -p /data && sudo chown "$USER" /data
V=$HOME/venv
if ! $V/bin/python -c "import torch, transformers, airon, boto3, sklearn" 2>/dev/null; then
  python3 -m pip install -q --user virtualenv
  python3 -m virtualenv -q $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  $V/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba accelerate boto3 scikit-learn "airon>=0.6.0"
fi
export PATH=$V/bin:$PATH PYTHONPATH=$repo/src:$repo/scripts PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MALLOC_ARENA_MAX=2
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
cmd=${1:?command}; shift || true
case $cmd in
frames|texts) python "$repo/scripts/dy42_prep.py" $cmd "$@" ;;
head) python "$repo/scripts/dy42_head.py" "$@" ;;
*) echo "unknown command $cmd"; exit 2 ;;
esac
