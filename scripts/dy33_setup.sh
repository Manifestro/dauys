#!/usr/bin/env bash
# DY-33 track M: install the DY-32/33 evaluation environment on any GPU machine (idempotent).
# venv $VENV or ~/venv (torch 2.7.1 cu128, transformers 4.46.3), recogniser / Whisper / judge / GPT-2 / kNN-VC downloaded,
# prompt packs, cached references (ref.json, eval_base*.json, eval_controls.json, judge cache) and anchor folders.
#   bash scripts/dy33_setup.sh
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
[[ -f /etc/airon.env ]] && { set -a; . /etc/airon.env; set +a; }
V=${VENV:-$HOME/venv}  # evaluation venv (an own one on a shared machine)
B=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys
W=/data/dy32/a
sudo mkdir -p /data && sudo chown "$USER" /data
mkdir -p $W /data/dy32/prompts /data/dy32/prompts_x
if [[ ! -x $V/bin/python ]] || ! $V/bin/python -c "import torch, transformers, jiwer" 2>/dev/null; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > $V.constraints.txt
  $V/bin/pip install -q -c $V.constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba accelerate airon "datasets==3.6.0" jiwer onnxruntime
fi
export PATH=$V/bin:$PATH HF_HUB_DISABLE_PROGRESS_BARS=1
python - <<'EOF'
from huggingface_hub import snapshot_download as s
for m in ["facebook/wav2vec2-large-960h-lv60-self", "openai/whisper-large-v3-turbo", "gpt2-medium",
          "sentence-transformers/all-MiniLM-L6-v2", "Qwen/Qwen2.5-7B-Instruct"]:
    s(m, allow_patterns=None if m in ("gpt2-medium", "facebook/wav2vec2-large-960h-lv60-self") else
      ["*.json", "*.safetensors", "*.txt", "*.model"])
    print("got", m, flush=True)
import torch
torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cpu")
print("got knn-vc", flush=True)
EOF
aws s3 sync $B/dy32/prompts/ /data/dy32/prompts/ --only-show-errors
aws s3 sync $B/dy32/prompts_x/ /data/dy32/prompts_x/ --only-show-errors
aws s3 sync $B/dy32/a/centroids/ $W/centroids/ --only-show-errors
# cached references: DY-33 cache first, else the DY-32 originals on HF
aws s3 sync $B/dy33/eval/cache/ $W/ --only-show-errors
for f in ref.json eval_base.json eval_base_x.json eval_controls.json; do
  [[ -s $W/$f ]] || python -c "
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download('manifestro/dauys-runs', 'dy32/a/$f', token=os.environ['HF_TOKEN'])
shutil.copy(p, '$W/$f')"
done
mkdir -p $W/anchor_real
for i in $(seq -w 0 49); do [[ -e $W/anchor_real/$i.wav ]] || cp /data/dy32/prompts/real_$i.wav $W/anchor_real/$i.wav; done
ls $W
echo "DY33 SETUP DONE"
