#!/usr/bin/env bash
# Alphabet TTS data on an Airon AWS GPU instance (A10G is enough): venv (virtualenv, no apt), qwen-tts, then
#   PART=k/n nohup bash ~/dauys/scripts/tts_pod.sh > ~/tts.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
sudo mkdir -p /data && sudo chown "$USER" /data
if [[ ! -x ~/venv/bin/python ]] || ! ~/venv/bin/python -c "import qwen_tts" 2>/dev/null; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  ~/venv/bin/pip install -q qwen-tts soundfile scipy "datasets==3.6.0"
fi
~/venv/bin/python -c "import torch, qwen_tts; assert torch.cuda.is_available(); print('setup ok', torch.cuda.get_device_name(0))" 2>&1 | grep "setup ok" || { echo "SETUP FAILED"; exit 1; }
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
~/venv/bin/python "$repo/scripts/tts_alphabet.py" --part "$PART" --hours "${HOURS:-500}" ${TTS_ARGS:-} 2>&1 | grep --line-buffered -vE "Warning|warn|flash-attn|^\*+$|SoX|sox|onnxruntime|pad_token_id|^ *$" || { echo "TTS FAILED"; exit 1; }
echo "TTS POD DONE $PART"
