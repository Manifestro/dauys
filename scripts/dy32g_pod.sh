#!/usr/bin/env bash
# DY-32 track G machine setup: venv (torch 2.7.1 cu128), released GSLM / unit-vocoder weights into /data/m,
# textlesslib pinned (only its Tacotron2/WaveGlow modules are imported), the prompt pack into /data/pack.
set -uo pipefail
sudo mkdir -p /data && sudo chown "$USER" /data
if [[ ! -x ~/venv/bin/python ]]; then
  python3 -m pip install -q --user virtualenv && python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba accelerate airon "datasets==3.6.0" jiwer scikit-learn joblib librosa omegaconf unidecode inflect
fi
[[ -d /data/textlesslib ]] || git clone -q https://github.com/facebookresearch/textlesslib.git /data/textlesslib
git -C /data/textlesslib checkout -q ba33d669d8284b4f7bfe81e7384e83ab799fe384
mkdir -p /data/m && cd /data/m
B=https://dl.fbaipublicfiles.com
get() { [[ -s $2 ]] || aria2c -q -x8 -o "$2" "$1"; }
for k in 50 100 200; do
  get $B/textless_nlp/gslm/hubert/km$k/km.bin km$k.bin
  get $B/textless_nlp/gslm/hubert/lm_km$k/hubert${k}_lm.tgz hubert${k}_lm.tgz
  [[ -d lm$k ]] || { mkdir -p lm$k && tar xzf hubert${k}_lm.tgz -C lm$k; }
  get $B/textless_nlp/gslm/hubert/tts_km$k/tts_checkpoint_best.pt tts$k.pt
  get $B/textless_nlp/gslm/hubert/tts_km$k/code_dict tts${k}_code_dict
done
get $B/textless_nlp/gslm/hubert/km500/km.bin km500.bin
get $B/textless_nlp/gslm/waveglow_256channels_standalone.pt waveglow_256channels_standalone.pt
get $B/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000 hifigan100.pt
get $B/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json hifigan100_config.json
get $B/hubert/hubert_base_ls960_L9_km500.bin L9_km500.bin
E=$B/textless_nlp/expresso/checkpoints/hifigan_expresso_lj_vctk_hubert_base_ls960_L9_km500
get $E/generator.pt exp500_generator.pt
get $E/config.json exp500_config.json
get $E/speakers.txt exp500_speakers.txt
get $E/styles.txt exp500_styles.txt
aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/prompts/ /data/pack/ --only-show-errors
echo "DY32G SETUP DONE"
