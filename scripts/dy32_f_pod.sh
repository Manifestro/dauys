#!/usr/bin/env bash
# DY-32 F on one AWS g7e (RTX PRO 6000): GSLM2 decoder to audio (SpidR 2.0 six-stream units -> unit HiFi-GAN 16 kHz,
# plus the GSLM2 duration predictor), using the gslm2 code base (github.com/facebookresearch/gslm2, CC BY-NC 4.0).
# Each stage is a step that was run by hand in this order; marks make reruns skip finished stages.
#   bash ~/dauys/scripts/dy32_f_pod.sh <stage>     stages: setup lj vocoder ll duration serve
set -uo pipefail
W=/opt/dlami/nvme/w; R=$(cd "$(dirname "$0")/.." && pwd); C=$R/scripts/dy32_f
sudo chown "$USER" /opt/dlami/nvme; mkdir -p $W/marks; cd $W
envs() { . $W/venv/bin/activate; export LD_LIBRARY_PATH=$W/ff/lib PATH=$W/ff/bin:$PATH PYTHONWARNINGS=ignore; }
mark() { touch $W/marks/$1; echo "=== $1 done $(date -u +%H:%M) UTC"; }
case ${1:-} in
setup)  # Python 3.12 venv with gslm2[native] (torch 2.13 cu130), FFmpeg 8 from conda-forge for torchcodec, SpidR 2.0
  git clone -q https://github.com/facebookresearch/gslm2 && git -C gslm2 checkout -q e3020fe804075c6e17a22aa78c566f6ac7891c77
  curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
  uv venv -p 3.12 venv && . venv/bin/activate && (cd gslm2 && uv pip install -e ".[native]" --torch-backend cu130)
  uv pip install jiwer airon awscli huggingface_hub
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  ./bin/micromamba create -y -q -p $W/ff -c conda-forge "ffmpeg=8"
  mkdir -p ckpt/spidr && for f in final.pt config.json README.md manifest.json LICENSE SHA256SUMS; do
    curl -fsL https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/spidr-2.0/$f -o ckpt/spidr/$f; done
  (cd ckpt/spidr && sha256sum -c SHA256SUMS) && mark setup ;;
lj)  # LJ Speech 1.1 -> 16 kHz + fixed gslm2 splits -> full-rate six-stream codes (layers 5-10)
  envs
  [[ -d LJSpeech-1.1 ]] || { wget -q https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 && tar xjf LJSpeech-1.1.tar.bz2; }
  python gslm2/scripts/prepare_ljspeech_manifests.py --data-dir $W/LJSpeech-1.1 --output-dir $W/ljprep --workers 8
  gslm2-extract-units --config $C/extract_lj.yaml && mark lj ;;
vocoder)  # gslm2 unit HiFi-GAN trainer, one GPU, bf16; resumes from the latest checkpoint in the work dir
  envs; torchrun --standalone --nproc_per_node=1 -m gslm2.vocoder $C/vocoder.yaml ;;
ll)  # Libri-Light small -> 60 s VAD segments (WAV) -> manifest (5% of books held out for validation) -> codes
  envs; mkdir -p ll/raw ll/manifests
  [[ -d ll/raw/small ]] || { curl -fsSL https://dl.fbaipublicfiles.com/librilight/data/small.tar -o ll/raw/small.tar && tar -xf ll/raw/small.tar -C ll/raw; }
  [[ -d llrepo ]] || git clone -q https://github.com/facebookresearch/libri-light.git llrepo
  python llrepo/data_preparation/cut_by_vad.py --input_dir ll/raw/small --output_dir ll/small_segmented \
    --target_len_sec 60 --n_workers 6 --out_extension .wav
  python -m gslm2.encoder.data.write_manifest ll/small_segmented ll/manifests/all.csv --ext .wav --num-workers 6
  python - <<'EOF'
import csv, random
rows = list(csv.reader(open("ll/manifests/all.csv")))
head, body = rows[0], rows[1:]
books = sorted({r[1].split("/")[-3] + "/" + r[1].split("/")[-2] for r in body})
random.Random(0).shuffle(books)
dev = set(books[: max(1, len(books) // 20)])
for name, keep in (("small_segmented_60s", False), ("small_segmented_60s_dev", True)):
    with open(f"ll/manifests/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head)
        w.writerows(r for r in body if ((r[1].split("/")[-3] + "/" + r[1].split("/")[-2]) in dev) == keep)
EOF
  gslm2-extract-units --config $C/extract_ll.yaml && mark ll ;;
duration)
  envs; gslm2-train-duration $C/duration.yaml && mark duration ;;
serve)  # a decoding round on every new vocoder checkpoint / new track E units (S3 outputs/dauys/dy32/{resynth,gen,f})
  envs; python $R/scripts/dy32_f_decode.py serve ;;
*) echo "stage?"; exit 1 ;;
esac
