#!/usr/bin/env bash
# Alphabet test on a prepared DY-24 pod (after dy24_prep.sh): ZeroSpeech dev -> 500 h of phonemes -> DY-21-size LM
# on phonemes (diagnostic; text-trained recogniser as ruler) -> benchmarks -> upload HF phon500/.
#   nohup bash ~/dauys/scripts/phon_pod.sh > ~/phon.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until grep -q "PREP DONE" ~/prep.log 2>/dev/null; do sleep 20; done
set -a; . /etc/airon.env; set +a
export PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 PATH=$HOME/venv/bin:$PATH OMP_NUM_THREADS=8
W=/data/phon; P=/opt/dlami/nvme/phon; mkdir -p $W/marks
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }
if ! done_ zr; then
  pip install -q "datasets==3.6.0" 2>/dev/null
  command -v aria2c >/dev/null || sudo apt-get install -y -qq aria2 >/dev/null 2>&1
  mkdir -p /data/bench/zr
  aria2c -q -x 16 -s 16 -k 64M -d /data/bench -o sLM21.zip https://download.zerospeech.com/datasets/sLM21.dataset.zip || exit 1
  unzip -q /data/bench/sLM21.zip "*lexical/dev/*" "*syntactic/dev/*" -d /data/bench/zr || exit 1
  rm -f /data/bench/sLM21.zip
  mark zr
fi
if ! done_ phon; then
  python "$repo/scripts/phon_units.py" --hours 500 --held 4 --out $P > $W/phon.log 2>&1 || { echo "PHON FAILED"; tail -20 $W/phon.log; exit 1; }
  tail -2 $W/phon.log
  mark phon
fi
if ! done_ train; then
  python "$repo/scripts/dy21_train.py" --units $P --out $W/run --steps ${STEPS:-4000} --val-every 500 --tracking off > $W/train.log 2>&1 \
    || { echo "TRAIN FAILED"; tail -5 $W/train.log; exit 1; }
  mark train
fi
if ! done_ bench; then
  python "$repo/scripts/bench_phon.py" --zr /data/bench/zr --run $W/run/best.pt --out $W/bench.json > $W/bench.log 2>&1 \
    || { echo "BENCH FAILED"; tail -20 $W/bench.log; exit 1; }
  grep -E "^(swuggy|sblimp|tsc|ssc)" $W/bench.log
  mark bench
fi
python - <<'EOF2'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.repo_info("manifestro/dauys-runs").private
for f in ["bench.json", "train.log", "phon.log", "run/metadata.json", "run/best.pt"]:
    api.upload_file(path_or_fileobj=f"/data/phon/{f}", path_in_repo=f"phon500/{f}", repo_id="manifestro/dauys-runs")
print("UPLOADED phon500/")
EOF2
echo "PHON POD DONE"
