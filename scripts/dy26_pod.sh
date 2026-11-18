#!/usr/bin/env bash
# DY-26: fast "is this alphabet good for an LM" proxy, calibrated on phonemes vs DY-14 units, on one Airon AWS GPU.
# setup -> ZeroSpeech dev + DY-14 centroids -> ~100 h of phonemes (first shards of the phon500 order, the same 4
# held-out shards) -> DY-14 units of the same shards -> checklist -> dry runs (200 steps + 50-pair bench each)
# -> ~50M LM per alphabet (best by val) -> sWUGGY/sBLIMP 3000 pairs -> HF dy26/.
#   nohup bash ~/dauys/scripts/dy26_pod.sh > ~/dy26.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
for v in HF_TOKEN AIRON_API_KEY AIRON_URL; do [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }; done
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy26; mkdir -p $W/marks /data/dy14
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
EXP="DY-26: быстрый прокси «годится ли алфавит для языковой модели» — калибровка на фонемах и DY-14"
CFG=512,8,8,2048,1024
STEPS=${STEPS:-3000}
mark() { touch "$W/marks/$1"; echo "=== $1 done $(date -u +%H:%M:%S)"; }
done_() { [[ -e "$W/marks/$1" ]]; }
if ! done_ setup; then
  python3 -m pip install -q --user virtualenv 2>/dev/null
  python3 -m virtualenv -q ~/venv
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  printf "torch==2.7.1\ntorchaudio==2.7.1\n" > /tmp/constraints.txt
  ~/venv/bin/pip install -q -c /tmp/constraints.txt "numpy<2.1" scipy transformers==4.46.3 huggingface_hub safetensors \
    soundfile pyyaml numba sylber==0.1.4 accelerate airon "datasets==3.6.0"
  ~/venv/bin/python -c "import torch, sylber, airon; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))" || exit 1
  mark setup
fi
export PATH=$HOME/venv/bin:$PATH PYTHONPATH=$repo/src PYTHONWARNINGS=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2
if ! done_ zr; then
  mkdir -p /data/bench
  aws s3 cp $S3/dauys/bench/zr_dev.tar - --only-show-errors | tar -C /data/bench -xf - || exit 1
  aws s3 cp "$S3/dauys/dy14-grid/dy14-grid.tar" - --only-show-errors | tar -C /data/dy14 -xf - centroids.pt || exit 1
  mark zr
fi
if ! done_ phon; then
  python "$repo/scripts/phon_units.py" --hours 100 --held 4 --out $W/phon --work /data/phon_work > $W/phon.log 2>&1 || { echo "PHON FAILED"; tail -20 $W/phon.log; exit 1; }
  tail -1 $W/phon.log
  mark phon
fi
if ! done_ dy14; then
  mkdir -p $W/dy14
  for f in $W/phon/*.npz; do
    aws s3 cp "$S3/dauys/dc-mono-units/$(basename $f)" $W/dy14/ --only-show-errors || { echo "DY14 UNITS MISSING $(basename $f)"; exit 1; }
  done
  echo "dy14 shards: $(ls $W/dy14 | wc -l)"
  mark dy14
fi
if ! done_ checklist; then
  python "$repo/scripts/dy26_checklist.py" --phon $W/phon --dy14 $W/dy14 --out $W/checklist.json > $W/checklist.log 2>&1
  grep -E "^(LEAK|KNOWN|OVERFIT|CHECKLIST)" $W/checklist.log
  grep -q "CHECKLIST PASS" $W/checklist.log || { echo "CHECKLIST FAILED"; tail -5 $W/checklist.log; exit 1; }
  # dry run: 200 steps on 2 shards (one held out) + a 50-pair bench, per alphabet
  for a in phon dy14; do
    mkdir -p $W/dry_$a/data
    h=$(python -c "import sys; sys.path.insert(0, '$repo/scripts'); from dy21_train import held_out; import glob, os; fs=sorted(glob.glob('$W/$a/*.npz')); print(next(f for f in fs if held_out(os.path.basename(f)[:-4])), next(f for f in fs if not held_out(os.path.basename(f)[:-4])))")
    for f in $h; do ln -sf $f $W/dry_$a/data/; done
    python "$repo/scripts/dy21_train.py" --units $W/dry_$a/data --out $W/dry_$a/run --steps 200 --val-every 100 --warmup 50 --cfg $CFG --tracking off > $W/dry_$a.log 2>&1 || { echo "DRY $a TRAIN FAILED"; tail -5 $W/dry_$a.log; exit 1; }
  done
  python "$repo/scripts/bench_phon.py" --zr /data/bench/zr --run $W/dry_phon/run/best.pt --out $W/dry_bench_phon.json --pairs 50 --tasks swuggy,sblimp > $W/dry_bench_phon.log 2>&1 || { echo "DRY BENCH phon FAILED"; tail -5 $W/dry_bench_phon.log; exit 1; }
  python "$repo/scripts/bench_slm.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --dy21 $W/dry_dy14/run/best.pt --out $W/dry_bench_dy14.json --pairs 50 --tasks swuggy,sblimp > $W/dry_bench_dy14.log 2>&1 || { echo "DRY BENCH dy14 FAILED"; tail -5 $W/dry_bench_dy14.log; exit 1; }
  grep -hE "^(swuggy|sblimp)" $W/dry_bench_phon.log $W/dry_bench_dy14.log
  mark checklist
fi
for a in phon dy14; do
  if ! done_ train_$a; then
    python "$repo/scripts/dy21_train.py" --units $W/$a --out $W/run_$a --steps $STEPS --val-every 100 --patience 5 --warmup 500 \
      --cfg $CFG --experiment "$EXP" --name dy26-$a > $W/train_$a.log 2>&1 &
  fi
done
wait
for a in phon dy14; do
  [[ -s $W/run_$a/best.pt ]] || { echo "TRAIN $a FAILED"; tail -5 $W/train_$a.log; exit 1; }
  grep -aE "early_stop|TRAIN DONE" $W/train_$a.log | tail -1
  mark train_$a
done
if ! done_ bench; then
  python "$repo/scripts/bench_phon.py" --zr /data/bench/zr --run $W/run_phon/best.pt --out $W/bench_phon.json --tasks swuggy,sblimp > $W/bench_phon.log 2>&1 &
  python "$repo/scripts/bench_slm.py" --zr /data/bench/zr --centroids /data/dy14/centroids.pt --dy21 $W/run_dy14/best.pt --out $W/bench_dy14.json --tasks swuggy,sblimp > $W/bench_dy14.log 2>&1 &
  wait
  grep -E "^(swuggy|sblimp)" $W/bench_phon.log | sed 's/^/phon /'
  grep -E "^(swuggy|sblimp)" $W/bench_dy14.log | sed 's/^/dy14 /'
  grep -q "BENCH DONE" $W/bench_phon.log && grep -q "BENCH DONE" $W/bench_dy14.log || { echo "BENCH FAILED"; exit 1; }
  mark bench
fi
python - <<'EOF2'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.repo_info("manifestro/dauys-runs").private
w = "/data/dy26"
for f in glob.glob(f"{w}/*.json") + glob.glob(f"{w}/*.log") + glob.glob(f"{w}/run_*/best.pt") + glob.glob(f"{w}/run_*/metadata.json"):
    api.upload_file(path_or_fileobj=f, path_in_repo="dy26/" + os.path.relpath(f, w), repo_id="manifestro/dauys-runs")
print("UPLOADED dy26/")
EOF2
echo "DY26 POD DONE"
