#!/usr/bin/env bash
# DY-27 on one Airon AWS GPU: setup -> the DY-26 shards (same rule: 4 held out, then training shards in sorted order
# up to 100 h) -> WavLM L12 k-means 500 (training shards) -> raw codes -> checklist -> LM0 on K500 and the acoustic
# merge 500->96 (parallel) -> LM merge 500->96 from LM0 -> LM on it -> benchmarks of all three -> gate-R reference
# on the 200 live words -> HF dy27/.
#   nohup bash ~/dauys/scripts/dy27_pod.sh > ~/dy27.log 2>&1 &
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
until [[ -s /etc/airon.env ]]; do sleep 5; done
set -a; . /etc/airon.env; set +a
AIRON_URL=${AIRON_URL%/v1}
for v in HF_TOKEN AIRON_API_KEY AIRON_URL; do [[ -n "${!v:-}" ]] || { echo "secret $v is not set; not starting"; exit 1; }; done
export DY21_COMMIT=$(git -C "$repo" rev-parse HEAD)
sudo mkdir -p /data && sudo chown "$USER" /data
W=/data/dy27; mkdir -p $W/marks
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs
EXP="DY-27: петля сверху вниз, шаг 1 — объединение кодов WavLM по предсказаниям LM против объединения по звуку (прокси DY-26)"
CFG=512,8,8,2048,1024
STEPS=${STEPS:-3000}
TR="--steps $STEPS --val-every 100 --patience 5 --warmup 500 --cfg $CFG --experiment"
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
if ! done_ data; then
  mkdir -p /data/bench $W/shards
  aws s3 cp $S3/dauys/bench/zr_dev.tar - --only-show-errors | tar -C /data/bench -xf - || exit 1
  aws s3 cp $S3/dauys/bench/words_pick.pkl $W/words_pick.pkl --only-show-errors || exit 1
  python - <<'EOF3' || { echo "DATA FAILED"; exit 1; }
import subprocess, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/dauys/scripts")
from dy21_train import held_out
SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"
sh = lambda c: subprocess.run(c, shell=True, check=True, capture_output=True, text=True).stdout
shards = sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines() if l.strip().startswith("PRE"))
plan = [s for s in shards if held_out(s)][:4] + [s for s in shards if not held_out(s)]
h = 0.0
for s in plan:
    if not held_out(s) and h >= 100:
        break
    d = f"/data/dy27/shards/{s}"
    sh(f"mkdir -p {d} && aws s3 cp {SRC}/{s}/index.npy {d}/ --only-show-errors && aws s3 cp {SRC}/{s}/segs.bin {d}/ --only-show-errors")
    idx = np.load(f"{d}/index.npy")
    hs = sum(int(n) for _, n in idx if n >= 5 * 16000) / 16000 / 3600
    if not held_out(s):
        h += hs
    print(s, "held" if held_out(s) else "train", round(hs, 1), round(h, 1), flush=True)
EOF3
  echo "shards: $(ls $W/shards | tr '\n' ' ')"
  mark data
fi
if ! done_ kmeans; then
  python "$repo/scripts/dy27.py" kmeans --shards $W/shards --out $W/centroids.pt > $W/kmeans.log 2>&1 || { echo "KMEANS FAILED"; tail -5 $W/kmeans.log; exit 1; }
  grep KMEANS $W/kmeans.log
  python "$repo/scripts/dy27.py" tokenize --shards $W/shards --centroids $W/centroids.pt --out $W/raw > $W/tokenize.log 2>&1 || { echo "TOKENIZE FAILED"; tail -5 $W/tokenize.log; exit 1; }
  python "$repo/scripts/dy27.py" apply --raw $W/raw --out $W/k500 | tee -a $W/apply.log
  python "$repo/scripts/dy27.py" ac_merge --centroids $W/centroids.pt --to 96 --out $W/map_ac96.npy | tee -a $W/merge.log
  python "$repo/scripts/dy27.py" apply --raw $W/raw --out $W/ac96 --map $W/map_ac96.npy | tee -a $W/apply.log
  mark kmeans
fi
if ! done_ checklist; then
  python "$repo/scripts/dy26_checklist.py" --phon $W/k500 --dy14 $W/k500 --out $W/checklist_lm.json > $W/checklist.log 2>&1
  python "$repo/scripts/dy27.py" known --out $W/known.json >> $W/checklist.log 2>&1
  grep -E "^(LEAK|OVERFIT|KNOWN)" $W/checklist.log
  grep -q '"pass": true' $W/known.json && grep -q '"pass": true' <(python -c "import json; r=json.load(open('$W/checklist_lm.json')); print(json.dumps({'pass': r['overfit']['pass'] and r['leak']['pass']}))") \
    || { echo "CHECKLIST FAILED"; exit 1; }
  for a in k500 ac96; do
    mkdir -p $W/dry_$a/data
    h=$(python -c "import sys; sys.path.insert(0, '$repo/scripts'); from dy21_train import held_out; import glob, os; fs=sorted(glob.glob('$W/$a/*.npz')); print(next(f for f in fs if held_out(os.path.basename(f)[:-4])), next(f for f in fs if not held_out(os.path.basename(f)[:-4])))")
    for f in $h; do ln -sf $f $W/dry_$a/data/; done
    python "$repo/scripts/dy21_train.py" --units $W/dry_$a/data --out $W/dry_$a/run --steps 200 --val-every 100 --warmup 50 --cfg $CFG --tracking off > $W/dry_$a.log 2>&1 || { echo "DRY $a FAILED"; tail -5 $W/dry_$a.log; exit 1; }
  done
  python "$repo/scripts/dy27.py" lm_merge --units $W/dry_k500/data --run $W/dry_k500/run/best.pt --to 96 --windows 200 --out $W/dry_map_lm96.npy > $W/dry_merge.log 2>&1 || { echo "DRY LM MERGE FAILED"; tail -5 $W/dry_merge.log; exit 1; }
  for a in k500 ac96 lm96; do
    case $a in k500) m=""; r=$W/dry_k500/run/best.pt;; ac96) m="--map $W/map_ac96.npy"; r=$W/dry_ac96/run/best.pt;; lm96) m="--map $W/dry_map_lm96.npy"; r=$W/dry_ac96/run/best.pt;; esac
    python "$repo/scripts/dy27.py" bench --zr /data/bench/zr --centroids $W/centroids.pt $m --run $r --pairs 50 --out $W/dry_bench_$a.json > $W/dry_bench_$a.log 2>&1 || { echo "DRY BENCH $a FAILED"; tail -5 $W/dry_bench_$a.log; exit 1; }
    grep -E "^(swuggy|sblimp)" $W/dry_bench_$a.log | sed "s/^/dry $a /"
  done
  mark checklist
fi
for a in k500 ac96; do
  done_ train_$a || python "$repo/scripts/dy21_train.py" --units $W/$a --out $W/run_$a $TR "$EXP" --name dy27-$a > $W/train_$a.log 2>&1 &
done
wait
for a in k500 ac96; do [[ -s $W/run_$a/best.pt ]] || { echo "TRAIN $a FAILED"; tail -5 $W/train_$a.log; exit 1; }; grep -aE "early_stop|TRAIN DONE" $W/train_$a.log | tail -1 | sed "s/^/$a /"; mark train_$a; done
if ! done_ train_lm96; then
  python "$repo/scripts/dy27.py" lm_merge --units $W/k500 --run $W/run_k500/best.pt --to 96 --out $W/map_lm96.npy | tee -a $W/merge.log
  python "$repo/scripts/dy27.py" apply --raw $W/raw --out $W/lm96 --map $W/map_lm96.npy | tee -a $W/apply.log
  for a in k500 ac96; do
    m=""; [[ $a == ac96 ]] && m="--map $W/map_ac96.npy"
    python "$repo/scripts/dy27.py" bench --zr /data/bench/zr --centroids $W/centroids.pt $m --run $W/run_$a/best.pt --out $W/bench_$a.json > $W/bench_$a.log 2>&1 &
  done
  python "$repo/scripts/dy21_train.py" --units $W/lm96 --out $W/run_lm96 $TR "$EXP" --name dy27-lm96 > $W/train_lm96.log 2>&1
  [[ -s $W/run_lm96/best.pt ]] || { echo "TRAIN lm96 FAILED"; tail -5 $W/train_lm96.log; exit 1; }
  grep -aE "early_stop|TRAIN DONE" $W/train_lm96.log | tail -1 | sed "s/^/lm96 /"
  mark train_lm96
fi
if ! done_ bench; then
  python "$repo/scripts/dy27.py" bench --zr /data/bench/zr --centroids $W/centroids.pt --map $W/map_lm96.npy --run $W/run_lm96/best.pt --out $W/bench_lm96.json > $W/bench_lm96.log 2>&1 &
  python "$repo/scripts/dy27.py" words --cache $W/words_pick.pkl --centroids $W/centroids.pt --maps none,$W/map_lm96.npy,$W/map_ac96.npy --out $W/words.json > $W/words.log 2>&1 &
  wait
  for a in k500 ac96 lm96; do grep -E "^(swuggy|sblimp)" $W/bench_$a.log | sed "s/^/$a /"; grep -q "BENCH DONE" $W/bench_$a.log || { echo "BENCH $a FAILED"; exit 1; }; done
  grep -E "^WORDS " $W/words.log
  mark bench
fi
python - <<'EOF2'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
assert api.repo_info("manifestro/dauys-runs").private
w = "/data/dy27"
fs = glob.glob(f"{w}/*.json") + glob.glob(f"{w}/*.log") + glob.glob(f"{w}/map_*.npy") + [f"{w}/centroids.pt"] \
    + glob.glob(f"{w}/run_*/best.pt") + glob.glob(f"{w}/run_*/metadata.json")
for f in fs:
    api.upload_file(path_or_fileobj=f, path_in_repo="dy27/" + os.path.relpath(f, w), repo_id="manifestro/dauys-runs")
print("UPLOADED dy27/")
EOF2
echo "DY27 POD DONE"
