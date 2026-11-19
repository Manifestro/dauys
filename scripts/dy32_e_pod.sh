#!/usr/bin/env bash
# DY-32 track E machine steps, in the order they ran (under /opt/dlami/nvme). Each section was its own file there.

# ======== setup_env.sh
set -e
cd /opt/dlami/nvme
curl -sfL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o mf.sh
bash mf.sh -b -p /opt/dlami/nvme/mf
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh
conda create -y -n gslm2_lm python=3.12 ffmpeg=8 libsndfile cudnn=9.25.1.1 -c conda-forge
conda activate gslm2_lm
pip install -q uv
cd /opt/dlami/nvme/gslm2
uv pip install -e ".[lm]" --torch-backend cu128 --overrides <(printf "%s\n" "numpy>=2.1.3,<2.4")
uv pip install boto3 soundfile
python -c "import torch, fairseq2, numpy; print(torch.__version__, fairseq2.__version__, numpy.__version__, torch.cuda.is_available())"
echo ENV_DONE


# ======== dl.sh
B=https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints
cd /opt/dlami/nvme/ckpt
mkdir -p spidr-2.0; for f in final.pt config.json README.md manifest.json LICENSE SHA256SUMS; do curl -sfL $B/spidr-2.0/$f -o spidr-2.0/$f; done
(cd spidr-2.0 && sha256sum -c SHA256SUMS) ; echo ENC_DONE
for M in spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_cs-6mh_7b; do
 mkdir -p $M; for f in model.pt config.yaml units.json README.md manifest.json LICENSE SHA256SUMS; do curl -sfL $B/lm/$M/$f -o $M/$f; done
 (cd $M && sha256sum -c SHA256SUMS); echo DONE $M
done


# ======== dl7b.sh
B=https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_cs-6mh_7b
D=/opt/dlami/nvme/ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_cs-6mh_7b
mkdir -p $D/parts; cd $D
for f in config.yaml units.json README.md manifest.json LICENSE SHA256SUMS; do curl -sfL $B/$f -o $f; done
SZ=13128484783; N=48; CH=$(( (SZ + N - 1) / N ))
for i in $(seq 0 $((N-1))); do s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SZ ] && e=$((SZ-1));
  (until curl -sf --retry 5 -r $s-$e $B/model.pt -o parts/$(printf %03d $i) && [ $(stat -c%s parts/$(printf %03d $i)) -eq $((e-s+1)) ]; do sleep 2; done) & done
wait; cat parts/* > model.pt && rm -rf parts && sha256sum -c SHA256SUMS; echo DL7B_DONE


# ======== launch.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
set -a; . /etc/airon.env; set +a
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/gen_units
until grep -q "GEN DONE" logs/gen_7b.log; do sleep 10; done
aws s3 cp gen/cs6mh-7b/ $S3/cs6mh-7b/ --recursive --only-show-errors
python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt/fullrate --lm ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m --out gen/ll6kh-350m-t08 --group 5 --temperature 0.8 --top-p 0.95 > logs/gen_350m_t08.log 2>&1 && aws s3 cp gen/ll6kh-350m-t08/ $S3/ll6kh-350m-t08/ --recursive --only-show-errors
touch STOP_EXTRACT
while pgrep -f "[s]hards --shards lbig-00004" >/dev/null; do sleep 5; done
rm -rf units_train; mkdir units_train; ln -s /opt/dlami/nvme/work_tr/*.npz /opt/dlami/nvme/work_va/*.npz units_train/
export DY32_COMMIT=$(git -C dauys rev-parse HEAD)
python dauys/scripts/dy32_gslm2.py score --units units_train --lms ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m,ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b,ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_cs-6mh_7b --out score_released.json > logs/score.log 2>&1
python dauys/scripts/dy32_gslm2.py train --units units_train --init ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m --out runs/dy32-e --steps 100000 --until $(date -d "2026-11-19 07:05:00 UTC" +%s) --name dy32-e > logs/train.log 2>&1


# ======== train1b.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
set -a; . /etc/airon.env; set +a; export AIRON_URL=${AIRON_URL%/v1}
export DY32_COMMIT=$(git -C dauys rev-parse HEAD)
python dauys/scripts/dy32_gslm2.py train --units units_train --init ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b --out runs/dy32-e --steps 100000 --micro 8 --until $(date -d "2026-11-19 07:05:00 UTC" +%s) --name dy32-e > logs/train.log 2>&1


# ======== post.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/gen_units
G="python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt/fullrate --group 5"
snap() {  # $1 = variant name
  rm -rf snap_$1; cp -r runs/dy32-e/best snap_$1
  N=$(python -c "import json;print(json.load(open(\"snap_$1/step.json\"))[\"step\"])")
  $G --lm snap_$1 --out gen/$1 > logs/gen_$1.log 2>&1 || return
  python - <<PY
import json; p="gen/$1/info.json"; i=json.load(open(p)); i["train_step"]=$N; i["val"]=json.load(open("snap_$1/step.json"))["val"]
i["init"]="released GSLM2 Libri-Light 60K h 1B, fine-tuned on DuplexChat SpidR 2.0 units (dy32-e)"; json.dump(i,open(p,"w"),indent=2)
PY
  aws s3 cp gen/$1/ $S3/$1/ --recursive --only-show-errors
  aws s3 cp gen/$1/ $S3/$1/step$N/ --recursive --only-show-errors
  echo "$1 step $N uploaded $(date -u)"
}
until [ $(date -u +%H%M) -ge 0425 ]; do sleep 30; done
snap ours-early
until [ -f runs/dy32-e/DONE ]; do sleep 20; done
snap ours
echo POST_DONE
