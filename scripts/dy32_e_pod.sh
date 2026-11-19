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


# ======== qa.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
S3=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/gen_units
G="python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt/fullrate --temperature 0.8 --top-p 0.95"
$G --lm ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b --out gen/ll60kh-1b-t08 --group 10 > logs/gen_1b_t08.log 2>&1 && aws s3 cp gen/ll60kh-1b-t08/ $S3/ll60kh-1b-t08/ --recursive --only-show-errors && echo "1b-t08 up $(date -u)"
$G --lm ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_cs-6mh_7b --out gen/cs6mh-7b-t08 --group 5 > logs/gen_7b_t08.log 2>&1 && aws s3 cp gen/cs6mh-7b-t08/ $S3/cs6mh-7b-t08/ --recursive --only-show-errors && echo "7b-t08 up $(date -u)"
echo QA_DONE


# ======== qb.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
E=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e
mkdir -p pack_x && aws s3 sync s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/prompts_x/ pack_x/ --only-show-errors
ls pack_x/prompt_*.wav | wc -l
python dauys/scripts/dy32_gslm2.py wavs --in pack_x --glob "prompt_*.wav" --out u/prompt_x > logs/wavs_x.log 2>&1 && aws s3 cp u/prompt_x/ $E/prompt_units_x/ --recursive --only-show-errors && echo "prompt_x units up $(date -u)"
until grep -q "1b-t08 up" logs/qa.log; do sleep 10; done
python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt_x/fullrate --lm ckpt/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b --out gen/ll60kh-1b-x --group 10 > logs/gen_1b_x.log 2>&1 && aws s3 cp gen/ll60kh-1b-x/ $E/gen_units/ll60kh-1b-x/ --recursive --only-show-errors && echo "1b-x up $(date -u)"
echo QB_DONE


# ======== qc3.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
E=s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/gen_units
until [ -f runs/dy32-e/DONE ] || [ $(date -u +%H%M) -ge 0650 ]; do sleep 20; done
[ -f runs/dy32-e/DONE ] || touch STOP_TRAIN
until [ -f runs/dy32-e/DONE ] && [ -f snap_ours/step.json ]; do sleep 5; done
sleep 5
hours() { python - "$1" <<PY
import json,sys; p=sys.argv[1]; i=json.load(open(p)); st=json.load(open("snap_ours/step.json"))
i["train_step"]=st["step"]; i["val"]=st["val"]
i["init"]="released GSLM2 Libri-Light 60K h 1B, fine-tuned on DuplexChat SpidR 2.0 units (dy32-e)"
i["train_data_hours"]=824.3; i["train_audio_hours_seen_approx"]=round(st["step"]*32*1024/(0.737*50*3600),1)
i["train_batch"]="32 windows x 1024 dedup positions"; json.dump(i,open(p,"w"),indent=2)
PY
}
(python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt/fullrate --lm snap_ours --out gen/ours-t08 --group 10 --temperature 0.8 --top-p 0.95 > logs/gen_ours_t08.log 2>&1 && hours gen/ours-t08/info.json && aws s3 cp gen/ours-t08/ $E/ours-t08/ --recursive --only-show-errors && echo "ours-t08 up $(date -u)"
 python dauys/scripts/dy32_gslm2.py gen --prompts u/prompt_x/fullrate --lm snap_ours --out gen/ours-t08-x --group 25 --temperature 0.8 --top-p 0.95 > logs/gen_ours_t08_x.log 2>&1 && hours gen/ours-t08-x/info.json && aws s3 cp gen/ours-t08-x/ $E/ours-t08-x/ --recursive --only-show-errors && echo "ours-t08-x up $(date -u)") &
until grep -q "ours step" logs/post.log; do sleep 5; done
hours gen/ours/info.json; aws s3 cp gen/ours/info.json $E/ours/info.json --only-show-errors; echo "ours info up $(date -u)"
wait
echo QC_DONE


# ======== final.sh
. /opt/dlami/nvme/mf/etc/profile.d/conda.sh; conda activate gslm2_lm; cd /opt/dlami/nvme
until grep -q POST_DONE logs/post.log; do sleep 20; done
set -a; . /etc/airon.env; set +a
python hf_up.py > logs/hf.log 2>&1
aws s3 cp hf_stage/logs/ s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/logs/ --recursive --only-show-errors
aws s3 cp runs/dy32-e/metadata.json s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/e/logs/metadata.json --only-show-errors
echo FINAL_DONE >> logs/hf.log


# ======== hf_up.py (python)
: <<"PY"
import os, shutil, pathlib
from huggingface_hub import HfApi
root = pathlib.Path("/opt/dlami/nvme")
st = root / "hf_stage"
shutil.rmtree(st, ignore_errors=True)
(st / "best").mkdir(parents=True)
for f in ["model.pt", "config.yaml", "step.json"]:
    os.link(root / "runs/dy32-e/best" / f, st / "best" / f)
shutil.copy(root / "runs/dy32-e/metadata.json", st / "metadata.json")
(st / "logs").mkdir()
for f in ["train.log", "train_aborted315m.log", "shards_tr.log", "shards_va.log", "score.log", "gen_1b.log", "gen_7b.log", "gen_350m_g5.log", "gen_ours-early.log", "gen_ours.log", "post.log"]:
    if (root / "logs" / f).exists():
        shutil.copy(root / "logs" / f, st / "logs" / f)
shutil.copy(root / "score_released.json", st / "score_released.json")
(st / "LICENSE_NOTE.md").write_text("Fine-tuned from the GSLM2 release (github.com/facebookresearch/gslm2) Libri-Light 60K h 1B checkpoint; code and checkpoints CC BY-NC 4.0 -> research use only. Units: SpidR 2.0 layers 5..10 (same release).\n")
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(folder_path=str(st), repo_id="manifestro/dauys-runs", path_in_repo="dy32/e", repo_type="model", commit_message="DY-32 E: 1B GSLM2 fine-tuned on DuplexChat units, logs")
print("HF DONE")
PY
