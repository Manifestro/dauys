"""DY-41 mid-run listening: the newest best checkpoint of a DY-41 job from Airon's store -> continuations of the fixed
listening prompts (DY-32 pack `prompts`, ids of DY-34's listening set), both samplings, next to DY-40's continuations of
the same prompts; Whisper large-v3-turbo transcripts; everything to HF manifestro/dauys-runs dy41/listen-mid/step_<N>/.

Per prompt id and variant: <id>_<variant>.wav (24 kHz) = the last 3 s of the prompt, 0.3 s of silence, then the
10 s the model said; also <id>_prompt.wav (the whole 15 s prompt).

    python scripts/dy41_listen.py --job DY-41/24 --out /data/dy41/listen
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dauys.dy40_lm import DLConfig, DepthLM, generate
from dy40_mimi import BUCKET, MSR, SR, Mimi, read_pack, to16

STORE = "s3://airon-checkpoints-876108973178/dauys"
IDS = [47, 60, 115, 118, 133, 150, 153, 188]  # DY-34's listening set (dy32 prompts pack)


def run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def newest_best(job: str) -> tuple[str, int]:
    rows = run(f"aws s3 ls {STORE}/{job}/ ").split()
    steps = sorted({r.rstrip("/") for r in rows if r.startswith("step-")})
    for s in reversed(steps):
        if "best.pt" in run(f"aws s3 ls {STORE}/{job}/{s}/ || true"):
            return f"{STORE}/{job}/{s}/best.pt", int(s.split("-")[1])
    raise SystemExit(f"no best.pt under {STORE}/{job}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job", default="DY-41/24")
    p.add_argument("--out", default="/data/dy41/listen")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    uri, step = newest_best(a.job)
    ckf = out / "best.pt"
    run(f"aws s3 cp {uri} {ckf} --only-show-errors")
    ck = torch.load(ckf, map_location="cpu", weights_only=False)
    model = DepthLM(DLConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(json.dumps({"checkpoint": uri, "step": ck["step"], "val": ck["val"]["ce"]}), flush=True)

    mm = Mimi(dev)
    names, waves = read_pack(Path("/data/dy32/prompts"), "prompt", "prompts")
    keep = [i for i, n in enumerate(names) if int(n) in IDS]
    names, waves = [names[i] for i in keep], [waves[i] for i in keep]
    toks = mm.encode(waves)
    n = min(len(t) for t in toks)
    K = model.cfg.k
    prompts = np.stack([np.r_[np.full((1, 8), K), t[-n:]] for t in toks]).astype(np.int64)

    d = out / f"step_{ck['step']}"
    d.mkdir(exist_ok=True)
    import torchaudio

    ctx = {nm: torchaudio.functional.resample(torch.from_numpy(w[-3 * SR:]), SR, MSR).numpy() for nm, w in zip(names, waves)}
    gap = np.zeros(int(0.3 * MSR), np.float32)
    files16: dict[str, np.ndarray] = {}
    for nm, w in zip(names, waves):
        sf.write(d / f"{nm}_prompt.wav", w, SR)
    for var in ("t10", "ra08"):
        gen = generate(model, prompts, 125, var, a.seed, dev)
        for nm, g in zip(names, gen):
            w = mm.decode(g)
            sf.write(d / f"{nm}_dy41-{var}.wav", np.concatenate([ctx[nm], gap, w]), MSR)
            files16[f"{nm}_dy41-{var}"] = to16(w)
    for var in ("t10", "ra08"):  # DY-40 (7.2k h) on the same prompts, made at its end
        src = Path("/data/dy40_listen") / var
        src.mkdir(parents=True, exist_ok=True)
        run(f"aws s3 sync {BUCKET}/dauys/dy40/listen24/m-{var}/ {src}/ --only-show-errors")
        for nm in names:
            f = src / f"{nm}.wav"
            if f.exists():
                w, _ = sf.read(f, dtype="float32")
                sf.write(d / f"{nm}_dy40-{var}.wav", np.concatenate([ctx[nm], gap, w]), MSR)
                files16[f"{nm}_dy40-{var}"] = to16(w)

    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device=dev)
    text = {}
    for k in sorted(files16):
        text[k] = asr({"raw": files16[k], "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()
    for nm, w in zip(names, waves):
        text[f"{nm}_prompt"] = asr({"raw": w, "sampling_rate": SR}, generate_kwargs={"language": "en"})["text"].strip()
    lines = [f"# DY-41 {a.job}, step {ck['step']} (val CE {ck['val']['ce']:.3f}) vs DY-40 (7.2k h, end)", "",
             "Each file: last 3 s of the prompt, 0.3 s silence, then 10 s of the model. Whisper large-v3-turbo.", ""]
    for nm in names:
        lines += [f"## prompt {nm}", f"- prompt: {text[nm + '_prompt']}"]
        for k in (f"{nm}_dy41-t10", f"{nm}_dy41-ra08", f"{nm}_dy40-t10", f"{nm}_dy40-ra08"):
            if k in text:
                lines.append(f"- {k.split('_', 1)[1]}: {text[k]}")
        lines.append("")
    (d / "README.md").write_text("\n".join(lines))
    (d / "transcripts.json").write_text(json.dumps(text, indent=1, ensure_ascii=False))

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = "manifestro/dauys-runs"
    assert api.repo_info(repo).private
    api.upload_folder(folder_path=str(d), path_in_repo=f"dy41/listen-mid/{d.name}", repo_id=repo)
    print(f"LISTEN https://huggingface.co/manifestro/dauys-runs/tree/main/dy41/listen-mid/{d.name}", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
