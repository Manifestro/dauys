"""DY-33 track S: small listening set from one checkpoint (S1 sampling) for fixed prompt ids of the DY-32 packs.

Per id NNN: prompt_NNN (15 s, whole prompt seen) -> slow codes -> 125 new steps at temperature 1.0 / top-p 1.0 ->
codec decoder -> kNN-VC HiFi-GAN. File = last 5 s of the prompt, a 0.3 s beep, the 10 s continuation (each peak
normalised to 0.9) -> <tag>__NNN.mp3. Whisper (large-v3-turbo, measuring only) transcribes the continuation alone.

    python scripts/dy33s_listen.py --ckpt /data/dy33s/run/snapshot.pt --codec /data/codec/frozen.pt \
        --ids 047,060,115,118,133,150,153,188 --tag s-early --out /data/dy33s/listen_early --s3 eval/listen_early
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dauys.dy32c_lm import MSConfig, MultiStreamLM
from dy32c_resynth import whisper
from dy32c_units import RVQUnits, sh
from dy33s_codec import load_codec
from dy33s_gen import S3, SR, encode, generate

TAIL_S, BEEP_S = 5, 0.3


def beep() -> np.ndarray:
    t = np.arange(int(BEEP_S * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--codec", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--s3", default="")
    p.add_argument("--new-steps", type=int, default=125)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = a.ids.split(",")
    waves = []
    for i in ids:
        d = Path("/opt/dlami/nvme/prompts") if int(i) < 50 else Path("/opt/dlami/nvme/prompts_x")
        f = d / (f"prompt_{int(i):02d}.wav" if int(i) < 50 else f"prompt_{int(i):03d}.wav")
        w, sr = sf.read(f, dtype="float32")
        assert sr == SR
        waves.append(w if w.ndim == 1 else w.mean(1))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = MSConfig(**ck["cfg"])
    order = ck["order"]
    codec, cck = load_codec(Path(a.codec), dev)
    u = RVQUnits(dev)
    toks = encode(u, codec, waves, dev)
    n = min(len(t) for t in toks)
    prompts = np.stack([np.r_[np.full((1, cfg.streams), cfg.k), t[-n:][:, order]] for t in toks]).astype(np.int64)
    model = MultiStreamLM(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    gen = generate(model, prompts, a.new_steps, "t10", a.seed, dev)
    inv = [order.index(c) for c in sorted(order)]
    conts = []
    for gq in gen:
        c = torch.from_numpy(gq[:, inv].astype(np.int64)).to(dev)
        f = codec.decode(c[None])[0]
        conts.append(u.knn.vocode(f[None].float()).reshape(-1).cpu().numpy().astype(np.float32))
    del model, u
    torch.cuda.empty_cache()
    texts = whisper(conts, dev)
    pk = lambda v: 0.9 * v / max(float(np.abs(v).max()), 1e-6)
    res = {}
    for i, w, c, t in zip(ids, waves, conts, texts):
        y = np.concatenate([pk(w[-TAIL_S * SR :]), beep(), pk(c)])
        wav = out / f"{a.tag}__{i}.wav"
        sf.write(wav, y, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", "64k",
                        str(wav.with_suffix(".mp3"))], check=True)
        wav.unlink()
        res[i] = {"whisper": t, "words": len(t.split())}
    words = [r["words"] for r in res.values()]
    info = {"step": ck["step"], "val_ce": ck["val"]["ce"], "val": ck["val"], "checkpoint": a.ckpt,
            "sampling": "S1: temperature 1.0, top-p 1.0, all levels; EOS/PAD banned", "seed": a.seed,
            "prompt_seen": f"all {n} steps ({n / 12.5:.2f} s) of the 15 s prompt after one EOS step",
            "continuation_s": a.new_steps / 12.5, "file": "last 5 s of prompt + 0.3 s beep + 10 s continuation",
            "asr": "openai/whisper-large-v3-turbo on the continuation only (measuring only)",
            "words_per_10s_median": float(np.median(words)), "transcripts": res}
    (out / "transcripts.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps({"step": info["step"], "val_ce": info["val_ce"], "words_median": info["words_per_10s_median"]}), flush=True)
    if a.s3:
        sh(f"aws s3 cp {out}/ {S3}/{a.s3}/ --recursive --only-show-errors")
        print(f"UPLOADED {a.s3}", flush=True)


if __name__ == "__main__":
    main()
