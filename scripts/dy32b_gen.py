"""DY-32 track B: continuations of the common prompt pack with a UnitLM trained on one-voice units.

Each prompt_XX.wav (15 s) -> canonical codes (dy32b_channel, the whole prompt fits the context) -> EOS + codes ->
the LM samples --seconds of codes (temperature 0.8, top-p 0.95, EOS banned; one generator, seed 1234, all prompts in
one batch) -> decode table + kNN-VC vocoder -> XX.wav (16 kHz, the continuation only) and info.json.

    python scripts/dy32b_gen.py --ckpt /data/b/run/best.pt --cb /data/b/cb/lj_k2000_r25.pt --pack /data/b/pack \
        --out /data/b/gen
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy32b_channel import SR, Channel, Codebook

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cb", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--note", default="")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ch, cb = Channel(), Codebook(a.cb)
    files = sorted(glob.glob(f"{a.pack}/prompt_*.wav"))
    ids = [Path(f).stem.split("_")[1] for f in files]
    wavs = [sf.read(f, dtype="float32")[0] for f in files]
    codes = [cb.encode(f) for f in ch.feats(wavs)]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ck["cfg"])
    m = UnitLM(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    eos = cfg.vocab - 1
    L = min(len(c) for c in codes)
    n = int(round(a.seconds * cb.rate))
    keep = min(L, cfg.ctx - n - 1)
    x = torch.stack([torch.cat([torch.tensor([eos], device=dev), c[-keep:]]) for c in codes]).long()
    g = torch.Generator(device=dev).manual_seed(a.seed)
    with torch.no_grad():
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.states(x[:, -cfg.ctx :])
            lg = m.logits(h[:, -1])
            lg[:, eos] = -float("inf")
            x = torch.cat([x, sample_next(lg, a.temp, a.top_p, g)[:, None]], 1)
    gen = x[:, -n:]
    for pid, c in zip(ids, gen):
        sf.write(out / f"{pid}.wav", ch.vocode(cb.decode(c)), SR)
    u = gen.cpu().numpy()
    rep = float(np.mean([1 - len({tuple(r[i : i + 4]) for i in range(len(r) - 3)}) / (len(r) - 3) for r in u]))
    info = {"checkpoint": a.ckpt, "step": ck.get("step"), "val": ck.get("val"), "config": ck["cfg"],
            "params": sum(q.numel() for q in m.parameters()),
            "sampling": {"temperature": a.temp, "top_p": a.top_p, "seed": a.seed, "eos_banned": True,
                         "batch": "all prompts in one batch, one generator"},
            "prompt_seen_s": keep / cb.rate, "prompt_codes": keep, "generated_codes": n, "generated_s": n / cb.rate,
            "units": f"WavLM-Large layer 6 (kNN-VC), {cb.rate} Hz, K={cb.K}, cosine nearest centroid, codebook {a.cb}",
            "decoder": "centroid lookup -> kNN-VC prematched HiFi-GAN, 16 kHz",
            "rep4_mean": rep, "note": a.note, "codes": {pid: r.tolist() for pid, r in zip(ids, u)}}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print("GEN DONE", len(ids), json.dumps({k: info[k] for k in ("step", "val", "prompt_seen_s", "rep4_mean")}), flush=True)


if __name__ == "__main__":
    main()
