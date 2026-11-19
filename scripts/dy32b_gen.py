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


def rope_at(x: torch.Tensor, pos0: int, base: float = 10000.0) -> torch.Tensor:
    """dy21_lm.rope for positions pos0 .. pos0 + t - 1."""
    b, h, t, d = x.shape
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = (pos0 + torch.arange(t, device=x.device, dtype=torch.float32))[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def cached_states(m: UnitLM, ids: torch.Tensor, cache: list, pos0: int) -> torch.Tensor:
    """UnitLM.states for new tokens ids [B, t] at positions pos0.. with a key/value cache (one entry per block)."""
    import torch.nn.functional as F

    x = m.emb(ids)
    for j, blk in enumerate(m.blocks):
        b, t, d = x.shape
        q, k, v = blk.qkv(blk.n1(x)).view(b, t, 3, blk.heads, d // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = rope_at(q, pos0), rope_at(k, pos0)
        if len(cache) > j:
            k, v = torch.cat([cache[j][0], k], 2), torch.cat([cache[j][1], v], 2)
            cache[j] = (k, v)
        else:
            cache.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1)
        x = x + blk.out(a.transpose(1, 2).reshape(b, t, d))
        g, u = blk.up(blk.n2(x)).chunk(2, -1)
        x = x + blk.down(F.silu(g) * u)
    return m.norm(x)


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
    assert x.shape[1] + n <= cfg.ctx, "prompt + continuation must fit the context (no sliding window here)"
    cache: list = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = m.states(x)[:, -1]
            h = cached_states(m, x, cache, 0)[:, -1]
        diff = float((ref.float() - h.float()).abs().mean())
        print("cache check: mean abs diff of last states", round(diff, 5), flush=True)
        assert diff < 0.05, "cached forward differs from UnitLM.states"
        for i in range(n):
            lg = m.logits(h)
            lg[:, eos] = -float("inf")
            nxt = sample_next(lg, a.temp, a.top_p, g)[:, None]
            x = torch.cat([x, nxt], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = cached_states(m, nxt, cache, x.shape[1] - 1)[:, -1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full = m.states(x)[:, -1]
        diff2 = float((full.float() - h.float()).abs().mean())
        print("cache check after generation: mean abs diff", round(diff2, 5), flush=True)
        assert diff2 < 0.05, "incremental cached forward differs from UnitLM.states"
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
