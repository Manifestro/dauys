"""DY-33 track S continuations: the multi-stream LM on slow units continues the DY-32 prompt packs.

prompt_XX.wav (15 s) -> kNN-VC WavLM-Large layer 6 (as at unit time) -> frozen slow codec -> 4 codes per 80 ms step
-> EOS step + all prompt steps -> 125 new steps (10 s) -> codec decoder -> 50 Hz features -> kNN-VC HiFi-GAN.
Samplings (declared):
  t10  temperature 1.0, top-p 1.0 on all levels
  ra08 repeat-aware sharpening: the probability of repeating that level's previous code is kept as it is; the rest
       of the distribution gets temperature 0.8 and is renormalised to the remaining mass (top-p 1.0).
EOS/PAD are never sampled. Outputs --out/<name>/XX.wav + info.json; --upload sends them to dy33/gen/<name>/.

    python scripts/dy33s_gen.py --run /data/dy33s/run --ckpt best.pt --codec /data/codec/frozen.pt --pack-src prompts \
        --variants "s-t10:t10;s-ra08:ra08" --out /data/dy33s/gen --upload
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dauys.dy32c_lm import MSConfig, MultiStreamLM, delay, undelay
from dy32c_resynth import read_pack
from dy32c_units import RVQUnits, sh
from dy33s_codec import load_codec, roundtrip  # noqa: F401  (roundtrip kept for checks)

S3 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy33"
SR = 16000


def sample(lg: torch.Tensor, prev: torch.Tensor, mode: str, g: torch.Generator) -> torch.Tensor:
    """lg [B, V] float32 (banned ids already -inf), prev [B] previous code of this level -> [B]."""
    p = torch.softmax(lg, -1)
    if mode == "ra08":
        keep = p.gather(1, prev[:, None])  # [B, 1]
        q = torch.softmax(lg / 0.8, -1)
        q = q.scatter(1, prev[:, None], 0.0)
        q = q / q.sum(-1, keepdim=True).clamp(min=1e-12) * (1 - keep)
        p = q.scatter(1, prev[:, None], keep)
    return torch.multinomial(p, 1, generator=g)[:, 0]


@torch.no_grad()
def generate(model, prompts: np.ndarray, n_new: int, mode: str, seed: int, dev, hook=None) -> np.ndarray:
    """prompts [B, N, S] (EOS step first) -> [B, n_new, S]. Same delay-layout loop as dy32c_lm.generate."""
    cfg = model.cfg
    B, N, S = prompts.shape
    K, PAD = cfg.k, cfg.k + 1
    total = N + n_new
    full = np.full((B, total, S), PAD, np.int64)
    full[:, :N] = prompts
    d = np.stack([delay(f, PAD) for f in full])
    known = np.zeros_like(d, bool)
    for k in range(S):
        known[:, k : k + N, k] = True
    ids = torch.from_numpy(d).to(dev)
    kn = torch.from_numpy(known).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    for t in range(N, total + S - 1):
        lo = max(0, t - cfg.ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ids[:, lo:t])
        lg = model.logits(h[:, -1])
        lg[..., K:] = -float("inf")
        if hook is not None:
            lg = hook(t, ids, lg)
        nxt = torch.stack([sample(lg[:, k], ids[:, t - 1, k].clamp(max=K - 1), mode, g) for k in range(S)], 1)
        for k in range(S):
            if t - k < 0 or t - k >= total:
                nxt[:, k] = PAD
        ids[:, t] = torch.where(kn[:, t], ids[:, t], nxt)
    out = ids.cpu().numpy()
    return np.stack([undelay(o, total)[N:] for o in out])


HEAD = 0.05
GBUCKETS = {"0-1s": (0, 12), "1-3s": (12, 37), "3-10s": (37, 125)}


@torch.no_grad()
def generate_guided(model, prompts: np.ndarray, swaps: np.ndarray, n_new: int, w: float, seed: int, dev, stats: dict):
    """DY-34's content-contrastive guidance on the delay layout. Two branches share the generated codes: TRUE prompt +
    generated and SWAP prompt (same recording, >= 60 s away) + generated; prompt and swap have the same number of
    steps. Per level: H = {v: p_t(v) >= 0.05 max p_t}; inside H score = log p_t + w (log p_t - log p_s), softmax over
    H scaled to H's p_t mass; outside H p_t; one multinomial draw (temperature 1.0, top-p 1.0), levels in order from
    one generator as the t10 sampler."""
    cfg = model.cfg
    B, N, S = prompts.shape
    assert swaps.shape == prompts.shape
    K, PAD = cfg.k, cfg.k + 1
    total = N + n_new
    ids = []
    for pr in (prompts, swaps):
        full = np.full((B, total, S), PAD, np.int64)
        full[:, :N] = pr
        ids.append(torch.from_numpy(np.stack([delay(f, PAD) for f in full])).to(dev))
    known = np.zeros(ids[0].shape, bool)
    for k in range(S):
        known[:, k : k + N, k] = True
    kn = torch.from_numpy(known).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    for t in range(N, total + S - 1):
        lo = max(0, t - cfg.ctx)
        lg = []
        for x in ids:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.states(x[:, lo:t])
            l = model.logits(h[:, -1])
            l[..., K:] = -float("inf")
            lg.append(l)
        nxt = []
        for k in range(S):
            lpt, lps = torch.log_softmax(lg[0][:, k], -1), torch.log_softmax(lg[1][:, k], -1)
            pt = lpt.exp()
            Hm = pt >= HEAD * pt.max(-1, keepdim=True).values
            mass = (pt * Hm).sum(-1, keepdim=True)
            d = lpt - lps
            score = torch.where(Hm, lpt + w * d, torch.full_like(lpt, -float("inf")))
            q = torch.where(Hm, torch.softmax(score, -1) * mass, pt)
            nxt.append(torch.multinomial(q, 1, generator=g)[:, 0])
            frame = t - k
            if N <= frame < total:
                st = frame - N
                stats.setdefault(f"hsize_L{k + 1}", []).append(Hm.sum(-1).float().mean().item())
                stats.setdefault(f"hmass_L{k + 1}", []).append(mass.mean().item())
                bk = next(b for b, (s0, e0) in GBUCKETS.items() if s0 <= st < e0)
                stats.setdefault(f"absd_{bk}", []).append(d[Hm].abs().float().mean().item())
                if w == 0:
                    m0 = stats.setdefault("w0_max_abs_q_minus_p", [0.0])
                    m0[0] = max(m0[0], float((q - pt).abs().max()))
        nxt = torch.stack(nxt, 1)
        for k in range(S):
            if t - k < 0 or t - k >= total:
                nxt[:, k] = PAD
        for x in ids:
            x[:, t] = torch.where(kn[:, t], x[:, t], nxt)
    out = ids[0].cpu().numpy()
    return np.stack([undelay(o, total)[N:] for o in out])


def encode(u: RVQUnits, codec, waves: list[np.ndarray], dev) -> list[np.ndarray]:
    """float waves -> slow codes [steps, 4] (each wave alone, as the unit pass: features padded to a multiple of 4)."""
    from dy33s_feats import piece_feats

    feats = piece_feats(u, [(w * 32768.0).clip(-32768, 32767).astype(np.int16) for w in waves])
    out = []
    for f in feats:
        x = torch.from_numpy(f.astype(np.float32)).to(dev)
        pad = (-len(x)) % 4
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, -1)])
        out.append(codec.codes(x[None])[0].cpu().numpy())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--codec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pack-src", default="prompts")
    p.add_argument("--variants", default="s-t10:t10;s-ra08:ra08")
    p.add_argument("--new-steps", type=int, default=125)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--swaps", default="", help="guided: local dir with swap_XXX.wav (s3 dy34/swap/); variant mode gW, e.g. g4")
    p.add_argument("--check-codes", default="", help="guided w=0: gen_units.npz of the t10 variant to compare with")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckf = out / f"lm_{a.ckpt}"
    shutil.copy(Path(a.run) / a.ckpt, ckf)
    ck = torch.load(ckf, map_location="cpu", weights_only=False)
    cfg = MSConfig(**ck["cfg"])
    order = ck["order"]
    codec, cck = load_codec(Path(a.codec), dev)
    names, waves = read_pack(Path(f"/opt/dlami/nvme/{a.pack_src}"), "prompt", a.pack_src)
    u = RVQUnits(dev)
    toks = encode(u, codec, waves, dev)
    n = min(len(t) for t in toks)
    prompts = np.stack([np.r_[np.full((1, cfg.streams), cfg.k), t[-n:][:, order]] for t in toks]).astype(np.int64)
    swp = None
    if a.swaps:
        sw = [sf.read(f"{a.swaps}/swap_{int(nm):03d}.wav", dtype="float32")[0] for nm in names]
        st = encode(u, codec, sw, dev)
        ns = min(len(t) for t in st)
        assert ns == n, (ns, n)
        swp = np.stack([np.r_[np.full((1, cfg.streams), cfg.k), t[-n:][:, order]] for t in st]).astype(np.int64)
    model = MultiStreamLM(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    inv = [order.index(c) for c in sorted(order)]
    for v in a.variants.split(";"):
        name, mode = v.split(":")
        vo = out / name
        vo.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        stats: dict = {}
        if mode.startswith("g"):
            wv = float(mode[1:])
            gen = np.concatenate([generate_guided(model, prompts[i : i + a.batch], swp[i : i + a.batch], a.new_steps, wv,
                                                  a.seed + i, dev, stats) for i in range(0, len(prompts), a.batch)])
        else:
            gen = np.concatenate([generate(model, prompts[i : i + a.batch], a.new_steps, mode, a.seed + i, dev)
                                  for i in range(0, len(prompts), a.batch)])
        secs = time.time() - t0
        for nm, gq in zip(names, gen):
            c = torch.from_numpy(gq[:, inv].astype(np.int64)).to(dev)
            f = codec.decode(c[None])[0]
            sf.write(vo / f"{nm}.wav", u.knn.vocode(f[None].float()).reshape(-1).cpu().numpy(), SR)
        np.savez(vo / "gen_units.npz", **{nm: gq[:, inv] for nm, gq in zip(names, gen)})
        desc = {"t10": "temperature 1.0, top-p 1.0, all levels",
                "ra08": "repeat-aware: p(previous code of that level) kept; the rest temperature 0.8, renormalised to "
                        "the remaining mass; top-p 1.0"}
        info = {"track": "s", "variant": name, "sampling": desc.get(mode, f"guided (DY-34 method), w = {mode[1:]}, head "
                f"p_true >= {HEAD} max p_true per level, temperature 1.0, top-p 1.0; swap = dy34/swap/swap_XXX.wav encoded alone"), "banned": "EOS, PAD", "seed": f"{a.seed} + batch offset",
                "lm_checkpoint": a.ckpt, "lm_step": ck["step"], "lm_val": ck["val"], "lm_config": ck["cfg"],
                "prompts": f"s3 dy32/{a.pack_src}/ ({len(names)})",
                "prompt_seen": f"all {n} steps ({n / 12.5:.2f} s) of each 15 s prompt after one EOS step",
                "new_steps": a.new_steps, "continuation_s": a.new_steps / 12.5, "generation_seconds": round(secs),
                "codec": {"file": a.codec, "step": cck["step"], "config": cck["cfg"]},
                "channel": "kNN-VC WavLM-Large layer 6 -> slow codec (12.5 Hz, RVQ 4 x 1024) -> codec decoder -> kNN-VC HiFi-GAN"}
        if stats:
            info["guided"] = {"w": float(mode[1:]), "head_threshold": f"p_true >= {HEAD} * max p_true",
                              **{f"mean_head_size_L{k + 1}": round(float(np.mean(stats[f"hsize_L{k + 1}"])), 2) for k in range(cfg.streams)},
                              **{f"mean_head_mass_L{k + 1}": round(float(np.mean(stats[f"hmass_L{k + 1}"])), 4) for k in range(cfg.streams)},
                              **{f"mean_abs_logp_true_minus_logp_swap_in_head_{b}": round(float(np.mean(stats[f"absd_{b}"])), 4)
                                 for b in GBUCKETS if f"absd_{b}" in stats}}
            if "w0_max_abs_q_minus_p" in stats:
                info["guided"]["w0_max_abs_q_minus_p"] = stats["w0_max_abs_q_minus_p"][0]
            if a.check_codes:
                ref = np.load(a.check_codes)
                same = [bool(np.array_equal(ref[nm], gq[:, inv])) for nm, gq in zip(names, gen)]
                agree = [float((ref[nm] == gq[:, inv]).mean()) for nm, gq in zip(names, gen)]
                info["guided"]["w0_check"] = {"reference": a.check_codes, "identical_pieces": int(sum(same)), "pieces": len(same),
                                              "mean_code_agreement": round(float(np.mean(agree)), 4)}
        (vo / "info.json").write_text(json.dumps(info, indent=2))
        print("GEN", name, ck["step"], round(secs), flush=True)
        if a.upload:
            sh(f"aws s3 cp {vo}/ {S3}/gen/{name}/ --recursive --exclude '*' --include '*.wav' --include info.json --only-show-errors")
            print(f"UPLOADED gen/{name}", flush=True)


if __name__ == "__main__":
    main()
