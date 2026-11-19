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
    model = MultiStreamLM(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    inv = [order.index(c) for c in sorted(order)]
    for v in a.variants.split(";"):
        name, mode = v.split(":")
        vo = out / name
        vo.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        gen = np.concatenate([generate(model, prompts[i : i + a.batch], a.new_steps, mode, a.seed + i, dev)
                              for i in range(0, len(prompts), a.batch)])
        secs = time.time() - t0
        for nm, gq in zip(names, gen):
            c = torch.from_numpy(gq[:, inv].astype(np.int64)).to(dev)
            f = codec.decode(c[None])[0]
            sf.write(vo / f"{nm}.wav", u.knn.vocode(f[None].float()).reshape(-1).cpu().numpy(), SR)
        np.savez(vo / "gen_units.npz", **{nm: gq[:, inv] for nm, gq in zip(names, gen)})
        info = {"track": "s", "variant": name, "sampling": {"t10": "temperature 1.0, top-p 1.0, all levels",
                "ra08": "repeat-aware: p(previous code of that level) kept; the rest temperature 0.8, renormalised to "
                        "the remaining mass; top-p 1.0"}[mode], "banned": "EOS, PAD", "seed": f"{a.seed} + batch offset",
                "lm_checkpoint": a.ckpt, "lm_step": ck["step"], "lm_val": ck["val"], "lm_config": ck["cfg"],
                "prompts": f"s3 dy32/{a.pack_src}/ ({len(names)})",
                "prompt_seen": f"all {n} steps ({n / 12.5:.2f} s) of each 15 s prompt after one EOS step",
                "new_steps": a.new_steps, "continuation_s": a.new_steps / 12.5, "generation_seconds": round(secs),
                "codec": {"file": a.codec, "step": cck["step"], "config": cck["cfg"]},
                "channel": "kNN-VC WavLM-Large layer 6 -> slow codec (12.5 Hz, RVQ 4 x 1024) -> codec decoder -> kNN-VC HiFi-GAN"}
        (vo / "info.json").write_text(json.dumps(info, indent=2))
        print("GEN", name, ck["step"], round(secs), flush=True)
        if a.upload:
            sh(f"aws s3 cp {vo}/ {S3}/gen/{name}/ --recursive --exclude '*' --include '*.wav' --include info.json --only-show-errors")
            print(f"UPLOADED gen/{name}", flush=True)


if __name__ == "__main__":
    main()
