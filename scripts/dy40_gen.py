"""DY-40: continuations and the likelihood test of heard content for the depth LM on Mimi codes.

gen:   prompt_XX.wav (15 s, 16 kHz) of a DY-32 pack -> Mimi 8 codes (as at unit time) -> EOS frame + all prompt
       frames -> 125 new frames (10 s) -> Mimi decoder (24 kHz) -> XX.wav at 16 kHz for the evaluation (S3 dy40/gen/<name>/)
       and XX.wav at 24 kHz for listening (S3 dy40/listen24/<name>/). Samplings (declared in DY-40): t10 (temperature
       1.0 on every code) and ra08 (DY-33 S2: p(previous frame's code of that level) kept, the rest temperature 0.8).
heard: DY-33's likelihood test (windows copied from dy33s_heard_use: held-out shards, runs, own = next 10 s, heard H s
       under a = true preceding, b = same episode >= 60 s away, c = another episode); heard and own encoded separately,
       then joined and put in the delayed layout (acoustic codes one frame after code 0) after an EOS position;
       CE of the true own codes per level, buckets 0-1 s / 1-3 s / 3-10 s; also the loss-weighted mean.

    python scripts/dy40_gen.py gen --run /data/dy40/run --ckpt best.pt --pack-src prompts --variants "m-t10:t10;m-ra08:ra08" --upload
    python scripts/dy40_gen.py heard --held /opt/dlami/nvme/held --ckpt /data/dy40/run/best.pt --out /data/dy40/heard_use.json
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
from dauys.dy40_lm import W0, DLConfig, DepthLM, delay, generate, level_ce
from dy40_mimi import BUCKET, S3, SR, Mimi, read_pack, sh, to16

MSR = 24000


def load(ckpt: Path, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = DepthLM(DLConfig(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def cmd_gen(a) -> None:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckf = out / f"lm_{a.ckpt}"
    shutil.copy(Path(a.run) / a.ckpt, ckf)
    model, ck = load(ckf, dev)
    K = model.cfg.k
    mm = Mimi(dev)
    names, waves = read_pack(Path(f"/data/dy32/{a.pack_src}"), "prompt", a.pack_src)
    if a.limit:
        names, waves = names[: a.limit], waves[: a.limit]
    toks = mm.encode(waves)
    n = min(len(t) for t in toks)
    prompts = np.stack([np.r_[np.full((1, 8), K), t[-n:]] for t in toks]).astype(np.int64)
    for v in a.variants.split(";"):
        name, mode = v.split(":")
        vo, vl = out / name, out / "listen24" / name
        vo.mkdir(parents=True, exist_ok=True)
        vl.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        gen = np.concatenate([generate(model, prompts[i : i + a.batch], a.new_steps, mode, a.seed + i, dev)
                              for i in range(0, len(prompts), a.batch)])
        secs = time.time() - t0
        for nm, g in zip(names, gen):
            w = mm.decode(g)
            sf.write(vl / f"{nm}.wav", w, MSR)
            sf.write(vo / f"{nm}.wav", to16(w), SR)
        np.savez(vo / "gen_units.npz", **dict(zip(names, gen)))
        desc = {"t10": "temperature 1.0, top-p 1.0, every code",
                "ra08": "repeat-aware: p(previous frame's code of that level) kept; the rest temperature 0.8, "
                        "renormalised to the remaining mass"}
        info = {"track": "dy40", "variant": name, "sampling": desc[mode], "banned": "EOS", "seed": f"{a.seed} + batch offset",
                "lm_checkpoint": a.ckpt, "lm_step": ck["step"], "lm_val": ck["val"], "lm_config": ck["cfg"],
                "prompts": f"s3 dy32/{a.pack_src}/ ({len(names)})",
                "prompt_seen": f"all {n} frames ({n / 12.5:.2f} s) of each 15 s prompt after one EOS frame",
                "new_steps": a.new_steps, "continuation_s": a.new_steps / 12.5, "generation_seconds": round(secs),
                "channel": "16 kHz -> 24 kHz -> Mimi encoder (8 x 2048, 12.5 Hz) -> LM -> Mimi decoder 24 kHz -> 16 kHz"}
        (vo / "info.json").write_text(json.dumps(info, indent=2))
        print("GEN", name, ck["step"], round(secs), flush=True)
        if a.upload:
            sh(f"aws s3 cp {vo}/ {S3}/gen/{name}/ --recursive --exclude '*' --include '*.wav' --include info.json --only-show-errors")
            sh(f"aws s3 cp {vl}/ {S3}/listen24/{name}/ --recursive --only-show-errors")
            print(f"UPLOADED gen/{name}", flush=True)


def cmd_heard(a) -> None:
    import torch.nn.functional as F  # noqa: F401

    from dy33s_heard_use import BUCKETS, OWN_S, OWN_STEPS, mean_ci, wilson, windows

    dev = torch.device("cuda")
    Hs = [int(x) for x in a.heard.split(",")]
    hmax = max(Hs)
    pieces, wins, winfo = windows(a.held, a.n, a.seed, hmax)

    def chunk(i, off, n):
        segs, st, _, _, _ = pieces[i]
        return np.asarray(segs[st + off : st + off + n], np.int16)

    mm = Mimi(dev)
    model, ck = load(Path(a.ckpt), dev)
    K = model.cfg.k

    @torch.no_grad()
    def ce(heard: list[np.ndarray], own: list[np.ndarray]) -> np.ndarray:
        """EOS + delayed(heard frames + own frames) -> [B, n_own, 8] CE of the own frames' codes (code 0 of own frame f
        at run position n_h + f, its codes 1..7 one position later)."""
        n_own, n_h = min(len(o) for o in own), min(len(h) for h in heard)
        seqs = [np.r_[np.full((1, 8), K), delay(np.r_[h[-n_h:], o[:n_own]].astype(np.int64), K + 1)]
                for h, o in zip(heard, own)]
        ids = torch.from_numpy(np.stack(seqs).astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = level_ce(model, ids).float()  # CE index i = ids position i + 1 = run position i
        f = torch.arange(n_own, device=dev)
        c0 = c[:, n_h + f, 0]
        ca = c[:, n_h + f + 1, 1:]
        return torch.cat([c0[..., None], ca], -1).cpu().numpy()

    res = {(H, c): [] for H in Hs for c in "abc"}
    hs, os_ = hmax * SR, OWN_S * SR
    for b0 in range(0, len(wins), a.batch):
        W = wins[b0 : b0 + a.batch]
        own = mm.encode([chunk(i, off + hs, os_) for i, off, *_ in W])
        src = {"a": [(i, off) for i, off, *_ in W], "b": [(j, ob) for _, _, j, ob, _, _ in W],
               "c": [(k, oc) for *_, k, oc in W]}
        for H in Hs:
            for c, lst in src.items():
                heard = mm.encode([chunk(i, o + hs - H * SR, H * SR) for i, o in lst])
                res[(H, c)].append(ce(heard, own))
        if b0 % 160 == 0:
            print(json.dumps({"done": b0 + len(W)}), flush=True)
    res = {k: np.concatenate([x[:, :OWN_STEPS] for x in v]) for k, v in res.items()}
    wts = np.array([W0] + [1.0] * 7)
    wts = wts / wts.sum()
    names = [f"L{k + 1}" for k in range(8)] + ["mean", "weighted"]
    out = {"info": winfo, "step": ck["step"], "val": ck["val"], "table": []}
    for H in Hs:
        for bk, (lo, hi) in BUCKETS.items():
            for li, nm in enumerate(names):
                if nm == "mean":
                    sel = lambda x: x[:, lo:hi].mean(-1).mean(-1)  # noqa: E731
                elif nm == "weighted":
                    sel = lambda x: (x[:, lo:hi] * wts).sum(-1).mean(-1)  # noqa: E731
                else:
                    sel = lambda x, li=li: x[:, lo:hi, li].mean(-1)  # noqa: E731
                A, B, C = (sel(res[(H, c)]) for c in "abc")
                n = len(A)
                out["table"].append({"heard_s": H, "bucket": bk, "level": nm, "ce_a": round(float(A.mean()), 4),
                                     "ce_b": round(float(B.mean()), 4), "ce_c": round(float(C.mean()), 4),
                                     "b_minus_a": mean_ci(B - A), "c_minus_a": mean_ci(C - A),
                                     "b_minus_a_rel_pct": round(100 * float((B - A).mean() / A.mean()), 3),
                                     "share_a_lt_b": round(float((A < B).mean()), 4), "share_a_lt_b_ci": wilson(int((A < B).sum()), n),
                                     "share_a_lt_c": round(float((A < C).mean()), 4), "share_a_lt_c_ci": wilson(int((A < C).sum()), n)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    lines = [f"# DY-40: likelihood test of heard content (depth LM on Mimi 8 x 2048, 12.5 Hz, step {ck['step']})", "",
             f"windows {winfo['windows']}, episodes {winfo['episodes']}; (a) true preceding, (b) same episode >= 60 s away, "
             "(c) another episode. CE of the true own codes per level, mean, and the training loss weighting.", "",
             "| heard | bucket | level | CE a | CE b | CE c | b-a (95% CI) | (b-a)/a % | share a<b (CI) | share a<c (CI) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["table"]:
        lines.append(f"| {r['heard_s']} s | {r['bucket']} | {r['level']} | {r['ce_a']} | {r['ce_b']} | {r['ce_c']} | "
                     f"{r['b_minus_a']['mean']} {r['b_minus_a']['ci95']} | {r['b_minus_a_rel_pct']} | "
                     f"{r['share_a_lt_b']} {r['share_a_lt_b_ci']} | {r['share_a_lt_c']} {r['share_a_lt_c_ci']} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4] + [l for l in lines if "| mean |" in l or "| weighted |" in l]), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--run", required=True)
    g.add_argument("--ckpt", default="best.pt")
    g.add_argument("--out", default="/data/dy40/gen")
    g.add_argument("--pack-src", default="prompts")
    g.add_argument("--variants", default="m-t10:t10;m-ra08:ra08")
    g.add_argument("--new-steps", type=int, default=125)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--batch", type=int, default=50)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--upload", action="store_true")
    h = sp.add_parser("heard")
    h.add_argument("--held", required=True)
    h.add_argument("--ckpt", required=True)
    h.add_argument("--n", type=int, default=1000)
    h.add_argument("--heard", default="120,60,30,15,5")
    h.add_argument("--batch", type=int, default=16)
    h.add_argument("--seed", type=int, default=2026)
    h.add_argument("--out", required=True)
    a = p.parse_args()
    {"gen": cmd_gen, "heard": cmd_heard}[a.cmd](a)


if __name__ == "__main__":
    main()
