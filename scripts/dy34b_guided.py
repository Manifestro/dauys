"""DY-34 track B: content-contrastive guidance at generation (no training, no text).

Two branches of the same UnitLM (dy32 b, step 12257) share the generated codes: TRUE = EOS + true 15 s prompt codes +
generated, SWAP = EOS + swap 15 s codes (same recording, >= 60 s away) + generated. Prompt and swap are each encoded
alone. At every step (EOS banned in both): p_t, p_s; head H = {v: p_t(v) >= HEAD * max p_t}. Inside H:
score = log p_t + w (log p_t - log p_s), softmax over H, scaled to H's original mass under p_t; outside H: p_t.
Sampling: dauys.dy16_model.sample_next on log q at temperature 1.0, top-p 1.0 (one draw per step from one generator,
seed --seed, the same RNG path as dy32b_gen.py). w = 0 gives q = p_t up to float rounding.

    python scripts/dy34b_guided.py --ckpt best.pt --cb km-corpus-K2000.npy --prompts dir --swaps dir --ids 200-299 \
        --w 2 --out /data/dy34b/gen/b-g2-dev
    python scripts/dy34b_guided.py ... --w 0 --check-codes /path/gen_b-t10/info.json   # w = 0 vs the plain generator
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy32b_channel import SR, Channel, Codebook
from dy32b_gen import cached_states

from dauys.dy16_model import sample_next
from dauys.dy21_lm import LMConfig, UnitLM

HEAD = 0.05
BUCKETS = {"0-1s": (0, 50), "1-3s": (50, 150), "3-10s": (150, 500)}
LONG = 25


def runs_stats(seqs: list[np.ndarray]) -> dict:
    long_, tot = 0, 0
    for u in seqs:
        i = 0
        while i < len(u):
            j = i
            while j < len(u) and u[j] == u[i]:
                j += 1
            if j - i >= LONG:
                long_ += j - i
            i = j
        tot += len(u)
    return {"share_frames_in_runs_ge_0.5s": round(long_ / tot, 4),
            "distinct_codes_per_10s": round(float(np.mean([len(set(u.tolist())) for u in seqs])), 1)}


def parse_ids(s: str, width: int) -> list[str]:
    a, b = map(int, s.split("-"))
    return [f"{i:0{width}d}" for i in range(a, b + 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cb", required=True)
    p.add_argument("--prompts", required=True, help="dir with prompt_<id>.wav")
    p.add_argument("--swaps", required=True, help="dir with swap_<3-digit id>.wav")
    p.add_argument("--ids", required=True, help="a-b, e.g. 0-49, 50-199, 200-299")
    p.add_argument("--width", type=int, default=3, help="digits of the prompt ids (2 for the 00-49 pack)")
    p.add_argument("--w", type=float, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mem", type=float, default=0.30, help="cap of this process's share of GPU memory")
    p.add_argument("--check-codes", default="", help="info.json of the plain generator (w = 0 check)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t_start = time.time()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = parse_ids(a.ids, a.width)
    ch, cb = Channel(), Codebook(a.cb)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ck["cfg"])
    m = UnitLM(cfg).to(dev).eval()
    m.load_state_dict(ck["model"])
    eos = cfg.vocab - 1

    all_codes, hsize, hmass, maxdq = {}, [], [], []
    diffs = {k: [] for k in BUCKETS}
    for b0 in range(0, len(ids), a.batch):
        B_ids = ids[b0 : b0 + a.batch]
        pw = [sf.read(f"{a.prompts}/prompt_{i}.wav", dtype="float32")[0] for i in B_ids]
        sw = [sf.read(f"{a.swaps}/swap_{int(i):03d}.wav", dtype="float32")[0] for i in B_ids]
        pc = [cb.encode(f) for f in ch.feats(pw)]
        sc = [cb.encode(f) for f in ch.feats(sw)]
        Lp, Ls = min(map(len, pc)), min(map(len, sc))
        assert 1 + max(Lp, Ls) + a.n <= cfg.ctx
        xt = torch.stack([torch.cat([torch.tensor([eos], device=dev), c[-Lp:]]) for c in pc]).long()
        xs = torch.stack([torch.cat([torch.tensor([eos], device=dev), c[-Ls:]]) for c in sc]).long()
        g = torch.Generator(device=dev).manual_seed(a.seed)
        ct, cs = [], []
        gen = []
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ht = cached_states(m, xt, ct, 0)[:, -1]
                hs = cached_states(m, xs, cs, 0)[:, -1]
            pos_t, pos_s = xt.shape[1], xs.shape[1]
            for step in range(a.n):
                lt, ls = m.logits(ht), m.logits(hs)
                lt[:, eos] = -float("inf")
                ls[:, eos] = -float("inf")
                lpt, lps = torch.log_softmax(lt, -1), torch.log_softmax(ls, -1)
                pt = lpt.exp()
                H = pt >= HEAD * pt.max(-1, keepdim=True).values
                mass = (pt * H).sum(-1, keepdim=True)
                d = lpt - lps
                score = torch.where(H, lpt + a.w * d, torch.full_like(lpt, -float("inf")))
                q = torch.where(H, torch.softmax(score, -1) * mass, pt)
                if a.w == 0:
                    maxdq.append(float((q - pt).abs().max()))
                nxt = sample_next(torch.log(q), 1.0, 1.0, g)[:, None]
                gen.append(nxt)
                hsize.append(H.sum(-1).float().cpu().numpy())
                hmass.append(mass[:, 0].cpu().numpy())
                bk = next(k for k, (s, e) in BUCKETS.items() if s <= step < e)
                diffs[bk].append(d[H].abs().float().cpu().numpy())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ht = cached_states(m, nxt, ct, pos_t)[:, -1]
                    hs = cached_states(m, nxt, cs, pos_s)[:, -1]
                pos_t, pos_s = pos_t + 1, pos_s + 1
        gen = torch.cat(gen, 1)
        for i, c in zip(B_ids, gen):
            sf.write(out / f"{i}.wav", ch.vocode(cb.decode(c)), SR)
            all_codes[i] = c.cpu().numpy().astype(np.int16)
        del ct, cs
        torch.cuda.empty_cache()
        print(json.dumps({"done": b0 + len(B_ids), "of": len(ids), "s": round(time.time() - t_start)}), flush=True)

    np.savez(out / "codes.npz", **{f"g{i}": v for i, v in all_codes.items()})
    dd = {k: np.concatenate(v) if v else np.zeros(1) for k, v in diffs.items()}
    info = {"w": a.w, "head_threshold": f"p_true >= {HEAD} * max p_true", "seed": a.seed, "batch": a.batch,
            "temperature": 1.0, "top_p": 1.0, "eos_banned": True, "checkpoint": a.ckpt, "step": ck.get("step"),
            "val": ck.get("val"), "codebook": a.cb, "generated_codes": a.n, "prompt_codes": int(Lp),
            "swap_codes": int(Ls), "ids": a.ids,
            "mean_head_size": round(float(np.mean(np.concatenate(hsize))), 2),
            "mean_head_mass": round(float(np.mean(np.concatenate(hmass))), 4),
            "abs_logp_true_minus_logp_swap_in_head": {k: {"mean": round(float(v.mean()), 4),
                                                          "p95": round(float(np.percentile(v, 95)), 4)}
                                                      for k, v in dd.items()},
            **runs_stats(list(all_codes.values())), "seconds": round(time.time() - t_start),
            "codes": {i: v.tolist() for i, v in all_codes.items()}}
    if a.w == 0:
        info["w0_max_abs_q_minus_p_true"] = max(maxdq)
    if a.check_codes:
        ref = json.loads(Path(a.check_codes).read_text())["codes"]
        same = [bool(np.array_equal(np.asarray(ref[i]), all_codes[i])) for i in ids if i in ref]
        agree = [float(np.mean(np.asarray(ref[i]) == all_codes[i])) for i in ids if i in ref]
        first = [int(np.argmax(np.asarray(ref[i]) != all_codes[i])) if not s else a.n
                 for i, s in zip([i for i in ids if i in ref], same)]
        info["w0_check"] = {"reference": a.check_codes, "pieces": len(same), "identical_pieces": int(sum(same)),
                            "mean_code_agreement": round(float(np.mean(agree)), 4),
                            "median_first_divergence_step": float(np.median(first)),
                            "reference_stats": runs_stats([np.asarray(ref[i]) for i in ids if i in ref])}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print("GUIDED DONE", json.dumps({k: v for k, v in info.items() if k != "codes"}), flush=True)


if __name__ == "__main__":
    main()
