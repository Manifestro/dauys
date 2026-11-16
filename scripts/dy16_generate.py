"""DY-16: generate the own channel at reply moments of test and measure content, timing, form.

Moments (fixed with seed 1234, written before any generation): frame ``t`` where the own
channel was silent for 1 s, the partner spoke in the last 3 s, and in reality the own
channel starts a unit within 2 s; at least 60 s of context. From the real 60 s up to
``t`` the model writes 10 s of its own channel frame by frame; the partner channel is the
real continuation (offline: the partner does not react to what was generated).

Content (gate C): the DY-14 s0 judge reads the conversation from the partner's side and
picks, among the generated 2 s window (t+0.4…2.4 s) and 15 real windows of the same
person from the same conversation, the one that follows; again with the judge's own
channel swapped for another conversation. The real reply is scored the same way for
reference. Timing (T): first own unit after ``t``. Form (F): units per second of speech
and repeated 4-grams of units.

    python scripts/dy16_generate.py --grid /data/dy14/grid --anchors test_anchors.json --model /data/dy16/runs/f0/best.pt \
        --judge /data/dy14/runs/s0/best.pt --out /data/dy16/gen_f0.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dauys.dy14_data import (
    CTX,
    GAP,
    WIN,
    Conv,
    hard_negatives,
    load_convs,
)
from dauys.dy14_eval import bootstrap, correct_per_anchor
from dauys.dy14_model import Tower, TowerConfig
from dauys.dy16_model import Speaker, sample_next

GEN = 100  # 10 s


def moments(convs: list[Conv], n: int, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    per = []
    for ci, c in enumerate(convs):
        for ch in range(2):
            ts = []
            for t in range(CTX, c.length - GEN - GAP - WIN - 1):
                if (
                    c.active[ch, t - 10 : t + 1].any()
                    or not c.active[1 - ch, t - 30 : t + 1].any()
                ):
                    continue
                if not c.unit[ch, t + 1 : t + 21].any():
                    continue
                ts.append(t)
            rng.shuffle(ts)
            per.append([{"conv": ci, "ch": ch, "t": int(t)} for t in ts])
    out, k = [], 0
    while len(out) < n and any(len(p) > k for p in per):
        for p in per:
            if len(p) > k and len(out) < n:
                out.append(p[k])
        k += 1
    return out


@torch.no_grad()
def generate(
    model: Speaker,
    convs: list[Conv],
    items: list[dict],
    device,
    temperature: float,
    top_p: float,
    seed: int,
) -> np.ndarray:
    k = model.cfg.k
    gen = torch.Generator(device=device).manual_seed(seed)
    tok_self = np.stack(
        [
            convs[it["conv"]].tok[it["ch"], it["t"] - CTX + 1 : it["t"] + 1 + GEN]
            for it in items
        ]
    )
    tok_part = np.stack(
        [
            convs[it["conv"]].tok[1 - it["ch"], it["t"] - CTX + 1 : it["t"] + 1 + GEN]
            for it in items
        ]
    )
    pros_self = np.stack(
        [
            convs[it["conv"]].pros[it["ch"], it["t"] - CTX + 1 : it["t"] + 1 + GEN]
            for it in items
        ]
    )
    pros_part = np.stack(
        [
            convs[it["conv"]].pros[1 - it["ch"], it["t"] - CTX + 1 : it["t"] + 1 + GEN]
            for it in items
        ]
    )
    tok_self[:, CTX:] = k + 1
    pros_self[:, CTX:] = 0
    ts = torch.from_numpy(tok_self).to(device)
    tp = torch.from_numpy(tok_part).to(device)
    pr = torch.from_numpy(np.concatenate([pros_self, pros_part], -1)).to(device)
    model.eval()
    for g in range(GEN):
        lo = g + 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(
                ts[:, lo : lo + CTX], tp[:, lo : lo + CTX], pr[:, lo : lo + CTX]
            )
        nxt = sample_next(model.speech_logits(h[:, -1]), temperature, top_p, gen)
        ts[:, CTX + g] = nxt
    return ts[:, CTX:].cpu().numpy()


def timing_form(tok: np.ndarray, k: int) -> dict:
    unit = tok < k
    active = tok != k + 1
    first = int(np.argmax(unit)) if unit.any() else -1
    seq = tok[unit]
    grams = [tuple(seq[i : i + 4]) for i in range(len(seq) - 3)]
    rep = 1 - len(set(grams)) / len(grams) if grams else 0.0
    return {
        "first": first,
        "units": int(unit.sum()),
        "active_s": float(active.sum()) / 10,
        "rep4": rep,
    }


def judge_content(
    judge: Tower, convs: list[Conv], items: list[dict], gens: np.ndarray, device
) -> dict:
    """Judge accuracy for the generated and for the real reply, with real and swapped judge context."""
    rng = np.random.default_rng(7)
    rows = {"gen": [], "real": []}
    # the judge's swap takes the next list entry: order so that neighbours are other conversations
    order = sorted(range(len(items)), key=lambda i: (items[i]["ch"], items[i]["conv"]))
    assert all(
        items[order[j]]["conv"] != items[order[(j + 1) % len(order)]]["conv"]
        for j in range(len(order))
    )
    items_o = [items[i] for i in order]
    gens_o = gens[order]
    for kind in rows:
        mod, anc = [], []
        for i, it in enumerate(items_o):
            c = copy.copy(convs[it["conv"]])
            c.tok = c.tok.copy()
            c.pros = c.pros.copy()
            if kind == "gen":
                c.tok[it["ch"], it["t"] + 1 : it["t"] + 1 + GEN] = gens_o[i]
                c.pros[it["ch"], it["t"] + 1 : it["t"] + 1 + GEN] = 0
            c.__post_init__()
            negs = hard_negatives(convs[it["conv"]], it["ch"], it["t"], rng)
            if not negs or len(c.window(it["ch"], it["t"])) < 3:
                anc.append(None)
            else:
                anc.append(
                    {
                        "conv": i,
                        "ch": 1 - it["ch"],
                        "s": it["t"] - CTX + 1,
                        "t": it["t"],
                        "neg": negs,
                    }
                )
            mod.append(c)
        valid = [a for a in anc if a is not None]
        res_real = correct_per_anchor(
            judge, mod, {"items": valid}, swap=False, device=device
        )
        res_swap = correct_per_anchor(
            judge, mod, {"items": valid}, swap=True, device=device
        )
        real_full, swap_full, j = np.zeros(len(items_o)), np.zeros(len(items_o)), 0
        for i, a in enumerate(anc):
            if a is not None:
                real_full[i], swap_full[i] = res_real[j], res_swap[j]
                j += 1
        conv = np.array([it["conv"] for it in items_o])
        r = {"answered": float(np.mean([a is not None for a in anc]))}
        r["acc"], r["acc_lo"], r["acc_hi"] = bootstrap(real_full, conv)
        r["acc_swap"], _, _ = bootstrap(swap_full, conv)
        r["delta"], r["delta_lo"], r["delta_hi"] = bootstrap(
            real_full - swap_full, conv
        )
        rows[kind] = r
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument(
        "--anchors",
        required=True,
        help="DY-14 test_anchors.json: the test conversation list",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument(
        "--moments",
        default="",
        help="existing moments file; built and written if missing",
    )
    args = p.parse_args()
    device = torch.device("cuda")
    names = json.loads(Path(args.anchors).read_text())["names"]
    convs = load_convs(args.grid, names, args.k)
    mpath = Path(args.moments or Path(args.out).with_name("moments.json"))
    if not mpath.exists():
        mpath.write_text(json.dumps(moments(convs, args.n)))
    items = json.loads(mpath.read_text())

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = Speaker(TowerConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    gens = np.concatenate(
        [
            generate(model, convs, items[i : i + 100], device, 1.0, 0.9, seed=1234 + i)
            for i in range(0, len(items), 100)
        ]
    )
    jk = torch.load(args.judge, map_location="cpu", weights_only=False)
    judge = Tower(TowerConfig(**jk["cfg"])).to(device)
    judge.load_state_dict(jk["model"])
    content = judge_content(judge, convs, items, gens, device)

    gen_tf = [timing_form(g, args.k) for g in gens]
    real_tf = [
        timing_form(
            convs[it["conv"]].tok[it["ch"], it["t"] + 1 : it["t"] + 1 + GEN], args.k
        )
        for it in items
    ]

    def agg(tf: list[dict]) -> dict:
        firsts = [x["first"] for x in tf if 0 <= x["first"] < 30]
        units = sum(x["units"] for x in tf)
        act = sum(x["active_s"] for x in tf)
        return {
            "started_3s": len(firsts) / len(tf),
            "median_latency_s": float(np.median(firsts)) / 10 if firsts else None,
            "units_per_s_speech": units / max(act, 1e-6),
            "rep4": float(np.mean([x["rep4"] for x in tf])),
        }

    timing = {"gen": agg(gen_tf), "real": agg(real_tf)}
    g, r = timing["gen"], timing["real"]
    gates = {
        "C": content["gen"]["acc"] >= 0.125 and content["gen"]["delta_lo"] > 0,
        "T": (
            g["median_latency_s"] is not None
            and r["median_latency_s"]
            and 0.5 <= g["median_latency_s"] / r["median_latency_s"] <= 2.0
            and g["started_3s"] >= 0.5 * r["started_3s"]
        ),
        "F": 3 <= g["units_per_s_speech"] <= 6
        and g["rep4"] <= 2 * max(r["rep4"], 1e-3),
    }
    res = {
        "model": args.model,
        "judge": args.judge,
        "n": len(items),
        "content": content,
        "timing_form": timing,
        "gates": {k: bool(v) for k, v in gates.items()},
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    np.save(Path(args.out).with_suffix(".tokens.npy"), gens)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
