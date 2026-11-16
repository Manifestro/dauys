"""DY-17: the who-and-when head decides when to start speaking; the speech head writes what.

Same model, moments, judge and gates as DY-16 (``dy16_generate``); only the onset rule changes.
While the own channel is silent, at each frame the who-and-when head gives the probability that
the own channel is active in each of the next 10 frames. If their mean is above 0.5 the own
channel starts at the next frame with a unit sampled from the speech head restricted to units;
otherwise it stays silent. Once speaking, the speech head runs freely (it also decides to stop;
after a stop the rule above applies again).

Control moments (gate D): own channel silent for 1 s, partner spoke in the last 3 s, and in
reality the own channel stays silent for the next 3 s. A model that starts everywhere fails D.

``--oracle`` replaces the head's probabilities by the real future activity of the own channel
(checklist: known answer for the onset plumbing).

    python scripts/dy17_generate.py --grid /data/dy14/grid --anchors test_anchors.json --model /data/dy16/runs/f0/best.pt \
        --judge /data/dy14/runs/s0/best.pt --moments moments.json --control control.json --out /data/dy17/gen_f0.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy16_generate import GEN, judge_content, timing_form

from dauys.dy14_data import CTX, GAP, WIN, Conv, WHO_AHEAD, load_convs
from dauys.dy14_model import Tower, TowerConfig
from dauys.dy16_model import Speaker, sample_next

THRESHOLD = 0.5


def control_moments(convs: list[Conv], n: int, seed: int = 1234) -> list[dict]:
    """Like DY-16 moments, but the real own channel stays silent for the next 3 s."""
    rng = np.random.default_rng(seed)
    per = []
    for ci, c in enumerate(convs):
        for ch in range(2):
            ts = [
                t
                for t in range(CTX, c.length - GEN - GAP - WIN - 1)
                if not c.active[ch, t - 10 : t + 31].any()
                and c.active[1 - ch, t - 30 : t + 1].any()
            ]
            rng.shuffle(ts)
            per.append([{"conv": ci, "ch": ch, "t": int(t)} for t in ts])
    out, k = [], 0
    while len(out) < n and any(len(p) > k for p in per):
        for p in per:
            if len(p) > k and len(out) < n:
                out.append(p[k])
        k += 1
    return out


def _stack(convs, items, ch_of, field):
    return np.stack(
        [
            getattr(convs[it["conv"]], field)[
                ch_of(it), it["t"] - CTX + 1 : it["t"] + 1 + GEN
            ]
            for it in items
        ]
    )


@torch.no_grad()
def generate_gated(
    model: Speaker,
    convs: list[Conv],
    items: list[dict],
    device,
    seed: int,
    oracle: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns generated own tokens [B, GEN] and the onset score at every generated frame [B, GEN]."""
    k = model.cfg.k
    gen = torch.Generator(device=device).manual_seed(seed)
    own, part = (lambda it: it["ch"]), (lambda it: 1 - it["ch"])
    tok_self = _stack(convs, items, own, "tok")
    tok_part = _stack(convs, items, part, "tok")
    pros = np.concatenate(
        [_stack(convs, items, own, "pros"), _stack(convs, items, part, "pros")], -1
    )
    future = torch.from_numpy(
        np.stack(
            [
                convs[it["conv"]].active[
                    it["ch"], it["t"] + 1 : it["t"] + 1 + GEN + WHO_AHEAD
                ]
                for it in items
            ]
        ).astype(np.float32)
    ).to(device)
    tok_self[:, CTX:] = k + 1
    pros[:, CTX:, : pros.shape[-1] // 2] = 0
    ts = torch.from_numpy(tok_self).to(device)
    tp = torch.from_numpy(tok_part).to(device)
    pr = torch.from_numpy(pros).to(device)
    unit_only = torch.zeros(k + 2, device=device, dtype=torch.bool)
    unit_only[k:] = True
    scores = []
    model.eval()
    for g in range(GEN):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.states(ts[:, g : g + CTX], tp[:, g : g + CTX], pr[:, g : g + CTX])
        last = h[:, -1]
        if oracle:
            p_self = future[:, g : g + WHO_AHEAD]
        else:
            p_self = torch.sigmoid(model.who(last).float())[:, :WHO_AHEAD]
        score = p_self.mean(1)
        scores.append(score)
        logits = model.speech_logits(last)
        free = sample_next(logits, temperature, top_p, gen)
        first = sample_next(logits.masked_fill(unit_only, float("-inf")), temperature, top_p, gen)
        silent = ts[:, CTX + g - 1] == k + 1
        start = score > threshold
        nxt = torch.where(silent, torch.where(start, first, torch.full_like(free, k + 1)), free)
        ts[:, CTX + g] = nxt
    return ts[:, CTX:].cpu().numpy(), torch.stack(scores, 1).cpu().numpy()


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


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


def run_all(model, convs, items, device, oracle, threshold=THRESHOLD):
    outs = [
        generate_gated(model, convs, items[i : i + 100], device, 1234 + i, oracle, threshold=threshold)
        for i in range(0, len(items), 100)
    ]
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--judge", default="")
    p.add_argument("--moments", required=True)
    p.add_argument("--control", required=True, help="control moments; built and written if missing")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--threshold-file", default="", help="JSON with the calibrated threshold (DY-18)")
    args = p.parse_args()
    device = torch.device("cuda")
    names = json.loads(Path(args.anchors).read_text())["names"]
    convs = load_convs(args.grid, names, args.k)
    items = json.loads(Path(args.moments).read_text())
    cpath = Path(args.control)
    if not cpath.exists():
        cpath.write_text(json.dumps(control_moments(convs, args.n)))
    ctrl = json.loads(cpath.read_text())

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = Speaker(TowerConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    thr = args.threshold
    if args.threshold_file:
        thr = json.loads(Path(args.threshold_file).read_text())["threshold"]
    gens, s_reply = run_all(model, convs, items, device, args.oracle, thr)
    gens_c, s_ctrl = run_all(model, convs, ctrl, device, args.oracle, thr)
    s_reply, s_ctrl = s_reply[:, 0], s_ctrl[:, 0]

    real_tf = [
        timing_form(convs[it["conv"]].tok[it["ch"], it["t"] + 1 : it["t"] + 1 + GEN], args.k)
        for it in items
    ]
    timing = {
        "gen": agg([timing_form(g, args.k) for g in gens]),
        "real": agg(real_tf),
        "gen_control": agg([timing_form(g, args.k) for g in gens_c]),
        "onset_score_auc": auc(s_reply, s_ctrl),
        "onset_score_reply_mean": float(s_reply.mean()),
        "onset_score_control_mean": float(s_ctrl.mean()),
    }
    g, r, c = timing["gen"], timing["real"], timing["gen_control"]
    gates = {
        "T": (
            g["median_latency_s"] is not None
            and 0.5 <= g["median_latency_s"] / r["median_latency_s"] <= 2.0
            and g["started_3s"] >= 0.5 * r["started_3s"]
        ),
        "D": g["started_3s"] - c["started_3s"] >= 0.3 and c["started_3s"] <= 0.5,
        "F": 3 <= g["units_per_s_speech"] <= 6 and g["rep4"] <= 2 * max(r["rep4"], 1e-3),
    }
    res = {"model": args.model, "threshold": thr, "oracle": args.oracle, "n": len(items), "n_control": len(ctrl), "timing_form": timing}
    if args.judge:
        jk = torch.load(args.judge, map_location="cpu", weights_only=False)
        judge = Tower(TowerConfig(**jk["cfg"])).to(device)
        judge.load_state_dict(jk["model"])
        res["content"] = judge_content(judge, convs, items, gens, device)
        cg = res["content"]["gen"]
        gates["C"] = cg["acc"] >= 0.125 and cg["delta_lo"] > 0
    res["gates"] = {k: bool(v) for k, v in gates.items()}
    Path(args.out).write_text(json.dumps(res, indent=2))
    np.save(Path(args.out).with_suffix(".tokens.npy"), gens)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
