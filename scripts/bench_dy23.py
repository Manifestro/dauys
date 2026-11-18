"""Open spoken-LM benchmarks for DY-23 (same protocol as bench_slm.py: sWUGGY / sBLIMP dev 3000 pairs, seed 0;
topic / spoken StoryCloze full clips; ties count half). Each clip -> DY-14 grid (events) + 8x256 voice-invariant
codes exactly as dy23_units.py; score = log p(events) + log p(8 codes of every syllable).

    python scripts/bench_dy23.py --zr /data/bench/zr --centroids /data/dy14/centroids.pt --coder coder.pt \
        --run /data/dy23/run/best.pt --out /data/dy23/bench.json
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import to_int16, zr_pairs
from dy14_features import segments_of
from dy14_grid import nearest
from dy23_train import keep_mask
from units_vinv import Coder

from dauys.dy14_units import GRID, SR, build_track, grid_track, hidden_states, load_sylber, segment_f0
from dauys.dy21_lm import LMConfig
from dauys.dy23_lm import CONT, EOS, SIL, UNIT, CodeLM

K = 8192


class Enc:
    def __init__(self, centroids: str, coder: str, device) -> None:
        self.device = device
        self.syl = load_sylber(device)
        self.cent = torch.load(centroids).to(device)
        ck = torch.load(coder, map_location="cpu")
        self.coder = Coder().to(device).eval()
        self.coder.load_state_dict(ck["coder"])
        self.pq = [b.to(device) for b in ck["pq"]["8x256"]]

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(wav) < SR // 5:
            wav = np.pad(wav, (0, SR // 5 - len(wav)))
        states = hidden_states(self.syl, wav)
        seg = segments_of(states)
        if len(seg) == 0:
            return np.zeros(0, np.int64), np.zeros((0, 8), np.int64)
        f0, voiced = segment_f0(wav, seg, self.device)
        tr = build_track(states, seg, wav, np.zeros_like(wav), f0, voiced)
        f = torch.from_numpy(tr.features.astype(np.float32)).to(self.device)
        units = nearest(F.normalize(f, dim=1), self.cent).cpu().numpy()
        tok, _, _ = grid_track(tr.segments, tr.keep, units, tr.energy_db, tr.f0_st, tr.voiced, int(len(states)), K)
        z = self.coder.z(F.normalize(f, dim=1))
        c = torch.stack([torch.cdist(p.contiguous(), b).argmin(1) for p, b in zip(z.chunk(8, 1), self.pq)], 1).cpu().numpy()
        cd = np.zeros((len(tok), 8), np.int64)
        kept = np.flatnonzero(tr.keep)
        taken = np.zeros(len(tok), bool)
        for j, s0 in enumerate(tr.segments[kept][:, 0] // GRID):
            if not taken[s0]:
                taken[s0] = True
                cd[s0] = c[kept[j]]
        tok = tok.astype(np.int64)
        sp = np.flatnonzero(tok != K + 1)
        if len(sp) == 0:
            return np.zeros(0, np.int64), np.zeros((0, 8), np.int64)
        tok, cd = tok[sp[0] : sp[-1] + 1], cd[sp[0] : sp[-1] + 1]
        m = keep_mask(tok)
        tok, cd = tok[m], cd[m]
        ev = np.where(tok < K, UNIT, np.where(tok == K, CONT, SIL))
        return ev, cd


@torch.no_grad()
def logprob(model: CodeLM, ev: np.ndarray, cd: np.ndarray, device, groups: int = 8, events: bool = True,
            mean: bool = False) -> float:
    e = torch.from_numpy(np.r_[EOS, ev]).long()[None].to(device)
    c = torch.from_numpy(np.concatenate([np.zeros((1, 8), np.int64), cd])).long()[None].to(device)
    e, c = e[:, -(model.cfg.ctx + 1) :], c[:, -(model.cfg.ctx + 1) :]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.states(e[:, :-1], c[:, :-1])[0]
    y = e[0, 1:]
    lp = torch.log_softmax(model.ev_head(h).float(), -1).gather(1, y[:, None]).sum() if events else torch.zeros((), device=device)
    nxt = y == UNIT
    if nxt.any():
        yc = c[0, 1:][nxt]
        lc = torch.log_softmax(model.code_logits(h[nxt], yc), -1)
        lp = lp + lc.gather(2, yc[:, :, None])[:, :groups].sum()
    return float(lp) / (max(int((y == UNIT).sum()), 1) if mean else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--coder", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--sc-splits", default="tsc_bm,tsc_af,ssc_bm,ssc_af")
    p.add_argument("--tasks", default="swuggy,sblimp,sc")
    p.add_argument("--scores", default="all", help="comma list: all, g1, g2, g4, codes_only (no events), all_mean / g1_mean (per syllable)")
    args = p.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    enc = Enc(args.centroids, args.coder, device)
    ck = torch.load(args.run, map_location="cpu", weights_only=False)
    model = CodeLM(LMConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    results = {}

    modes = {"all": (8, True, False), "g1": (1, True, False), "g2": (2, True, False), "g4": (4, True, False),
             "codes_only": (8, False, False), "all_mean": (8, True, True), "g1_mean": (1, True, True)}
    scores = args.scores.split(",")

    def run(name, items):
        t0, n = time.time(), 0
        right = {s: 0.0 for s in scores}
        for good, bad in items:
            if (good[0] == UNIT).sum() == 0 or (bad[0] == UNIT).sum() == 0:
                continue
            n += 1
            for s in scores:
                g, e, mn = modes[s]
                a, b = logprob(model, *good, device, g, e, mn), logprob(model, *bad, device, g, e, mn)
                right[s] += 1.0 if a > b else 0.5 if a == b else 0.0
        results[name] = {"n": n, "seconds": round(time.time() - t0)}
        for s in scores:
            acc = right[s] / max(n, 1)
            key = "dy23" if s == "all" else f"dy23_{s}"
            results[name][key] = {"acc": round(acc, 4), "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / max(n, 1)), 4)}
        print(name, json.dumps(results[name]), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    tasks = args.tasks.split(",")
    for task in ("swuggy", "sblimp"):
        if task in tasks:
            pairs = zr_pairs(Path(args.zr), {"swuggy": "lexical", "sblimp": "syntactic"}[task], args.pairs, rng)
            run(task, ((enc(to_int16(*sf.read(g))), enc(to_int16(*sf.read(b)))) for g, b in pairs))
    if "sc" in tasks:
        from datasets import load_dataset

        for split in args.sc_splits.split(","):
            kind, voice = split.split("_")
            ds = load_dataset("slprl/multispeaker-storycloze", data_files=f"{kind[0]}SC/{voice}/*.parquet", split="train")
            a = lambda x: enc(to_int16(np.asarray(x["array"], np.float32), x["sampling_rate"]))
            run(split, ((a(r["correct_audio"]), a(r["negative_audio"])) for r in ds))
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
