"""Open spoken-LM benchmarks for the DY-21 / DY-22 unit LMs: sWUGGY, sBLIMP (ZeroSpeech 2021 dev) and
spoken / topic StoryCloze (slprl/multispeaker-storycloze).

Every clip is turned into DY-14 units exactly as the training data (Sylber layer-9 states, segments, DY-14
track with no other channel, nearest DY-14 centroid, 10 Hz grid). A pair is right when the model gives the
correct item the higher total log-probability (sWUGGY, sBLIMP: the whole clip after an EOS; StoryCloze: the
ending given the story context). Models:
  dy21   — deduplicated units (the grid with continue/pause removed);
  dy22   — the grid with pauses cut to 2 s; scored twice: ``all`` tokens and ``unit`` targets only.
Pairs are subsampled with a fixed seed (``--pairs``); accuracy with a 95 % binomial interval.

    python scripts/bench_slm.py --zr /data/bench/sLM21-dataset --centroids /data/dy14/centroids.pt \
        --dy21 /data/dy21/run/best.pt --dy22 /data/dy22/run/best.pt --out /data/bench/out.json
"""

import argparse
import csv
import io
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
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from dy14_features import segments_of
from dy14_grid import nearest
from dy22_train import cut_pauses

from dauys.dy14_units import SR, build_track, grid_track, hidden_states, load_sylber, segment_f0
from dauys.dy21_lm import LMConfig, UnitLM
from dauys.dy22_lm import EOS as EOS22
from dauys.dy22_lm import SIL, GridLM

K = 8192


def to_int16(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr)
    return np.clip(wav * 32768, -32768, 32767).astype(np.int16)


class Units:
    def __init__(self, centroids: str, device) -> None:
        self.device = device
        self.model = load_sylber(device)
        self.cent = torch.load(centroids).to(device)

    def grid(self, wav: np.ndarray) -> np.ndarray:
        """10 Hz grid tokens of one clip, leading and trailing pauses removed, inner pauses cut to 2 s."""
        if len(wav) < SR // 5:
            wav = np.pad(wav, (0, SR // 5 - len(wav)))
        states = hidden_states(self.model, wav)
        seg = segments_of(states)
        if len(seg) == 0:
            return np.zeros(0, np.int64)
        f0, voiced = segment_f0(wav, seg, self.device)
        tr = build_track(states, seg, wav, np.zeros_like(wav), f0, voiced)
        f = torch.from_numpy(tr.features.astype(np.float32)).to(self.device)
        units = nearest(F.normalize(f, dim=1), self.cent).cpu().numpy() if len(f) else np.zeros(0, np.int64)
        tok, pros, _ = grid_track(tr.segments, tr.keep, units, tr.energy_db, tr.f0_st, tr.voiced, int(len(states)), K)
        tok = tok.astype(np.int64)
        speech = np.flatnonzero(tok != SIL)
        if len(speech) == 0:
            return np.zeros(0, np.int64)
        tok = tok[speech[0] : speech[-1] + 1]
        tok, _ = cut_pauses(tok, np.zeros((len(tok), 3), np.float32))
        return tok


@torch.no_grad()
def logprob(model, ctx: np.ndarray, cont: np.ndarray, device, unit_only: bool) -> float:
    """Sum of log p(cont | EOS, ctx) under the model; ``unit_only`` sums only where the target is a unit."""
    ids = torch.from_numpy(np.concatenate([ctx, cont]).astype(np.int64))[None].to(device)
    ids = ids[:, -(model.cfg.ctx + 1) :]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.states(ids[:, :-1])
    lp = torch.log_softmax(model.logits(h), -1)[0]
    tgt = ids[0, 1:]
    tok_lp = lp.gather(1, tgt[:, None])[:, 0]
    n = min(len(cont), len(tgt))
    tok_lp, tgt = tok_lp[-n:], tgt[-n:]
    if unit_only:
        tok_lp = tok_lp[tgt < K]
    return float(tok_lp.sum())


def zr_pairs(root: Path, task: str, n: int, rng) -> list[tuple[Path, Path]]:
    d = root / task / "dev"
    rows = list(csv.DictReader(open(d / "gold.csv")))
    groups = defaultdict(dict)
    for r in rows:
        key = (r["id"], r.get("voice", ""))
        groups[key][int(r["correct"])] = d / f"{r['filename']}.wav"
    pairs = [(g[1], g[0]) for g in groups.values() if 0 in g and 1 in g]
    pick = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return [pairs[i] for i in pick]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True, help="unpacked ZeroSpeech sLM21-dataset")
    p.add_argument("--centroids", required=True)
    p.add_argument("--dy21", required=True)
    p.add_argument("--dy22", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--sc-splits", default="tsc_bm,tsc_af,ssc_bm,ssc_af")
    p.add_argument("--tasks", default="swuggy,sblimp,sc")
    args = p.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    U = Units(args.centroids, device)

    models = {}
    ck = torch.load(args.dy21, map_location="cpu", weights_only=False)
    m = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    models["dy21"] = m
    ck = torch.load(args.dy22, map_location="cpu", weights_only=False)
    m = GridLM(LMConfig(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    models["dy22"] = m
    scorers = {
        "dy21": lambda ctx, cont: logprob(models["dy21"], dedup(ctx, 8192), dedup(cont, None), device, False),
        "dy22_all": lambda ctx, cont: logprob(models["dy22"], ctx22(ctx), cont, device, False),
        "dy22_unit": lambda ctx, cont: logprob(models["dy22"], ctx22(ctx), cont, device, True),
    }

    def dedup(t: np.ndarray, eos) -> np.ndarray:
        u = t[t < K]
        return np.r_[eos, u] if eos is not None else u

    def ctx22(t: np.ndarray) -> np.ndarray:
        return np.r_[EOS22, t]

    results = {}
    tasks = args.tasks.split(",")

    def run(name: str, items):
        t0 = time.time()
        right = defaultdict(int)
        n = 0
        for ctx, good, bad in items:
            if len(good[good < K]) == 0 or len(bad[bad < K]) == 0:
                continue
            n += 1
            for s, fn in scorers.items():
                right[s] += fn(ctx, good) > fn(ctx, bad)
        res = {}
        for s in scorers:
            a = right[s] / max(n, 1)
            res[s] = {"acc": round(a, 4), "ci95": round(1.96 * math.sqrt(a * (1 - a) / max(n, 1)), 4)}
        res["n"] = n
        res["seconds"] = round(time.time() - t0)
        results[name] = res
        print(name, json.dumps(res), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    empty = np.zeros(0, np.int64)
    for task in ("swuggy", "sblimp"):
        if task not in tasks:
            continue
        sub = {"swuggy": "lexical", "sblimp": "syntactic"}[task]
        pairs = zr_pairs(Path(args.zr), sub, args.pairs, rng)

        def items(pairs=pairs):
            for g, b in pairs:
                yield empty, U.grid(to_int16(*sf.read(g))), U.grid(to_int16(*sf.read(b)))
        run(task, items())

    if "sc" in tasks:
        from datasets import load_dataset

        for split in args.sc_splits.split(","):
            ds = load_dataset("slprl/multispeaker-storycloze", split=split)
            cols = ds.column_names
            print(split, cols, flush=True)
            audio = [c for c in cols if ds.features[c].__class__.__name__ == "Audio"]
            pc = next(c for c in audio if "prompt" in c)
            gc = next(c for c in audio if c != pc and ("correct" in c or "positive" in c) and "in" not in c.split("_")[0])
            bc = next(c for c in audio if c not in (pc, gc))
            print(split, "prompt", pc, "good", gc, "bad", bc, flush=True)

            def items(ds=ds, pc=pc, gc=gc, bc=bc):
                for r in ds:
                    enc = lambda a: U.grid(to_int16(np.asarray(a["array"], np.float32), a["sampling_rate"]))
                    yield enc(r[pc]), enc(r[gc]), enc(r[bc])
            run(split, items())
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
