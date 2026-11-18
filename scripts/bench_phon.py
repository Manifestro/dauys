"""Alphabet test, benchmarks: the phoneme LM on sWUGGY / sBLIMP (ZeroSpeech dev, 3000 pairs, seed 0) and topic /
spoken StoryCloze — same protocol as bench_slm.py (total log-probability after EOS, ties half), clips turned
into phonemes by the same recogniser as the training data (phon_units.Phonemes).

    python scripts/bench_phon.py --zr /data/bench/zr --run /data/phon/run/best.pt --out /data/phon/bench.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import logprob, to_int16, zr_pairs
from phon_units import Phonemes

from dauys.dy21_lm import LMConfig, UnitLM

EOS = 8192


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--sc-splits", default="tsc_bm,tsc_af,ssc_bm,ssc_af")
    p.add_argument("--tasks", default="swuggy,sblimp,sc")
    args = p.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    ph = Phonemes(device)
    ck = torch.load(args.run, map_location="cpu", weights_only=False)
    model = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    results = {}

    def run(name, items):
        t0, right, n = time.time(), 0.0, 0
        for good, bad in items:
            if len(good) == 0 or len(bad) == 0:
                continue
            n += 1
            a = logprob(model, np.array([EOS]), good, device, False)
            b = logprob(model, np.array([EOS]), bad, device, False)
            right += 1.0 if a > b else 0.5 if a == b else 0.0
        acc = right / max(n, 1)
        results[name] = {"phon": {"acc": round(acc, 4), "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / max(n, 1)), 4)},
                         "n": n, "seconds": round(time.time() - t0)}
        print(name, json.dumps(results[name]), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    tasks = args.tasks.split(",")
    for task in ("swuggy", "sblimp"):
        if task in tasks:
            pairs = zr_pairs(Path(args.zr), {"swuggy": "lexical", "sblimp": "syntactic"}[task], args.pairs, rng)
            run(task, ((ph(to_int16(*sf.read(g))), ph(to_int16(*sf.read(b)))) for g, b in pairs))
    if "sc" in tasks:
        from datasets import load_dataset

        for split in args.sc_splits.split(","):
            kind, voice = split.split("_")
            ds = load_dataset("slprl/multispeaker-storycloze", data_files=f"{kind[0]}SC/{voice}/*.parquet", split="train")
            a = lambda x: ph(to_int16(np.asarray(x["array"], np.float32), x["sampling_rate"]))
            run(split, ((a(r["correct_audio"]), a(r["negative_audio"])) for r in ds))
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
