"""Alphabet test, consistency: the phoneme recogniser on the same sWUGGY items as the DY-14 unit check.

units_diag.py protocol: sWUGGY dev ids with all four voices for both the word and the nonword, shuffled with
seed 0, first 400. Reports (1) the share of voice pairs (A/C/D/F, a < b) whose phoneme sequences for the WORD are
identical (DY-14 units: 28.6 %), (2) the same for the nonword, (3) ABX with normalised edit distance as in
units_diag (X = item in v1, A = same item in v2, B = other item in v1; DY-14 units 69.9, Sylber continuous 90.0).

    python scripts/phon_consistency.py --zr /data/bench/zr --out /data/phon/consistency.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import to_int16
from phon_units import Phonemes
from units_diag import edit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", type=int, default=400)
    args = p.parse_args()
    d = Path(args.zr) / "lexical" / "dev"
    items = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(d / "gold.csv")):
        items[r["id"]][int(r["correct"])][r["voice"]] = d / f"{r['filename']}.wav"
    ids = [i for i, g in items.items() if len(g[0]) == 4 and len(g[1]) == 4]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    ids = ids[: args.ids]
    ph = Phonemes(torch.device("cuda"))
    voices = ["A", "C", "D", "F"]
    seq = {(i, c, v): ph(to_int16(*sf.read(items[i][c][v]))) for i in ids for c in (0, 1) for v in voices}
    same = lambda c: float(np.mean([np.array_equal(seq[(i, c, a)], seq[(i, c, b)]) for i in ids for a in voices
                                    for b in voices if a < b]))
    right = n = 0.0
    for i in ids:
        for c in (0, 1):
            for v1 in voices:
                for v2 in voices:
                    if v1 != v2:
                        a = edit(seq[(i, c, v1)], seq[(i, c, v2)])
                        b = edit(seq[(i, c, v1)], seq[(i, 1 - c, v1)])
                        right += 1.0 if a < b else 0.5 if a == b else 0.0
                        n += 1
    res = {"same_word_identical_across_voices": round(same(1), 4), "same_nonword_identical_across_voices": round(same(0), 4),
           "word_vs_nonword_identical_same_voice": round(float(np.mean([np.array_equal(seq[(i, 1, v)], seq[(i, 0, v)])
                                                                       for i in ids for v in voices])), 4),
           "abx_edit": round(right / n, 4), "phonemes_per_item": round(float(np.mean([len(s) for s in seq.values()])), 2),
           "ids": len(ids), "dy14_reference": {"same_word_identical": 0.2858, "abx": 0.6986}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("CONSISTENCY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
