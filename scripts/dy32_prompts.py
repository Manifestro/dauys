"""DY-32 track A, job 1: the common prompt pack.

Held-out selection as dy31_eval (every held-out shard of the source, pieces >= 45 s, seed 1234 order); walked in
that order, taking pieces round-robin over shards (first one piece per shard, then a second, ...) until 50.
prompt_XX.wav = piece 0-15 s, real_XX.wav = 15-25 s (16 kHz mono int16), manifest.json (id, shard, piece, offsets).

    python scripts/dy32_prompts.py --out /data/dy32/prompts --s3 s3://.../outputs/dauys/dy32/prompts/
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_eval import MIN_S, SEED, SR, SRC, candidates, read_pieces

PROMPT_S, REAL_S, N = 15, 10, 50


def shards() -> list[str]:
    out = subprocess.run(f"aws s3 ls {SRC}/", shell=True, check=True, capture_output=True, text=True).stdout
    return sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))


def round_robin(order: list[tuple], n: int) -> list[tuple]:
    seen: dict[str, int] = {}
    ranked = []
    for pos, c in enumerate(order):
        r = seen.get(c[0], 0)
        seen[c[0]] = r + 1
        ranked.append((r, pos, c))
    return [c for _, _, c in sorted(ranked)[:n]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--s3", default="")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--first", type=int, default=0, help="extension pack: skip the first pieces of the same round-robin")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent))
    val = [s for s in shards() if held_out(s)]
    cands = candidates(val, MIN_S, tmp)
    rng = np.random.default_rng(SEED)
    order = [cands[i] for i in rng.permutation(len(cands))]
    sel = round_robin(order, a.n)[a.first :]
    fmt = "03d" if a.first else "02d"
    audio = read_pieces(sel, tmp)
    man = []
    for i, ((s, pc, st, ln), w) in enumerate(zip(sel, audio), start=a.first):
        assert held_out(s) and len(w) >= (PROMPT_S + REAL_S) * SR
        sf.write(out / f"prompt_{i:{fmt}}.wav", w[: PROMPT_S * SR], SR, subtype="PCM_16")
        sf.write(out / f"real_{i:{fmt}}.wav", w[PROMPT_S * SR : (PROMPT_S + REAL_S) * SR], SR, subtype="PCM_16")
        man.append({"id": f"{i:{fmt}}", "shard": s, "piece": pc, "piece_start_sample": st, "piece_len_s": round(ln / SR, 2),
                    "prompt_s": [0, PROMPT_S], "real_s": [PROMPT_S, PROMPT_S + REAL_S]})
    info = {"source": SRC, "held_out_shards": val, "candidates": len(cands), "min_piece_s": MIN_S, "seed": SEED,
            "rule": "held-out shards (dy21_train.held_out), pieces >= 45 s, seed-1234 permutation, round-robin over shards",
            "distinct_shards": len({m["shard"] for m in man}), "sr": SR, "pieces": man}
    (out / "manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "pieces"}), flush=True)
    if a.s3:
        subprocess.run(f"aws s3 sync {out}/ {a.s3} --only-show-errors", shell=True, check=True)
        print("UPLOADED", a.s3, flush=True)


if __name__ == "__main__":
    main()
