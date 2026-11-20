"""DY-34 packs (track M): dev prompts 200-299 and one "swap" chunk for each of the 300 prompts.

Dev: dy32_prompts with the same held-out round-robin continued after piece 199 (pieces 200-299, no overlap).
Swap for prompt XXX (pack 000-049 dy32/prompts, 050-199 dy32/prompts_x, 200-299 dy34/prompts_dev): 15 s of the SAME
episode (pieces.jsonl "ep" of the prompt's shard) whose whole span is >= 60 s away from the span [prompt start, end of
the real continuation] = the first 25 s of the prompt's piece, in episode time. Candidates: every piece of that
episode, every 15 s chunk start inside it that keeps the gap; one candidate interval is drawn, then a start uniformly
inside it (one generator, seed 1234, prompts in id order) - as dy32b_heard_use does. If no chunk keeps 60 s, the
farthest possible chunk is taken and flagged.

    python scripts/dy34_packs.py
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import SR, SRC

B = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys"
D = Path("/data/dy34")
SWAP_S, GAP_S, SPAN_S = 15, 60.0, 25.0


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def shard_meta(s: str) -> tuple[np.ndarray, list[dict]]:
    d = D / "meta" / s
    d.mkdir(parents=True, exist_ok=True)
    for f in ("index.npy", "pieces.jsonl"):
        if not (d / f).exists():
            sh(f"aws s3 cp {SRC}/{s}/{f} {d / f} --only-show-errors")
    return np.load(d / "index.npy"), [json.loads(l) for l in open(d / "pieces.jsonl")]


def main() -> None:
    D.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    if not (D / "prompts_dev" / "manifest.json").exists():
        subprocess.run([sys.executable, str(here / "dy32_prompts.py"), "--n", "300", "--first", "200",
                        "--out", str(D / "prompts_dev"), "--s3", f"{B}/dy34/prompts_dev/"], check=True)
    man = []
    for p in ("/data/dy32/prompts/manifest.json", "/data/dy32/prompts_x/manifest.json", str(D / "prompts_dev" / "manifest.json")):
        man += json.loads(Path(p).read_text())["pieces"]
    man = sorted(man, key=lambda m: int(m["id"]))
    assert [int(m["id"]) for m in man] == list(range(300)), "packs must hold ids 000-299"
    assert len({(m["shard"], m["piece"]) for m in man}) == 300, "pieces overlap"
    rng = np.random.default_rng(1234)
    hs = SWAP_S * SR
    out, reads = [], []
    for m in man:
        idx, meta = shard_meta(m["shard"])
        p = m["piece"]
        ep, t0 = meta[p]["ep"], float(meta[p]["start"])
        assert int(idx[p][0]) == m["piece_start_sample"]
        w0, w1 = t0, t0 + SPAN_S
        same = [j for j in range(len(meta)) if meta[j]["ep"] == ep and int(idx[j][1]) >= hs]
        cand = []
        for j in same:
            lo, hi = float(meta[j]["start"]), float(meta[j]["start"]) + (int(idx[j][1]) - hs) / SR
            a, b = max(lo, w1 + GAP_S), hi
            if b >= a:
                cand.append((j, a, b))
            a, b = lo, min(hi, w0 - GAP_S - SWAP_S)
            if b >= a:
                cand.append((j, a, b))
        flagged = not cand
        if cand:
            j, lo, hi = cand[int(rng.integers(len(cand)))]
            t = float(rng.uniform(lo, hi))
        else:  # farthest possible chunk of the episode
            best = None
            for j in same:
                lo, hi = float(meta[j]["start"]), float(meta[j]["start"]) + (int(idx[j][1]) - hs) / SR
                for t_ in (lo, hi):
                    gap = max(t_ - w1, w0 - (t_ + SWAP_S))
                    if best is None or gap > best[0]:
                        best = (gap, j, t_)
            _, j, t = best
        off = min(max(int(round((t - float(meta[j]["start"])) * SR)), 0), int(idx[j][1]) - hs)
        t = float(meta[j]["start"]) + off / SR
        gap = max(t - w1, w0 - (t + SWAP_S))
        reads.append((m["id"], m["shard"], int(idx[j][0]) + off))
        out.append({"id": m["id"], "shard": m["shard"], "episode": ep, "prompt_piece": p,
                    "prompt_span_episode_s": [round(w0, 3), round(w1, 3)], "swap_piece": j,
                    "swap_offset_in_piece_samples": off, "swap_span_episode_s": [round(t, 3), round(t + SWAP_S, 3)],
                    "distance_s": round(gap, 3), "flagged_under_60s": bool(flagged or gap < GAP_S)})
    bucket, prefix = SRC[5:].split("/", 1)
    sw = D / "swap"
    sw.mkdir(parents=True, exist_ok=True)

    def one(r):
        i, s, st = r
        f = sw / f"tmp_{i}.bin"
        sh(f"aws s3api get-object --bucket {bucket} --key {prefix}/{s}/segs.bin --range bytes={2 * st}-{2 * (st + hs) - 1} {f}")
        w = np.fromfile(f, dtype=np.int16)
        f.unlink()
        assert len(w) == hs, (i, len(w))
        sf.write(sw / f"swap_{int(i):03d}.wav", w, SR, subtype="PCM_16")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, reads))
    nflag = sum(o["flagged_under_60s"] for o in out)
    info = {"rule": __doc__.split("\n\n")[0].strip(), "swap_s": SWAP_S, "gap_s": GAP_S, "span_s": SPAN_S, "seed": 1234,
            "flagged": nflag, "distance_min_s": min(o["distance_s"] for o in out),
            "distance_median_s": float(np.median([o["distance_s"] for o in out])), "items": out}
    (sw / "manifest.json").write_text(json.dumps(info, indent=1))
    sh(f"aws s3 sync {sw}/ {B}/dy34/swap/ --only-show-errors")
    print(json.dumps({k: v for k, v in info.items() if k != "items"}), flush=True)
    sh(f"echo done | aws s3 cp - {B}/dy34/PACKS_DONE")
    print("PACKS DONE", flush=True)


if __name__ == "__main__":
    main()
