"""DY-33 track M: run-length statistics of generated codes. Share of frames (steps) that lie in runs of one repeated
code (all levels equal, for multi-level steps) lasting >= 0.5 s. Codes come from each folder's info.json ("codes":
id -> list, 1-D or [steps][levels]; rate from info "rate_hz"/"steps_per_s", else 50 Hz) for the folder and its -x
twin; the real reference is the held-out real_XX (and 050-199) coded with km-corpus-K2000 at 50 Hz (cosine).

    python scripts/dy33_runs.py --round b_ref --folders b-t10,b-ra08
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy32_channel import S3, W, log, sh

MIN_S = 0.5


def run_share(seqs: list[np.ndarray], rate: float) -> float:
    need = int(round(MIN_S * rate))
    tot = inside = 0
    for s in seqs:
        s = np.asarray(s)
        if s.ndim == 2:
            s = np.array([hash(tuple(r)) for r in s])
        if len(s) == 0:
            continue
        b = np.r_[True, s[1:] != s[:-1]]
        starts = np.flatnonzero(b)
        lens = np.diff(np.r_[starts, len(s)])
        inside += int(lens[lens >= need].sum())
        tot += len(s)
    return inside / tot if tot else float("nan")


def folder_codes(d: Path) -> tuple[list, float, int]:
    seqs, rate = [], 50.0
    for x in (d, Path(str(d) + "-x")):
        f = x / "info.json"
        if f.exists():
            j = json.loads(f.read_text())
            rate = float(j.get("rate_hz") or j.get("steps_per_s") or rate)
            seqs += [np.asarray(v) for v in (j.get("codes") or {}).values()]
    return seqs, rate, len(seqs)


def real_codes() -> list:
    import soundfile as sf
    import torch  # noqa: F401

    from dy32b_channel import Channel, Codebook

    ch, cb = Channel(), Codebook(str(W / "centroids" / "km-corpus-K2000.npy"))
    files = sorted(Path("/data/dy32/prompts").glob("real_*.wav")) + sorted(Path("/data/dy32/prompts_x").glob("real_*.wav"))
    wavs = [sf.read(f, dtype="float32")[0] for f in files]
    return [cb.encode(f).cpu().numpy() for f in ch.feats(wavs)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True)
    p.add_argument("--folders", required=True)
    a = p.parse_args()
    E = W / "eval" / a.round
    E.mkdir(parents=True, exist_ok=True)
    cf = W / "runs_real.json"
    if cf.exists():
        real = json.loads(cf.read_text())
    else:
        rc = real_codes()
        real = {"share": run_share(rc, 50.0), "n": len(rc), "units": "km-corpus-K2000, 50 Hz"}
        cf.write_text(json.dumps(real))
    res = {"definition": f"share of steps inside runs of one repeated code (all levels equal) lasting >= {MIN_S} s",
           "real": real, "folders": {}}
    for n in a.folders.split(","):
        seqs, rate, k = folder_codes(W / "gen" / n)
        res["folders"][n] = {"share": run_share(seqs, rate) if seqs else None, "n": k, "rate": rate}
    (E / "runs.json").write_text(json.dumps(res, indent=1))
    L = ["", "## Run-length statistics of generated codes", "", res["definition"] + ".", "",
         "| folder | share in runs >= 0.5 s | sequences | steps/s |", "|---|---|---|---|",
         f"| real (held-out, km-corpus-K2000) | {real['share']:.4f} | {real['n']} | 50 |"]
    for n, r in res["folders"].items():
        share = "-" if r["share"] is None else f"{r['share']:.4f}"
        L.append(f"| {n} | {share} | {r['n']} | {r['rate']:g} |")
    sj = E / "summary.json"
    if sj.exists():
        j = json.loads(sj.read_text())
        j["runs"] = res
        sj.write_text(json.dumps(j, indent=2, ensure_ascii=False))
    sm = E / "summary.md"
    if sm.exists():
        sm.write_text(sm.read_text() + "\n".join(L) + "\n")
    log("RUNS", json.dumps(res))
    sh(f"aws s3 sync {E}/ {S3}/eval/{a.round}/ --only-show-errors --exclude '*' --include 'runs.json' --include 'summary.md' --include 'summary.json'")


if __name__ == "__main__":
    main()
