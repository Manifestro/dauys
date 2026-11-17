"""Richer syllable tokens: residual codes over the Sylber segment feature, checked by the sWUGGY-voice ABX.

Level 1 is the DY-14 unit (nearest of 8192 centroids, as now); levels 2..L are k-means codebooks (1024 each)
fitted on the residual of podcast segment features (DuplexChat training shards, as in DY-21/22). For L in
LEVELS: ABX (see units_diag.py) with
  recon — the feature rebuilt from L codes, DTW with cosine distance (what the codes still carry);
  codes — edit distance over syllables, substitution cost = share of the L codes that differ (what an LM
          over these discrete codes would see);
plus the mean cosine between a held-out podcast feature and its L-code rebuild.

    python scripts/units_rvq.py --zr /data/bench/zr --centroids /data/dy14/centroids.pt --out /data/rvq/out.json
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import to_int16
from dy14_features import segments_of
from dy21_train import held_out
from units_diag import cos_dtw, kmeans

from dauys.dy14_units import SR, build_track, hidden_states, load_sylber, segment_f0

SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"
LEVELS = (1, 2, 3, 4, 6, 8)


@numba.njit(cache=True)
def edit_codes(a: np.ndarray, b: np.ndarray) -> float:
    """Edit distance over code tuples [n, L]; substitution costs the share of differing codes."""
    n, m, L = a.shape[0], b.shape[0], a.shape[1]
    if max(n, m) == 0:
        return 0.0
    d = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = 0.0
            for k in range(L):
                s += a[i - 1, k] != b[j - 1, k]
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + s / L)
    return d[n, m] / max(n, m)


def seg_features(syl, wav: np.ndarray, device) -> torch.Tensor:
    states = hidden_states(syl, wav)
    seg = segments_of(states)
    if len(seg) == 0:
        return torch.zeros(0, 768, device=device)
    f0, voiced = segment_f0(wav, seg, device)
    tr = build_track(states, seg, wav, np.zeros_like(wav), f0, voiced)
    return F.normalize(torch.from_numpy(tr.features.astype(np.float32)).to(device), dim=1)


def podcast_features(syl, device, n_seg: int, held: bool, work: Path) -> torch.Tensor:
    out = subprocess.run(f"aws s3 ls {SRC}/", shell=True, capture_output=True, text=True).stdout
    shards = sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))
    shards = [s for s in shards if held_out(s) == held]
    feats, total = [], 0
    for s in shards:
        d = work / s
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            subprocess.run(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors", shell=True, check=True)
        idx = np.load(d / "index.npy")
        segs = np.fromfile(d / "segs.bin", dtype=np.int16)
        for st, ln in idx[:200]:
            if ln < 5 * SR:
                continue
            f = seg_features(syl, segs[st : st + ln], device)
            feats.append(f)
            total += len(f)
            if total >= n_seg:
                return torch.cat(feats)
        subprocess.run(f"rm -rf {d}", shell=True)
        print("shard", s, "segments", total, flush=True)
    return torch.cat(feats)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit-seg", type=int, default=300000)
    p.add_argument("--ids", type=int, default=400)
    p.add_argument("--k", type=int, default=1024)
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    syl = load_sylber(device)
    cn = F.normalize(torch.load(args.centroids).to(device), dim=1)

    t0 = time.time()
    fit = podcast_features(syl, device, args.fit_seg, False, out.parent / "work")
    test = podcast_features(syl, device, 20000, True, out.parent / "work")
    print("podcast segments fit", len(fit), "held-out", len(test), "in", round(time.time() - t0), "s", flush=True)

    books = []
    r = fit - cn[(fit @ cn.T).argmax(1)]
    for lvl in range(2, max(LEVELS) + 1):
        c = kmeans(r, args.k, cosine=False, iters=25, seed=lvl)
        books.append(c)
        r = r - c[torch.cdist(r, c).argmin(1)]
        print("level", lvl, "residual norm", round(float(r.norm(dim=1).mean()), 4), flush=True)
    torch.save({"level1": "dy14 centroids", "books": [b.cpu() for b in books]}, out.parent / "rvq_books.pt")

    def encode(f: torch.Tensor) -> tuple[np.ndarray, list[torch.Tensor]]:
        """Codes [n, 8] and the rebuild after each level."""
        a = (f @ cn.T).argmax(1)
        codes, rec = [a], [cn[a]]
        r = f - cn[a]
        for c in books:
            j = torch.cdist(r, c).argmin(1)
            codes.append(j)
            rec.append(rec[-1] + c[j])
            r = r - c[j]
        return torch.stack(codes, 1).cpu().numpy(), rec

    res = {"recon_cos_heldout": {}}
    _, rec = encode(test)
    for L in LEVELS:
        res["recon_cos_heldout"][L] = round(float(F.cosine_similarity(rec[L - 1], test, dim=1).mean()), 4)
    print("recon cos", json.dumps(res["recon_cos_heldout"]), flush=True)

    d = Path(args.zr) / "lexical" / "dev"
    items = defaultdict(lambda: defaultdict(dict))
    for row in csv.DictReader(open(d / "gold.csv")):
        items[row["id"]][int(row["correct"])][row["voice"]] = d / f"{row['filename']}.wav"
    ids = [i for i, g in items.items() if len(g[0]) == 4 and len(g[1]) == 4]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    ids = ids[: args.ids]
    reps = {}
    for i in ids:
        for c in (0, 1):
            for v, path in items[i][c].items():
                w = to_int16(*sf.read(path))
                f = seg_features(syl, np.pad(w, (0, max(0, SR // 5 - len(w)))), device)
                codes, rec = encode(f) if len(f) else (np.zeros((0, 8), np.int64), [torch.zeros(0, 768, device=device)] * 8)
                reps[(i, c, v)] = {"cont": f.cpu().numpy(), "codes": codes,
                                   "rec": [F.normalize(x, dim=1).cpu().numpy() if len(x) else np.zeros((0, 768), np.float32) for x in rec]}
    voices = ["A", "C", "D", "F"]

    def abx(dist) -> float:
        right = n = 0.0
        for i in ids:
            for c in (0, 1):
                for v1 in voices:
                    for v2 in voices:
                        if v1 != v2:
                            a, b = dist((i, c, v1), (i, c, v2)), dist((i, c, v1), (i, 1 - c, v1))
                            right += 1.0 if a < b else 0.5 if a == b else 0.0
                            n += 1
        return round(right / n, 4)

    res["abx_sylber_cont"] = abx(lambda x, y: cos_dtw(reps[x]["cont"], reps[y]["cont"]))
    print("abx sylber_cont", res["abx_sylber_cont"], flush=True)
    res["abx_recon"], res["abx_codes"] = {}, {}
    for L in LEVELS:
        res["abx_recon"][L] = abx(lambda x, y: cos_dtw(reps[x]["rec"][L - 1], reps[y]["rec"][L - 1]))
        res["abx_codes"][L] = abx(lambda x, y: edit_codes(np.ascontiguousarray(reps[x]["codes"][:, :L]),
                                                          np.ascontiguousarray(reps[y]["codes"][:, :L])))
        print("L", L, "abx recon", res["abx_recon"][L], "abx codes", res["abx_codes"][L], flush=True)
    out.write_text(json.dumps(res, indent=2))
    print("SUMMARY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
