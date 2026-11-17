"""Are our units phonetic? ABX on the sWUGGY dev voices, no training.

Each sWUGGY id has a word and a one-phoneme-different nonword, each spoken by four synthetic voices. Triplet:
X = item in voice v1, A = the same item in voice v2, B = the other item in voice v1 (same voice, one phoneme
apart). Right when d(X, A) < d(X, B) (ties = 0.5): a good unit sequence ignores the voice and keeps the
phoneme. Representations:
  dy14_k8192   — our DY-14 units (deduplicated), normalised edit distance;
  dy14_k2048 / dy14_k512 — the same units merged by k-means over the DY-14 centroids;
  sylber_cont  — Sylber segment features before quantisation, DTW with cosine distance;
  hubert_cont  — HuBERT-base layer 6 frames at 50 Hz, DTW with cosine distance;
  hubert_k500  — HuBERT-base layer 6 k-means 500 (fitted here on other sWUGGY ids), deduplicated, edit distance.
Also: share of words whose DY-14 unit sequence is identical across two voices, and segments per item.

    python scripts/units_diag.py --zr /data/bench/zr --centroids /data/dy14/centroids.pt --out /data/diag/out.json
"""

import argparse
import csv
import json
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
from dy14_grid import nearest

from dauys.dy14_units import SR, build_track, hidden_states, load_sylber, segment_f0


@numba.njit(cache=True)
def dtw(c: np.ndarray) -> float:
    n, m = c.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = c[i - 1, j - 1] + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    return d[n, m] / (n + m)


@numba.njit(cache=True)
def edit(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 0.0
    d = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return d[n, m] / max(n, m)


def cos_dtw(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 1.0
    return float(dtw((1 - x @ y.T).astype(np.float64)))


def dedup(u: np.ndarray) -> np.ndarray:
    return u[np.r_[True, u[1:] != u[:-1]]] if len(u) else u


def kmeans(x: torch.Tensor, k: int, cosine: bool, iters: int = 30, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1) if cosine else torch.cdist(x, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).float()[:, None]
        c = torch.where(n > 0, s / n.clamp(min=1), c)
        if cosine:
            c = F.normalize(c, dim=1)
    return c


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", type=int, default=400)
    p.add_argument("--fit-ids", type=int, default=600)
    args = p.parse_args()
    device = torch.device("cuda")
    d = Path(args.zr) / "lexical" / "dev"
    items = defaultdict(lambda: defaultdict(dict))  # id -> correct -> voice -> path
    for r in csv.DictReader(open(d / "gold.csv")):
        items[r["id"]][int(r["correct"])][r["voice"]] = d / f"{r['filename']}.wav"
    ids = [i for i, g in items.items() if len(g[0]) == 4 and len(g[1]) == 4]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    test, fit = ids[: args.ids], ids[args.ids : args.ids + args.fit_ids]

    syl = load_sylber(device)
    cent = torch.load(args.centroids).to(device)
    cn = F.normalize(cent, dim=1)
    merged = {k: kmeans(cn, k, cosine=True) for k in (2048, 512)}
    merge_map = {k: (cn @ F.normalize(c, dim=1).T).argmax(1) for k, c in merged.items()}
    from transformers import HubertModel

    hub = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()

    @torch.no_grad()
    def hubert6(wav16: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav16.astype(np.float32) / 32768)[None].to(device)
        x = (x - x.mean()) / (x.std() + 1e-7)
        return hub(x, output_hidden_states=True).hidden_states[6][0]

    def load(path: Path) -> np.ndarray:
        w = to_int16(*sf.read(path))
        return np.pad(w, (0, max(0, SR // 5 - len(w))))

    t0 = time.time()
    feats = []
    for i in fit:
        for c in (0, 1):
            feats.append(hubert6(load(items[i][c]["A"])))
    feats = torch.cat(feats)
    hk = kmeans(feats, 500, cosine=False)
    print("hubert k500 fitted on", len(feats), "frames in", round(time.time() - t0), "s", flush=True)

    reps = {}
    for i in test:
        for c in (0, 1):
            for v, path in items[i][c].items():
                wav = load(path)
                states = hidden_states(syl, wav)
                seg = segments_of(states)
                r = {}
                if len(seg):
                    f0, voiced = segment_f0(wav, seg, device)
                    tr = build_track(states, seg, wav, np.zeros_like(wav), f0, voiced)
                    f = F.normalize(torch.from_numpy(tr.features.astype(np.float32)).to(device), dim=1)
                    u = nearest(f, cent)
                    r["sylber_cont"] = f.cpu().numpy()
                    r["dy14_k8192"] = dedup(u.cpu().numpy())
                    for k, mm in merge_map.items():
                        r[f"dy14_k{k}"] = dedup(mm[u].cpu().numpy())
                else:
                    r["sylber_cont"] = np.zeros((0, 768), np.float32)
                    for k in ("dy14_k8192", "dy14_k2048", "dy14_k512"):
                        r[k] = np.zeros(0, np.int64)
                h = hubert6(wav)
                r["hubert_cont"] = F.normalize(h, dim=1).cpu().numpy()
                r["hubert_k500"] = dedup(torch.cdist(h, hk).argmin(1).cpu().numpy())
                reps[(i, c, v)] = r
    print("encoded", len(reps), "clips in", round(time.time() - t0), "s", flush=True)

    names = ["dy14_k8192", "dy14_k2048", "dy14_k512", "sylber_cont", "hubert_cont", "hubert_k500"]

    def dist(name, a, b):
        x, y = reps[a][name], reps[b][name]
        return cos_dtw(x, y) if name.endswith("cont") else edit(np.asarray(x, np.int64), np.asarray(y, np.int64))

    res = {}
    voices = ["A", "C", "D", "F"]
    for name in names:
        right = n = 0.0
        for i in test:
            for c in (0, 1):
                for v1 in voices:
                    for v2 in voices:
                        if v1 == v2:
                            continue
                        dxa, dxb = dist(name, (i, c, v1), (i, c, v2)), dist(name, (i, c, v1), (i, 1 - c, v1))
                        right += 1.0 if dxa < dxb else 0.5 if dxa == dxb else 0.0
                        n += 1
        res[name] = {"abx_acc": round(right / n, 4), "triplets": int(n)}
        print(name, json.dumps(res[name]), flush=True)
    same = [np.array_equal(reps[(i, 1, a)]["dy14_k8192"], reps[(i, 1, b)]["dy14_k8192"])
            for i in test for a in voices for b in voices if a < b]
    res["dy14_same_word_identical_across_voices"] = round(float(np.mean(same)), 4)
    res["dy14_units_per_item"] = round(float(np.mean([len(r["dy14_k8192"]) for r in reps.values()])), 2)
    res["hubert_k500_units_per_item"] = round(float(np.mean([len(r["hubert_k500"]) for r in reps.values()])), 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("SUMMARY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
