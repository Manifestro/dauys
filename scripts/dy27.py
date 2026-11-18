"""DY-27: top-down loop, step 1 — merge WavLM k-means codes by what an LM predicts after them, vs merging by sound.

Tokens: WavLM-large hidden_states[12] (frozen) at 50 Hz -> mean of frame pairs (25 Hz) -> L2 -> nearest of K=500
spherical k-means centroids -> (optional merge map 500 -> K') -> repeats collapsed. Data: the DY-26 subset (same 10
shards of DuplexChat en-16k, 4 held out); pieces >= 5 s, as phon_units.py.

    python scripts/dy27.py kmeans   --shards DIR --out C.pt               # centroids from training shards only
    python scripts/dy27.py tokenize --shards DIR --centroids C.pt --out RAW # raw 25 Hz codes per shard
    python scripts/dy27.py apply    --raw RAW --out DIR [--map M.npy]      # DY-21 format (collapsed), optional merge
    python scripts/dy27.py lm_merge --units DIR --run best.pt --k 500 --to 96 --out M.npy
    python scripts/dy27.py ac_merge --centroids C.pt --to 96 --out M.npy
    python scripts/dy27.py known    --out known.json                       # known answer for the LM merge
    python scripts/dy27.py bench    --zr ZR --centroids C.pt [--map M.npy] --run best.pt --out B.json
    python scripts/dy27.py words    --cache words_pick.pkl --centroids C.pt --maps none,M1.npy,M2.npy --out W.json
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from alphabet import Feat, word_scores
from bench_slm import logprob, to_int16, zr_pairs
from dy21_train import EOS, held_out, load_stream
from units_diag import kmeans as sph_kmeans

from dauys.dy21_lm import LMConfig, UnitLM

SR = 16000
SPEC = "microsoft/wavlm-large:12"
FIT_FRAMES = 2_000_000  # k-means sample (25 Hz frames from training shards), seed 0


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def frames25(fe: Feat, w16: np.ndarray) -> torch.Tensor:
    h = torch.from_numpy(fe(w16))
    h = h[: len(h) // 2 * 2].view(-1, 2, h.shape[1]).mean(1)
    return F.normalize(h, dim=1)


def pieces(shard_dir: Path) -> list[np.ndarray]:
    idx = np.load(shard_dir / "index.npy")
    segs = np.fromfile(shard_dir / "segs.bin", dtype=np.int16)
    return [segs[st : st + ln] for st, ln in idx if ln >= 5 * SR]


def collapse(c: np.ndarray) -> np.ndarray:
    return c[np.r_[True, c[1:] != c[:-1]]] if len(c) else c


class Tok:
    def __init__(self, centroids: str, device, merge: str = "") -> None:
        self.fe = Feat(SPEC, device)
        self.c = torch.load(centroids).to(device)
        self.map = np.load(merge) if merge and merge != "none" else None
        self.device = device

    def raw(self, w16: np.ndarray) -> np.ndarray:
        w16 = np.pad(w16, (0, max(0, SR // 5 - len(w16))))
        return (frames25(self.fe, w16).to(self.device) @ self.c.T).argmax(1).cpu().numpy()

    def __call__(self, w16: np.ndarray) -> np.ndarray:
        r = self.raw(w16)
        return collapse(self.map[r] if self.map is not None else r).astype(np.int64)


def cmd_kmeans(a) -> None:
    device = torch.device("cuda")
    fe = Feat(SPEC, device)
    dirs = [d for d in sorted(Path(a.shards).iterdir()) if not held_out(d.name)]
    xs, n = [], 0
    for d in dirs:
        for w in pieces(d):
            x = frames25(fe, w).half()
            xs.append(x)
            n += len(x)
        print(json.dumps({"shard": d.name, "frames": n}), flush=True)
    x = torch.cat(xs)
    g = torch.Generator().manual_seed(0)
    x = x[torch.randperm(len(x), generator=g)[:FIT_FRAMES]].float().to(device)
    c = sph_kmeans(x, a.k, cosine=True, iters=40, seed=0)
    use = torch.bincount((x @ c.T).argmax(1), minlength=a.k)
    torch.save(c.cpu(), a.out)
    print("KMEANS DONE", json.dumps({"frames_total": n, "fit": len(x), "k": a.k, "empty": int((use == 0).sum()),
                                     "train_shards": [d.name for d in dirs]}), flush=True)


def cmd_tokenize(a) -> None:
    device = torch.device("cuda")
    t = Tok(a.centroids, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(Path(a.shards).iterdir()):
        t0 = time.time()
        rs = [t.raw(w).astype(np.int16) for w in pieces(d)]
        lens = np.array([len(r) for r in rs])
        np.savez(out / f"{d.name}.npz", raw=np.concatenate(rs), index=np.stack([np.r_[0, np.cumsum(lens)[:-1]], lens], 1))
        print(json.dumps({"shard": d.name, "held": held_out(d.name), "frames": int(lens.sum()), "s": round(time.time() - t0)}), flush=True)
    print("TOKENIZE DONE", flush=True)


def cmd_apply(a) -> None:
    m = np.load(a.map) if a.map else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tot_f = tot_t = 0
    for f in sorted(Path(a.raw).glob("*.npz")):
        z = np.load(f)
        raw, index = z["raw"], z["index"]
        toks, idx, pos = [], [], 0
        for st, n in index:
            r = raw[st : st + n]
            c = collapse(m[r] if m is not None else r).astype(np.int16)
            toks.append(c)
            idx.append((pos, len(c)))
            pos += len(c)
            tot_f += n
            tot_t += len(c)
        np.savez(out / f.name, tok=np.concatenate(toks), index=np.asarray(idx, np.int64).reshape(-1, 2))
    print("APPLY DONE", json.dumps({"map": a.map or "none", "tokens_per_s": round(tot_t / (tot_f / 25), 2)}), flush=True)


@torch.no_grad()
def next_dists(model, stream: np.ndarray, k: int, device, max_windows: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """v_c = mean LM-predicted next-token distribution (over the k codes + EOS) at positions holding code c."""
    T = model.cfg.ctx
    acc = torch.zeros(k, k + 1, device=device, dtype=torch.float64)
    cnt = torch.zeros(k, device=device, dtype=torch.float64)
    starts = np.arange(0, len(stream) - T, T)[:max_windows]
    cols = torch.tensor(list(range(k)) + [EOS], device=device)
    for i in range(0, len(starts), 32):
        ids = torch.from_numpy(np.stack([stream[s : s + T] for s in starts[i : i + 32]]).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model.logits(model.states(ids))
        p = torch.softmax(lg.float()[..., cols], -1).double()  # renormalised over the alphabet
        cur = ids.flatten()
        keep = cur < k
        acc.index_add_(0, cur[keep], p.flatten(0, 1)[keep])
        cnt.index_add_(0, cur[keep], torch.ones(int(keep.sum()), device=device, dtype=torch.float64))
    v = (acc / cnt.clamp(min=1)[:, None]).cpu().numpy()
    return v, cnt.cpu().numpy()


def js_matrix(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    v = v / v.sum(1, keepdims=True)
    lv = np.log(v)
    k = len(v)
    d = np.zeros((k, k))
    for i in range(k):
        m = 0.5 * (v[i][None] + v)
        lm = np.log(m)
        d[i] = 0.5 * (v[i][None] * (lv[i][None] - lm)).sum(1) + 0.5 * (v * (lv - lm)).sum(1)
    d = np.maximum(0.5 * (d + d.T), 0)
    np.fill_diagonal(d, 0)
    return d


def agglo(d: np.ndarray, to: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    lab = fcluster(linkage(squareform(d, checks=False), method="average"), to, criterion="maxclust") - 1
    return lab.astype(np.int16)


def cmd_lm_merge(a) -> None:
    device = torch.device("cuda")
    ck = torch.load(a.run, map_location="cpu", weights_only=False)
    model = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    files = [f for f in sorted(Path(a.units).glob("*.npz")) if not held_out(f.stem)]  # training shards only
    v, cnt = next_dists(model, load_stream(files), a.k, device, a.windows)
    lab = agglo(js_matrix(v), a.to)
    np.save(a.out, lab)
    print("LM MERGE DONE", json.dumps({"k": a.k, "to": int(lab.max()) + 1, "unseen_codes": int((cnt == 0).sum()),
                                       "group_sizes": sorted(np.bincount(lab).tolist(), reverse=True)[:10]}), flush=True)


def cmd_ac_merge(a) -> None:
    c = F.normalize(torch.load(a.centroids).float(), dim=1).numpy()
    d = np.clip(1 - c @ c.T, 0, None)
    np.fill_diagonal(d, 0)
    lab = agglo(0.5 * (d + d.T), a.to)
    np.save(a.out, lab)
    print("AC MERGE DONE", json.dumps({"to": int(lab.max()) + 1, "group_sizes": sorted(np.bincount(lab).tolist(), reverse=True)[:10]}), flush=True)


def cmd_known(a) -> None:
    """96 'phonemes' (Markov chain, no self loops) x random 'allophones' (500 codes in all, uniform choice): the
    allophones of one phoneme have the same next distribution, so the LM merge must recover the groups."""
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    P, K = 96, 500
    owner = np.r_[np.arange(P), rng.integers(0, P, K - P)]
    rng.shuffle(owner)
    allo = [np.flatnonzero(owner == p) for p in range(P)]
    tr = rng.dirichlet(np.full(P, 0.1), P)
    np.fill_diagonal(tr, 0)
    tr /= tr.sum(1, keepdims=True)
    n = 500_000
    cdf = np.cumsum(tr, 1)
    u = rng.random(n)
    ph = np.zeros(n, np.int64)
    for i in range(1, n):
        ph[i] = min(int(np.searchsorted(cdf[ph[i - 1]], u[i])), P - 1)
    stream = np.array([allo[p][rng.integers(len(allo[p]))] for p in ph], np.int16)
    torch.manual_seed(0)
    cfg = LMConfig(d=256, layers=4, heads=4, ffn=1024, ctx=256)
    model = UnitLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1500):
        st = rng.integers(0, n - 258, 64)
        ids = torch.from_numpy(np.stack([stream[s : s + 257] for s in st]).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model.logits(model.states(ids[:, :-1]))
        ce = F.cross_entropy(lg.float().flatten(0, 1), ids[:, 1:].flatten())
        opt.zero_grad()
        ce.backward()
        opt.step()
    model.eval()
    v, _ = next_dists(model, stream, K, device, 2000)
    lab = agglo(js_matrix(v), P)
    purity = sum(np.bincount(owner[lab == g]).max() for g in np.unique(lab)) / K
    res = {"ce": round(ce.item(), 4), "purity": round(float(purity), 4), "pass": purity >= 0.9}
    Path(a.out).write_text(json.dumps(res))
    print("KNOWN", json.dumps(res), flush=True)


def cmd_bench(a) -> None:
    device = torch.device("cuda")
    t = Tok(a.centroids, device, a.map)
    ck = torch.load(a.run, map_location="cpu", weights_only=False)
    model = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    rng = np.random.default_rng(0)
    results = {}
    for task in a.tasks.split(","):
        pairs = zr_pairs(Path(a.zr), {"swuggy": "lexical", "sblimp": "syntactic"}[task], a.pairs, rng)
        t0, right, n = time.time(), 0.0, 0
        for g, b in pairs:
            good, bad = t(to_int16(*sf.read(g))), t(to_int16(*sf.read(b)))
            if len(good) == 0 or len(bad) == 0:
                continue
            n += 1
            x, y = logprob(model, np.array([EOS]), good, device, False), logprob(model, np.array([EOS]), bad, device, False)
            right += 1.0 if x > y else 0.5 if x == y else 0.0
        acc = right / max(n, 1)
        results[task] = {"acc": round(acc, 4), "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / max(n, 1)), 4), "n": n,
                         "seconds": round(time.time() - t0)}
        print(task, json.dumps(results[task]), flush=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
    print("BENCH DONE", flush=True)


def cmd_words(a) -> None:
    import pickle

    device = torch.device("cuda")
    names, pick = pickle.loads(Path(a.cache).read_bytes())
    res = {"words": len(names)}
    base = Tok(a.centroids, device)
    for m in a.maps.split(","):
        mp = np.load(m) if m != "none" else None
        res["k500" if m == "none" else Path(m).stem] = (lambda mp: lambda w: collapse(mp[base.raw(w)] if mp is not None else base.raw(w)))(mp)
    word_scores(res, names, pick, a.out)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("kmeans"); x.add_argument("--shards", required=True); x.add_argument("--out", required=True); x.add_argument("--k", type=int, default=500)
    x = s.add_parser("tokenize"); x.add_argument("--shards", required=True); x.add_argument("--centroids", required=True); x.add_argument("--out", required=True)
    x = s.add_parser("apply"); x.add_argument("--raw", required=True); x.add_argument("--out", required=True); x.add_argument("--map", default="")
    x = s.add_parser("lm_merge"); x.add_argument("--units", required=True); x.add_argument("--run", required=True); x.add_argument("--k", type=int, default=500)
    x.add_argument("--to", type=int, default=96); x.add_argument("--out", required=True); x.add_argument("--windows", type=int, default=4000)
    x = s.add_parser("ac_merge"); x.add_argument("--centroids", required=True); x.add_argument("--to", type=int, default=96); x.add_argument("--out", required=True)
    x = s.add_parser("known"); x.add_argument("--out", required=True)
    x = s.add_parser("bench"); x.add_argument("--zr", required=True); x.add_argument("--centroids", required=True); x.add_argument("--map", default="")
    x.add_argument("--run", required=True); x.add_argument("--out", required=True); x.add_argument("--pairs", type=int, default=3000); x.add_argument("--tasks", default="swuggy,sblimp")
    x = s.add_parser("words"); x.add_argument("--cache", required=True); x.add_argument("--centroids", required=True); x.add_argument("--maps", required=True); x.add_argument("--out", required=True)
    a = p.parse_args()
    {"kmeans": cmd_kmeans, "tokenize": cmd_tokenize, "apply": cmd_apply, "lm_merge": cmd_lm_merge, "ac_merge": cmd_ac_merge,
     "known": cmd_known, "bench": cmd_bench, "words": cmd_words}[a.cmd](a)


if __name__ == "__main__":
    main()
