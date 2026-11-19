"""DY-36 thought dictionary: every real utterance (2-20 s speaker segment of dy35_utt.py) -> a 64-d unit vector.

table:  utterance table over all <shard>.npz of dy35/utt (training + held-out), in (file, episode, t0) order:
        file, piece, s, e (frames in piece), piece_orig, ep (global episode = recording), spk (global speaker of the
        recording = ep * 64 + cluster), t0, t1 (s in the episode), len (frames), held; codes (alone-encoded K2000) as
        one int16 array with offsets.
A:      "which sounds were in the utterance": counts of the utterance's codes (consecutive repeats merged) and of code
        bigrams (bigrams with document frequency >= MIN_DF in training), tf-idf weighted (smooth idf from training,
        raw counts), rows L2-normalised; minus the mean row of that speaker in that recording; truncated SVD to 64 dims
        fitted on the speaker-centred rows of a training subsample (whole recordings, 1/3, seed 0; randomized SVD with
        3 power iterations); L2-normalised. Because centring and projection are linear, raw = X V is stored and
        thought = normalise(raw - mean of raw over the speaker's utterances in the recording).
C:      DY-35 gist encoder z of the utterance codes (snapshot); raw = z, thought = normalise(z - speaker mean);
        "C-raw" = z itself (no centring).
Output: <out>/table.npz, <out>/dict_<name>.npz (raw float32 [n, 64], thought float32 [n, 64], valid bool: speaker has
>= 2 utterances in the recording and the centred vector is not ~0).

    python scripts/dy36_dict.py table --utt /data/utt --out /data/dy36/dict
    python scripts/dy36_dict.py A --out /data/dy36/dict
    python scripts/dy36_dict.py C --out /data/dy36/dict --ckpt snap.pt --key enc --name C2400
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out

K = 2000
UMIN, UMAX = 100, 1000
MIN_DF = 20
DIM = 64


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cmd_table(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.utt).glob("*.npz"))
    cols = {k: [] for k in ("file", "piece", "s", "e", "piece_orig", "ep", "spk", "t0", "t1", "held")}
    codes, offs, pos, ep_base = [], [], 0, 0
    for fi, f in enumerate(files):
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        L = z["seg_e"] - z["seg_s"]
        m = np.flatnonzero((L >= UMIN) & (L <= UMAX))
        p = z["seg_piece"][m]
        s, e = z["seg_s"][m].astype(np.int64), z["seg_e"][m].astype(np.int64)
        ep = ep_base + z["piece_ep"][p].astype(np.int64)
        t0 = z["piece_t0"][p] + s / 50
        o = np.lexsort((t0, ep))
        m, p, s, e, ep, t0 = m[o], p[o], s[o], e[o], ep[o], t0[o]
        for pi, si, ei in zip(p, s, e):
            a0 = int(index[pi][0])
            codes.append(tok[a0 + si : a0 + ei].astype(np.int16))
            offs.append(pos)
            pos += ei - si
        cols["file"].append(np.full(len(m), fi))
        cols["piece"].append(p)
        cols["s"].append(s)
        cols["e"].append(e)
        cols["piece_orig"].append(z["piece_orig"][p])
        cols["ep"].append(ep)
        cols["spk"].append(ep * 64 + z["seg_spk"][m].astype(np.int64))
        cols["t0"].append(t0)
        cols["t1"].append(z["piece_t0"][p] + e / 50)
        cols["held"].append(np.full(len(m), held_out(f.stem)))
        ep_base += len(json.loads(str(z["eps"])))
        log(f.stem, len(m))
    T = {k: np.concatenate(v) for k, v in cols.items()}
    T["len"] = T["e"] - T["s"]
    T["code_off"] = np.asarray(offs + [pos], np.int64)
    T["codes"] = np.concatenate(codes)
    T["files"] = np.array(json.dumps([f.stem for f in files]))
    # time order inside every episode; episodes contiguous
    assert np.all(np.diff(T["ep"]) >= 0)
    np.savez(out / "table.npz", **T)
    log("TABLE", len(T["ep"]), "train", int((~T["held"]).sum()), "held", int(T["held"].sum()),
        "episodes", len(np.unique(T["ep"])), "speakers", len(np.unique(T["spk"])))


def load_table(d: Path) -> dict:
    z = np.load(Path(d) / "table.npz")
    return {k: z[k] for k in z.files}


def group_mean(x: np.ndarray, grp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise mean of x over rows of the same group; also the group sizes per row."""
    _, inv, cnt = np.unique(grp, return_inverse=True, return_counts=True)
    s = np.zeros((len(cnt), x.shape[1]), np.float64)
    np.add.at(s, inv, x)
    return (s / cnt[:, None])[inv], cnt[inv]


def centre(raw: np.ndarray, spk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, n = group_mean(raw.astype(np.float64), spk)
    r = raw - mu
    nr = np.linalg.norm(r, axis=1)
    valid = (n >= 2) & (nr > 1e-6)
    th = np.where(valid[:, None], r / np.maximum(nr, 1e-12)[:, None], 0.0)
    return th.astype(np.float32), valid


def features(T: dict):
    """Per utterance: unique (utt, key) with counts; key < K unigram, else K + a*K + b bigram."""
    co, off = T["codes"].astype(np.int64), T["code_off"]
    n = len(off) - 1
    uid = np.repeat(np.arange(n), np.diff(off))
    keep = np.ones(len(co), bool)
    keep[1:] = (co[1:] != co[:-1]) | (uid[1:] != uid[:-1])
    c, u = co[keep], uid[keep]
    same = u[1:] == u[:-1]
    bk = K + c[:-1][same] * K + c[1:][same]
    keys = np.concatenate([u * (K + K * K) + c, u[:-1][same] * (K + K * K) + bk])
    uk, cnt = np.unique(keys, return_counts=True)
    return uk // (K + K * K), uk % (K + K * K), cnt


def cmd_A(a) -> None:
    import scipy.sparse as sp

    out = Path(a.out)
    T = load_table(out)
    n = len(T["ep"])
    train = ~T["held"]
    log("features ...")
    rows, keys, cnt = features(T)
    log("nnz", len(rows))
    df = np.bincount(keys[train[rows]], minlength=K + K * K)
    vocab = np.flatnonzero((np.arange(K + K * K) < K) | (df >= MIN_DF))
    col = np.full(K + K * K, -1, np.int64)
    col[vocab] = np.arange(len(vocab))
    m = col[keys] >= 0
    rows, cols, cnt = rows[m], col[keys[m]], cnt[m].astype(np.float32)
    del keys
    N = int(train.sum())
    idf = (np.log((1 + N) / (1 + df[vocab])) + 1).astype(np.float32)
    X = sp.csr_matrix((cnt * idf[cols], (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
    del rows, cols, cnt
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    X = sp.diags(1 / np.maximum(nr, 1e-12)).astype(np.float32) @ X
    log("X", X.shape, "nnz", X.nnz, "bigrams kept", len(vocab) - K)
    # randomized SVD of the speaker-centred training subsample
    rng = np.random.default_rng(0)
    eps = np.unique(T["ep"][train])
    sub_eps = rng.choice(eps, len(eps) // 3, replace=False)
    sub = np.flatnonzero(train & np.isin(T["ep"], sub_eps))
    Xs, gs = X[sub], T["spk"][sub]
    _, inv, gcnt = np.unique(gs, return_inverse=True, return_counts=True)
    G = sp.csr_matrix((1 / gcnt[inv], (inv, np.arange(len(sub)))), shape=(len(gcnt), len(sub)))  # group means
    E = sp.csr_matrix((np.ones(len(sub)), (np.arange(len(sub)), inv)), shape=(len(sub), len(gcnt)))  # expand
    R = lambda Y: Xs @ Y - E @ (G @ (Xs @ Y))
    Rt = lambda Q: Xs.T @ (Q - E @ (G @ Q))  # (I - E G) is a symmetric projection
    k = DIM + 16
    Y = R(rng.standard_normal((X.shape[1], k)).astype(np.float32))
    for it in range(3):
        Q, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(Rt(Q))
        Y = R(Z)
        log("power", it)
    Q, _ = np.linalg.qr(Y)
    B = np.asarray(Rt(Q)).T  # [k, feat]
    _, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt[:DIM].T.astype(np.float32)
    raw = np.asarray(X @ V, np.float32)
    th, valid = centre(raw, T["spk"])
    np.savez(out / "dict_A.npz", raw=raw, thought=th, valid=valid, sv=S[:DIM], V=V, idf=idf, vocab=vocab,
             info=np.array(json.dumps({"rule": __doc__.split("C:")[0], "min_df": MIN_DF, "svd_rows": len(sub),
                                       "features": int(X.shape[1])})))
    log("A DONE", "valid", float(valid.mean()), "sv", np.round(S[:8], 3).tolist(), "explained top64 share",
        float((S[:DIM] ** 2).sum() / (S ** 2).sum()))


def cmd_C(a) -> None:
    import torch

    from dauys.dy35_thought import GistEncoder

    out = Path(a.out)
    T = load_table(out)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck[a.key] if a.key else ck
    enc = GistEncoder().cuda().eval()
    enc.load_state_dict(sd)
    co, off = T["codes"], T["code_off"]
    n = len(off) - 1
    raw = np.zeros((n, 64), np.float32)
    step = 20000
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, n, step):
            us = [torch.from_numpy(co[off[i] : off[i + 1]].astype(np.int64)).cuda() for i in range(b, min(n, b + step))]
            raw[b : b + len(us)] = enc.encode_many(us).float().cpu().numpy()
    th, valid = centre(raw, T["spk"])
    np.savez(out / f"dict_{a.name}.npz", raw=raw, thought=th, valid=valid,
             info=np.array(json.dumps({"ckpt": a.ckpt, "step": ck.get("step") if isinstance(ck, dict) else None})))
    # C-raw: z itself (no centring)
    np.savez(out / f"dict_{a.name}raw.npz", raw=raw, thought=raw, valid=valid)
    mu, _ = group_mean(raw.astype(np.float64), T["spk"])
    log(a.name, "DONE valid", float(valid.mean()), "mean cos to own speaker mean", float((raw * mu).sum(1).mean()))


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("table")
    x.add_argument("--utt", required=True)
    x.add_argument("--out", required=True)
    x = s.add_parser("A")
    x.add_argument("--out", required=True)
    x = s.add_parser("C")
    x.add_argument("--out", required=True)
    x.add_argument("--ckpt", required=True)
    x.add_argument("--key", default="enc")
    x.add_argument("--name", default="C")
    a = p.parse_args()
    {"table": cmd_table, "A": cmd_A, "C": cmd_C}[a.cmd](a)


if __name__ == "__main__":
    main()
