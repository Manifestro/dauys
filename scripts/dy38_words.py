"""DY-38: a dictionary of words from audio, without text: one word said by any voice -> one token.

Steps (all on one GPU machine; text only measures dev/test, it never reaches segmentation, encoder or dictionary):
  select   utterances of the DY-35/36 table: ~200 h of training recordings (seed 0), dev ~2 h and test ~5 h from
           held-out recordings (split by episode hash)
  feats    WavLM-Large layer 18 frames (50 Hz) -> PCA 256 (fitted on the first ~3 h of training utterances),
           fp16 memmap; Sylber syllable segments; test/dev audio kept for listening
  ref      dev/test only: Whisper large-v3-turbo transcript -> torchaudio MMS_FA forced alignment -> word spans
  seg      W: peaks of neighbouring-window dissimilarity (3 frames each side) on L18-PCA, min distance 8 frames,
           threshold set so the training rate is 3.0 pieces per second; U: equal pieces at the same rate; S: Sylber
  g1       word-boundary F1 on test (tolerance 40 ms, also 20 ms); chooses the arm by the pre-registered rule
  base     training-free vectors of the chosen arm: mean L18-PCA of the piece, minus the speaker mean (per split),
           without the top 10 principal directions of the training vectors
  check    pre-run checklist: leakage, known-answer loss, overfit one batch
  train    self-bootstrapping: vectors -> k-means K=20000 -> for each piece the closest of 32 random members of its
           cluster from ANOTHER recording -> top half of pairs -> InfoNCE with the next piece of the same utterance as
           a hard negative; 3 rounds x 4000 steps; best round by dev AP_cross
  dict     k-means K=20000 over final training vectors (and over training-free vectors, the reference dictionary)
  eval     G2 (AP same-different across recordings / within speaker), G3 (NMI, purity), oracle-span diagnostics,
           measure checks, listening clips

    python scripts/dy38_words.py select --table /data/dy36/dict --out /data/dy38
    python scripts/dy38_words.py feats --split dev --out /data/dy38      (then test, train)
    ...
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SR, HOP = 16000, 320
LAYER = 18
PCA_DIM = 256
RATE = 3.0  # pieces per second (DY-32: live speech 29-31 words per 10 s)
WIN = 3
MIN_DIST = 8
MAXF = 60  # frames of a piece fed to the encoder (plus 1 context frame each side)
K = 20000
TOL40, TOL20 = 2, 1  # frames
EDGE = 2  # boundaries this close to an utterance edge are not scored
MIN_LETTERS, MIN_WORD_FR = 3, 10
PER_WORD = 20
EXPERIMENT = "Словарь слов из звука: одно слово любым голосом — один токен (200 ч, без текста)"


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ep_hash(ep: int) -> int:
    return int(hashlib.sha1(str(int(ep)).encode()).hexdigest(), 16)


# ----------------------------------------------------------------------------------------------------------- select


def cmd_select(a) -> None:
    from dy36_dict import load_table

    T = load_table(Path(a.table))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = T["held"].astype(bool)
    sec = T["len"] / 50.0
    rng = np.random.default_rng(0)
    tr_eps = rng.permutation(np.unique(T["ep"][~held]))
    ep_sec = {}
    for e, s in zip(T["ep"][~held], sec[~held]):
        ep_sec[int(e)] = ep_sec.get(int(e), 0.0) + float(s)
    pick, tot = [], 0.0
    for e in tr_eps:
        if tot >= a.hours * 3600:
            break
        pick.append(int(e))
        tot += ep_sec[int(e)]
    train = np.flatnonzero(~held & np.isin(T["ep"], pick))
    h_eps = np.unique(T["ep"][held])
    dev_eps = [int(e) for e in h_eps if ep_hash(e) % 7 == 0]
    test_eps = [int(e) for e in h_eps if ep_hash(e) % 7 != 0]
    rng2 = np.random.default_rng(1)

    def take(eps, hours):
        eps = list(rng2.permutation(eps))
        rows, t = [], 0.0
        for e in eps:
            if t >= hours * 3600:
                break
            r = np.flatnonzero(held & (T["ep"] == e))
            r = r[np.cumsum(sec[r]) <= a.per_rec_min * 60]  # at most per_rec_min minutes per recording: more voices
            rows.append(r)
            t += float(sec[r].sum())
        return np.sort(np.concatenate(rows))

    dev, test = take(dev_eps, a.dev_hours), take(test_eps, a.test_hours)
    assert not set(T["ep"][train]) & set(T["ep"][test]) and not set(T["ep"][dev]) & set(T["ep"][test])
    assert not set(T["ep"][train]) & set(T["ep"][dev])
    np.savez(out / "split.npz", train=train, dev=dev, test=test)
    for k, r in (("train", train), ("dev", dev), ("test", test)):
        log(k, "utterances", len(r), "hours", round(float(sec[r].sum()) / 3600, 2), "recordings", len(np.unique(T["ep"][r])))


# ------------------------------------------------------------------------------------------------------------ feats


def n_frames(n: int) -> int:
    return max(0, (n + 80 - 400) // 320 + 1)


def cmd_feats(a) -> None:
    import torch

    import dy36_common
    from dauys.dy14_units import get_segment, hidden_states, load_sylber
    from dy36_common import utt_job
    from dy36_dict import load_table

    T = load_table(Path(a.table))
    out = Path(a.out)
    sp = np.load(out / "split.npz")[a.split]
    if a.limit_hours:
        c = np.cumsum(T["len"][sp] / 50.0)
        sp = sp[: int(np.searchsorted(c, a.limit_hours * 3600)) + 1]
    d = out / a.split
    d.mkdir(parents=True, exist_ok=True)
    dy36_common.IDX_DIR = out / "idx"
    au = dy36_common.Audio(json.loads(str(T["files"])))
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cuda",
                         verbose=False)
    wavlm = knn.wavlm.eval()
    syl = load_sylber("cuda")
    pca_f = out / "pca.npz"
    lens = []
    # audio lengths need the audio; fetch in blocks, keep order of sp
    block = 4000
    frames_list, sylseg, nfr_all, keep_audio = [], [], [], []
    P = None
    if pca_f.exists():
        z = np.load(pca_f)
        P = (torch.from_numpy(z["mu"]).cuda(), torch.from_numpy(z["W"]).cuda())
    buf, buf_rows = [], 0
    mm = None
    total_written = 0
    off = [0]
    t0 = time.time()
    tmp_raw = []  # raw 1024-d frames before the PCA exists (train only)

    def project(x):
        mu, W = P
        return ((x.float() - mu) @ W).half()

    for b0 in range(0, len(sp), block):
        rows = sp[b0 : b0 + block]
        wavs = au.many([utt_job(T, int(u)) for u in rows])
        if a.split != "train":
            keep_audio.extend(wavs)
        nfr = [n_frames(len(w)) for w in wavs]
        nfr_all.extend(nfr)
        order = np.argsort(-np.array([len(w) for w in wavs]))
        feats = [None] * len(wavs)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            i = 0
            while i < len(order):
                L0 = len(wavs[order[i]])
                nb = max(1, int(a.batch_sec * SR // max(L0, 1)))
                idx = order[i : i + nb]
                i += nb
                x = torch.zeros(len(idx), L0 + 80)
                pad = torch.ones(len(idx), L0 + 80, dtype=torch.bool)
                for j, k in enumerate(idx):
                    x[j, : len(wavs[k])] = torch.from_numpy(wavs[k].astype(np.float32) / 32768.0)
                    pad[j, : len(wavs[k]) + 80] = False
                x, pad = x.cuda(), pad.cuda()
                (_, lr), _ = wavlm.extract_features(x, padding_mask=pad if pad.any() else None, output_layer=LAYER,
                                                    ret_layer_results=True)
                h = lr[LAYER][0].transpose(0, 1)
                for j, k in enumerate(idx):
                    feats[k] = h[j, : nfr[k]].half()
        # Sylber syllables of every utterance (its own 50 Hz frames)
        for k, w in enumerate(wavs):
            st = hidden_states(syl, w)
            segs = get_segment(st) if len(st) else np.zeros((0, 2), int)
            for s, e in segs:
                if e - s >= 1:
                    sylseg.append((b0 + k, int(s), int(min(e, nfr[k]))))
        if P is None:
            assert a.split == "train", "PCA must be fitted on training data first"
            tmp_raw.extend(f.cpu() for f in feats)
            if sum(len(f) for f in tmp_raw) >= a.pca_hours * 3600 * 50 or b0 + block >= len(sp):
                X = torch.cat(tmp_raw).cuda().float()
                mu = X.mean(0)
                _, _, V = torch.linalg.svd(X[torch.randperm(len(X), device=X.device)[:400000]] - mu, full_matrices=False)
                W = V[:PCA_DIM].T.contiguous()
                np.savez(pca_f, mu=mu.cpu().numpy(), W=W.cpu().numpy())
                P = (mu, W)
                log("PCA fitted on", len(X), "frames; variance kept",
                    round(float(((X - mu) @ W).var(0).sum() / (X - mu).var(0).sum()), 4))
                feats_out = [project(f.cuda()) for f in tmp_raw]
                tmp_raw = []
                del X
            else:
                log("buffering for PCA", b0 + len(rows), "of", len(sp))
                continue
        else:
            feats_out = [project(f) for f in feats]
        tot_new = sum(len(f) for f in feats_out)
        if mm is None:
            est = int(sum(T["len"][sp]) * 1.02) + 10000
            mm = np.lib.format.open_memmap(d / "frames.npy", "w+", np.float16, (est, PCA_DIM))
        cat = torch.cat(feats_out).cpu().numpy()
        mm[total_written : total_written + tot_new] = cat
        total_written += tot_new
        for f in feats_out:
            off.append(off[-1] + len(f))
        log(a.split, "done", b0 + len(rows), "of", len(sp), "frames", total_written, round(time.time() - t0), "s")
    mm.flush()
    assert off[-1] == total_written == sum(nfr_all), (off[-1], total_written, sum(nfr_all))
    np.save(d / "frames_n.npy", np.array([total_written]))
    np.save(d / "off.npy", np.array(off, np.int64))
    np.save(d / "rows.npy", sp)
    s = np.array(sylseg, np.int64).reshape(-1, 3)
    np.save(d / "seg_S.npy", s)
    if keep_audio:
        aoff = np.concatenate([[0], np.cumsum([len(w) for w in keep_audio])]).astype(np.int64)
        np.save(d / "audio_off.npy", aoff)
        np.save(d / "audio.npy", np.concatenate(keep_audio).astype(np.int16))
    log("FEATS DONE", a.split, "utterances", len(sp), "frames", total_written, "syllables", len(s),
        "syl per s", round(len(s) / (total_written / 50), 3))


def load_split(out: Path, split: str):
    d = out / split
    n = int(np.load(d / "frames_n.npy")[0])
    fr = np.load(d / "frames.npy", mmap_mode="r")[:n]
    return fr, np.load(d / "off.npy"), np.load(d / "rows.npy")


# -------------------------------------------------------------------------------------------------------------- ref


def norm_word(w: str) -> str:
    return "".join(c for c in w.lower() if "a" <= c <= "z")


def cmd_ref(a) -> None:
    import torch
    import torchaudio
    from transformers import pipeline

    out = Path(a.out)
    d = out / a.split
    audio, aoff = np.load(d / "audio.npy", mmap_mode="r"), np.load(d / "audio_off.npy")
    n = len(aoff) - 1
    wavs = [np.asarray(audio[aoff[i] : aoff[i + 1]], np.float32) / 32768.0 for i in range(n)]
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0,
                   torch_dtype=torch.float16, chunk_length_s=30)
    texts = [r["text"].strip() for r in asr([{"raw": w, "sampling_rate": SR} for w in wavs], batch_size=16,
                                           generate_kwargs={"language": "en", "task": "transcribe"})]
    del asr
    torch.cuda.empty_cache()
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).cuda().eval()
    tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    ratio = bundle.sample_rate / SR
    assert ratio == 1.0
    res, skipped = [], 0
    for i, (w, t) in enumerate(zip(wavs, texts)):
        if any(c.isdigit() for c in t) or len(w) < SR // 2:
            res.append(None)
            skipped += 1
            continue
        words = [norm_word(x) for x in t.split()]
        words = [x for x in words if x]
        if not words:
            res.append(None)
            skipped += 1
            continue
        with torch.inference_mode():
            em, _ = model(torch.from_numpy(w)[None].cuda())
            spans = aligner(em[0], tok(words))
        fps = em.shape[1] / (len(w) / SR)  # emission frames per second (50)
        lst = []
        for word, sp in zip(words, spans):
            s0 = sp[0].start / fps * 50
            e0 = sp[-1].end / fps * 50
            lst.append([word, int(round(s0)), int(round(e0))])
        res.append(lst)
    (d / "ref.json").write_text(json.dumps({"text": texts, "words": res}))
    nw = sum(len(r) for r in res if r)
    log("REF DONE", a.split, "utterances", n, "skipped", skipped, "words", nw)


# -------------------------------------------------------------------------------------------------------------- seg


def dissim(x: np.ndarray) -> np.ndarray:
    """1 - cos(mean of WIN frames before t, mean of WIN frames from t), t = 1..len-1 (index t)."""
    n = len(x)
    d = np.zeros(n, np.float32)
    if n < 2:
        return d
    c = np.concatenate([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, 0, dtype=np.float64)])
    t = np.arange(1, n)
    lo, hi = np.maximum(0, t - WIN), np.minimum(n, t + WIN)
    left = (c[t] - c[lo]) / (t - lo)[:, None]
    right = (c[hi] - c[t]) / (hi - t)[:, None]
    cs = (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-8)
    d[1:] = 1 - cs
    return d


def peaks(d: np.ndarray, thr: float) -> np.ndarray:
    from scipy.signal import find_peaks

    p, _ = find_peaks(d, height=thr, distance=MIN_DIST)
    return p


def to_segs(u: int, bounds: np.ndarray, n: int) -> list:
    b = [0] + [int(x) for x in bounds if 0 < x < n] + [n]
    return [(u, b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def cmd_seg(a) -> None:
    out = Path(a.out)
    # threshold on training utterances: rate RATE pieces per second
    fr, off, _ = load_split(out, "train")
    rng = np.random.default_rng(0)
    sample = rng.permutation(len(off) - 1)[:3000]
    D = [dissim(np.asarray(fr[off[u] : off[u + 1]], np.float32)) for u in sample]
    sec = sum(len(x) for x in D) / 50

    def rate(th):
        return sum(len(peaks(x, th)) + 1 for x in D) / sec

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if rate(mid) > RATE:
            lo = mid
        else:
            hi = mid
    thr = (lo + hi) / 2
    log("W threshold", round(thr, 5), "rate", round(rate(thr), 3))
    (out / "seg_params.json").write_text(json.dumps({"W_threshold": thr, "rate_target": RATE}))
    ulen = 50 / RATE
    for split in ("train", "dev", "test"):
        fr, off, _ = load_split(out, split)
        W, U = [], []
        for u in range(len(off) - 1):
            n = int(off[u + 1] - off[u])
            x = np.asarray(fr[off[u] : off[u + 1]], np.float32)
            W += to_segs(u, peaks(dissim(x), thr), n)
            U += to_segs(u, np.round(np.arange(1, int(n / ulen) + 1) * ulen).astype(int), n)
        np.save(out / split / "seg_W.npy", np.array(W, np.int64).reshape(-1, 3))
        np.save(out / split / "seg_U.npy", np.array(U, np.int64).reshape(-1, 3))
        nsec = off[-1] / 50
        log("SEG", split, "W per s", round(len(W) / nsec, 3), "U per s", round(len(U) / nsec, 3))


# --------------------------------------------------------------------------------------------------------------- g1


def match_count(ref: list, hyp: list, tol: int) -> int:
    """Greedy one-to-one matching of sorted boundary lists within tol frames."""
    i = j = m = 0
    while i < len(ref) and j < len(hyp):
        if abs(ref[i] - hyp[j]) <= tol:
            m += 1
            i += 1
            j += 1
        elif ref[i] < hyp[j]:
            i += 1
        else:
            j += 1
    return m


def ref_bounds(words: list, n: int) -> list:
    b = sorted({w[1] for w in words} | {w[2] for w in words})
    out = []
    for x in b:
        if EDGE <= x <= n - EDGE and (not out or x - out[-1] > 1):
            out.append(x)
    return out


def seg_bounds(segs: np.ndarray, n: int) -> list:
    b = sorted({int(s) for s in segs[:, 1]} | {int(e) for e in segs[:, 2]})
    return [x for x in b if EDGE <= x <= n - EDGE]


def boundary_scores(out: Path, split: str, arm: str) -> dict:
    ref = json.loads((out / split / "ref.json").read_text())["words"]
    off = np.load(out / split / "off.npy")
    S = np.load(out / split / f"seg_{arm}.npy")
    by = {}
    for r in S:
        by.setdefault(int(r[0]), []).append(r)
    res = {}
    for name, tol in (("40ms", TOL40), ("20ms", TOL20)):
        nr = nh = m = 0
        for u, words in enumerate(ref):
            if not words:
                continue
            n = int(off[u + 1] - off[u])
            rb = ref_bounds(words, n)
            hb = seg_bounds(np.array(by.get(u, np.zeros((0, 3), int))).reshape(-1, 3), n)
            nr += len(rb)
            nh += len(hb)
            m += match_count(rb, hb, tol)
        p, r = m / max(nh, 1), m / max(nr, 1)
        res[name] = {"precision": round(p, 4), "recall": round(r, 4), "F1": round(2 * p * r / max(p + r, 1e-9), 4),
                     "ref": nr, "hyp": nh}
    return res


def cmd_g1(a) -> None:
    out = Path(a.out)
    R = {arm: boundary_scores(out, "test", arm) for arm in ("W", "U", "S")}
    fw, fu = R["W"]["40ms"]["F1"], R["U"]["40ms"]["F1"]
    if fw >= 0.35 and fw >= fu + 0.10:
        arm, passed = "W", True
    elif R["S"]["40ms"]["recall"] >= 0.70:
        arm, passed = "S", True
    else:
        arm, passed = None, False
    R["decision"] = {"arm": arm, "G1_pass": passed, "rule": "W if F1_40(W)>=0.35 and >=F1_40(U)+0.10; else S if "
                     "recall_40(S)>=0.70; else stop"}
    (out / "g1.json").write_text(json.dumps(R, indent=1))
    log("G1", json.dumps(R))


# ------------------------------------------------------------------------------------------------------------- base


def seg_info(out: Path, split: str, arm: str, T: dict):
    """segments [n,3] (utt, s, e), their global frame start/end, recording and speaker ids."""
    off, rows = np.load(out / split / "off.npy"), np.load(out / split / "rows.npy")
    S = np.load(out / split / f"seg_{arm}.npy")
    g0 = off[S[:, 0]] + S[:, 1]
    g1 = off[S[:, 0]] + S[:, 2]
    return S, g0, g1, T["ep"][rows[S[:, 0]]], T["spk"][rows[S[:, 0]]]


def frames_tensor(fr: np.ndarray):
    """Frames on the GPU when they fit with 20 GB to spare, otherwise in RAM (gathered there, then moved)."""
    import torch

    need = fr.shape[0] * fr.shape[1] * 2
    free, _ = torch.cuda.mem_get_info()
    X = torch.from_numpy(np.asarray(fr, np.float16))
    if need + 20 * 2**30 < free:
        return X.cuda()
    log("frames stay in RAM:", round(need / 2**30, 1), "GB; GPU free", round(free / 2**30, 1), "GB")
    return X


def gather(X, idx):
    """X[idx] as float32 on the GPU, whichever device X is on."""
    if X.is_cuda:
        return X[idx].float()
    return X[idx.cpu()].cuda(non_blocking=True).float()


def mean_pool(fr: np.ndarray, g0: np.ndarray, g1: np.ndarray, chunk: int = 200000) -> np.ndarray:
    """Mean frame of every piece [g0, g1); pieces processed in chunks with index_add (no long cumulative sums)."""
    import torch

    X = frames_tensor(fr)
    out = np.zeros((len(g0), X.shape[1]), np.float32)
    for i in range(0, len(g0), chunk):
        a_ = torch.from_numpy(g0[i : i + chunk]).cuda()
        b_ = torch.from_numpy(g1[i : i + chunk]).cuda()
        L = (b_ - a_).clamp(min=1)
        seg = torch.repeat_interleave(torch.arange(len(a_), device="cuda"), L)
        first = torch.repeat_interleave(torch.cumsum(L, 0) - L, L)
        fidx = torch.repeat_interleave(a_, L) + torch.arange(len(seg), device="cuda") - first
        fidx = fidx.clamp(max=len(X) - 1)
        S = torch.zeros(len(a_), X.shape[1], device="cuda").index_add_(0, seg, gather(X, fidx))
        out[i : i + chunk] = (S / L[:, None].float()).cpu().numpy()
    del X
    torch.cuda.empty_cache()
    return out


def base_vectors(V: np.ndarray, spk: np.ndarray, pcs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    from dy36_dict import group_mean

    m, _ = group_mean(V, spk)
    X = V - m
    X = X - mu
    X = X - (X @ pcs.T) @ pcs
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def cmd_base(a) -> None:
    from dy36_dict import group_mean, load_table

    out = Path(a.out)
    T = load_table(Path(a.table))
    arm = json.loads((out / "g1.json").read_text())["decision"]["arm"]
    assert arm, "G1 failed: no arm"
    fr, _, _ = load_split(out, "train")
    S, g0, g1, ep, spk = seg_info(out, "train", arm, T)
    V = mean_pool(fr, g0, g1)
    m, _ = group_mean(V, spk)
    X = V - m
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd((X - mu)[np.random.default_rng(0).permutation(len(X))[:300000]], full_matrices=False)
    pcs = Vt[:10]
    np.savez(out / "base_params.npz", mu=mu, pcs=pcs)
    np.save(out / "train" / "base.npy", base_vectors(V, spk, pcs, mu))
    for split in ("dev", "test"):
        frs, _, _ = load_split(out, split)
        _, h0, h1, _, sp2 = seg_info(out, split, arm, T)
        np.save(out / split / "base.npy", base_vectors(mean_pool(frs, h0, h1), sp2, pcs, mu))
    log("BASE DONE", arm, "train pieces", len(S))


# ---------------------------------------------------------------------------------------------------------- encoder


def make_encoder():
    import torch
    import torch.nn as nn
    import torch.utils.checkpoint

    class Enc(nn.Module):
        def __init__(self, d_in=PCA_DIM, d=384, layers=4, heads=6, ff=1536, out=256, maxlen=MAXF + 2):
            super().__init__()
            self.inp = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(maxlen, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, layers)
            self.norm = nn.LayerNorm(d)
            self.att = nn.Linear(d, 1)
            self.out = nn.Linear(d, out)

        def forward(self, x, mask):  # x [B, L, 256], mask [B, L] True = valid
            h = self.inp(x) + self.pos[: x.shape[1]]
            pad = ~mask
            for layer in self.tr.layers:
                if self.training and torch.is_grad_enabled():  # recompute activations: batch 2048 x 3 fits a 24 GB GPU
                    h = torch.utils.checkpoint.checkpoint(lambda t, l=layer: l(t, src_key_padding_mask=pad), h,
                                                          use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
            h = self.norm(h)
            w = self.att(h).squeeze(-1).masked_fill(~mask, -1e4)
            z = (torch.softmax(w, -1).unsqueeze(-1) * h).sum(1)
            return nn.functional.normalize(self.out(z), dim=-1)

    return Enc()


class Pieces:
    """Training pieces on the GPU: frames of the split, piece bounds, utterance bounds; batches with augmentation."""

    def __init__(self, fr: np.ndarray, off: np.ndarray, S: np.ndarray, g0: np.ndarray, g1: np.ndarray) -> None:
        import torch

        self.t = torch
        self.X = frames_tensor(fr)
        self.g0, self.g1 = torch.from_numpy(g0).cuda(), torch.from_numpy(g1).cuda()
        self.u0 = torch.from_numpy(off[S[:, 0]]).cuda()
        self.u1 = torch.from_numpy(off[S[:, 0] + 1]).cuda()

    def batch(self, idx, aug: bool):
        t = self.t
        s, e = self.g0[idx].clone(), self.g1[idx].clone()
        if aug:
            s = s + t.randint(-2, 3, s.shape, device="cuda")
            e = e + t.randint(-2, 3, e.shape, device="cuda")
        s = t.maximum(s - 1, self.u0[idx])
        e = t.minimum(e + 1, self.u1[idx])
        e = t.maximum(e, s + 2)
        e = t.minimum(e, self.u1[idx])
        s = t.minimum(s, e - 1)
        n = (e - s).float()
        L = MAXF + 2
        if aug:
            r = t.empty_like(n).uniform_(0.8, 1.25)
            m = (n / r).round().clamp(1, L)
        else:
            m = n.clamp(1, L)
        pos = t.arange(L, device="cuda")[None].float()
        if aug:
            src = s[:, None].float() + pos * (n / m)[:, None]
        else:  # centre crop
            src = s[:, None].float() + ((n - m) / 2).floor()[:, None] + pos
        src = t.minimum(src.long(), (e - 1)[:, None])
        mask = pos < m[:, None]
        if aug:
            drop = t.rand(mask.shape, device="cuda") < 0.10
            drop[:, 0] = False
            mask = mask & ~drop
        x = gather(self.X, src)
        return x, mask


def embed_all(model, P: Pieces, n: int, bs: int = 8192) -> np.ndarray:
    import torch

    model.eval()
    outs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, bs):
            idx = torch.arange(i, min(n, i + bs), device="cuda")
            x, m = P.batch(idx, aug=False)
            outs.append(model(x, m).float())
    model.train()
    return torch.cat(outs).cpu().numpy()


def kmeans(V: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means on the GPU; returns (centroids [k, d], assignment [n])."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.nn.functional.normalize(torch.from_numpy(V).cuda().float(), dim=1)
    C = X[torch.randperm(len(X), generator=g, device="cuda")[:k]].clone()
    for it in range(iters + 1):
        a = torch.empty(len(X), dtype=torch.long, device="cuda")
        for i in range(0, len(X), 65536):
            a[i : i + 65536] = (X[i : i + 65536] @ C.T).argmax(1)
        if it == iters:
            break
        S = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        S[empty] = X[torch.randint(0, len(X), (int(empty.sum()),), device="cuda", generator=g)]
        C = torch.nn.functional.normalize(S, dim=1)
    return C.cpu().numpy(), a.cpu().numpy()


def mine_pairs(V: np.ndarray, assign: np.ndarray, ep: np.ndarray, cand: int = 32, seed: int = 0):
    """For each piece: the closest of `cand` random members of its cluster from another recording; top half kept."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.from_numpy(V).cuda().float()
    a = torch.from_numpy(assign).cuda()
    e = torch.from_numpy(ep).cuda()
    order = torch.argsort(a)
    cnt = torch.bincount(a, minlength=int(a.max()) + 1)
    start = torch.cumsum(cnt, 0) - cnt
    anc, pos, sim = [], [], []
    for i in range(0, len(X), 65536):
        idx = torch.arange(i, min(len(X), i + 65536), device="cuda")
        c = a[idx]
        r = torch.randint(0, 2**31 - 1, (len(idx), cand), device="cuda", generator=g)
        mem = order[start[c][:, None] + r % cnt[c][:, None]]
        s = (X[idx][:, None] * X[mem]).sum(-1)
        bad = (e[mem] == e[idx][:, None]) | (mem == idx[:, None])
        s = s.masked_fill(bad, -2)
        best, j = s.max(1)
        ok = best > -2
        anc.append(idx[ok])
        pos.append(mem[ok.nonzero().squeeze(1), j[ok]])
        sim.append(best[ok])
    anc, pos, sim = torch.cat(anc), torch.cat(pos), torch.cat(sim)
    keep = sim >= sim.median()
    return anc[keep], pos[keep], {"pieces": len(X), "with_pair": len(sim), "kept": int(keep.sum()),
                                  "median_cos": round(float(sim.median()), 4)}


def info_nce(za, zp, zn, tau: float):
    import torch
    import torch.nn.functional as F

    lab = torch.arange(len(za), device=za.device)
    la = torch.cat([za @ zp.T, (za * zn).sum(-1, keepdim=True)], 1) / tau
    lp = (zp @ za.T) / tau
    return (F.cross_entropy(la, lab) + F.cross_entropy(lp, lab)) / 2


def next_piece(S: np.ndarray) -> np.ndarray:
    """Index of the next piece of the same utterance (or the previous one at the end)."""
    n = len(S)
    nx = np.arange(1, n + 1)
    nx[-1] = n - 2
    same_next = np.zeros(n, bool)
    same_next[:-1] = S[1:, 0] == S[:-1, 0]
    nx[~same_next] = np.arange(n)[~same_next] - 1
    nx = np.clip(nx, 0, n - 1)
    return nx


# ------------------------------------------------------------------------------------------------------------- eval


def ap_score(scores, labels) -> float:
    import torch

    o = torch.argsort(scores, descending=True)
    l = labels[o].float()
    tp = torch.cumsum(l, 0)
    prec = tp / torch.arange(1, len(l) + 1, device=l.device)
    return float((prec * l).sum() / l.sum().clamp(min=1))


def same_different(Z: np.ndarray, word: np.ndarray, ep: np.ndarray, spk: np.ndarray) -> dict:
    """AP over pairs from different recordings (cross) and from the same speaker of one recording (within)."""
    import torch

    Zt = torch.nn.functional.normalize(torch.from_numpy(Z).cuda().float(), dim=1)
    w, e, s = (torch.from_numpy(x).cuda() for x in (word, ep, spk))
    up = torch.ones(len(Z), len(Z), dtype=torch.bool, device="cuda").triu(1)
    sc = (Zt @ Zt.T)[up]
    pos = (w[:, None] == w[None])[up]
    cross = (e[:, None] != e[None])[up]
    within = (s[:, None] == s[None])[up]
    out = {}
    for name, m in (("cross", cross), ("within", within)):
        out[f"AP_{name}"] = round(ap_score(sc[m], pos[m]), 4)
        out[f"pos_rate_{name}"] = round(float(pos[m].float().mean()), 5)
        out[f"pairs_{name}"] = int(m.sum())
        out[f"pos_{name}"] = int((pos & m).sum())
    out["ratio_cross_within"] = round(out["AP_cross"] / max(out["AP_within"], 1e-9), 4)
    return out


def tokens(out: Path, split: str, arm: str, T: dict, oracle: bool, seed: int = 0):
    """Eval tokens: pieces of the arm whose edges are within TOL40 of one reference word (or the word spans)."""
    ref = json.loads((out / split / "ref.json").read_text())["words"]
    off, rows = np.load(out / split / "off.npy"), np.load(out / split / "rows.npy")
    S = np.load(out / split / f"seg_{arm}.npy")
    by = {}
    for k, r in enumerate(S):
        by.setdefault(int(r[0]), []).append(k)
    items = []  # (word, g0, g1, utt, piece index or -1)
    for u, words in enumerate(ref):
        if not words:
            continue
        for wd, ws, we in words:
            if len(wd) < MIN_LETTERS or we - ws < MIN_WORD_FR:
                continue
            if oracle:
                items.append((wd, off[u] + ws, off[u] + we, u, -1))
                continue
            for k in by.get(u, []):
                if abs(S[k, 1] - ws) <= TOL40 and abs(S[k, 2] - we) <= TOL40:
                    items.append((wd, off[u] + S[k, 1], off[u] + S[k, 2], u, k))
                    break
    rng = np.random.default_rng(seed)
    byw = {}
    for it in items:
        byw.setdefault(it[0], []).append(it)
    keep = []
    for wd, lst in byw.items():
        if len(lst) < 2:
            continue
        if len(lst) > PER_WORD:
            lst = [lst[i] for i in rng.permutation(len(lst))[:PER_WORD]]
        keep += lst
    vocab = {w: i for i, w in enumerate(sorted({k[0] for k in keep}))}
    word = np.array([vocab[k[0]] for k in keep], np.int64)
    g0 = np.array([k[1] for k in keep], np.int64)
    g1 = np.array([k[2] for k in keep], np.int64)
    utt = np.array([k[3] for k in keep], np.int64)
    pidx = np.array([k[4] for k in keep], np.int64)
    return {"word": word, "g0": g0, "g1": g1, "utt": utt, "piece": pidx, "ep": T["ep"][rows[utt]],
            "spk": T["spk"][rows[utt]], "vocab": sorted(vocab, key=vocab.get), "matched_items": len(items)}


def cmd_check(a) -> None:
    import torch

    from dy36_dict import load_table

    out = Path(a.out)
    T = load_table(Path(a.table))
    arm = json.loads((out / "g1.json").read_text())["decision"]["arm"]
    R = {}
    sp = np.load(out / "split.npz")
    tr_eps = set(T["ep"][sp["train"]].tolist())
    R["leakage"] = {"pass": not (tr_eps & set(T["ep"][sp["test"]].tolist())) and not (tr_eps & set(T["ep"][sp["dev"]].tolist())),
                    "note": "dev/test recordings not in training; PCA and directions fitted on training only"}
    fr, off, _ = load_split(out, "train")
    S, g0, g1, ep, _ = seg_info(out, "train", arm, T)
    P = Pieces(fr, off, S, g0, g1)
    torch.manual_seed(0)
    model = make_encoder().cuda()
    npar = sum(p.numel() for p in model.parameters())
    R["params"] = npar
    B = 2048
    idx = torch.randint(0, len(S), (B,), device="cuda")
    x, m = P.batch(idx, aug=True)
    x2, m2 = P.batch(idx, aug=True)  # true pairs: the same pieces, another augmentation
    xn, mn = P.batch(torch.randint(0, len(S), (B,), device="cuda"), aug=True)
    model.eval()
    with torch.no_grad():
        za, zp, zn = model(x, m), model(x2, m2), model(xn, mn)
        l_true = float(info_nce(za, zp, zn, 0.07))
        l_perm = float(info_nce(za, zp[torch.randperm(B, device="cuda")], zn, 0.07))
    model.train()
    # known answer: an untrained encoder already maps the same piece (two augmentations) closer than other pieces,
    # so the loss on true pairs is below the loss with the positives shuffled; with shuffled positives the loss is
    # the no-information level for this temperature (not ln(B+1): random-init vectors are not spread uniformly)
    R["init_loss"] = {"true_pairs": round(l_true, 4), "shuffled_pairs": round(l_perm, 4),
                      "ln_B+1": round(math.log(B + 1), 4), "pass": l_true < l_perm}
    # known answer: the piece with itself (two augmentations) is learnable fast
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    for st in range(200):
        idx = torch.randint(0, len(S), (512,), device="cuda")
        xa, ma = P.batch(idx, aug=True)
        xb, mb = P.batch(idx, aug=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            za, zb = model(xa, ma), model(xb, mb)
        loss = info_nce(za.float(), zb.float(), zb.float().roll(1, 0), 0.07)
        opt.zero_grad()
        loss.backward()
        opt.step()
    R["self_pairs"] = {"loss_after_200": round(float(loss), 4), "start": round(math.log(513), 4),
                       "pass": float(loss) < 0.5 * math.log(513)}
    # overfit one fixed batch of different pieces paired arbitrarily
    torch.manual_seed(1)
    model = make_encoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    ia = torch.randint(0, len(S), (256,), device="cuda")
    ib = torch.randint(0, len(S), (256,), device="cuda")
    xa, ma = P.batch(ia, aug=False)
    xb, mb = P.batch(ib, aug=False)
    first = None
    for st in range(300):
        loss = info_nce(model(xa, ma), model(xb, mb), model(xb, mb).roll(1, 0), 0.07)
        first = first or float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    R["overfit_one_batch"] = {"start": round(first, 4), "end": round(float(loss), 4), "pass": float(loss) < 0.3 * first}
    # measure checks on test tokens
    Tk = tokens(out, "test", arm, T, oracle=False)
    rng = np.random.default_rng(0)
    V = len(Tk["vocab"])  # answer vectors + noise of total norm ~0.3 whatever the vocabulary size
    oh = np.eye(V, dtype=np.float32)[Tk["word"]] + (0.3 / math.sqrt(V)) * rng.standard_normal((len(Tk["word"]), V)).astype(np.float32)
    ka = same_different(oh, Tk["word"], Tk["ep"], Tk["spk"])
    rnd = same_different(rng.standard_normal((len(Tk["word"]), 64)).astype(np.float32), Tk["word"], Tk["ep"], Tk["spk"])
    R["measure_known_answer"] = {"AP_cross": ka["AP_cross"], "pass": ka["AP_cross"] > 0.95}
    R["measure_random"] = {"AP_cross": rnd["AP_cross"], "pos_rate": rnd["pos_rate_cross"],
                           "pass": rnd["AP_cross"] < rnd["pos_rate_cross"] * 3 + 0.01}
    R["test_tokens"] = {"tokens": len(Tk["word"]), "words": len(Tk["vocab"]), "matched_items": Tk["matched_items"]}
    R["all_pass"] = all(v["pass"] for v in R.values() if isinstance(v, dict) and "pass" in v)
    (out / "checklist.json").write_text(json.dumps(R, indent=1))
    log("CHECKLIST", json.dumps(R))


def dev_ap(model, out: Path, arm: str, T: dict, cache: dict) -> dict:
    if "dev" not in cache:
        fr, off, _ = load_split(out, "dev")
        Tk = tokens(out, "dev", arm, T, oracle=False)
        P = Pieces(fr, off, np.stack([Tk["utt"], Tk["g0"] - off[Tk["utt"]], Tk["g1"] - off[Tk["utt"]]], 1), Tk["g0"], Tk["g1"])
        cache["dev"] = (Tk, P)
    Tk, P = cache["dev"]
    Z = embed_all(model, P, len(Tk["word"]))
    return same_different(Z, Tk["word"], Tk["ep"], Tk["spk"])


def cmd_train(a) -> None:
    import airon
    import torch

    from dy36_dict import load_table

    out = Path(a.out)
    run_dir = out / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    T = load_table(Path(a.table))
    arm = json.loads((out / "g1.json").read_text())["decision"]["arm"]
    fr, off, _ = load_split(out, "train")
    S, g0, g1, ep, _ = seg_info(out, "train", arm, T)
    P = Pieces(fr, off, S, g0, g1)
    nx = torch.from_numpy(next_piece(S)).cuda()
    torch.manual_seed(0)
    model = make_encoder().cuda()
    npar = sum(p.numel() for p in model.parameters())
    meta_f = run_dir / "metadata.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"airon_run_id": None}
    import uuid

    meta.update({"arm": arm, "params": npar, "rounds": a.rounds, "steps": a.steps, "batch": a.batch, "tau": a.tau,
                 "lr": a.lr, "K": K, "pieces": len(S)})
    fresh = not meta["airon_run_id"]
    meta["airon_run_id"] = meta["airon_run_id"] or str(uuid.uuid4())
    meta_f.write_text(json.dumps(meta, indent=1))
    log("encoder params", npar, "pieces", len(S), "arm", arm)
    run = airon.init(project="dauys", name=f"dy38-{a.name}", config=meta, experiment=EXPERIMENT, tags=["dy38"],
                     total_steps=a.rounds * a.steps, run_id=meta["airon_run_id"], resume="never" if fresh else "allow",
                     mode="disabled" if a.dry else None)
    V = np.load(out / "train" / "base.npy")
    cache, hist, best = {}, [], (-1.0, None)
    r0 = dev_ap(model, out, arm, T, cache)  # untrained encoder, for reference
    hist.append({"round": -1, "step": 0, "dev": r0})
    gstep = 0
    for rnd in range(a.rounds):
        _, assign = kmeans(V, min(K, len(V) // 4), seed=rnd)
        anc, pos, st = mine_pairs(V, assign, ep, seed=rnd)
        log("round", rnd, "pairs", json.dumps(st))
        opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
        for step in range(a.steps):
            j = torch.randint(0, len(anc), (a.batch,), device="cuda")
            ia, ip = anc[j], pos[j]
            xa, ma = P.batch(ia, aug=True)
            xp, mp = P.batch(ip, aug=True)
            xn, mn = P.batch(nx[ia], aug=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                za, zp, zn = model(xa, ma), model(xp, mp), model(xn, mn)
            loss = info_nce(za.float(), zp.float(), zn.float(), a.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            if step % 100 == 0:
                run.log({"loss": float(loss), "round": rnd, "lr": sched.get_last_lr()[0]}, step=gstep)
            if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                r = dev_ap(model, out, arm, T, cache)
                hist.append({"round": rnd, "step": gstep, "loss": round(float(loss), 4), "dev": r})
                run.log({"dev_AP_cross": r["AP_cross"], "dev_AP_within": r["AP_within"]}, step=gstep)
                log("round", rnd, "step", gstep, "loss", round(float(loss), 4), "dev", json.dumps(r))
                if r["AP_cross"] > best[0]:
                    best = (r["AP_cross"], gstep)
                    torch.save({"model": model.state_dict(), "step": gstep, "round": rnd}, run_dir / "best.pt")
                (run_dir / "val_history.json").write_text(json.dumps(hist, indent=1))
        V = embed_all(model, P, len(S))
    torch.save({"model": model.state_dict(), "step": gstep}, run_dir / "last.pt")
    run.finish()
    (run_dir / "DONE").write_text(json.dumps({"best_dev_AP_cross": best[0], "best_step": best[1]}))
    log("TRAIN DONE best dev AP_cross", best)


def cmd_dict(a) -> None:
    import torch

    from dy36_dict import load_table

    out = Path(a.out)
    T = load_table(Path(a.table))
    arm = json.loads((out / "g1.json").read_text())["decision"]["arm"]
    fr, off, _ = load_split(out, "train")
    S, g0, g1, _, _ = seg_info(out, "train", arm, T)
    model = make_encoder().cuda()
    model.load_state_dict(torch.load(out / a.name / "best.pt", map_location="cuda")["model"])
    Z = embed_all(model, Pieces(fr, off, S, g0, g1), len(S))
    C, _ = kmeans(Z, K, iters=20, seed=0)
    Cb, _ = kmeans(np.load(out / "train" / "base.npy"), K, iters=20, seed=0)
    np.savez(out / a.name / "dict.npz", trained=C, base=Cb)
    log("DICT DONE", C.shape)


def nmi_purity(word: np.ndarray, tok: np.ndarray) -> dict:
    from sklearn.metrics import normalized_mutual_info_score

    pur = 0
    for t in np.unique(tok):
        m = tok == t
        pur += np.bincount(word[m]).max()
    cons = 0  # word consistency: tokens of a word that fall into that word's most common dictionary token
    for w in np.unique(word):
        cons += np.bincount(tok[word == w]).max()
    return {"NMI": round(float(normalized_mutual_info_score(word, tok)), 4), "purity": round(pur / len(word), 4),
            "consistency": round(cons / len(word), 4), "tokens_used": int(len(np.unique(tok)))}


def cmd_eval(a) -> None:
    import torch

    from dy36_dict import load_table

    out = Path(a.out)
    T = load_table(Path(a.table))
    arm = json.loads((out / "g1.json").read_text())["decision"]["arm"]
    model = make_encoder().cuda()
    model.load_state_dict(torch.load(out / a.name / "best.pt", map_location="cuda")["model"])
    D = np.load(out / a.name / "dict.npz")
    bp = np.load(out / "base_params.npz")
    fr, off, _ = load_split(out, "test")
    _, h0, h1, _, hspk = seg_info(out, "test", arm, T)
    raw_all = mean_pool(fr, h0, h1)  # speaker means for training-free vectors use every test piece of the arm
    spk_ids, inv = np.unique(hspk, return_inverse=True)
    smean = np.zeros((len(spk_ids), raw_all.shape[1]))
    np.add.at(smean, inv, raw_all)
    smean /= np.bincount(inv)[:, None]
    smap = {int(s): i for i, s in enumerate(spk_ids)}
    R = {"arm": arm}
    for kind in ("own", "oracle"):
        Tk = tokens(out, "test", arm, T, oracle=(kind == "oracle"))
        S3 = np.stack([Tk["utt"], Tk["g0"] - off[Tk["utt"]], Tk["g1"] - off[Tk["utt"]]], 1)
        Zt = embed_all(model, Pieces(fr, off, S3, Tk["g0"], Tk["g1"]), len(Tk["word"]))
        raw = mean_pool(fr, Tk["g0"], Tk["g1"])
        X = raw - np.stack([smean[smap[int(s)]] if int(s) in smap else raw[i] * 0 for i, s in enumerate(Tk["spk"])])
        X = X - bp["mu"]
        X = X - (X @ bp["pcs"].T) @ bp["pcs"]
        Zb = (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)).astype(np.float32)
        res = {"tokens": len(Tk["word"]), "words": len(Tk["vocab"]),
               "trained": same_different(Zt, Tk["word"], Tk["ep"], Tk["spk"]),
               "base": same_different(Zb, Tk["word"], Tk["ep"], Tk["spk"])}
        for nm, Z, C in (("trained", Zt, D["trained"]), ("base", Zb, D["base"])):
            tok = (torch.from_numpy(Z).cuda() @ torch.from_numpy(C).cuda().T).argmax(1).cpu().numpy()
            res[nm]["dict"] = nmi_purity(Tk["word"], tok)
            if kind == "own" and nm == "trained":
                np.savez(out / a.name / "test_tokens.npz", tok=tok, word=Tk["word"], g0=Tk["g0"], g1=Tk["g1"],
                         utt=Tk["utt"], ep=Tk["ep"], vocab=np.array(Tk["vocab"]))
        R[kind] = res
    o, t, b = R["own"], R["own"]["trained"], R["own"]["base"]
    g2 = t["AP_cross"] >= 0.50 and t["AP_cross"] >= b["AP_cross"] + 0.10 and t["ratio_cross_within"] >= 0.75
    # purity is near 1 by construction when K=20000 >> test tokens, so G3 uses consistency (fixed before the full run)
    g3 = t["dict"]["NMI"] >= b["dict"]["NMI"] + 0.05 and t["dict"]["consistency"] >= b["dict"]["consistency"] + 0.05
    R["gates"] = {"G2": g2, "G3": g3}
    (out / a.name / "eval.json").write_text(json.dumps(R, indent=1))
    log("EVAL", json.dumps(R))
    listening(out, a.name)


def listening(out: Path, name: str) -> None:
    """10 dictionary tokens with test pieces from the most recordings: up to 5 clips each, from different recordings."""
    import wave

    z = np.load(out / name / "test_tokens.npz")
    audio, aoff = np.load(out / "test" / "audio.npy", mmap_mode="r"), np.load(out / "test" / "audio_off.npy")
    off = np.load(out / "test" / "off.npy")
    vocab = z["vocab"]
    d = out / name / "listen"
    d.mkdir(exist_ok=True)
    toks = [t for t in np.unique(z["tok"])]
    toks.sort(key=lambda t: -len(np.unique(z["ep"][z["tok"] == t])))
    index = []
    for t in toks[:10]:
        idx = np.flatnonzero(z["tok"] == t)
        seen, clips = set(), []
        for i in idx:
            if z["ep"][i] in seen:
                continue
            seen.add(z["ep"][i])
            u = int(z["utt"][i])
            s = int(z["g0"][i] - off[u]) * HOP
            e = int(z["g1"][i] - off[u]) * HOP
            clips.append(np.asarray(audio[aoff[u] + s : aoff[u] + e], np.int16))
            index.append({"token": int(t), "clip": len(clips), "word_whisper": str(vocab[z["word"][i]]),
                          "recording": int(z["ep"][i])})
            if len(clips) == 5:
                break
        gap = np.zeros(SR // 3, np.int16)
        full = np.concatenate([x for c in clips for x in (c, gap)])
        with wave.open(str(d / f"token_{int(t)}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(full.tobytes())
    (d / "index.json").write_text(json.dumps(index, indent=1))
    log("LISTEN DONE", len(index))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--table", default="/data/dy36/dict")
    p.add_argument("--out", default="/data/dy38")
    p.add_argument("--split", default="train")
    p.add_argument("--hours", type=float, default=200)
    p.add_argument("--dev_hours", type=float, default=2)
    p.add_argument("--test_hours", type=float, default=5)
    p.add_argument("--per_rec_min", type=float, default=10)
    p.add_argument("--limit_hours", type=float, default=0)
    p.add_argument("--pca_hours", type=float, default=3)
    p.add_argument("--batch_sec", type=float, default=600)
    p.add_argument("--name", default="run")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    {"select": cmd_select, "feats": cmd_feats, "ref": cmd_ref, "seg": cmd_seg, "g1": cmd_g1, "base": cmd_base,
     "check": cmd_check, "train": cmd_train, "dict": cmd_dict, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
