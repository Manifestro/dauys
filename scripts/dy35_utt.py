"""DY-35 data: utterances (without text) and separately encoded codes over the DY-32 b shards.

Per shard of DuplexChat en-16k (pieces >= 5 s, as DY-32 b):
  1. voice activity: silero-vad (MIT) probabilities per 32 ms chunk, batched over all pieces of the shard; hysteresis
     0.5 / 0.35 at the 20 ms frame grid; speech runs < 250 ms dropped, pauses < 300 ms bridged.
  2. speaker vectors: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, Apache-2.0, speaker labels only) on 1.5 s
     windows, hop 0.75 s, inside speech runs; clustered per episode (pieces.jsonl "ep"): average-linkage cosine
     agglomerative clustering (threshold --thr) on <= 3000 windows, clusters < 3 % merged into the nearest one, every
     window assigned to the nearest centroid, labels smoothed by the majority of 3 neighbouring windows.
  3. segments: maximal runs of speech frames with one speaker label (a pause >= 300 ms or a label change ends one).
     Utterance = segment of 2-20 s (100-1000 frames).
  4. codes: the piece is cut into chunks at every utterance start and end; each utterance is one chunk, the stretches
     between them are chunks of <= 20 s. Every chunk is encoded by WavLM-Large layer 6 ALONE (padded by 80 zero
     samples so a chunk of 320 m samples gives exactly m frames on the global 50 Hz grid) and quantised with the
     K2000 corpus codebook. So the codes of an utterance never saw the audio before or after it, and the codes before
     an utterance never saw the utterance. The ordinary training codes are these same chunk codes (the DY-32 b codes
     are NOT reused: their 20 s windows cross utterance boundaries).

``<shard>.npz``: tok int16 [frames], index [pieces, 2] (start, len) into tok, piece_orig, piece_ep, piece_t0, eps (json),
vad uint8 and edb int8 (frame RMS dBFS) per frame, seg_piece/seg_s/seg_e (frames within the piece)/seg_spk/seg_emb
(float16 [n, 192], mean normalised window vectors), chunk_s/chunk_e/chunk_piece.

    python scripts/dy35_utt.py run --part 0/3 --work /opt/dlami/nvme/w0 --cb /data/cb/km-corpus-K2000.npy
    python scripts/dy35_utt.py one --shard la-00030 --work ... --cb ...   # one shard, local only, timings
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import SRC, listing, sh
from dy32b_channel import SR, Channel, Codebook

S3 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys"
DST = f"{S3}/dy35/utt"
SHARDS = f"{S3}/dy32/b/units50/_PASS_DONE"
HOP = 320
CH = 512  # silero chunk at 16 kHz
UMIN, UMAX = 100, 1000  # utterance length in frames (2-20 s)
WINF, WHOP = 75, 37  # speaker window / hop in frames
PAUSE = 15  # 300 ms
MINSP = 13  # 250 ms
GAPMAX = 1000  # gap chunks <= 20 s


def rss_gb() -> float:
    try:
        return int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * 4096 / 1e9
    except Exception:
        return -1.0


def fetch(s: str, work: Path):
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin", "pieces.jsonl"):
        sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    meta = [json.loads(l) for l in open(d / "pieces.jsonl")]
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    sh(f"rm -rf {d}")
    assert len(meta) == len(idx)
    keep = [(j, int(st), int(ln)) for j, (st, ln) in enumerate(idx) if ln >= 5 * SR]
    pieces = [segs[st : st + ln] for _, st, ln in keep]
    return s, pieces, [j for j, _, _ in keep], [meta[j] for j, _, _ in keep]


# ---------------------------------------------------------------- voice activity
class Vad:
    def __init__(self, device: str) -> None:
        from silero_vad import load_silero_vad

        self.m = load_silero_vad(onnx=False).to(device)
        self.dev = device

    @torch.inference_mode()
    def probs(self, pieces: list[np.ndarray], seg: int = 312, warm: int = 16, batch: int = 8192) -> list[np.ndarray]:
        """Chunk probabilities per piece. Each piece is cut into streams of ``seg`` chunks (10 s), each stream starts
        ``warm`` chunks (0.5 s) earlier to warm the recurrent state (those outputs are dropped); all streams of all
        pieces run as one batch (the per-step cost is launch-bound, so wide batches are nearly free)."""
        jobs = []  # (piece, first chunk)
        for k, p in enumerate(pieces):
            n = -(-len(p) // CH)
            jobs += [(k, c) for c in range(0, n, seg)]
        L = (seg + warm) * CH
        out = [np.zeros(-(-len(p) // CH), np.float32) for p in pieces]
        for b in range(0, len(jobs), batch):
            js = jobs[b : b + batch]
            x = torch.zeros(len(js), L)
            for i, (k, c) in enumerate(js):
                a = max(0, (c - warm) * CH)
                w = pieces[k][a : (c + seg) * CH]
                off = a - (c - warm) * CH  # chunk c - warm sits at position 0 (zeros before the piece start)
                x[i, off : off + len(w)] = torch.from_numpy(w.astype(np.float32) / 32768.0)
            x = x.to(self.dev)
            self.m.reset_states()
            pr = torch.stack([self.m(x[:, j * CH : (j + 1) * CH], SR)[:, 0] for j in range(seg + warm)], 1)
            pr = pr[:, warm:].float().cpu().numpy()
            for i, (k, c) in enumerate(js):
                m = min(seg, len(out[k]) - c)
                out[k][c : c + m] = pr[i, :m]
        return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def speech_frames(pr: np.ndarray, nf: int) -> np.ndarray:
    """Chunk probabilities -> speech mask on the 20 ms frame grid (hysteresis, drop < 250 ms, bridge < 300 ms)."""
    ci = np.minimum((np.arange(nf) * HOP + HOP // 2) // CH, len(pr) - 1)
    p = pr[ci]
    sp = np.zeros(nf, bool)
    on = False
    for t in range(nf):
        on = p[t] >= 0.5 if not on else p[t] >= 0.35
        sp[t] = on
    for s, e in runs(sp):
        if e - s < MINSP:
            sp[s:e] = False
    for s, e in runs(~sp):
        if s > 0 and e < nf and e - s < PAUSE:
            sp[s:e] = True
    return sp


# ---------------------------------------------------------------- speakers
class Spk:
    def __init__(self, device: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/data/ecapa",
                                                run_opts={"device": device})
        self.dev = device

    @torch.inference_mode()
    def embed(self, wavs: list[np.ndarray], batch: int = 256) -> np.ndarray:
        order = sorted(range(len(wavs)), key=lambda i: -len(wavs[i]))
        out = np.zeros((len(wavs), 192), np.float32)
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            L = len(wavs[idx[0]])
            x = torch.zeros(len(idx), L)
            lens = torch.zeros(len(idx))
            for j, i in enumerate(idx):
                x[j, : len(wavs[i])] = torch.from_numpy(wavs[i].astype(np.float32) / 32768.0)
                lens[j] = len(wavs[i]) / L
            e = self.m.encode_batch(x.to(self.dev), lens.to(self.dev))[:, 0].float()
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
            out[idx] = e
        return out


def cluster(x: np.ndarray, thr: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """x [n, 192] normalised -> labels [n], centroids [k, 192] (normalised)."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(x)
    if n == 1:
        return np.zeros(1, np.int64), x.copy()
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, 3000, replace=False)) if n > 3000 else np.arange(n)
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=thr).fit_predict(x[sub])
    cnt = np.bincount(lab)
    cent = np.stack([x[sub][lab == k].mean(0) for k in range(len(cnt))])
    big = np.flatnonzero(cnt >= max(2, 0.03 * len(sub)))
    if len(big) == 0:
        big = np.array([int(cnt.argmax())])
    cent = cent[big]
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    return (x @ cent.T).argmax(1), cent


# ---------------------------------------------------------------- one shard
def frame_db(w: np.ndarray, nf: int) -> np.ndarray:
    x = w[: nf * HOP].astype(np.float32).reshape(nf, HOP) / 32768.0
    return np.clip(10 * np.log10((x * x).mean(1) + 1e-10), -100, 0).astype(np.int8)


def process(s: str, pieces, orig, meta, vad: Vad, spk: Spk, ch: Channel, cb: Codebook, thr: float, half: bool) -> dict:
    tm = {}
    t = time.time()
    nfs = [len(p) // HOP for p in pieces]
    prs = vad.probs(pieces)
    sps = [speech_frames(pr, nf) for pr, nf in zip(prs, nfs)]
    tm["vad"] = time.time() - t
    # speaker windows
    t = time.time()
    win_p, win_s, win_e, wavs = [], [], [], []
    for k, (p, sp) in enumerate(zip(pieces, sps)):
        for s0, e0 in runs(sp):
            if e0 - s0 <= WINF:
                starts = [s0]
            else:
                starts = list(range(s0, e0 - WINF + 1, WHOP))
                if starts[-1] + WINF < e0:
                    starts.append(e0 - WINF)
            for a in starts:
                b = min(a + WINF, e0)
                win_p.append(k)
                win_s.append(a)
                win_e.append(b)
                wavs.append(p[a * HOP : b * HOP])
    emb = spk.embed(wavs) if wavs else np.zeros((0, 192), np.float32)
    del wavs
    win_p, win_s, win_e = map(np.asarray, (win_p, win_s, win_e))
    tm["spk"] = time.time() - t
    t = time.time()
    eps = sorted({m["ep"] for m in meta})
    ep_of = np.array([eps.index(m["ep"]) for m in meta])
    lab = np.full(len(emb), -1, np.int64)
    c_ep, c_vec = [], []
    for ei in range(len(eps)):
        w = np.flatnonzero(ep_of[win_p] == ei) if len(win_p) else np.zeros(0, int)
        if len(w) == 0:
            continue
        l, cent = cluster(emb[w], thr)
        lab[w] = l
        c_ep += [ei] * len(cent)
        c_vec.append(cent)
    # smoothing: majority of 3 neighbouring windows within one speech run (windows are in time order per piece)
    sm = lab.copy()
    for i in range(1, len(lab) - 1):
        if win_p[i - 1] == win_p[i] == win_p[i + 1] and win_s[i] < win_e[i - 1] and win_s[i + 1] < win_e[i]:
            if lab[i - 1] == lab[i + 1] != lab[i]:
                sm[i] = lab[i - 1]
    lab = sm
    tm["cluster"] = time.time() - t
    # segments
    t = time.time()
    seg = {"piece": [], "s": [], "e": [], "spk": [], "emb": []}
    chunks = []  # (piece, s, e)
    wi_by_piece = {k: np.flatnonzero(win_p == k) for k in range(len(pieces))}
    for k, (sp, nf) in enumerate(zip(sps, nfs)):
        fl = np.full(nf, -1, np.int64)
        wi = wi_by_piece[k]
        if len(wi):
            ctr = (win_s[wi] + win_e[wi]) / 2
            fr = np.flatnonzero(sp)
            near = np.abs(fr[:, None] - ctr[None]).argmin(1) if len(fr) and len(fr) * len(wi) < 5e7 else None
            if near is None and len(fr):
                near = np.searchsorted(ctr, fr).clip(0, len(ctr) - 1)
            fl[fr] = lab[wi[near]]
        # segments = runs of equal label among speech frames
        b = np.flatnonzero(np.diff(np.concatenate([[-2], fl, [-2]])) != 0)
        cuts, uset = [0], set()
        for s0, e0 in zip(b[:-1], b[1:]):
            if fl[s0] < 0:
                continue
            ws = wi[(win_s[wi] < e0) & (win_e[wi] > s0)] if len(wi) else wi
            v = emb[ws].mean(0) if len(ws) else np.zeros(192, np.float32)
            v = v / max(np.linalg.norm(v), 1e-6)
            seg["piece"].append(k)
            seg["s"].append(s0)
            seg["e"].append(e0)
            seg["spk"].append(fl[s0])
            seg["emb"].append(v.astype(np.float16))
            if UMIN <= e0 - s0 <= UMAX:
                cuts += [s0, e0]
                uset.add((int(s0), int(e0)))
        cuts.append(nf)
        cuts = sorted(set(int(c) for c in cuts))
        for a, c in zip(cuts[:-1], cuts[1:]):
            if (a, c) in uset:
                chunks.append((k, a, c))
            else:
                for x in range(a, c, GAPMAX):
                    chunks.append((k, x, min(x + GAPMAX, c)))
    tm["segments"] = time.time() - t
    # codes, chunk by chunk
    t = time.time()
    toks = [np.zeros(nf, np.int16) for nf in nfs]
    order = sorted(range(len(chunks)), key=lambda i: -(chunks[i][2] - chunks[i][1]))
    for b0 in range(0, len(order), 256):
        part = [chunks[i] for i in order[b0 : b0 + 256]]
        x = [np.concatenate([pieces[k][a * HOP : c * HOP].astype(np.float32) / 32768.0, np.zeros(80, np.float32)])
             for k, a, c in part]
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            fs = ch.feats(x, batch=32)
        for (k, a, c), f in zip(part, fs):
            assert f.shape[0] == c - a, (f.shape, a, c)
            toks[k][a:c] = cb.encode(f.float()).cpu().numpy().astype(np.int16)
    tm["codes"] = time.time() - t
    index, pos = [], 0
    for nf in nfs:
        index.append((pos, nf))
        pos += nf
    res = {
        "tok": np.concatenate(toks), "index": np.asarray(index, np.int64),
        "piece_orig": np.asarray(orig, np.int32), "piece_ep": ep_of.astype(np.int32),
        "piece_t0": np.asarray([m["start"] for m in meta], np.float64), "eps": np.array(json.dumps(eps)),
        "vad": np.concatenate(sps).astype(np.uint8),
        "edb": np.concatenate([frame_db(p, nf) for p, nf in zip(pieces, nfs)]),
        "seg_piece": np.asarray(seg["piece"], np.int32), "seg_s": np.asarray(seg["s"], np.int32),
        "seg_e": np.asarray(seg["e"], np.int32), "seg_spk": np.asarray(seg["spk"], np.int16),
        "seg_emb": np.stack(seg["emb"]) if seg["emb"] else np.zeros((0, 192), np.float16),
        "chunk_piece": np.asarray([c[0] for c in chunks], np.int32),
        "chunk_s": np.asarray([c[1] for c in chunks], np.int32), "chunk_e": np.asarray([c[2] for c in chunks], np.int32),
        "c_ep": np.asarray(c_ep, np.int32), "c_vec": np.concatenate(c_vec).astype(np.float16) if c_vec else np.zeros((0, 192), np.float16),
    }
    L = res["seg_e"] - res["seg_s"]
    stats = {"pieces": len(pieces), "hours": round(sum(nfs) / 50 / 3600, 2), "eps": len(eps), "windows": len(emb),
             "segments": len(L), "utterances": int(((L >= UMIN) & (L <= UMAX)).sum()), "chunks": len(chunks),
             "speech_share": round(float(res["vad"].mean()), 3),
             "spk_per_ep": round(len(c_ep) / max(len(eps), 1), 2), "t": {k: round(v, 1) for k, v in tm.items()}}
    return res, stats


def check_chunks(res: dict) -> None:
    """Every utterance is exactly one chunk; chunks tile every piece."""
    ch = set(zip(res["chunk_piece"].tolist(), res["chunk_s"].tolist(), res["chunk_e"].tolist()))
    L = res["seg_e"] - res["seg_s"]
    for p, s, e, l in zip(res["seg_piece"], res["seg_s"], res["seg_e"], L):
        if UMIN <= l <= UMAX:
            assert (int(p), int(s), int(e)) in ch, ("utterance is not one chunk", p, s, e)
    for k, (_, nf) in enumerate(res["index"]):
        m = res["chunk_piece"] == k
        s, e = np.sort(res["chunk_s"][m]), np.sort(res["chunk_e"][m])
        assert s[0] == 0 and e[-1] == nf and np.all(s[1:] == e[:-1]), ("chunks do not tile piece", k)


def models(a):
    torch.backends.cuda.matmul.allow_tf32 = True
    return Vad("cuda"), Spk("cuda"), Channel(), Codebook(a.cb)


def cmd_run(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    vad, spk, ch, cb = models(a)
    names = [l.strip().removesuffix(".npz") for l in sh(f"aws s3 cp {SHARDS} -").splitlines() if l.strip().endswith(".npz")]
    order = [n for n in names if held_out(n)] + [n for n in names if not held_out(n)]
    k, n = map(int, a.part.split("/"))
    done = set(listing(a.dst, dirs=False))
    mine = [x for x in (a.shards.split(",") if a.shards else order[k::n]) if x not in done]
    print(json.dumps({"part": a.part, "todo": len(mine), "all": len(order), "held": [x for x in order if held_out(x)]}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    ups, failed, lock = [], [], threading.Lock()

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
            f.unlink()
        except Exception as e:
            with lock:
                failed.append((s, str(e)[-200:]))

    nxt = dl.submit(fetch, mine[0], work) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        t0 = time.time()
        s, pieces, orig, meta = nxt.result()
        wait = time.time() - t0
        nxt = None
        res, st = process(s, pieces, orig, meta, vad, spk, ch, cb, a.thr, a.half)
        check_chunks(res)
        del pieces
        nxt = dl.submit(fetch, mine[j + 1], work) if j + 1 < len(mine) else None
        out = work / f"{s}.npz"
        np.savez(out, **res)
        del res
        ups.append(up.submit(upload, out, s))
        h_all += st["hours"]
        print(json.dumps({"part": a.part, "shard": s, "held": held_out(s), "done": j + 1, "of": len(mine), **st,
                          "wait_s": round(wait, 1), "x_rt": round(st["hours"] * 3600 / (time.time() - t0), 1),
                          "hours_total": round(h_all, 1), "rss_gb": round(rss_gb(), 2)}), flush=True)
    for f in ups:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"UTT DONE part {a.part}", round(h_all, 1), "h", flush=True)


def cmd_one(a) -> None:
    vad, spk, ch, cb = models(a)
    t = time.time()
    s, pieces, orig, meta = fetch(a.shard, Path(a.work))
    if a.max_pieces:
        pieces, orig, meta = pieces[: a.max_pieces], orig[: a.max_pieces], meta[: a.max_pieces]
    print("fetch", round(time.time() - t, 1), flush=True)
    res, st = process(s, pieces, orig, meta, vad, spk, ch, cb, a.thr, a.half)
    check_chunks(res)
    print("ONE", json.dumps(st), "rss_gb", round(rss_gb(), 2), flush=True)
    np.savez(Path(a.work) / f"{s}.npz", **res)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "one"):
        x = s.add_parser(name)
        x.add_argument("--work", required=True)
        x.add_argument("--cb", required=True)
        x.add_argument("--thr", type=float, default=0.7, help="cosine distance threshold of the clustering")
        x.add_argument("--half", action="store_true")
    s.choices["run"].add_argument("--part", default="0/1")
    s.choices["run"].add_argument("--dst", default=DST)
    s.choices["run"].add_argument("--shards", default="", help="explicit comma-separated shard list (overrides --part)")
    s.choices["one"].add_argument("--shard", required=True)
    s.choices["one"].add_argument("--max-pieces", type=int, default=0)
    a = p.parse_args()
    {"run": cmd_run, "one": cmd_one}[a.cmd](a)


if __name__ == "__main__":
    main()
