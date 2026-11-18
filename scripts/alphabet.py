"""DY-24: a textless phone-level alphabet, learned from TTS pairs "same sentence, other voice".

Frozen Sylber layer-9 frames (50 Hz) -> small encoder (2 transformer layers, stride 2 -> 25 Hz, 256-d) -> K codes.
Pairs come from S3 outputs/dauys/tts-alphabet (each sentence in 4 cloned voices; text is never used here, only the
grouping). Frames of two voices are aligned by DTW on the frozen features; the loss is SPIN-style swapped
prediction: each voice predicts the other's balanced (Sinkhorn) code assignment at the aligned frame, so codes
must carry what the voices share (the sounds) and not who speaks. Tokens = argmax codes, consecutive repeats merged.

    python scripts/alphabet.py feats --files 8 --out /data/alpha/pairs.pt
    python scripts/alphabet.py train --pairs /data/alpha/pairs.pt --out /data/alpha/run --k 256
    python scripts/alphabet.py eval --run /data/alpha/run --zr /data/bench/zr --out /data/alpha/eval.json
"""

import argparse
import csv
import io
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
from scipy.signal import resample_poly
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import to_int16
from units_diag import edit

from dauys.dy14_units import SR, hidden_states, load_sylber

TTS = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/tts-alphabet"
SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


@numba.njit(cache=True)
def dtw_map(c: np.ndarray) -> np.ndarray:
    """DTW over cost c [n, m]; for each row i the matched column (middle of its run on the path)."""
    n, m = c.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = c[i - 1, j - 1] + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    lo = np.full(n, m, np.int64)
    hi = np.zeros(n, np.int64)
    i, j = n, m
    while i > 0 and j > 0:
        lo[i - 1] = min(lo[i - 1], j - 1)
        hi[i - 1] = max(hi[i - 1], j - 1)
        a, b, e = d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]
        if a <= b and a <= e:
            i, j = i - 1, j - 1
        elif b <= e:
            i -= 1
        else:
            j -= 1
    return (lo + hi) // 2


# ---------------------------------------------------------------- features + pairs
def feats(args) -> None:
    device = torch.device("cuda")
    syl = load_sylber(device)
    files = [l.split()[-1] for l in sh(f"aws s3 ls --recursive {TTS}/").splitlines() if l.endswith(".npz")]
    files = sorted(files, key=lambda f: (int(Path(f).stem), f))[: args.files]  # first files of every part
    out_f, groups, voices = [], [], []
    t0 = time.time()
    for f in files:
        b = sh(f"aws s3 cp s3://manifestro-dataset-876108973178-us-east-1-an/{f} /tmp/a.npz --only-show-errors")
        z = np.load("/tmp/a.npz")
        meta = [json.loads(l) for l in str(z["meta"]).splitlines()]
        by_text = defaultdict(list)
        for i, m in enumerate(meta):
            by_text[m["text_id"]].append(i)
        for tid, idx in by_text.items():
            if len(idx) < 2:
                continue
            g = []
            for i in idx:
                s, n = z["index"][i]
                st = hidden_states(syl, z["audio"][s : s + n])
                out_f.append(torch.from_numpy(st.astype(np.float16)))
                voices.append(meta[i]["voice"])
                g.append(len(out_f) - 1)
            groups.append(g)
        print(json.dumps({"file": f, "clips": len(out_f), "groups": len(groups), "hours": round(
            sum(len(x) for x in out_f) / 50 / 3600, 2), "s": round(time.time() - t0)}), flush=True)
    align = {}
    for g in groups:
        nf = [F.normalize(out_f[i].float(), dim=1).numpy() for i in g]
        for a in range(len(g)):
            for b in range(len(g)):
                if a != b:
                    align[(g[a], g[b])] = torch.from_numpy(dtw_map((1 - nf[a] @ nf[b].T).astype(np.float64)).astype(np.int32))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": out_f, "groups": groups, "voices": voices, "align": align}, args.out)
    print("FEATS DONE", len(out_f), "clips", len(groups), "groups", len(align), "pairs", flush=True)


# ---------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, k: int, d: int = 512, z: int = 256) -> None:
        super().__init__()
        self.inp = nn.Linear(768, d)
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 1024, 0.1, batch_first=True, norm_first=True), 2)
        self.down = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.out = nn.Linear(d, z)
        self.codes = nn.Parameter(F.normalize(torch.randn(k, z), dim=1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, 768] frames at 50 Hz -> logits [B, ceil(T/2), K] (cosine / tau)."""
        h = self.tr(self.inp(x), src_key_padding_mask=None if mask is None else ~mask)
        h = self.down(h.transpose(1, 2)).transpose(1, 2)
        z = F.normalize(self.out(h), dim=-1)
        return z @ F.normalize(self.codes, dim=1).T / 0.1

    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> np.ndarray:
        c = self.forward(x[None])[0].argmax(-1).cpu().numpy()
        return c[np.r_[True, c[1:] != c[:-1]]] if len(c) else c


@torch.no_grad()
def sinkhorn(logits: torch.Tensor, iters: int = 3, eps: float = 0.05) -> torch.Tensor:
    q = torch.exp((logits - logits.max()) * 0.1 / eps).T  # [K, N]
    q /= q.sum()
    k, n = q.shape
    for _ in range(iters):
        q /= q.sum(1, keepdim=True)
        q /= k
        q /= q.sum(0, keepdim=True)
        q /= n
    return (q * n).T


def train(args) -> None:
    device = torch.device("cuda")
    data = torch.load(args.pairs)
    feats_, align = data["feats"], data["align"]
    pairs = list(align.keys())
    print("clips", len(feats_), "pairs", len(pairs), flush=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = Encoder(args.k).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        sel = [pairs[i] for i in rng.integers(0, len(pairs), args.batch)]
        clips = sorted({c for p in sel for c in p})
        pos = {c: i for i, c in enumerate(clips)}
        T = max(len(feats_[c]) for c in clips)
        x = torch.zeros(len(clips), T, 768)
        mask = torch.zeros(len(clips), T, dtype=torch.bool)
        for c, i in pos.items():
            x[i, : len(feats_[c])] = feats_[c].float()
            mask[i, : len(feats_[c])] = True
        x, mask = x.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x, mask).float()  # [C, T2, K]
        m2 = mask[:, ::2]
        q = torch.zeros_like(lg)
        q[m2] = sinkhorn(lg[m2])
        ia, ta, ib, tb = [], [], [], []
        for a, b in sel:
            al = align[(a, b)].numpy()
            t = np.arange(0, len(al), 2)
            ia.append(np.full(len(t), pos[a]))
            ta.append(t // 2)
            ib.append(np.full(len(t), pos[b]))
            tb.append(np.minimum(al[t] // 2, (len(feats_[b]) - 1) // 2))
        ia, ta, ib, tb = (torch.from_numpy(np.concatenate(v)).to(device) for v in (ia, ta, ib, tb))
        loss = -(q[ia, ta] * F.log_softmax(lg[ib, tb], -1)).sum(-1).mean() - (q[ib, tb] * F.log_softmax(lg[ia, ta], -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                agree = (lg[ia, ta].argmax(-1) == lg[ib, tb].argmax(-1)).float().mean().item()
                used = lg[m2].argmax(-1).unique().numel()
            print(json.dumps({"step": step + 1, "loss": round(loss.item(), 4), "aligned_agree": round(agree, 4),
                              "codes_used": used, "s": round(time.time() - t0)}), flush=True)
    torch.save({"model": m.state_dict(), "k": args.k}, out / "alpha.pt")
    print("TRAIN DONE", flush=True)


# ---------------------------------------------------------------- evaluation
def evaluate(args) -> None:
    device = torch.device("cuda")
    syl = load_sylber(device)
    ck = torch.load(Path(args.run) / "alpha.pt")
    m = Encoder(ck["k"]).to(device).eval()
    m.load_state_dict(ck["model"])

    def tok(w16: np.ndarray) -> np.ndarray:
        w16 = np.pad(w16, (0, max(0, SR // 5 - len(w16))))
        return m.tokens(torch.from_numpy(hidden_states(syl, w16)).float().to(device))

    res = {"k": ck["k"]}
    d = Path(args.zr) / "lexical" / "dev"
    items = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(d / "gold.csv")):
        items[r["id"]][int(r["correct"])][r["voice"]] = d / f"{r['filename']}.wav"
    ids = [i for i, g in items.items() if len(g[0]) == 4 and len(g[1]) == 4]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    ids = ids[:400]
    voices = ["A", "C", "D", "F"]
    seq, dur = {}, 0.0
    for i in ids:
        for c in (0, 1):
            for v in voices:
                w = to_int16(*sf.read(items[i][c][v]))
                seq[(i, c, v)] = tok(w)
                dur += len(w) / SR
    res["swuggy_same_word_identical"] = round(float(np.mean([np.array_equal(seq[(i, 1, a)], seq[(i, 1, b)])
                                                             for i in ids for a in voices for b in voices if a < b])), 4)
    right = n = 0.0
    for i in ids:
        for c in (0, 1):
            for v1 in voices:
                for v2 in voices:
                    if v1 != v2:
                        a, b = edit(seq[(i, c, v1)], seq[(i, c, v2)]), edit(seq[(i, c, v1)], seq[(i, 1 - c, v1)])
                        right += 1.0 if a < b else 0.5 if a == b else 0.0
                        n += 1
    res["swuggy_abx_edit"] = round(right / n, 4)
    res["tokens_per_s_swuggy"] = round(sum(len(s) for s in seq.values()) / dur, 2)
    print("SWUGGY", json.dumps(res), flush=True)

    # real speech: token stability under augmentation, next to the phoneme recogniser on the same clips
    from phon_units import Phonemes

    ph = Phonemes(device)
    shard = sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines() if l.strip().startswith("PRE"))
    from dy21_train import held_out

    s = [x for x in shard if held_out(x)][1]
    for fn in ("index.npy", "segs.bin"):
        sh(f"aws s3 cp {SRC}/{s}/{fn} /tmp/{fn} --only-show-errors")
    idx, segs = np.load("/tmp/index.npy"), np.fromfile("/tmp/segs.bin", dtype=np.int16)
    clips = [segs[st + 2 * SR : st + 12 * SR] for st, ln in idx if ln > 14 * SR][:100]
    arng = np.random.default_rng(1)

    def aug(w16: np.ndarray, kind: str) -> np.ndarray:
        x = w16.astype(np.float32)
        if kind == "noise":
            x = x + arng.standard_normal(len(x)) * x.std() / 10 ** (15 / 20)
        elif kind == "speed":
            x = resample_poly(x, 100, 108) if arng.random() < 0.5 else resample_poly(x, 108, 100)
        elif kind == "lowpass":
            x = resample_poly(resample_poly(x, 1, 2), 2, 1)  # 4 kHz band, like a phone line
        return np.clip(x, -32768, 32767).astype(np.int16)

    for name, fn in (("alpha", tok), ("phonemes", lambda w: ph(w))):
        r = {}
        base = [fn(c) for c in clips]
        r["tokens_per_s"] = round(sum(len(b) for b in base) / (len(clips) * 10), 2)
        for kind in ("noise", "speed", "lowpass"):
            r[f"edit_{kind}"] = round(float(np.mean([edit(np.asarray(b, np.int64), np.asarray(fn(aug(c, kind)), np.int64))
                                                     for b, c in zip(base, clips)])), 4)
        res[f"real_{name}"] = r
        print("REAL", name, json.dumps(r), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("EVAL", json.dumps(res), flush=True)


# ---------------------------------------------------------------- real words, relative to the phoneme oracle
STOP = set("""the and that this with have from they what were there their would about which when will your just like
know yeah really because people think going right some then them than also been into more over only very even well
much here where these those other could should does doing being after before something things thing actually""".split())


def words(args) -> None:
    """~200 content words of held-out podcasts, each cut (Whisper word timestamps, a ruler only) from several different
    pieces (different people); tokens from the alphabet and from the phoneme oracle on the same cuts. Per word: share
    of occurrence pairs with identical tokens and mean normalised edit distance; plus word ABX (X = occurrence of w,
    A = other occurrence of w, B = occurrence of another word of similar length)."""
    from transformers import pipeline

    from dy21_train import held_out
    from phon_units import Phonemes

    device = torch.device("cuda")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=0, torch_dtype=torch.float16)
    shards = [x for x in sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines()
                                if l.strip().startswith("PRE")) if held_out(x)][: args.shards]
    occ = defaultdict(list)  # word -> [(piece key, audio int16)]
    for s in shards:
        for fn in ("index.npy", "segs.bin"):
            sh(f"aws s3 cp {SRC}/{s}/{fn} /tmp/w_{fn} --only-show-errors")
        idx, segs = np.load("/tmp/w_index.npy"), np.fromfile("/tmp/w_segs.bin", dtype=np.int16)
        for p, (st, ln) in enumerate(idx[: args.pieces]):
            if ln < 20 * SR:
                continue
            w = segs[st + 2 * SR : st + 32 * SR]
            r = asr({"raw": w.astype(np.float32) / 32768, "sampling_rate": SR}, return_timestamps="word",
                    generate_kwargs={"language": "en"})
            for ch in r.get("chunks", []):
                t = "".join(c for c in ch["text"].lower() if c.isalpha())
                a, b = ch["timestamp"]
                if len(t) >= 4 and t not in STOP and a is not None and b is not None and 0.2 <= b - a <= 1.2:
                    occ[t].append((s, w[max(0, int((a - 0.03) * SR)) : int((b + 0.03) * SR)]))
        print(json.dumps({"shard": s, "words": len(occ)}), flush=True)
    pick = {}
    for t, lst in occ.items():
        seen, keep = set(), []
        for k, a in lst:
            if k not in seen:
                seen.add(k)
                keep.append(a)
        if len(keep) >= 3:
            pick[t] = keep[:4]
    names = sorted(pick, key=lambda t: -len(pick[t]))[: args.n]
    print("words with >= 3 different pieces:", len(pick), "used", len(names), flush=True)

    syl = load_sylber(device)
    ph = Phonemes(device)
    res = {"words": len(names), "occurrences": sum(len(pick[t]) for t in names)}
    for run in args.runs:
        ck = torch.load(Path(run) / "alpha.pt")
        m = Encoder(ck["k"]).to(device).eval()
        m.load_state_dict(ck["model"])
        res[f"alpha_k{ck['k']}"] = (lambda m: lambda w: m.tokens(torch.from_numpy(hidden_states(syl, np.pad(w, (0, max(0, SR // 5 - len(w)))))).float().to(device)))(m)
    res["phonemes"] = lambda w: ph(w)
    rng = np.random.default_rng(0)
    out = {k: v for k, v in res.items() if not callable(v)}
    for name in [k for k, v in res.items() if callable(v)]:
        fn = res[name]
        tok = {t: [np.asarray(fn(a), np.int64) for a in pick[t]] for t in names}
        same, dist = [], []
        for t in names:
            q = tok[t]
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    same.append(np.array_equal(q[i], q[j]))
                    dist.append(edit(q[i], q[j]))
        right = n = 0.0
        for t in names:
            others = [u for u in names if u != t and abs(len(u) - len(t)) <= 1] or [u for u in names if u != t]
            for i in range(len(tok[t])):
                for j in range(len(tok[t])):
                    if i != j:
                        u = others[rng.integers(len(others))]
                        x, a_, b_ = tok[t][i], tok[t][j], tok[u][rng.integers(len(tok[u]))]
                        da, db = edit(x, a_), edit(x, b_)
                        right += 1.0 if da < db else 0.5 if da == db else 0.0
                        n += 1
        out[name] = {"same_word_identical": round(float(np.mean(same)), 4), "mean_edit": round(float(np.mean(dist)), 4),
                     "word_abx": round(right / n, 4), "tokens_per_word": round(float(np.mean([len(x) for v in tok.values() for x in v])), 2)}
        print("WORDS", name, json.dumps(out[name]), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("WORDS DONE", json.dumps(out), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("feats")
    a.add_argument("--files", type=int, default=8)
    a.add_argument("--out", required=True)
    b = sub.add_parser("train")
    b.add_argument("--pairs", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=256)
    b.add_argument("--steps", type=int, default=6000)
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--lr", type=float, default=5e-4)
    c = sub.add_parser("eval")
    c.add_argument("--run", required=True)
    c.add_argument("--zr", required=True)
    c.add_argument("--out", required=True)
    w = sub.add_parser("words")
    w.add_argument("--runs", nargs="+", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--shards", type=int, default=12)
    w.add_argument("--pieces", type=int, default=80)
    w.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    {"feats": feats, "train": train, "eval": evaluate, "words": words}[args.cmd](args)


if __name__ == "__main__":
    main()
