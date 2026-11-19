"""DY-32 track A, job 2: channel lab. Do the words of real conversational speech survive units -> decoder -> audio?

For every channel: real_XX.wav (pack from dy32_prompts) -> units -> decoder -> 16 kHz audio -> recogniser; WER/CER
against the same recogniser's transcript of the original real_XX.wav. Recognisers (measurement only): the letters
CTC model of dy31 (facebook/wav2vec2-large-960h-lv60-self, greedy) and Whisper large-v3-turbo (second opinion).

Channels built here (all textless; features = kNN-VC's WavLM-Large layer 6, 50 Hz, its prematched HiFi-GAN decodes):
  knn-cont            continuous features straight to the vocoder (decoder ceiling)
  km-corpus-K         k-means on ~6 h of DuplexChat training shards, decode by centroid lookup
  km-lj-K             k-means on LJSpeech only (one canonical voice); every speaker tokenised by nearest centroid
  knnvc-lj-k4         kNN-VC proper: each frame -> mean of its 4 nearest LJSpeech frames (cosine), no quantisation
  km25-{corpus,lj}-K  25 Hz: pairs of frames concatenated (2048-d) before k-means; a unit decodes into 2 frames
  rvq-corpus-L{2,4}   residual k-means (K=1024 per level, L2), decode by summing centroids
  mimi-q{1,2,4,8}     Mimi codec, first n codebooks (reference)
Assignment for single-level k-means is by cosine (as kNN-VC matching); centroids are means of raw features.
Also measures any extra folder of XX.wav given by --measure name=dir.

    python scripts/dy32_channel.py lab --stage all
    python scripts/dy32_channel.py measure --dirs b=/data/dy32/ext/b ...
"""

import argparse
import glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from dy30_eval import ASR, SR, SRC, candidates, norm_text, read_pieces

S3 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32"
W = Path("/data/dy32/a")
NV = Path("/opt/dlami/nvme/dy32a")
PACK = Path("/data/dy32/prompts")
LJ = Path("/opt/dlami/nvme/lj/LJSpeech-1.1/wavs")
WIN = 20 * SR
KS = (500, 2000, 8000)
SURVIVE, IDEA_GAP = 0.30, 0.05
REF_ROWS = [("old Sylber k-means (DY-31 era, reference)", 0.72), ("continuous Sylber features (reference)", 0.53)]


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def reals() -> list[np.ndarray]:
    fs = sorted(PACK.glob("real_*.wav"))
    assert len(fs) == 50, len(fs)
    return [sf.read(f, dtype="float32")[0] for f in fs]


def write(d: Path, i: int, w: np.ndarray, sr: int = SR) -> None:
    d.mkdir(parents=True, exist_ok=True)
    w = np.asarray(w, np.float32)
    m = float(np.abs(w).max()) if len(w) else 0.0
    if m > 0.99:
        w = w * (0.95 / m)
    sf.write(d / f"{i:02d}.wav", w, sr, subtype="PCM_16")


# ---------------------------------------------------------------- recognisers + scoring

class Rec:
    def __init__(self, device, whisper: bool = True) -> None:
        import torch
        from transformers import pipeline

        from dy31_units import Letters, Text

        self.letters, self.text = Letters(device), Text()
        self.asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16,
                            chunk_length_s=30) if whisper else None

    def __call__(self, waves: list[np.ndarray]) -> dict:
        lt = [self.text(self.letters(w.astype(np.float32))) for w in waves]
        wt = [""] * len(waves)
        if self.asr is not None:
            idx = [i for i, w in enumerate(waves) if len(w) >= SR // 10]
            res = self.asr([{"raw": waves[i].astype(np.float32), "sampling_rate": SR} for i in idx], batch_size=16,
                           generate_kwargs={"language": "en", "task": "transcribe"})
            for i, r in zip(idx, res):
                wt[i] = r["text"].strip()
        return {"letters": lt, "whisper": wt}


def err(ref: str, hyp: str, kind: str) -> float:
    import jiwer

    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return float("nan")
    if not h:
        return 1.0
    return float(jiwer.wer(r, h) if kind == "wer" else jiwer.cer(r, h))


def score(ref: dict, hyp: dict) -> dict:
    out = {}
    for rk in ("letters", "whisper"):
        for kind in ("wer", "cer"):
            v = np.array([err(r, h, kind) for r, h in zip(ref[rk], hyp[rk])])
            out[f"{rk}_{kind}"] = {"median": round(float(np.nanmedian(v)), 4), "mean": round(float(np.nanmean(v)), 4),
                                   "per_piece": [None if not np.isfinite(x) else round(float(x), 4) for x in v]}
    return out


def load_wavs(d: Path) -> list[np.ndarray] | None:
    import torch
    import torchaudio

    out = []
    for i in range(50):
        f = d / f"{i:02d}.wav"
        if not f.exists():
            return None
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        if sr != SR:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        out.append(w)
    return out


class Table:
    def __init__(self) -> None:
        self.f = W / "channel.json"
        self.rows = json.loads(self.f.read_text()) if self.f.exists() else {}

    def add(self, name: str, desc: str, rate: str, sc: dict, texts: dict) -> None:
        if self.f.exists():  # several processes add rows: merge with what is on disk
            self.rows = json.loads(self.f.read_text())
        self.rows[name] = {"desc": desc, "rate": rate, **{k: {x: y for x, y in v.items()} for k, v in sc.items()},
                           "texts_first5": {k: v[:5] for k, v in texts.items()}}
        self.f.write_text(json.dumps(self.rows, indent=1))
        self.md()

    def med(self, n: str, k: str = "letters_wer") -> float:
        return self.rows[n][k]["median"] if n in self.rows else float("nan")

    def md(self) -> None:
        L = ["# DY-32 channel lab (track A): do the words of real conversational speech survive?", "",
             "50 held-out pieces (pack dy32/prompts, real_XX.wav, 10 s each). Each channel: audio -> units -> decoder -> "
             "audio -> recogniser; error vs the same recogniser on the original audio. Letters = "
             "wav2vec2-large-960h-lv60-self greedy CTC (as dy31); Whisper = large-v3-turbo (second opinion). "
             f"Updated {time.strftime('%Y-%m-%d %H:%M')} UTC.", "",
             "| channel | rate | letters WER median | mean | letters CER median | Whisper WER median | mean | description |",
             "|---|---|---|---|---|---|---|---|"]
        for n, r in sorted(self.rows.items(), key=lambda x: x[1]["letters_wer"]["median"]):
            L.append(f"| {n} | {r['rate']} | {r['letters_wer']['median']:.3f} | {r['letters_wer']['mean']:.3f} | "
                     f"{r['letters_cer']['median']:.3f} | {r['whisper_wer']['median']:.3f} | {r['whisper_wer']['mean']:.3f} | {r['desc']} |")
        L += ["", "Reference values from earlier experiments (letters WER): " + "; ".join(f"{a} {b}" for a, b in REF_ROWS), "",
              "## Pre-registered thresholds (letters WER median; Whisper shown for agreement)", ""]
        surv = [n for n, r in self.rows.items() if r["letters_wer"]["median"] <= SURVIVE and not n.startswith("orig")]
        L.append(f"- Words survive (median WER <= {SURVIVE}): " + (", ".join(sorted(surv)) if surv else "none so far"))
        for K in KS:
            for p in ("km", "km25"):
                a, b = f"{p}-lj-K{K}", f"{p}-corpus-K{K}"
                if a in self.rows and b in self.rows:
                    gap = self.med(b) - self.med(a)
                    gw = self.med(b, "whisper_wer") - self.med(a, "whisper_wer")
                    L.append(f"- Idea 2 at {p} K={K}: corpus {self.med(b):.3f} vs one-voice {self.med(a):.3f}, gain {gap:+.3f} "
                             f"(Whisper {gw:+.3f}) -> {'CONFIRMED' if gap >= IDEA_GAP else 'not confirmed'} (needs >= {IDEA_GAP})")
        nstreams = lambda n: (int(n.split("-L")[-1]) if n.startswith("rvq-") else len(n.split("-L")[-1].split("-"))
                              if n.startswith("sb-") else 1)
        singles = [n for n in self.rows if n.startswith(("km-", "km25-", "rvq-", "sb-")) and nstreams(n) == 1]
        multi = [n for n in self.rows if n.startswith(("rvq-", "sb-")) and nstreams(n) > 1]
        if singles and multi:
            bs = min(singles, key=self.med)
            for m in sorted(multi):
                gap = self.med(bs) - self.med(m)
                L.append(f"- Idea 1, {m} vs best single stream {bs} ({self.med(bs):.3f}): {self.med(m):.3f}, gain {gap:+.3f} -> "
                         f"{'CONFIRMED' if gap >= IDEA_GAP else 'not confirmed'} (needs >= {IDEA_GAP})")
        L += ["", "Notes: Whisper means are dominated by a few pieces where Whisper's text of the original is near empty or "
              "hallucinated (WER > 1 there); read medians. Corpus k-means = 6.6 h of 48 DuplexChat training shards (same "
              "domain as the test pieces); one-voice k-means = all of LJSpeech (read speech, 24 h). Idea 1 compares every "
              "multi-stream row with the best single-stream row (km, km25, rvq-L1, single-layer sb)."]
        (W / "channel.md").write_text("\n".join(L) + "\n")
        try:
            sh(f"aws s3 cp {W}/channel.md {S3}/a/channel.md --only-show-errors && aws s3 cp {W}/channel.json {S3}/a/channel.json --only-show-errors")
        except subprocess.CalledProcessError as e:
            log("upload failed", e)


# ---------------------------------------------------------------- features (kNN-VC WavLM-Large layer 6)

class KNN:
    def __init__(self, device) -> None:
        import torch

        self.torch, self.device = torch, device
        self.m = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=str(device))

    def feats(self, w: np.ndarray):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(w), WIN):
                x = w[i : i + WIN]
                if len(x) < 400:
                    continue
                out.append(self.m.get_features(torch.from_numpy(x.astype(np.float32))[None]).half())
        return torch.cat(out) if out else torch.zeros(0, 1024, device=self.device, dtype=torch.half)

    def vocode(self, f) -> np.ndarray:
        with self.torch.inference_mode():
            return self.m.vocode(f[None].float().to(self.device)).squeeze().cpu().numpy()


def corpus_feats(knn: KNN, hours: float, f: Path):
    import torch

    from dy21_train import held_out

    if f.exists():
        return torch.from_numpy(np.load(f)).to(knn.device)
    shards = sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines() if l.strip().startswith("PRE"))
    tr = [s for s in shards if not held_out(s)]
    rng = np.random.default_rng(0)
    pick = [tr[i] for i in rng.choice(len(tr), 48, replace=False)]
    tmp = NV / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    per = hours * 3600 * SR / len(pick)
    sel = []
    for s in pick:
        c = candidates([s], 4, tmp)
        got = 0
        for j in rng.permutation(len(c)):
            if got >= per:
                break
            sel.append(c[j])
            got += c[j][3]
    log("corpus pieces", len(sel), "hours", round(sum(x[3] for x in sel) / SR / 3600, 2), "shards", len(pick))
    out = []
    for i in range(0, len(sel), 256):
        for w in read_pieces(sel[i : i + 256], tmp):
            out.append(knn.feats(w.astype(np.float32) / 32768.0))
    x = torch.cat(out)
    np.save(f, x.cpu().numpy())
    (NV / "corpus_shards.json").write_text(json.dumps(pick))
    return x


def lj_feats(knn: KNN, f: Path):
    import torch
    import torchaudio

    if f.exists():
        return torch.from_numpy(np.load(f)).to(knn.device)
    out = []
    for p in sorted(LJ.glob("*.wav")):
        w, sr = sf.read(p, dtype="float32")
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        out.append(knn.feats(w))
    x = torch.cat(out)
    np.save(f, x.cpu().numpy())
    return x


# ---------------------------------------------------------------- k-means (GPU)

def assign(x, c, cos: bool, chunk: int = 131072):
    import torch
    import torch.nn.functional as F

    out = []
    cn = F.normalize(c.float(), dim=-1) if cos else c.float()
    c2 = (c.float() ** 2).sum(-1)
    for i in range(0, len(x), chunk):
        xx = x[i : i + chunk].float()
        if cos:
            out.append((F.normalize(xx, dim=-1) @ cn.T).argmax(-1))
        else:
            out.append((c2[None] - 2 * xx @ cn.T).argmin(-1))
    return torch.cat(out)


def kmeans(x, k: int, cos: bool, iters: int = 25, seed: int = 0):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].float().clone()
    for it in range(iters):
        a = assign(x, c, cos)
        s = torch.zeros_like(c)
        n = torch.zeros(k, device=x.device)
        for i in range(0, len(x), 262144):
            s.index_add_(0, a[i : i + 262144], x[i : i + 262144].float())
        n.index_add_(0, a, torch.ones(len(a), device=x.device))
        empty = n == 0
        c = torch.where(empty[:, None], x[torch.randint(0, len(x), (k,), generator=g, device=x.device)].float(), s / n.clamp(min=1)[:, None])
    used = int((n > 0).sum())
    return c, used


def pairs(f):
    import torch

    if len(f) % 2:
        f = torch.cat([f, f[-1:]])
    return f.reshape(len(f) // 2, -1)


# ---------------------------------------------------------------- lab

def lab(a) -> None:
    import torch

    device = torch.device("cuda")
    for d in (W, NV):
        d.mkdir(parents=True, exist_ok=True)
    R = reals()
    rec = Rec(device)
    tab = Table()
    rf = W / "ref.json"
    if rf.exists():
        ref = json.loads(rf.read_text())
    else:
        ref = rec(R)
        rf.write_text(json.dumps(ref, indent=1))
    log("reference transcripts ready; first:", ref["letters"][0][:120])

    def measure(name: str, desc: str, rate: str, waves: list[np.ndarray], save: bool = True) -> None:
        if save:
            for i, w in enumerate(waves):
                write(W / "resynth" / name, i, w)
        h = rec(waves)
        sc = score(ref, h)
        tab.add(name, desc, rate, sc, h)
        log("ROW", name, json.dumps({k: {"median": v["median"], "mean": v["mean"]} for k, v in sc.items()}))
        if save:
            try:
                sh(f"aws s3 sync {W}/resynth/{name}/ {S3}/a/resynth/{name}/ --only-show-errors")
            except subprocess.CalledProcessError as e:
                log("sync failed", e)

    if "orig-16k" not in tab.rows:
        measure("orig-16k", "original audio (sanity: should be ~0)", "-", R, save=False)

    knn = KNN(device)
    F = [knn.feats(w) for w in R]
    log("real feats", [tuple(f.shape) for f in F[:3]])
    st = set(a.stage.split(","))
    allst = "all" in st

    def todo(n):
        return n not in tab.rows or a.redo

    if (allst or "cont" in st) and todo("knn-cont"):
        measure("knn-cont", "WavLM-Large L6 continuous -> kNN-VC prematched HiFi-GAN (decoder ceiling)", "50 Hz x 1024-d float",
                [knn.vocode(f) for f in F])

    if (allst or "mimi" in st):
        mimi(device, R, measure, todo)

    if not (allst or st & {"lj", "corpus", "km25", "rvq", "knnvc"}):
        return
    lj = lj_feats(knn, NV / "lj_l6.npy")
    log("LJ frames", len(lj))
    if (allst or "knnvc" in st) and todo("knnvc-lj-k4"):
        waves = []
        ljn = torch.nn.functional.normalize(lj.float(), dim=-1).half()
        for f in F:
            q = torch.nn.functional.normalize(f.float(), dim=-1).half()
            best_s, best_i = None, None
            for i in range(0, len(lj), 1_000_000):
                s = q @ ljn[i : i + 1_000_000].T
                v, ix = s.float().topk(4, dim=-1)
                ix = ix + i
                if best_s is None:
                    best_s, best_i = v, ix
                else:
                    cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, ix], 1)
                    best_s, o = cs.topk(4, dim=-1)
                    best_i = ci.gather(1, o)
            waves.append(knn.vocode(lj[best_i].float().mean(1)))
        del ljn
        measure("knnvc-lj-k4", "kNN-VC proper: each frame -> mean of 4 nearest LJSpeech frames (cosine), no quantisation",
                "50 Hz continuous", waves)

    corp = corpus_feats(knn, a.hours, NV / "corpus_l6.npy") if (allst or st & {"corpus", "km25", "rvq"}) else None
    if corp is not None:
        log("corpus frames", len(corp))
    cdir = W / "centroids"
    cdir.mkdir(exist_ok=True)

    def km_row(src, tag, K, p25: bool) -> None:
        name = f"{'km25' if p25 else 'km'}-{tag}-K{K}"
        if not todo(name):
            return
        t0 = time.time()
        X = pairs(src) if p25 else src
        c, used = kmeans(X, K, cos=True)
        np.save(cdir / f"{name}.npy", c.half().cpu().numpy())
        waves = []
        for f in F:
            q = pairs(f) if p25 else f
            u = assign(q, c, cos=True)
            d = c[u]
            if p25:
                d = d.reshape(-1, 1024)[: len(f)]
            waves.append(knn.vocode(d))
        who = "LJSpeech only (one voice)" if tag == "lj" else f"{a.hours:g} h DuplexChat training shards"
        measure(name, f"k-means K={K} ({used} used) on {who}; cosine assignment; centroid lookup -> HiFi-GAN"
                + ("; 25 Hz = frame pairs concatenated" if p25 else ""),
                f"{25 if p25 else 50} Hz x log2 {K} = {(25 if p25 else 50) * math.log2(K):.0f} bit/s", waves)
        log(name, "seconds", round(time.time() - t0))

    for K in KS:
        if allst or "lj" in st:
            km_row(lj, "lj", K, False)
        if corp is not None and (allst or "corpus" in st):
            km_row(corp, "corpus", K, False)
    if allst or "km25" in st:
        for K in KS:
            km_row(lj, "lj", K, True)
            if corp is not None:
                km_row(corp, "corpus", K, True)
    if corp is not None and (allst or "rvq" in st) and (todo("rvq-corpus-L4") or todo("rvq-corpus-L2")):
        r = corp.float()
        books = []
        for lvl in range(4):
            c, _ = kmeans(r, 1024, cos=False, iters=20, seed=lvl)
            books.append(c)
            r = r - c[assign(r, c, cos=False)]
        np.save(cdir / "rvq-corpus-1024x4.npy", torch.stack(books).half().cpu().numpy())
        for L in (1, 2, 4):
            waves = []
            for f in F:
                rr, d = f.float(), torch.zeros_like(f.float())
                for c in books[:L]:
                    q = c[assign(rr, c, cos=False)]
                    d, rr = d + q, rr - q
                waves.append(knn.vocode(d))
            measure(f"rvq-corpus-L{L}", f"residual k-means on corpus L6 features, {L} level(s) x K=1024 (L2), decode by summed centroids",
                    f"50 Hz x {L} x 10 bit = {500 * L} bit/s", waves)
    try:
        sh(f"aws s3 sync {cdir}/ {S3}/a/centroids/ --only-show-errors")
    except subprocess.CalledProcessError as e:
        log("centroid sync failed", e)
    log("LAB DONE")


def mimi(device, R, measure, todo) -> None:
    import torch
    import torchaudio
    from transformers import MimiModel

    m = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    for n in (1, 2, 4, 8):
        name = f"mimi-q{n}"
        if not todo(name):
            continue
        waves = []
        with torch.no_grad():
            for w in R:
                x = torchaudio.functional.resample(torch.from_numpy(w), SR, 24000)[None, None].to(device)
                codes = m.encode(x, num_quantizers=n).audio_codes
                y = m.decode(codes).audio_values[0, 0].float().cpu()
                waves.append(torchaudio.functional.resample(y, 24000, SR).numpy())
        measure(name, f"Mimi codec (kyutai/mimi), first {n} codebook(s) of 8 (reference)", f"12.5 Hz x {n} x 11 bit = {137.5 * n:.0f} bit/s", waves)
    del m


def measure_dirs(a) -> None:
    import torch

    device = torch.device("cuda")
    ref = json.loads((W / "ref.json").read_text())
    rec = Rec(device)
    tab = Table()
    for spec in a.dirs:
        name, d = spec.split("=", 1)
        waves = load_wavs(Path(d))
        if waves is None:
            log("incomplete folder, skipped", name)
            continue
        desc = a.desc or f"track output {d}"
        info = Path(d) / "info.json"
        if info.exists():
            try:
                j = json.loads(info.read_text())
                desc = str(j.get("desc") or j.get("description") or j.get("units") or j)[:200]
            except Exception:
                pass
        h = rec(waves)
        sc = score(ref, h)
        tab.add(name, desc, a.rate, sc, h)
        log("ROW", name, json.dumps({k: {"median": v["median"], "mean": v["mean"]} for k, v in sc.items()}))


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("lab")
    x.add_argument("--stage", default="all", help="all or comma list of cont,mimi,knnvc,lj,corpus,km25,rvq")
    x.add_argument("--hours", type=float, default=6.0)
    x.add_argument("--redo", action="store_true")
    x = s.add_parser("measure")
    x.add_argument("--dirs", nargs="+", required=True, help="name=folder with 00.wav..49.wav")
    x.add_argument("--desc", default="")
    x.add_argument("--rate", default="?")
    a = p.parse_args()
    {"lab": lab, "measure": measure_dirs}[a.cmd](a)


if __name__ == "__main__":
    main()
