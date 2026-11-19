"""DY-32 track B: one-voice units. Every frame of any speaker is coded by its nearest centroid among centroids fitted
on ONE reference speaker (LJSpeech), so all speech lives in that speaker's feature space; decoding is a table lookup
plus the kNN-VC vocoder (audio in the reference voice).

Features: WavLM-Large layer 6 at 50 Hz exactly as kNN-VC computes them (github.com/bshall/knn-vc, torch.hub
"knn_vc", prematched=True): 16 kHz float waveform in [-1, 1], no waveform normalisation,
``wavlm.extract_features(x, output_layer=6)``. Distance: cosine (kNN-VC matches by cosine distance).
Rate: ``--rate 50`` codes every frame; ``--rate 25`` codes the mean of each pair of frames (2t, 2t+1).
Codebook file (``.pt``): ``cent`` [K, 1024] mean of the raw member features (in pooled space for 25 Hz), ``cent_n``
= cent L2-normalised (used for assignment: argmax cosine), ``dec`` [K, F, 1024] the decode table (F = 1 at 50 Hz;
F = 2 at 25 Hz: mean of the first and of the second frame of member pairs), meta.

    python scripts/dy32b_channel.py fit-lj --out /data/b/cb --ks 500,2000,4000
    python scripts/dy32b_channel.py fit-corpus --out /data/b/cb --ks 2000
    python scripts/dy32b_channel.py check --cb /data/b/cb --out /data/b/check
    python scripts/dy32b_channel.py resynth --cb /data/b/cb/lj_k2000_r25.pt --pack /data/b/pack --out /data/b/rs_lj
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000
LAYER = 6
HOP = 320
LJ = "/opt/dlami/nvme/lj/LJSpeech-1.1"
PACK = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/prompts"


def n_frames(n: int) -> int:
    return max(0, (n - 400) // HOP + 1)


class Channel:
    """WavLM-Large layer 6 (encoder) and the prematched kNN-VC HiFi-GAN (decoder)."""

    def __init__(self, device: str = "cuda") -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True,
                             device=device, verbose=False)
        self.wavlm, self.voc, self.device = knn.wavlm.eval(), knn.hifigan.eval(), torch.device(device)

    @torch.inference_mode()
    def feats(self, wavs: list[np.ndarray], batch: int = 32) -> list[torch.Tensor]:
        """Layer-6 features [T, 1024] (float32, on GPU) of each float32 waveform; batched by length with a padding
        mask (True = pad), each output cropped to its own frame count."""
        order = sorted(range(len(wavs)), key=lambda i: -len(wavs[i]))
        out: list[torch.Tensor | None] = [None] * len(wavs)
        for b in range(0, len(order), batch):
            idx = order[b : b + batch]
            L = len(wavs[idx[0]])
            x = torch.zeros(len(idx), L)
            pad = torch.ones(len(idx), L, dtype=torch.bool)
            for j, i in enumerate(idx):
                x[j, : len(wavs[i])] = torch.from_numpy(wavs[i])
                pad[j, : len(wavs[i])] = False
            x, pad = x.to(self.device), pad.to(self.device)
            f = self.wavlm.extract_features(x, padding_mask=pad if pad.any() else None, output_layer=LAYER,
                                            ret_layer_results=False)[0]
            f = f.float()
            for j, i in enumerate(idx):
                out[i] = f[j, : n_frames(len(wavs[i]))]
        return out

    @torch.inference_mode()
    def vocode(self, f: torch.Tensor) -> np.ndarray:
        """[T, 1024] features at 50 Hz -> int16 16 kHz waveform."""
        y = self.voc(f[None].to(self.device).float()).squeeze().float().cpu().numpy()
        return (np.clip(y, -1, 1) * 32767).astype(np.int16)


def pool(f: torch.Tensor, rate: int) -> torch.Tensor:
    if rate == 50:
        return f
    t = f.shape[0] // 2
    return f[: 2 * t].reshape(t, 2, f.shape[1]).mean(1)


class Codebook:
    def __init__(self, path: str, device: str = "cuda") -> None:
        if path.endswith(".npy"):  # track A centroids (K, 1024): 50 Hz, cosine assignment, decode = centroid lookup
            c = torch.from_numpy(np.load(path).astype(np.float32)).to(device)
            z = {"cent_n": F.normalize(c, dim=-1), "dec": c[:, None], "rate": 50, "meta": {"source": path}}
        else:
            z = torch.load(path, map_location=device)
        self.cent_n, self.dec, self.rate, self.meta = z["cent_n"], z["dec"], z["rate"], z["meta"]
        self.K = self.cent_n.shape[0]

    @torch.inference_mode()
    def encode(self, f50: torch.Tensor) -> torch.Tensor:
        """50 Hz layer-6 features [T, 1024] -> codes [T'] (T' = T at 50 Hz, T // 2 at 25 Hz), argmax cosine."""
        x = F.normalize(pool(f50, self.rate), dim=-1)
        return torch.cat([(c @ self.cent_n.T).argmax(-1) for c in x.split(65536)]) if len(x) else x.new_zeros(0, dtype=torch.long)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes -> 50 Hz features for the vocoder."""
        return self.dec[codes.long()].flatten(0, 1)


def kmeans(x: torch.Tensor, K: int, iters: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical-assignment k-means: assign by cosine, centroid = mean of raw members. Returns (cent raw, labels)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    xn = F.normalize(x, dim=-1)
    c = x[torch.randperm(len(x), generator=g)[:K].to(x.device)].clone()
    for it in range(iters):
        cn = F.normalize(c, dim=-1).half()
        lab = torch.cat([(p.half() @ cn.T).argmax(-1) for p in xn.split(32768)])
        cnt = torch.bincount(lab, minlength=K).float()
        s = torch.zeros_like(c).index_add_(0, lab, x)
        empty = cnt == 0
        c = s / cnt.clamp(min=1)[:, None]
        if empty.any():
            c[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=g).to(x.device)]
        if it % 5 == 0 or it == iters - 1:
            print(json.dumps({"K": K, "iter": it, "empty": int(empty.sum())}), flush=True)
    lab = torch.cat([(p.half() @ F.normalize(c, dim=-1).half().T).argmax(-1) for p in xn.split(32768)])
    return c, lab


def build(f50s: list[torch.Tensor], K: int, rate: int, iters: int, meta: dict) -> dict:
    if rate == 50:
        x = torch.cat(f50s)
        pairs = x[:, None]
    else:
        pairs = torch.cat([f[: 2 * (f.shape[0] // 2)].view(-1, 2, f.shape[1]) for f in f50s])
        x = pairs.mean(1)
    cent, lab = kmeans(x, K, iters)
    cnt = torch.bincount(lab, minlength=K).float().clamp(min=1)
    dec = torch.zeros(K, pairs.shape[1], x.shape[1], device=x.device).index_add_(0, lab, pairs) / cnt[:, None, None]
    zero = torch.bincount(lab, minlength=K) == 0
    dec[zero] = cent[zero][:, None]
    return {"cent": cent.cpu(), "cent_n": F.normalize(cent, dim=-1).cpu(), "dec": dec.cpu(), "rate": rate,
            "meta": {**meta, "K": K, "rate_hz": rate, "frames": int(len(x)), "iters": iters}}


def lj_feats(ch: Channel, hours: float, seed: int = 0) -> list[torch.Tensor]:
    import soundfile as sf
    import torchaudio

    files = sorted(glob.glob(f"{LJ}/wavs/*.wav"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    wavs, tot = [], 0.0
    for f in files:
        w, sr = sf.read(f, dtype="float32")
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
        wavs.append(w)
        tot += len(w) / SR
        if tot >= hours * 3600:
            break
    print(json.dumps({"lj_clips": len(wavs), "hours": round(tot / 3600, 2)}), flush=True)
    return ch.feats(wavs, batch=64)


def corpus_wavs(hours: float, seed: int = 0, n_shards: int = 60) -> list[np.ndarray]:
    """A multi-speaker sample of training (never held-out) DuplexChat pieces: up to 40 s from pieces of n_shards
    random training shards."""
    import tempfile

    from dy21_train import held_out
    from dy30_eval import candidates, read_pieces
    from dy30_units import SRC, listing

    rng = np.random.default_rng(seed)
    tr = [s for s in listing(SRC, dirs=True) if not held_out(s)]
    shards = list(rng.choice(tr, n_shards, replace=False))
    with tempfile.TemporaryDirectory() as tmp:
        cand = candidates(shards, 5, Path(tmp))
        rng.shuffle(cand)
        need, sel = hours * 3600, []
        for s, p, st, ln in cand:
            ln = min(ln, 40 * SR)
            sel.append((s, p, st, ln))
            need -= ln / SR
            if need <= 0:
                break
        w = read_pieces(sel, Path(tmp))
    print(json.dumps({"corpus_pieces": len(w), "shards": n_shards, "hours": round(sum(map(len, w)) / SR / 3600, 2)}), flush=True)
    return [x.astype(np.float32) / 32768.0 for x in w]


def cmd_fit(a, src: str) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ch = Channel()
    t0 = time.time()
    if src == "lj":
        f = lj_feats(ch, a.hours)
        meta = {"source": "LJSpeech-1.1 (one speaker), random clips seed 0, resampled 22050->16000"}
    else:
        f = []
        for w in corpus_wavs(a.hours):
            f += ch.feats([w[i : i + 20 * SR] for i in range(0, len(w), 20 * SR)])
        meta = {"source": "DuplexChat en-16k training shards (multi-speaker), 60 shards seed 0"}
    print(json.dumps({"features_s": round(time.time() - t0, 1), "frames": sum(len(x) for x in f)}), flush=True)
    del ch
    torch.cuda.empty_cache()
    meta.update({"encoder": "WavLM-Large layer 6 (kNN-VC hub checkpoint), raw waveform, no normalisation",
                 "distance": "cosine (argmax of normalised dot product)", "hours": a.hours})
    for K in map(int, a.ks.split(",")):
        for r in map(int, a.rates.split(",")):
            cb = build(f, K, r, a.iters, meta)
            torch.save(cb, out / f"{src}_k{K}_r{r}.pt")
            print("SAVED", out / f"{src}_k{K}_r{r}.pt", flush=True)


def heldout_pieces(n: int) -> list[tuple[str, np.ndarray]]:
    """Fallback check set when the prompt pack is not there: 10 s (30-40 s) of held-out pieces >= 45 s, seed 1234."""
    import tempfile

    from dy21_train import held_out
    from dy30_eval import candidates, read_pieces
    from dy30_units import SRC, listing

    with tempfile.TemporaryDirectory() as tmp:
        cand = candidates([s for s in listing(SRC, dirs=True) if held_out(s)], 45, Path(tmp))
        sel = [cand[i] for i in np.random.default_rng(1234).permutation(len(cand))[:n]]
        w = read_pieces(sel, Path(tmp))
    return [(f"h{j:02d}", x[30 * SR : 40 * SR].astype(np.float32) / 32768.0) for j, x in enumerate(w)]


def load_pack(d: Path, n: int = 0) -> list[tuple[str, np.ndarray]]:
    import soundfile as sf

    if n and not glob.glob(f"{d}/real_*.wav"):
        print("NO PACK: using held-out pieces", flush=True)
        return heldout_pieces(n)
    return [(Path(f).stem.split("_")[1], sf.read(f, dtype="float32")[0]) for f in sorted(glob.glob(f"{d}/real_*.wav"))]


def whisper(audio: list[np.ndarray]) -> list[str]:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cuda",
                   torch_dtype=torch.float16, chunk_length_s=30)
    res = asr([{"raw": w.astype(np.float32), "sampling_rate": SR} for w in audio], batch_size=16,
              generate_kwargs={"language": "en", "task": "transcribe"})
    return [r["text"].strip() for r in res]


def wer(ref: str, hyp: str) -> float:
    from dy30_eval import wer as w

    return w(ref, hyp)


def cmd_check(a) -> None:
    """Whisper WER of resynthesised held-out pieces against Whisper on the original, for every codebook in --cb,
    plus the raw-feature vocoder ceiling (no quantisation)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pieces = load_pack(Path(a.pack), a.n)[: a.n]
    ch = Channel()
    f50 = ch.feats([w for _, w in pieces])
    variants = {"raw": [ch.vocode(f) for f in f50]}
    for p in sorted(glob.glob(f"{a.cb}/*.pt")):
        cb = Codebook(p)
        variants[Path(p).stem] = [ch.vocode(cb.decode(cb.encode(f))) for f in f50]
    del ch
    torch.cuda.empty_cache()
    import soundfile as sf

    ref = whisper([w for _, w in pieces])
    res = {}
    for name, ws in variants.items():
        hyp = whisper([w.astype(np.float32) / 32768 for w in ws])
        res[name] = {"median_wer": float(np.nanmedian([wer(r, h) for r, h in zip(ref, hyp)])),
                     "mean_wer": float(np.nanmean([wer(r, h) for r, h in zip(ref, hyp)])),
                     "examples": [{"ref": r, "hyp": h} for r, h in list(zip(ref, hyp))[:3]]}
        print("CHECK", name, json.dumps({k: round(v, 3) for k, v in res[name].items() if k != "examples"}), flush=True)
        for (pid, _), w in list(zip(pieces, ws))[:4]:
            sf.write(out / f"{name}_{pid}.wav", w, SR)
    (out / "check.json").write_text(json.dumps({"n": len(pieces), "results": res}, indent=2))


def cmd_resynth(a) -> None:
    import soundfile as sf

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ch, cb = Channel(), Codebook(a.cb)
    pieces = load_pack(Path(a.pack))
    for (pid, _), f in zip(pieces, ch.feats([w for _, w in pieces])):
        sf.write(out / f"{pid}.wav", ch.vocode(cb.decode(cb.encode(f))), SR)
    print("RESYNTH DONE", len(pieces), flush=True)


def cmd_export(a) -> None:
    """The same codebook as .npz (float32 cent, cent_n, dec; rate) for users without torch."""
    z = torch.load(a.cb, map_location="cpu")
    np.savez(a.out, cent=z["cent"].numpy(), cent_n=z["cent_n"].numpy(), dec=z["dec"].numpy(), rate=z["rate"],
             meta=json.dumps(z["meta"]))
    print("EXPORTED", a.out, json.dumps(z["meta"]), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit-lj", "fit-corpus"):
        x = s.add_parser(name)
        x.add_argument("--out", required=True)
        x.add_argument("--ks", default="2000")
        x.add_argument("--rates", default="50,25")
        x.add_argument("--hours", type=float, default=6.0)
        x.add_argument("--iters", type=int, default=25)
    x = s.add_parser("check")
    x.add_argument("--cb", required=True)
    x.add_argument("--pack", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--n", type=int, default=30)
    x = s.add_parser("resynth")
    x.add_argument("--cb", required=True)
    x.add_argument("--pack", required=True)
    x.add_argument("--out", required=True)
    x = s.add_parser("export")
    x.add_argument("--cb", required=True)
    x.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "fit-lj":
        cmd_fit(a, "lj")
    elif a.cmd == "fit-corpus":
        cmd_fit(a, "corpus")
    else:
        {"check": cmd_check, "resynth": cmd_resynth, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
