"""DY-32 track G: the classic textless GSLM pipeline with its released weights, on our conversational audio.

    channel   real_XX.wav -> units -> vocoder (no LM), one folder per channel; quick Whisper WER vs the original
    released  prompt_XX.wav -> units -> released GSLM unit LM continues ~10 s -> vocoder
    layers    which HF HuBERT layer matches a released km.bin (lowest k-means inertia)
    units     unit pass over DuplexChat shards for our own model (dy30_units plumbing, --part k/n)
    gen       our UnitLM (dy21_lm, trained by dy21_train.py --k) continues the prompts -> vocoder

Models live in --models (downloaded by scripts/dy32g_pod.sh); paths below are file names inside it.
Channel names: km100 (HuBERT L6 km100 -> fairseq unit HiFi-GAN LJ, full 50 Hz units), km100-dedup (same, runs
collapsed, vocoder's duration predictor restores timing: the input the released uLM produces), km50 / km200
(released GSLM Tacotron2 + WaveGlow, deduplicated units), L9km500 (HuBERT L9 km500 -> Expresso unit HiFi-GAN,
speaker "lj", style "read-default", full rate). HuBERT L6 km500 has no released decoder.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dauys.gslm import SR, CodeHiFiGAN, FairseqULM, Hubert, KMeans, Tacotron, dedup, encode, sample_top_p

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
OUT = f"{BUCKET}/dauys/dy32"
ASR = "openai/whisper-large-v3-turbo"
TEXTLESSLIB = os.environ.get("TEXTLESSLIB", "/data/textlesslib")

CHANNELS = {  # name: (hubert layer, km file, decoder, dedup)
    "km100": (6, "km100.bin", "hifigan100", False),
    "km100-dedup": (6, "km100.bin", "hifigan100", True),
    "km200": (6, "km200.bin", "tts200", True),
    "km50": (6, "km50.bin", "tts50", True),
    "L9km500": (9, "L9_km500.bin", "expresso500", False),
}


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def read16(path: Path) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    return w if w.ndim == 1 else w.mean(1)


def write16(path: Path, w: np.ndarray) -> None:
    sf.write(path, np.clip(w, -1, 1).astype(np.float32), SR, subtype="PCM_16")


class Decoders:
    """Lazily loaded vocoders by name."""

    def __init__(self, models: Path, device) -> None:
        self.m, self.device, self.cache = models, device, {}

    def get(self, name: str):
        if name not in self.cache:
            m = self.m
            if name == "hifigan100":
                self.cache[name] = CodeHiFiGAN.load(m / "hifigan100.pt", m / "hifigan100_config.json", self.device)
            elif name == "expresso500":
                self.cache[name] = CodeHiFiGAN.load(m / "exp500_generator.pt", m / "exp500_config.json", self.device)
            elif name in ("tts50", "tts100", "tts200"):
                k = name[3:]
                self.cache[name] = Tacotron(TEXTLESSLIB, str(m / f"tts{k}.pt"), str(m / f"tts{k}_code_dict"),
                                            str(m / "waveglow_256channels_standalone.pt"), self.device)
            else:
                raise KeyError(name)
        return self.cache[name]

    def speak(self, name: str, units: np.ndarray, deduped: bool) -> np.ndarray:
        d = self.get(name)
        if name == "hifigan100":
            return d(units, dur_prediction=deduped)
        if name == "expresso500":
            spk = [l.strip() for l in open(self.m / "exp500_speakers.txt")].index("lj")
            sty = [l.strip() for l in open(self.m / "exp500_styles.txt")].index("read-default")
            return d(units, spkr=spk, style=sty)
        return d(units)


def whisper(audio: list[np.ndarray], device) -> list[str]:
    """Plain Whisper generate on <= 30 s clips (no chunking pipeline), batches of 16."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(ASR)
    m = WhisperForConditionalGeneration.from_pretrained(ASR, torch_dtype=torch.float16).to(device).eval()
    out = []
    for b in range(0, len(audio), 16):
        f = proc([w.astype(np.float32)[: 30 * SR] for w in audio[b : b + 16]], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            ids = m.generate(f.input_features.to(device, torch.float16), language="en", task="transcribe")
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    del m
    torch.cuda.empty_cache()
    return out


def norm(t: str) -> str:
    import re

    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split())


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


# ---------------------------------------------------------------- channel check

def cmd_layers(a) -> None:
    """k-means inertia of each HF HuBERT layer against each km.bin: the released layer should be the minimum."""
    dev = torch.device("cuda")
    models = Path(a.models)
    enc = Hubert(12, dev)
    wavs = [read16(Path(a.pack) / f"real_{i:02d}.wav") for i in range(8)]
    feats = []
    for w in wavs:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = enc.model(torch.from_numpy(w.astype(np.float32))[None].to(dev), output_hidden_states=True).hidden_states
        feats.append([h[0].float() for h in hs])
    res = {}
    for kmf in a.km.split(","):
        km = KMeans(str(models / kmf), dev)
        res[kmf] = {L: round(float(np.mean([km.inertia(f[L]) for f in feats])), 2) for L in range(13)}
        print("LAYERS", kmf, json.dumps(res[kmf]), "best", min(res[kmf], key=res[kmf].get), flush=True)


def cmd_channel(a) -> None:
    dev = torch.device("cuda")
    models, pack, work = Path(a.models), Path(a.pack), Path(a.work)
    ids = [f"{i:02d}" for i in range(a.n)]
    reals = [read16(pack / f"real_{i}.wav") for i in ids]
    dec = Decoders(models, dev)
    encs, summary = {}, {}
    ref_t = whisper(reals, dev) if a.wer else None
    for name in a.channels.split(","):
        layer, kmf, decoder, dd = CHANNELS[name]
        if layer not in encs:
            encs[layer] = Hubert(layer, dev)
        km = KMeans(str(models / kmf), dev)
        out = work / f"g-{name}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        audio, rates = [], []
        for i, w in zip(ids, reals):
            u = encode(encs[layer], km, w)
            uu, _ = dedup(u)
            rates.append(len(uu) / (len(w) / SR))
            y = dec.speak(decoder, uu if dd else u, dd)
            write16(out / f"{i}.wav", y)
            audio.append(y)
        info = {"channel": name, "hubert": "facebook/hubert-base-ls960", "layer": layer, "kmeans": kmf, "decoder": decoder,
                "deduplicated_units": dd, "units_per_s_after_dedup": round(float(np.mean(rates)), 1),
                "n": len(ids), "seconds": round(time.time() - t0, 1)}
        if a.wer:
            hyp = whisper(audio, dev)
            ws = [wer(r, h) for r, h in zip(ref_t, hyp)]
            info["quick_wer_vs_whisper_on_original"] = {"median": round(float(np.median(ws)), 3),
                                                        "mean": round(float(np.mean(ws)), 3)}
            info["examples"] = [{"id": i, "original": r, "resynth": h} for i, r, h in list(zip(ids, ref_t, hyp))[:5]]
        (out / "info.json").write_text(json.dumps(info, indent=2))
        summary[name] = info.get("quick_wer_vs_whisper_on_original")
        print("CHANNEL", json.dumps({k: v for k, v in info.items() if k != "examples"}), flush=True)
        if a.upload:
            sh(f"aws s3 sync {out} {OUT}/resynth/g-{name}/ --only-show-errors")
    print("CHANNELS DONE", json.dumps(summary), flush=True)


# ---------------------------------------------------------------- released uLM on our prompts

@torch.no_grad()
def continue_ulm(lm: FairseqULM, voc: CodeHiFiGAN, prompt: np.ndarray, seconds: float, temp: float, top_p: float,
                 seed: int, chunk: int = 32, max_tokens: int = 1500) -> np.ndarray:
    """Deduplicated prompt units -> continuation units, sampled until the vocoder's duration predictor says >= seconds
    (EOS and special symbols banned)."""
    dev = lm.emb.weight.device
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.tensor([FairseqULM.EOS] + [int(x) + FairseqULM.OFF for x in prompt], device=dev)[None]
    n0 = ids.shape[1]
    while ids.shape[1] - n0 < max_tokens:
        for _ in range(chunk):
            lg = lm.logits(ids[:, -3000:])[:, -1]
            lg[:, : FairseqULM.OFF] = -float("inf")
            ids = torch.cat([ids, sample_top_p(lg, temp, top_p, g)[:, None]], 1)
        cont = ids[0, n0:] - FairseqULM.OFF
        if len(voc.frames(cont)) >= seconds * 50:
            break
    return (ids[0, n0:] - FairseqULM.OFF).cpu().numpy()


def cmd_released(a) -> None:
    dev = torch.device("cuda")
    models, pack, work = Path(a.models), Path(a.pack), Path(a.work)
    k = a.k
    enc = Hubert(6, dev)
    km = KMeans(str(models / f"km{k}.bin"), dev)
    lm = FairseqULM.load(str(models / f"lm{k}/hubert{k}_lm/checkpoint_best.pt"), dev)
    voc100 = CodeHiFiGAN.load(models / "hifigan100.pt", models / "hifigan100_config.json", dev)
    dec = Decoders(models, dev)
    out = work / a.variant
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        w = read16(pack / f"prompt_{i:02d}.wav")
        u, _ = dedup(encode(enc, km, w))
        t0 = time.time()
        # the duration predictor of the km100 HiFi-GAN measures length for km100; for km50/km200 the unit rate of
        # the prompt converts seconds into a token budget
        if k == 100:
            cont = continue_ulm(lm, voc100, u, a.seconds, a.temp, a.top_p, a.seed + i)
            y = voc100(cont, dur_prediction=True)
        else:
            n_tok = int(round(len(u) / (len(w) / SR) * a.seconds))
            cont = continue_tokens(lm, u, n_tok, a.temp, a.top_p, a.seed + i)
            y = dec.speak(f"tts{k}", cont, True)
        y = y[: int(a.seconds * SR)]
        write16(out / f"{i:02d}.wav", y)
        rows.append({"id": f"{i:02d}", "prompt_units": int(len(u)), "cont_units": int(len(cont)),
                     "cont_s": round(len(y) / SR, 2), "gen_s": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    info = {"variant": a.variant, "pipeline": "GSLM released (Lakhotia et al. 2021), no training by us",
            "units": f"HuBERT-base-ls960 layer 6 (HF facebook/hubert-base-ls960) + GSLM k-means {k} (km.bin), deduplicated",
            "lm": f"released GSLM unit LM hubert{k}_lm (fairseq transformer_lm_big, 12 layers, 1024 d, trained on Libri-Light 6k h), loaded into a PyTorch re-implementation",
            "decoder": "fairseq unit HiFi-GAN hubert_base_100_lj with duration predictor" if k == 100 else f"GSLM Tacotron2 tts_km{k} + WaveGlow (textlesslib)",
            "sampling": {"temperature": a.temp, "top_p": a.top_p, "seed": f"{a.seed} + prompt index",
                         "banned": "EOS and special symbols", "stop": f"when the predicted duration reaches {a.seconds} s (km100) / token budget = prompt unit rate x {a.seconds} s"},
            "prompt_seen": "the whole 15 s prompt_XX.wav", "output": f"continuation only, cut to <= {a.seconds} s, 16 kHz mono",
            "training_step": None, "validation_loss": None, "note": "GSLM paper sampled with temperature; we use the DY-32 default 0.8 / top-p 0.95",
            "rows": rows}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    if a.upload:
        sh(f"aws s3 sync {out} {OUT}/gen/{a.variant}/ --only-show-errors")
    print("RELEASED DONE", a.variant, flush=True)


@torch.no_grad()
def continue_tokens(lm: FairseqULM, prompt: np.ndarray, n: int, temp: float, top_p: float, seed: int) -> np.ndarray:
    dev = lm.emb.weight.device
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.tensor([FairseqULM.EOS] + [int(x) + FairseqULM.OFF for x in prompt], device=dev)[None]
    for _ in range(n):
        lg = lm.logits(ids[:, -3000:])[:, -1]
        lg[:, : FairseqULM.OFF] = -float("inf")
        ids = torch.cat([ids, sample_top_p(lg, temp, top_p, g)[:, None]], 1)
    return (ids[0, -n:] - FairseqULM.OFF).cpu().numpy()


# ---------------------------------------------------------------- unit pass over training shards

def listing(prefix: str, dirs: bool) -> list[str]:
    out = sh(f"aws s3 ls {prefix}/ || true")
    if dirs:
        return sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))
    return sorted(l.split()[-1].removesuffix(".npz") for l in out.splitlines() if l.strip().endswith(".npz"))


def fetch(src: str, s: str, work: Path) -> tuple[str, list[np.ndarray], float]:
    """One shard -> its pieces >= 5 s (int16), as dy30_units.fetch."""
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin"):
        sh(f"aws s3 cp {src}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    sh(f"rm -rf {d}")
    pieces = [segs[st : st + ln] for st, ln in idx if ln >= 5 * SR]
    return s, pieces, sum(len(w) for w in pieces) / SR


class MultiUnits:
    """One HuBERT pass, several (layer, k-means) unit sets: windows [B, N] -> {channel: units [B, T]}."""

    def __init__(self, channels: list[str], models: Path, dev) -> None:
        self.ch = {c: (CHANNELS[c][0], KMeans(str(models / CHANNELS[c][1]), dev)) for c in channels}
        self.enc = Hubert(max(l for l, _ in self.ch.values()), dev)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.enc.model(x.to(self.enc.device), output_hidden_states=True).hidden_states
        return {c: km(hs[l].float()).cpu().numpy() for c, (l, km) in self.ch.items()}


def shard_units(mu: MultiUnits, pieces: list[np.ndarray], win: int, batch: int) -> dict[str, list[np.ndarray]]:
    """Full-rate units per piece and channel. Full 20 s windows go in batches; the shorter last window of each piece
    alone (HuBERT base has no attention mask and uses group norm, so windows are never padded)."""
    full = [(k, i) for k, w in enumerate(pieces) for i in range(0, len(w) - win + 1, win)]
    per = {c: [dict() for _ in pieces] for c in mu.ch}
    for b in range(0, len(full), batch):
        part = full[b : b + batch]
        x = torch.from_numpy(np.stack([pieces[k][i : i + win] for k, i in part]).astype(np.float32) / 32768.0)
        for c, u in mu(x).items():
            for (k, i), r in zip(part, u):
                per[c][k][i] = r
    for k, w in enumerate(pieces):
        i = (len(w) // win) * win
        if len(w) - i >= 400:
            for c, u in mu(torch.from_numpy(w[i:].astype(np.float32)[None] / 32768.0)).items():
                per[c][k][i] = u[0]
    return {c: [np.concatenate([d[i] for i in sorted(d)]) if d else np.zeros(0, np.int64) for d in ds]
            for c, ds in per.items()}


def cmd_units(a) -> None:
    from dy21_train import held_out

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    chans = a.channels.split(",")
    mu = MultiUnits(chans, Path(a.models), dev)
    dst = {c: f"{OUT}/g/units-{c}" for c in chans}
    done = set(listing(dst[chans[-1]], dirs=False))
    k, n = map(int, a.part.split("/"))
    allsh = [s for s in listing(a.src, dirs=True) if held_out(s) == a.held]
    allsh = [allsh[j] for j in np.random.default_rng(a.order_seed).permutation(len(allsh))]  # diverse prefix
    mine = [s for s in allsh[k::n] if s not in done][: a.max_shards or None]
    print(json.dumps({"part": a.part, "todo": len(mine), "already": len(done), "channels": chans}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(2)
    uploads, failed, lock = [], [], threading.Lock()

    def upload(fs: dict[str, Path], s: str) -> None:
        try:
            for c, f in fs.items():  # last channel last: it marks the shard done
                sh(f"aws s3 cp {f} {dst[c]}/{s}.npz --only-show-errors")
                if not a.keep:
                    f.unlink()
        except Exception as e:
            with lock:
                failed.append((s, str(e)[-200:]))

    nxt = dl.submit(fetch, a.src, mine[0], work) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        t0 = time.time()
        s, pieces, secs = nxt.result()
        wait = time.time() - t0
        nxt = dl.submit(fetch, a.src, mine[j + 1], work) if j + 1 < len(mine) else None
        units = shard_units(mu, pieces, int(20 * SR), a.batch)
        fs = {}
        for c, us in units.items():
            index, pos = [], 0
            for u in us:
                index.append((pos, len(u)))
                pos += len(u)
            fs[c] = work / f"{s}.{c}.npz"
            np.savez(fs[c], tok=np.concatenate(us).astype(np.int16), index=np.asarray(index, np.int64).reshape(-1, 2))
        del pieces, units
        uploads.append(up.submit(upload, fs, s))
        h_all += secs / 3600
        print(json.dumps({"shard": s, "done": j + 1, "of": len(mine), "hours": round(secs / 3600, 2),
                          "x_realtime": round(secs / (time.time() - t0), 1), "waited_s": round(wait, 1),
                          "hours_total": round(h_all, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    for f in uploads:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"G UNITS DONE part {a.part}", round(h_all, 1), "h", flush=True)


# ---------------------------------------------------------------- our model on this channel

def cmd_gen(a) -> None:
    from dauys.dy21_lm import LMConfig, UnitLM

    dev = torch.device("cuda")
    models, pack, work = Path(a.models), Path(a.pack), Path(a.work)
    layer, kmf, decoder, _ = CHANNELS[a.channel]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = LMConfig(**ck["cfg"])
    model = UnitLM(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    K = cfg.vocab - 1
    EOS = K
    enc = Hubert(layer, dev)
    km = KMeans(str(models / kmf), dev)
    dec = Decoders(models, dev)
    out = work / a.variant
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        w = read16(pack / f"prompt_{i:02d}.wav")
        u = encode(enc, km, w)
        if a.dedup:
            u, _ = dedup(u)
        n_tok = int(round(len(u) / (len(w) / SR) * a.seconds))
        g = torch.Generator(device=dev).manual_seed(a.seed + i)
        ids = torch.from_numpy(np.r_[EOS, u].astype(np.int64))[None].to(dev)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_tok):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model.states(ids[:, -cfg.ctx :])
                lg = model.logits(h[:, -1])
                lg[:, EOS] = -float("inf")
                ids = torch.cat([ids, sample_top_p(lg, a.temp, a.top_p, g)[:, None]], 1)
        cont = ids[0, -n_tok:].cpu().numpy()
        y = dec.speak(decoder, cont, a.dedup)[: int(a.seconds * SR)]
        write16(out / f"{i:02d}.wav", y)
        rows.append({"id": f"{i:02d}", "prompt_units": int(len(u)), "cont_units": n_tok, "cont_s": round(len(y) / SR, 2),
                     "gen_s": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    meta = {}
    mp = Path(a.ckpt).parent / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    info = {"variant": a.variant, "model": "our UnitLM (src/dauys/dy21_lm.py), trained from scratch by scripts/dy21_train.py",
            "params": meta.get("params"), "config": ck["cfg"], "training_step": ck["step"], "validation": ck["val"],
            "train_tokens": meta.get("train_tokens"), "git": meta.get("git"),
            "units": f"HuBERT-base-ls960 layer {layer} + k-means {kmf}" + (", deduplicated" if a.dedup else ", full 50 Hz"),
            "decoder": decoder, "sampling": {"temperature": a.temp, "top_p": a.top_p, "seed": f"{a.seed} + prompt index",
                                             "banned": "EOS", "tokens": f"prompt unit rate x {a.seconds} s"},
            "prompt_seen": f"the whole 15 s prompt (last {cfg.ctx} tokens of the context)",
            "output": f"continuation only, cut to <= {a.seconds} s", "rows": rows}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    if a.upload:
        for dst in a.upload.split(","):
            sh(f"aws s3 sync {out} {OUT}/{dst}/ --only-show-errors")
    print("GEN DONE", a.variant, flush=True)


def cmd_transcribe(a) -> None:
    """Whisper transcripts of a folder of XX.wav (measuring only), next to the prompt and real-continuation transcripts."""
    dev = torch.device("cuda")
    d, pack = Path(a.work), Path(a.pack)
    ids = sorted(f.stem for f in d.glob("[0-9][0-9].wav"))
    hyp = whisper([read16(d / f"{i}.wav") for i in ids], dev)
    pr = whisper([read16(pack / f"prompt_{i}.wav") for i in ids], dev)
    re_ = whisper([read16(pack / f"real_{i}.wav") for i in ids], dev)
    rows = [{"id": i, "prompt": p, "real": r, "generated": h, "words": len(norm(h).split()),
             "distinct_word_share": round(len(set(norm(h).split())) / max(len(norm(h).split()), 1), 3)}
            for i, p, r, h in zip(ids, pr, re_, hyp)]
    summ = {"n": len(rows), "median_words": float(np.median([r["words"] for r in rows])),
            "median_distinct_word_share": float(np.median([r["distinct_word_share"] for r in rows])),
            "empty": sum(r["words"] == 0 for r in rows)}
    (d / "transcripts.json").write_text(json.dumps({"asr": ASR, "summary": summ, "rows": rows}, indent=2))
    print("TRANSCRIBED", d.name, json.dumps(summ), flush=True)
    if a.upload:
        sh(f"aws s3 cp {d / 'transcripts.json'} {OUT}/g/transcripts/{d.name}.json --only-show-errors")


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("channel", "released", "layers", "units", "gen", "transcribe"):
        x = s.add_parser(name)
        x.add_argument("--models", default="/data/m")
        x.add_argument("--pack", default="/data/pack")
        x.add_argument("--work", default="/data/g")
        x.add_argument("--n", type=int, default=50)
        x.add_argument("--upload", default="")
        x.add_argument("--seconds", type=float, default=10.0)
        x.add_argument("--temp", type=float, default=0.8)
        x.add_argument("--top-p", type=float, default=0.95)
        x.add_argument("--seed", type=int, default=1234)
    s.choices["channel"].add_argument("--channels", default=",".join(CHANNELS))
    s.choices["channel"].add_argument("--wer", type=int, default=1)
    s.choices["layers"].add_argument("--km", default="km100.bin,km500.bin,L9_km500.bin")
    s.choices["released"].add_argument("--k", type=int, default=100)
    s.choices["released"].add_argument("--variant", default="g-released")
    x = s.choices["units"]
    x.add_argument("--channels", default="km100,L9km500")
    x.add_argument("--order-seed", type=int, default=0)
    x.add_argument("--part", default="0/1")
    x.add_argument("--src", default=SRC)
    x.add_argument("--max-shards", type=int, default=0)
    x.add_argument("--batch", type=int, default=32)
    x.add_argument("--held", action="store_true", help="only held-out shards (validation units)")
    x.add_argument("--keep", action="store_true")
    x = s.choices["gen"]
    x.add_argument("--channel", default="km100")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--variant", default="g-ours")
    x.add_argument("--dedup", type=int, default=1)
    a = p.parse_args()
    {"channel": cmd_channel, "released": cmd_released, "layers": cmd_layers, "units": cmd_units, "gen": cmd_gen,
     "transcribe": cmd_transcribe}[a.cmd](a)


if __name__ == "__main__":
    main()
