"""DY-32 track D: heard speech as continuous WavLM features, own speech as discrete units (src/dauys/dy32_heard.py).

Audio: DuplexChat en-16k shards downloaded to --audio (``<shard>/index.npy``, ``segs.bin`` int16, ``DONE``); held-out
shards (dy21_train.held_out) are never trained on. Frozen WavLM-Large (knn-vc release, layer 6, fp16) runs online
inside the training loop on random audio windows; unit ids are the nearest codebook centroid of the same features.

  kmeans  temporary whole-corpus codebook (K=2000, Euclidean) on layer-6 features of random 10 s windows
  train   the HeardLM; half of every batch in continuous-heard mode, half in discrete-heard mode; validation CE on the
          own part (fixed held-out windows, split at 750 frames = 15 s) in both modes; best.pt by their mean
  gen     continue the prompt pack in both modes from one checkpoint, decode by centroid lookup -> kNN-VC prematched
          HiFi-GAN; also the round trip of real_XX.wav through the units (--resynth)

    python scripts/dy32d.py kmeans --audio /opt/dlami/nvme/audio --out /data/dy32d/centroids_tmp.npy
    python scripts/dy32d.py train --audio /opt/dlami/nvme/audio --held /opt/dlami/nvme/heldaudio \
        --centroids c.npy --out /data/dy32d/run --steps 20000 --deadline 07:10
    python scripts/dy32d.py gen --run /data/dy32d/run --centroids c.npy --prompts /data/dy32d/prompts --out /data/dy32d/gen
"""

import argparse
import datetime as dt
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dauys.dy32_heard import HOP, SR, HeardConfig, HeardLM, generate, window_loss

EXPERIMENT = "DY-32: канал, через который слова выживают, без текста — три идеи и повтор GSLM параллельными ветками"
LAYER = 6


# ---------------------------------------------------------------- frozen WavLM / vocoder (knn-vc release, MIT)

def load_wavlm(device):
    m = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, progress=False, device=str(device))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def load_vocoder(device):
    voc, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, progress=False,
                            device=str(device))
    return voc.eval()


@torch.no_grad()
def wavlm_feats(wavlm, wav: torch.Tensor, chunk: int = 16, fp32: bool = False) -> torch.Tensor:
    """wav [B, N] float in [-1, 1] (as knn-vc: no waveform normalisation, no padding) -> [B, frames, 1024] fp16
    (computed in fp16 autocast for training speed; fp32 for evaluation, as track B's channel)."""
    out = []
    for i in range(0, len(wav), chunk):
        with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
            f = wavlm.extract_features(wav[i : i + chunk].float(), output_layer=LAYER, ret_layer_results=False)[0]
        out.append(f.float() if fp32 else f.half())
    return torch.cat(out)


class Codebook:
    """A plain [K, 1024] .npy (50 Hz, decode = centroid) or track B's .npz (cent, cent_n, dec [K, r, 1024], rate):
    at rate 25 the 50 Hz frames are mean-pooled in pairs before assignment and each code decodes to r = 2 frames."""

    def __init__(self, path: str, metric: str, device) -> None:
        if path.endswith(".npz"):
            z = np.load(path)
            c, dec, self.rate = z["cent"], z["dec"], int(z["rate"])
            metric = "cos"
        else:
            c = np.load(path)
            dec, self.rate = c[:, None], 50
        self.c = torch.from_numpy(c).float().to(device)
        self.dec = torch.from_numpy(dec).float().to(device)
        self.metric = metric
        self.cn = F.normalize(self.c, dim=-1)
        self.c2 = (self.c**2).sum(-1)

    def pool(self, f: torch.Tensor) -> torch.Tensor:
        """50 Hz features [B, T, D] -> model frames (mean of pairs at 25 Hz; an odd last frame is dropped)."""
        if self.rate == 50:
            return f
        r = 50 // self.rate
        t = f.shape[1] // r * r
        return f[:, :t].float().reshape(f.shape[0], t // r, r, f.shape[2]).mean(2).to(f.dtype)

    @torch.no_grad()
    def ids(self, f: torch.Tensor) -> torch.Tensor:
        sh = f.shape[:-1]
        x = f.reshape(-1, f.shape[-1]).float()
        out = []
        for i in range(0, len(x), 65536):
            z = x[i : i + 65536]
            if self.metric == "cos":
                out.append((F.normalize(z, dim=-1) @ self.cn.T).argmax(-1))
            else:
                out.append((self.c2[None] - 2 * z @ self.c.T).argmin(-1))
        return torch.cat(out).reshape(sh)

    def feats(self, ids: torch.Tensor) -> torch.Tensor:
        """codes [B, n] -> 50 Hz features [B, n * r, 1024] for the vocoder."""
        d = self.dec[ids]
        return d.reshape(ids.shape[0], -1, d.shape[-1])


# ---------------------------------------------------------------- audio windows

def n_samples(frames: int) -> int:
    """Samples giving exactly `frames` WavLM frames (conv stack: floor((N - 400) / 320) + 1)."""
    return (frames - 1) * HOP + 400


class Audio:
    """Random fixed-length windows from downloaded shards (memmap); rescans the directory for newly finished shards."""

    def __init__(self, root: str, length: int, train: bool, seed: int) -> None:
        self.root, self.length, self.train = Path(root), length, train
        self.rng = np.random.default_rng(seed)
        self.shards: dict[str, tuple] = {}
        self.scan()

    def scan(self) -> None:
        for d in sorted(self.root.iterdir()):
            if d.name in self.shards or not (d / "segs.bin").exists():
                continue
            if self.train and (held_out(d.name) or not (d / "DONE").exists()):
                continue
            if not self.train and not held_out(d.name):
                continue
            idx = np.load(d / "index.npy")
            idx = idx[idx[:, 1] >= self.length]
            if len(idx):
                self.shards[d.name] = (np.memmap(d / "segs.bin", dtype=np.int16, mode="r"), idx)
        self.names = sorted(self.shards)
        w = np.array([len(self.shards[s][1]) for s in self.names], np.float64)
        self.p = w / w.sum()
        self.hours = sum(float(self.shards[s][1][:, 1].sum()) for s in self.names) / SR / 3600

    def batch(self, n: int, rng=None) -> np.ndarray:
        rng = rng or self.rng
        out = np.empty((n, self.length), np.int16)
        for i in range(n):
            segs, idx = self.shards[self.names[rng.choice(len(self.names), p=self.p)]]
            st, ln = idx[rng.integers(len(idx))]
            o = st + rng.integers(0, ln - self.length + 1)
            out[i] = segs[o : o + self.length]
        return out


def to_wav(x: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(x).to(device, non_blocking=True).float() / 32768.0


# ---------------------------------------------------------------- kmeans

def cmd_kmeans(a) -> None:
    device = torch.device("cuda")
    wavlm = load_wavlm(device)
    au = Audio(a.audio, n_samples(500), True, 0)
    print(json.dumps({"shards": len(au.names), "hours": round(au.hours, 1)}), flush=True)
    feats = []
    for _ in range(a.windows // 64):
        feats.append(wavlm_feats(wavlm, to_wav(au.batch(64), device)).reshape(-1, 1024))
    x = torch.cat(feats).float()
    print("frames", len(x), flush=True)
    g = torch.Generator(device=device).manual_seed(0)
    c = x[torch.randperm(len(x), generator=g, device=device)[: a.k]].clone()
    for it in range(a.iters):
        c2 = (c**2).sum(-1)
        assign = torch.cat([(c2[None] - 2 * x[i : i + 65536] @ c.T).argmin(-1) for i in range(0, len(x), 65536)])
        s = torch.zeros_like(c).index_add_(0, assign, x)
        n = torch.bincount(assign, minlength=a.k).float()
        empty = n == 0
        c = torch.where(empty[:, None], x[torch.randint(len(x), (a.k,), device=device, generator=g)], s / n.clamp(min=1)[:, None])
        print(json.dumps({"iter": it, "empty": int(empty.sum())}), flush=True)
    np.save(a.out, c.cpu().numpy().astype(np.float32))
    print("KMEANS DONE", a.out, flush=True)


# ---------------------------------------------------------------- train

def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(steps - warmup, 1)))))


def deadline_ts(hhmm: str) -> float:
    if not hhmm:
        return math.inf
    now = dt.datetime.now(dt.timezone.utc)
    h, m = map(int, hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t < now - dt.timedelta(hours=2):
        t += dt.timedelta(days=1)
    return t.timestamp()


def cmd_train(a) -> None:
    import airon

    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_at = deadline_ts(a.deadline)
    cb = Codebook(a.centroids, a.metric, device)
    r = 50 // cb.rate
    if a.small:
        cfg = HeardConfig(units=len(cb.c), d=256, layers=4, heads=4, ffn=1024, ctx=a.ctx)
    else:
        cfg = HeardConfig(units=len(cb.c), ctx=a.ctx)
    T = cfg.ctx
    L = n_samples(r * (T + 1))
    wavlm = load_wavlm(device)
    tr = Audio(a.audio, L, True, a.seed)
    va = Audio(a.held, L, False, 0)
    # fixed validation windows; split at 15 s (the prompt length of the eval pack)
    vwav = va.batch(a.val_n, np.random.default_rng(1234))
    vfeat = torch.cat([cb.pool(wavlm_feats(wavlm, to_wav(vwav[i : i + 16], device))) for i in range(0, len(vwav), 16)])
    vids = cb.ids(vfeat)
    vsplit = torch.full((len(vwav),), min(15 * cb.rate, T - 1), device=device)

    model = HeardLM(cfg).to(device)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    meta = {"run_id": str(uuid.uuid4()), "params": model.parameters_count(), "config": vars(cfg), "args": vars(a),
            "train_shards": len(tr.names), "train_hours": round(tr.hours, 1), "val_shards": va.names,
            "frame_rate_hz": cb.rate, "codebook": a.centroids, "metric": cb.metric, "features": f"WavLM-Large layer {LAYER} (knn-vc release), frozen, fp16",
            "git": os.environ.get("DY21_COMMIT", "")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    run = airon.init(project="dauys", name=a.name, config=meta, experiment=a.experiment, tags=["dy32", "seed-0"],
                     total_steps=a.steps, run_id=meta["run_id"], mode="disabled" if a.tracking == "off" else None)

    # loader thread: int16 windows ready in pinned memory
    q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def loader() -> None:
        rng = np.random.default_rng(a.seed + 1)
        k = 0
        while not stop.is_set():
            k += 1
            if k % 400 == 0:
                tr.scan()
            q.put(torch.from_numpy(tr.batch(a.batch, rng)).pin_memory())

    th = threading.Thread(target=loader, daemon=True)
    th.start()

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        res = {}
        for mode, name in ((0, "cont"), (1, "disc")):
            tot, n = 0.0, 0
            for i in range(0, len(vids), 16):
                m = torch.full((len(vids[i : i + 16]),), mode, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s, k = window_loss(model, vids[i : i + 16], vfeat[i : i + 16], vsplit[i : i + 16], m, "sum")
                tot, n = tot + s.item(), n + k.item()
            res[f"ce_{name}"] = tot / n
        res["ce"] = (res["ce_cont"] + res["ce_disc"]) / 2
        res["gap_disc_minus_cont"] = res["ce_disc"] - res["ce_cont"]
        model.train()
        return res

    best, t0, waited = math.inf, time.time(), 0.0
    rng = torch.Generator(device=device).manual_seed(a.seed)
    lo, hi = a.split_min, T - a.split_min
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        tw = time.time()
        wav = q.get()
        waited += time.time() - tw
        wav = wav.to(device, non_blocking=True).float() / 32768.0
        feats = cb.pool(wavlm_feats(wavlm, wav))
        ids = cb.ids(feats)
        B = len(ids)
        split = torch.randint(lo, hi + 1, (B,), device=device, generator=rng)
        mode = (torch.arange(B, device=device) % 2)
        opt.zero_grad(set_to_none=True)
        logs = {"ce_cont": 0.0, "ce_disc": 0.0}
        nmic = math.ceil(B / a.micro)
        for i in range(0, B, a.micro):
            sl = slice(i, i + a.micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _ = window_loss(model, ids[sl], feats[sl], split[sl], mode[sl])
            (l / nmic).backward()
            logs["ce"] = logs.get("ce", 0.0) + l.item() / nmic
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if (step + 1) % 50 == 0:
            logs.pop("ce_cont"), logs.pop("ce_disc")
            logs["lr"] = opt.param_groups[0]["lr"]
            el = time.time() - t0
            logs["steps_per_s"] = 50 / el
            logs["loader_wait_frac"] = waited / el
            t0, waited = time.time(), 0.0
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "train_hours": round(tr.hours, 1),
                              **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        last = step + 1 == a.steps or time.time() > stop_at
        if (step + 1) % a.val_every == 0 or last:
            res = validate()
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res, "utc": time.strftime("%H:%M", time.gmtime())}), flush=True)
            if res["ce"] < best:
                best = res["ce"]
                torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res,
                            "metric": cb.metric, "rate": cb.rate}, out / "best.pt.tmp")
                os.replace(out / "best.pt.tmp", out / "best.pt")
        if last:
            if time.time() > stop_at:
                print(json.dumps({"deadline_stop": step + 1}), flush=True)
            break
    stop.set()
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


# ---------------------------------------------------------------- gen

def read_wav(p: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(p, dtype="float32")
    assert sr == SR, (p, sr)
    return x if x.ndim == 1 else x.mean(1)


def cmd_gen(a) -> None:
    import soundfile as sf

    device = torch.device("cuda")
    ck = torch.load(Path(a.run) / a.ckpt, map_location="cpu")
    cfg = HeardConfig(**ck["cfg"])
    model = HeardLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    cb = Codebook(a.centroids, ck.get("metric", a.metric), device)
    r = 50 // cb.rate
    frames = a.frames or 10 * cb.rate
    wavlm, voc = load_wavlm(device), load_vocoder(device)
    pdir = Path(a.prompts)
    names = sorted(p.stem.split("_")[1] for p in pdir.glob("prompt_*.wav"))
    prompts = [read_wav(pdir / f"prompt_{n}.wav") for n in names]
    P = min(len(x) for x in prompts)
    P = min(P, n_samples(r * (cfg.ctx - frames)))
    wav = torch.from_numpy(np.stack([x[-P:] for x in prompts])).to(device)  # last P samples (all 15 s if they fit)
    feats = cb.pool(wavlm_feats(wavlm, wav, fp32=True))
    ids = cb.ids(feats)
    out = Path(a.out)

    def decode(u: torch.Tensor) -> np.ndarray:
        res = []
        for i in range(0, len(u), 8):
            with torch.no_grad():
                y = voc(cb.feats(u[i : i + 8]).float()).squeeze(1)
            res.append(y.float().cpu().numpy())
        return np.concatenate(res)

    for mode, tag in ((0, "cont"), (1, "disc")):
        d = out / f"d-{tag}{a.suffix}"
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        gen = []
        for i in range(0, len(ids), a.gen_batch):  # chunk i uses seed + i
            sl = slice(i, i + a.gen_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen.append(generate(model, ids[sl], feats[sl] if mode == 0 else None, mode, frames, a.temp, a.top_p,
                                    a.seed + i))
        gen = torch.cat(gen)
        y = decode(gen)
        for n, w in zip(names, y):
            sf.write(d / f"{n}.wav", np.clip(w, -1, 1), SR, subtype="PCM_16")
        info = {"track": "d", "variant": f"d-{tag}{a.suffix}",
                "heard_mode": "continuous WavLM-Large layer-6 features (no quantisation)" if mode == 0 else
                "discrete: the same frames as unit ids (nearest centroid)",
                "checkpoint": a.ckpt, "step": ck["step"], "val": ck["val"], "config": ck["cfg"],
                "params": model.parameters_count(),
                "sampling": {"temperature": a.temp, "top_p": a.top_p, "seed": f"{a.seed} + first index of each chunk of {a.gen_batch}", "frames": frames,
                             "frame_rate_hz": cb.rate},
                "prompt_seen_s": P / SR, "prompt_frames": int(ids.shape[1]),
                "units": f"K={len(cb.c)} codebook on WavLM-Large layer {LAYER} (knn-vc), metric {cb.metric}: {a.unit_desc}",
                "decoder": "unit -> centroid feature -> kNN-VC prematched HiFi-GAN (github.com/bshall/knn-vc, MIT), 16 kHz",
                "gen_seconds": round(time.time() - t0, 1)}
        (d / "info.json").write_text(json.dumps(info, indent=2))
        print("GEN", tag, json.dumps({k: info[k] for k in ("step", "gen_seconds")}), flush=True)
    if a.resynth:
        d = out / "resynth-d"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            x = torch.from_numpy(read_wav(pdir / f"real_{n}.wav")).to(device)[None]
            y = decode(cb.ids(cb.pool(wavlm_feats(wavlm, x, fp32=True))))
            sf.write(d / f"{n}.wav", np.clip(y[0], -1, 1), SR, subtype="PCM_16")
        print("RESYNTH DONE", flush=True)
    print("GEN DONE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("kmeans")
    x.add_argument("--audio", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--k", type=int, default=2000)
    x.add_argument("--windows", type=int, default=2048)
    x.add_argument("--iters", type=int, default=25)
    x = s.add_parser("train")
    x.add_argument("--audio", required=True)
    x.add_argument("--held", required=True)
    x.add_argument("--centroids", required=True)
    x.add_argument("--metric", default="euclid", choices=["euclid", "cos"])
    x.add_argument("--out", required=True)
    x.add_argument("--steps", type=int, default=20000)
    x.add_argument("--deadline", default="", help="HH:MM UTC: stop (with a final validation) after this time")
    x.add_argument("--batch", type=int, default=32)
    x.add_argument("--micro", type=int, default=16)
    x.add_argument("--ctx", type=int, default=1280)
    x.add_argument("--split-min", type=int, default=200)
    x.add_argument("--lr", type=float, default=3e-4)
    x.add_argument("--warmup", type=int, default=1000)
    x.add_argument("--val-every", type=int, default=1000)
    x.add_argument("--val-n", type=int, default=128)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--small", action="store_true")
    x.add_argument("--tracking", default="on", choices=["on", "off"])
    x.add_argument("--experiment", default=EXPERIMENT)
    x.add_argument("--name", default="dy32-d")
    x = s.add_parser("gen")
    x.add_argument("--run", required=True)
    x.add_argument("--ckpt", default="best.pt")
    x.add_argument("--centroids", required=True)
    x.add_argument("--metric", default="euclid")
    x.add_argument("--unit-desc", default="")
    x.add_argument("--prompts", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--suffix", default="")
    x.add_argument("--frames", type=int, default=0, help="own frames to generate (default 10 s)")
    x.add_argument("--temp", type=float, default=0.8)
    x.add_argument("--top-p", type=float, default=0.95)
    x.add_argument("--seed", type=int, default=1234)
    x.add_argument("--resynth", action="store_true")
    x.add_argument("--gen-batch", type=int, default=25)
    a = p.parse_args()
    {"kmeans": cmd_kmeans, "train": cmd_train, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
