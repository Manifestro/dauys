"""DY-33 track S: train the slow codec (src/dauys/dy33s_codec.py) and check it on the prompt pack.

train: random 5.1 s crops (256 frames) of the layer-6 features written by dy33s_feats.py (training shards only; the
list of finished shards is re-read every 500 steps while the feature pass is still running). AdamW, fp32 (TF32).
Checkpoint every --save-every steps to --out/<name>/last.pt.

resynth: the 50 real_XX.wav of the DY-32 pack -> layer-6 features (as at unit time) -> codes -> decoder -> kNN-VC
vocoder -> --out/<name>/resynth/XX.wav, info.json, quick Whisper corpus WER against Whisper on the original (own
check only; track M's letters WER is the official gate). --s3 uploads to dy33/resynth/<name>/.

    python scripts/dy33s_codec.py train --name A --cfg 512,2,4,256 --feats /opt/dlami/nvme/feats --out /data/codec
    python scripts/dy33s_codec.py resynth --name A --out /data/codec --s3
"""

import argparse
import json
import random
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out

from dauys.dy33s_codec import Codec, CodecConfig

S3 = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy33"
CROP = 256


class Crops:
    def __init__(self, feats: str, batch: int, seed: int) -> None:
        self.dir, self.batch, self.rng = Path(feats), batch, random.Random(seed)
        self.mm, self.items = {}, []
        self.refresh()

    def refresh(self) -> None:
        for d in sorted(self.dir.iterdir()):
            if d.name in self.mm or held_out(d.name) or not (d / "DONE").exists():
                continue
            self.mm[d.name] = np.load(d / "feats.npy", mmap_mode="r")
            for st, ln in np.load(d / "index.npy"):
                if ln >= CROP:
                    self.items.append((d.name, int(st), int(ln)))

    def hours(self) -> float:
        return sum(len(m) for m in self.mm.values()) / 50 / 3600

    def get(self) -> np.ndarray:
        out = np.empty((self.batch, CROP, 1024), np.float16)
        for b in range(self.batch):
            s, st, ln = self.items[self.rng.randrange(len(self.items))]
            o = st + self.rng.randrange(ln - CROP + 1)
            out[b] = self.mm[s][o : o + CROP]
        return out


def cmd_train(a) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    d, n50, n12, dq = map(int, a.cfg.split(","))
    cfg = CodecConfig(d=d, n50=n50, n12=n12, dq=dq)
    src = Crops(a.feats, a.batch, a.seed)
    while not src.items:
        time.sleep(30)
        src.refresh()
    m = Codec(cfg).to(dev)
    x0 = torch.from_numpy(np.concatenate([src.get() for _ in range(16)]).reshape(-1, 1024).astype(np.float32))
    m.mu.copy_(x0.mean(0))
    m.sd.copy_(x0.std(0) + 1e-3)
    npar = sum(p.numel() for p in m.parameters())
    print(json.dumps({"name": a.name, "config": vars(cfg), "params": npar, "hours_now": round(src.hours(), 1)}), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    q: queue.Queue = queue.Queue(maxsize=8)

    def feeder():
        while True:
            q.put(src.get())

    for _ in range(3):
        threading.Thread(target=feeder, daemon=True).start()
    t0, t_start = time.time(), time.time()
    logs: dict = {}
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / 500)
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.from_numpy(q.get()).to(dev).float()
        nl = random.randint(1, cfg.levels) if random.random() < 0.5 else None
        loss, part = m.loss(x, nl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if nl is None:
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) * 0.98 + v * 0.02 if k in logs else v
        if (step + 1) % 100 == 0:
            idle = (m.rvq.idle > 50).float().mean(1).tolist()
            print(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}, "idle_share": [round(x, 3) for x in idle],
                              "steps_per_s": round(100 / (time.time() - t0), 2)}), flush=True)
            t0 = time.time()
        if (step + 1) % 500 == 0:
            src.refresh()
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or time.time() - t_start > a.minutes * 60:
            torch.save({"model": m.state_dict(), "cfg": vars(cfg), "step": step + 1, "hours": src.hours(), "logs": logs},
                       out / "last.tmp")
            Path(out / "last.tmp").replace(out / "last.pt")
            if time.time() - t_start > a.minutes * 60:
                break
    print("CODEC TRAIN DONE", a.name, step + 1, flush=True)


def load_codec(path: Path, dev) -> tuple[Codec, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Codec(CodecConfig(**ck["cfg"])).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


@torch.no_grad()
def roundtrip(m: Codec, f: torch.Tensor) -> torch.Tensor:
    """features [T, 1024] -> codes -> decoded features [T, 1024] (padded to a multiple of 4, then cut)."""
    T = len(f)
    pad = (-T) % 4
    x = torch.cat([f, f[-1:].expand(pad, -1)]) if pad else f
    return m.decode(m.codes(x[None].float()))[0, :T]


def cmd_resynth(a) -> None:
    import soundfile as sf

    from dy32c_resynth import corpus_wer, read_pack, whisper
    from dy32c_units import RVQUnits, sh

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    out = Path(a.out) / a.name
    if a.ckpt:
        ckp = Path(a.ckpt)
    else:  # snapshot last.pt first so the checked weights are kept under their step
        ck0 = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        ckp = out / f"snap{ck0['step']}.pt"
        torch.save(ck0, ckp)
    m, ck = load_codec(ckp, dev)
    names, waves = read_pack(Path(a.pack))
    u = RVQUnits(dev)
    rd = out / f"resynth{ck['step']}"
    rd.mkdir(parents=True, exist_ok=True)
    res = []
    for n, w in zip(names, waves):
        f, nf = u.feats([w.astype(np.float32)])
        y = roundtrip(m, f[0, : nf[0]])
        wav = u.knn.vocode(y[None].float()).reshape(-1).cpu().numpy()
        sf.write(rd / f"{n}.wav", wav, 16000)
        res.append(wav)
    info = {"codec": a.name, "checkpoint": str(ckp), "checkpoint_step": ck["step"], "train_hours_seen_pool": round(ck["hours"], 1), "config": ck["cfg"],
            "params": sum(v.numel() for k, v in ck["model"].items() if not k.startswith("rvq.") and k not in ("mu", "sd")),
            "path": "real_XX.wav -> kNN-VC WavLM-Large layer 6 (50 Hz) -> codec encoder (x4 down, 12.5 Hz) -> RVQ 4 x 1024 -> "
                    "codec decoder (x4 up) -> 50 Hz features -> kNN-VC prematched HiFi-GAN (frozen)",
            "rate": "12.5 steps/s x 4 codes x 10 bit = 500 bit/s"}
    del u
    torch.cuda.empty_cache()
    if a.asr:
        ref = whisper(waves, dev)
        hyp = whisper(res, dev)
        info["own_check_whisper_corpus_wer_vs_original"] = round(corpus_wer(ref, hyp), 4)
        info["own_check_hyp_first3"] = hyp[:3]
        print("OWN WER", a.name, info["own_check_whisper_corpus_wer_vs_original"], hyp[:2], flush=True)
    (rd / "info.json").write_text(json.dumps(info, indent=2))
    if a.s3:
        sh(f"aws s3 cp {rd}/ {S3}/resynth/{a.s3}/ --recursive --only-show-errors")
        print(f"UPLOADED resynth/{a.s3}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("train")
    x.add_argument("--name", required=True)
    x.add_argument("--cfg", default="512,2,4,256", help="d,n50,n12,dq")
    x.add_argument("--feats", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--steps", type=int, default=100000)
    x.add_argument("--minutes", type=float, default=150)
    x.add_argument("--batch", type=int, default=64)
    x.add_argument("--lr", type=float, default=5e-4)
    x.add_argument("--save-every", type=int, default=1000)
    x.add_argument("--seed", type=int, default=0)
    x = s.add_parser("resynth")
    x.add_argument("--name", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--pack", default="/opt/dlami/nvme/pack")
    x.add_argument("--asr", action="store_true")
    x.add_argument("--ckpt", default="", help="codec checkpoint (default: snapshot of last.pt)")
    x.add_argument("--s3", default="", help="upload to dy33/resynth/<this>/")
    a = p.parse_args()
    {"train": cmd_train, "resynth": cmd_resynth}[a.cmd](a)


if __name__ == "__main__":
    main()
