"""Alphabet test, data: DuplexChat audio to phoneme sequences from an off-the-shelf phoneme recogniser.

Diagnostic only (the user's "alphabet first" test, 2026-11-18): text-trained phoneme recognition is a ruler that
gives clean symbols, it is not part of the Dauys core. facebook/wav2vec2-lv-60-espeak-cv-ft (CTC over eSpeak
phonemes), greedy: argmax per 20 ms frame, repeats collapsed, blanks dropped. Audio in 20 s windows.
Output per shard ``<shard>.npz``: ``tok`` int16 phoneme ids (pieces concatenated), ``index`` [n, 2] (start, len) —
the DY-21 unit format, so dy21_train.py trains on it unchanged (ids < 8192).

    python scripts/phon_units.py --hours 500 --held 4 --out /opt/dlami/nvme/phon
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out

SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"
SR = 16000
WIN = 20 * SR
MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class Phonemes:
    def __init__(self, device) -> None:
        from transformers import Wav2Vec2ForCTC

        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(device).half().eval()
        self.blank = self.model.config.pad_token_id

    @torch.no_grad()
    def windows(self, wins: list[np.ndarray], batch: int = 32) -> list[np.ndarray]:
        """Collapsed phoneme ids per window (each window float32 16 kHz, <= WIN samples)."""
        out = []
        for i in range(0, len(wins), batch):
            w = wins[i : i + batch]
            n = max(len(x) for x in w)
            x = np.zeros((len(w), n), np.float32)
            m = np.zeros((len(w), n), np.int64)
            for j, a in enumerate(w):
                a = (a - a.mean()) / (a.std() + 1e-7)
                x[j, : len(a)] = a
                m[j, : len(a)] = 1
            xt = torch.from_numpy(x).to(self.device).half()
            mt = torch.from_numpy(m).to(self.device)
            ids = self.model(xt, attention_mask=mt).logits.argmax(-1).cpu().numpy()
            for j, a in enumerate(w):
                k = ids[j][: max(1, len(a) // 320)]
                k = k[np.r_[True, k[1:] != k[:-1]]]
                out.append(k[k != self.blank].astype(np.int64))
        return out

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """Phoneme ids of one clip of any length (int16 or float)."""
        a = wav.astype(np.float32) / (32768.0 if wav.dtype == np.int16 else 1.0)
        wins = [a[i : i + WIN] for i in range(0, max(len(a), 1), WIN)]
        return np.concatenate(self.windows(wins)) if wins else np.zeros(0, np.int64)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=500)
    p.add_argument("--held", type=int, default=4, help="held-out shards for validation")
    p.add_argument("--out", required=True)
    p.add_argument("--work", default="/opt/dlami/nvme/phon_work")
    args = p.parse_args()
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    ph = Phonemes(device)
    shards = sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines() if l.strip().startswith("PRE"))
    plan = [s for s in shards if held_out(s)][: args.held] + [s for s in shards if not held_out(s)]
    t0, train_h = time.time(), 0.0
    for s in plan:
        if not held_out(s) and train_h >= args.hours:
            break
        d = work / s
        d.mkdir(exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
        idx = np.load(d / "index.npy")
        segs = np.fromfile(d / "segs.bin", dtype=np.int16)
        pieces = [segs[st : st + ln] for st, ln in idx if ln >= 5 * SR]
        wins, owner = [], []
        for k, w in enumerate(pieces):
            a = w.astype(np.float32) / 32768.0
            for i in range(0, len(a), WIN):
                wins.append(a[i : i + WIN])
                owner.append(k)
        res = ph.windows(wins)
        toks, index, pos = [], [], 0
        per = [[] for _ in pieces]
        for k, r in zip(owner, res):
            per[k].append(r)
        for r in per:
            t = np.concatenate(r) if r else np.zeros(0, np.int64)
            toks.append(t.astype(np.int16))
            index.append((pos, len(t)))
            pos += len(t)
        np.savez(out / f"{s}.npz", tok=np.concatenate(toks), index=np.asarray(index, np.int64).reshape(-1, 2))
        sh(f"rm -rf {d}")
        hours = sum(len(w) for w in pieces) / SR / 3600
        if not held_out(s):
            train_h += hours
        print(json.dumps({"shard": s, "held": held_out(s), "hours": round(hours, 2), "train_hours": round(train_h, 1),
                          "phonemes": pos, "per_s": round(pos / max(hours * 3600, 1), 1),
                          "x_realtime": round(hours * 3600 / (time.time() - t0), 1)}), flush=True)
        t0 = time.time()
    print("PHON DONE", round(train_h, 1), "h", flush=True)


if __name__ == "__main__":
    main()
