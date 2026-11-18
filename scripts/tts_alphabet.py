"""Alphabet data: clean synthetic speech — the same sentence in four cloned real voices — for training a voice-invariant
quantiser (the user's "alphabet first" direction, 2026-11-18). Text is used only to make these data; it never
enters the Dauys model.

Voices: 6 s windows from DuplexChat podcast pieces (training shards; each worker takes its own shards, so the bank
is many different real people). TTS: Qwen3-TTS-12Hz-1.7B-Base, voice clone from the speaker embedding only
(x_vector_only_mode), bf16. Texts: conversational utterances from allenai/soda (6-30 words, ASCII), split between
workers by index. Each sentence is spoken by 4 voices of this worker's bank. Clips longer than 0.75 s per word
(TTS runaway) or shorter than 0.12 s per word are dropped. 24 kHz output resampled to 16 kHz.
Output per batch of ~2000 clips to S3 ``outputs/dauys/tts-alphabet/p{k}/{j:04d}.npz``: ``audio`` int16 (clips
concatenated), ``index`` [n, 2] (start, length), ``meta`` JSON lines (text_id, text, voice).

    python scripts/tts_alphabet.py --part 3/11 --hours 500
"""

import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"
DST = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/tts-alphabet"
SR = 16000
VOICES_PER_TEXT = 4


def held_out(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 50 == 0


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def voice_bank(k: int, n: int, size: int, work: Path, rng) -> list[np.ndarray]:
    shards = [s for s in sorted(l.split()[-1].rstrip("/") for l in sh(f"aws s3 ls {SRC}/").splitlines()
                                if l.strip().startswith("PRE")) if not held_out(s)][k::n]
    bank = []
    for s in shards:
        for fn in ("index.npy", "segs.bin"):
            sh(f"aws s3 cp {SRC}/{s}/{fn} {work}/{fn} --only-show-errors")
        idx = np.load(work / "index.npy")
        segs = np.fromfile(work / "segs.bin", dtype=np.int16)
        for st, ln in rng.permutation(idx)[:200]:
            if ln < 20 * SR:
                continue
            o = st + int(rng.integers(2 * SR, ln - 8 * SR))
            w = segs[o : o + 6 * SR].astype(np.float32) / 32768
            if np.sqrt((w ** 2).mean()) > 0.01:
                bank.append(w)
            if len(bank) >= size:
                return bank
    return bank


def texts(k: int, n: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/soda", split="train", streaming=True)
    i = 0
    for r in ds:
        for u in r["dialogue"]:
            u = u.strip()
            if 6 <= len(u.split()) <= 30 and u.isascii():
                if i % n == k:
                    yield i, u
                i += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True)
    p.add_argument("--hours", type=float, default=500, help="total over all parts")
    p.add_argument("--voices", type=int, default=400)
    p.add_argument("--batch-texts", type=int, default=16)
    p.add_argument("--clips-per-file", type=int, default=2000)
    p.add_argument("--work", default="/data/tts")
    args = p.parse_args()
    k, n = map(int, args.part.split("/"))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + k)
    bank = voice_bank(k, n, args.voices, work, rng)
    print(json.dumps({"part": args.part, "voices": len(bank)}), flush=True)

    from qwen_tts import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
    target_s = args.hours * 3600 / n
    done_s, file_j, t0 = 0.0, 0, time.time()
    buf_a, buf_m = [], []
    gen = texts(k, n)

    def flush():
        nonlocal file_j, buf_a, buf_m
        if not buf_a:
            return
        lens = np.array([len(a) for a in buf_a])
        index = np.stack([np.r_[0, np.cumsum(lens)[:-1]], lens], 1)
        b = io.BytesIO()
        np.savez(b, audio=np.concatenate(buf_a), index=index,
                 meta=np.array("\n".join(json.dumps(x) for x in buf_m)))
        path = work / f"{file_j:04d}.npz"
        path.write_bytes(b.getvalue())
        sh(f"aws s3 cp {path} {DST}/p{k}/{file_j:04d}.npz --only-show-errors && rm {path}")
        file_j += 1
        buf_a, buf_m = [], []

    while done_s < target_s:
        batch = [next(gen) for _ in range(args.batch_texts)]
        tx, ra, meta = [], [], []
        for tid, t in batch:
            for v in rng.choice(len(bank), VOICES_PER_TEXT, replace=False):
                tx.append(t)
                ra.append((bank[v], SR))
                meta.append({"text_id": int(tid), "text": t, "voice": f"p{k}v{int(v)}"})
        try:
            wavs, sr = m.generate_voice_clone(text=tx, language=["English"] * len(tx), ref_audio=ra,
                                              x_vector_only_mode=[True] * len(tx), max_new_tokens=480)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM, batch skipped", flush=True)
            continue
        for w, mt in zip(wavs, meta):
            a = resample_poly(np.asarray(w, np.float32), SR, sr)
            words = len(mt["text"].split())
            if not (0.12 * words * SR <= len(a) <= 0.75 * words * SR):
                continue
            buf_a.append(np.clip(a * 32768, -32768, 32767).astype(np.int16))
            buf_m.append(mt)
            done_s += len(a) / SR
        if len(buf_a) >= args.clips_per_file:
            flush()
            el = time.time() - t0
            print(json.dumps({"part": args.part, "files": file_j, "hours": round(done_s / 3600, 2),
                              "target_h": round(target_s / 3600, 1), "x_realtime": round(done_s / el, 2)}), flush=True)
    flush()
    print(f"TTS ALPHABET DONE part {args.part} hours {done_s / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
