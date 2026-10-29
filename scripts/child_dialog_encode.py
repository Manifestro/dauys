"""Encode synthesized child dialogues into Stage 1 A→B caches, in batches.

Each spoken pair becomes one recording: the question A, a pause of 4–8 silent
frames, the reply B, then 25–38 silent frames, so the model also learns to
stop after it has answered. A sits on the user channel and B on the model
channel, as ``scripts/v02_stage1_build.py`` lays out LibriSpeech; the cut is the
start of the pause. Clips are encoded with Mimi in batches of similar length.

Per split, writes ``tokens/`` (A→B caches), ``clips/`` (codes of each clip, so
an evaluation can pair any question with any reply) and ``manifest.jsonl``.
The chapter is the family group, so same-chapter pools hold different concepts
spoken by one voice.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile
import torch

from dauys.codec import encode, load_mimi
from dauys.v02_data import sha256
from dauys.v02_stage1 import ab_layout

FRAME = 1920


def activity(audio: np.ndarray, frames: int) -> torch.Tensor:
    padded = np.pad(audio, (0, frames * FRAME - len(audio)))
    rms = np.sqrt(np.mean(padded.reshape(frames, FRAME) ** 2, axis=1))
    return torch.from_numpy((rms > 10 ** (-45 / 20)).astype(np.uint8))


@torch.no_grad()
def encode_all(codec, wavs: Path, names: list[str], batch: int) -> dict:
    audio = {}
    for name in names:
        data, rate = soundfile.read(wavs / name, dtype="float32")
        if rate != 24_000:
            raise ValueError(f"{name} is {rate} Hz")
        audio[name] = data
    out = {}
    order = sorted(names, key=lambda n: len(audio[n]))
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        frames = [math.ceil(len(audio[n]) / FRAME) for n in chunk]
        longest = max(frames)
        pcm = np.stack(
            [np.pad(audio[n], (0, longest * FRAME - len(audio[n]))) for n in chunk]
        )
        codes = encode(codec, torch.from_numpy(pcm)[:, None].cuda()).cpu()
        for row, (name, count) in enumerate(zip(chunk, frames)):
            out[name] = (codes[row, :, :count].long(), activity(audio[name], count))
        if (start // batch) % 50 == 0:
            print(
                json.dumps({"encoded": start + len(chunk), "of": len(order)}),
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, required=True, help="synth output dir")
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    plan = [
        json.loads(line)
        for line in (args.synth / "plan.jsonl").read_text().splitlines()
    ]
    names = sorted({p["a_wav"] for p in plan} | {p["b_wav"] for p in plan})
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    clips = encode_all(codec, args.synth / "wavs", names, args.batch)
    manifests: dict[str, list] = {}
    for item in plan:
        root = args.output / item["split"]
        (root / "tokens").mkdir(parents=True, exist_ok=True)
        (root / "clips").mkdir(parents=True, exist_ok=True)
        a_codes, a_active = clips[item["a_wav"]]
        b_codes, b_active = clips[item["b_wav"]]
        for name, (codes, active) in (
            (item["a_wav"], (a_codes, a_active)),
            (item["b_wav"], (b_codes, b_active)),
        ):
            path = root / "clips" / (name.removesuffix(".wav") + ".pt")
            if not path.exists():
                torch.save({"codes": codes.short(), "activity": active}, path)
        rng = random.Random(item["id"])
        pause, tail = rng.randint(4, 8), rng.randint(25, 38)
        codes = torch.cat(
            (
                a_codes,
                silence[:, None].expand(8, pause),
                b_codes,
                silence[:, None].expand(8, tail),
            ),
            -1,
        )
        active = torch.cat(
            (
                a_active,
                torch.zeros(pause, dtype=torch.uint8),
                b_active,
                torch.zeros(tail, dtype=torch.uint8),
            )
        )
        cut = a_codes.shape[-1]
        layout = ab_layout(codes, active.long(), cut, silence)
        token = root / "tokens" / f"{item['id']}.pt"
        torch.save(
            {
                "user": layout.user.short(),
                "target": layout.target.short(),
                "user_activity": layout.user_activity.to(torch.uint8),
                "activity": layout.target_activity.to(torch.uint8),
            },
            token,
        )
        row = {
            "id": item["id"],
            "speaker": item["voice"],
            "chapter": item["group"],
            "topic": item["group"],
            "word": item["key"],
            "concept": item["concept"],
            "family": item["family"],
            "frame": item["frame"],
            "frames": codes.shape[-1],
            "cut_frame": cut,
            "tokens_path": f"tokens/{item['id']}.pt",
            "sha256": sha256(token),
            "a_clip": "clips/" + item["a_wav"].removesuffix(".wav") + ".pt",
            "b_clip": "clips/" + item["b_wav"].removesuffix(".wav") + ".pt",
            "question": item["question"],
            "reply": item["reply"],
        }
        for key in ("rank", "voice", "speed"):
            if key in item:
                row[key] = item[key]
        manifests.setdefault(item["split"], []).append(row)
    for split, rows in manifests.items():
        (args.output / split / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    print(
        json.dumps({split: len(rows) for split, rows in manifests.items()}), flush=True
    )


if __name__ == "__main__":
    main()
