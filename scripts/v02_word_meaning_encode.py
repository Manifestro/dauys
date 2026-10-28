"""Encode the synthesized word-meaning clips into Stage 1 A→B caches.

Each item becomes one recording: the user phrase A, a pause of 4–8 silent
frames, then the reply B. A sits on the user channel and B on the model
channel, exactly as ``scripts/v02_stage1_build.py`` lays out LibriSpeech, so
the Stage 1 loader, losses and metrics apply unchanged. The cut is the start of
the pause. Every split gets its own root with ``manifest.jsonl``; the chapter
is ``topic-frame``, so a same-chapter pool is a minimal-pair pool: one frame,
different slot words. A and B codes are also kept separately per clip, so the
evaluation can pair any A with any B. Control items (nonsense words) have no
reply and only keep their A codes.
"""

import argparse
import hashlib
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

FRAME_SAMPLES = 1920


def clip_id(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]


def activity(audio: np.ndarray, frames: int) -> torch.Tensor:
    padded = np.pad(audio, (0, frames * FRAME_SAMPLES - len(audio)))
    rms = np.sqrt(np.mean(padded.reshape(frames, FRAME_SAMPLES) ** 2, axis=1))
    return torch.from_numpy((rms > 10 ** (-45 / 20)).astype(np.uint8))


@torch.no_grad()
def encode_clip(codec, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    audio, rate = soundfile.read(path, dtype="float32")
    if rate != 24_000:
        raise ValueError(f"{path} is {rate} Hz, not 24 kHz")
    frames = math.ceil(len(audio) / FRAME_SAMPLES)
    pcm = np.pad(audio, (0, frames * FRAME_SAMPLES - len(audio)))
    codes = encode(codec, torch.from_numpy(pcm)[None, None].cuda())[0].cpu()
    return codes.long(), activity(audio, frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True, help="synth output dir")
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    clips = {
        (row["text"], row["voice"]): args.clips / "wavs" / row["path"]
        for row in map(
            json.loads, (args.clips / "clips.jsonl").read_text().splitlines()
        )
    }
    encoded: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(text: str, voice: str):
        key = (text, voice)
        if key not in encoded:
            encoded[key] = encode_clip(codec, clips[key])
        return encoded[key]

    items = [json.loads(line) for line in args.plan.read_text().splitlines()]
    manifests: dict[str, list[dict]] = {}
    for count, item in enumerate(items, start=1):
        root = args.output / item["split"]
        (root / "tokens").mkdir(parents=True, exist_ok=True)
        (root / "clips").mkdir(parents=True, exist_ok=True)
        a_codes, a_active = get(item["a_text"], item["voice"])
        a_name = f"clips/{clip_id(item['a_text'], item['voice'])}.pt"
        if not (root / a_name).exists():
            torch.save({"codes": a_codes.short(), "activity": a_active}, root / a_name)
        row = {
            "id": item["id"],
            "speaker": item["voice"],
            "chapter": f"{item['topic']}-f{item['frame']}",
            "topic": item["topic"],
            "word": item["word"],
            "frame": item["frame"],
            "a_clip": a_name,
        }
        if item["b_text"] is not None:
            b_codes, b_active = get(item["b_text"], item["reply_voice"])
            b_name = f"clips/{clip_id(item['b_text'], item['reply_voice'])}.pt"
            if not (root / b_name).exists():
                torch.save(
                    {"codes": b_codes.short(), "activity": b_active}, root / b_name
                )
            pause = random.Random(item["id"]).randint(4, 8)
            gap = silence[:, None].expand(8, pause)
            codes = torch.cat((a_codes, gap, b_codes), dim=-1)
            active = torch.cat(
                (a_active, torch.zeros(pause, dtype=torch.uint8), b_active)
            )
            cut = a_codes.shape[-1]
            layout = ab_layout(codes, active.long(), cut, silence)
            token_name = f"tokens/{item['id']}.pt"
            torch.save(
                {
                    "user": layout.user.short(),
                    "target": layout.target.short(),
                    "user_activity": layout.user_activity.to(torch.uint8),
                    "activity": layout.target_activity.to(torch.uint8),
                },
                root / token_name,
            )
            row.update(
                {
                    "frames": codes.shape[-1],
                    "cut_frame": cut,
                    "tokens_path": token_name,
                    "sha256": sha256(root / token_name),
                    "b_clip": b_name,
                }
            )
        manifests.setdefault(item["split"], []).append(row)
        if count % 2000 == 0:
            print(json.dumps({"items": count, "of": len(items)}), flush=True)
    for split, rows in manifests.items():
        (args.output / split / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    print(json.dumps({s: len(r) for s, r in manifests.items()}), flush=True)


if __name__ == "__main__":
    main()
