"""Turn hohere-1m conversations into Stage 2 records, one per assistant turn.

Each conversation keeps its two real channels in one token file: A is the user,
B the assistant. A record points at the frame where an assistant turn starts,
so the trainer's Stage 1 batches read it as heard speech followed by a reply.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile
import torch

from dauys.codec import encode, load_mimi
from dauys.v02_data import sha256

FRAME_SAMPLES = 1920
MIN_BEFORE = 38
MIN_AFTER = 25


def activity_from_events(events: list[dict], speaker: str, frames: int) -> torch.Tensor:
    """Frames covered by this speaker's labelled speech."""
    active = torch.zeros(frames, dtype=torch.uint8)
    for event in events:
        if event["speaker"] == speaker:
            start = event["start_sample"] // FRAME_SAMPLES
            end = -(-event["end_sample"] // FRAME_SAMPLES)
            active[start:end] = 1
    return active


def reply_cuts(events: list[dict], frames: int, user: torch.Tensor) -> list[int]:
    """Assistant turn starts with heard speech before and room to reply after."""
    cuts = []
    for event in events:
        if event["speaker"] != "B":
            continue
        cut = event["start_sample"] // FRAME_SAMPLES
        if cut < MIN_BEFORE or frames - cut < MIN_AFTER or not user[:cut].any():
            continue
        cuts.append(cut)
    return sorted(set(cuts))


@torch.inference_mode()
def read_channel(mimi, cell: dict) -> torch.Tensor:
    audio, rate = soundfile.read(io.BytesIO(cell["bytes"]), dtype="float32")
    if rate != 24_000:
        raise ValueError(f"Expected 24 kHz audio, got {rate}")
    pad = (-len(audio)) % FRAME_SAMPLES
    wav = torch.from_numpy(np.pad(audio, (0, pad)))[None, None].cuda()
    return encode(mimi, wav)[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (args.output / "tokens").mkdir(parents=True, exist_ok=True)
    mimi = load_mimi(args.mimi_checkpoint, "cuda")
    rows = []
    conversations = 0
    for path in args.parquet:
        for row in pq.read_table(path).to_pylist():
            a = read_channel(mimi, row["channel_a"])
            b = read_channel(mimi, row["channel_b"])
            frames = min(a.shape[-1], b.shape[-1])
            a, b = a[:, :frames], b[:, :frames]
            user_activity = activity_from_events(row["events"], "A", frames)
            activity = activity_from_events(row["events"], "B", frames)
            cuts = reply_cuts(row["events"], frames, user_activity)
            if not cuts:
                continue
            token = args.output / "tokens" / f"{row['id']}.pt"
            torch.save(
                {
                    "user": a.short(),
                    "target": b.short(),
                    "user_activity": user_activity,
                    "activity": activity,
                },
                token,
            )
            digest = sha256(token)
            for cut in cuts:
                rows.append(
                    {
                        "id": f"{row['id']}-{cut}",
                        "speaker": row["voice_b"],
                        "chapter": row["skill"],
                        "frames": frames,
                        "cut_frame": cut,
                        "tokens_path": f"tokens/{row['id']}.pt",
                        "sha256": digest,
                        "source": "hohere",
                    }
                )
            conversations += 1
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    print(json.dumps({"conversations": conversations, "records": len(rows)}))


if __name__ == "__main__":
    main()
