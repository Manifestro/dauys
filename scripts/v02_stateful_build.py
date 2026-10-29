"""Build long alternating-channel streams for stateful Stage 1 training."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from dauys.v02_stage1 import FRAME_RATE

PAUSE_FRAMES = 6


def _real(cached: dict, source: str) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "1a_on_A":
        return cached["user"].long(), cached["user_activity"].long()
    if source == "1a_on_B":
        return cached["target"].long(), cached["activity"].long()
    raise ValueError(f"Stateful training expects single-stream speech, not {source}")


def utterance_number(identity: str) -> int:
    tail = identity.rsplit("-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 2**31 - 1


def chapter_of(identity: str) -> str:
    return identity.rsplit("-", 1)[0] if "-" in identity else identity


def speaker_of(chapter: str) -> str:
    return chapter.rsplit("-", 1)[0] if "-" in chapter else chapter


def assemble_chapter(
    phrases: list[dict], silence: torch.Tensor, pause: int = PAUSE_FRAMES
) -> dict:
    """Alternate phrases between user and model channels with silent gaps."""
    if not phrases:
        raise ValueError("A chapter needs at least one phrase")
    if silence.shape != (8,):
        raise ValueError("Silence codes must have eight Mimi indices")
    lengths = [int(phrase["codes"].shape[-1]) for phrase in phrases]
    total = sum(lengths) + pause * (len(phrases) - 1)
    user = silence[:, None].expand(8, total).clone()
    target = user.clone()
    user_activity = torch.zeros(total, dtype=torch.long)
    output_activity = torch.zeros(total, dtype=torch.long)
    position = 0
    for index, phrase in enumerate(phrases):
        codes = phrase["codes"].long()
        activity = phrase["activity"].long()
        length = codes.shape[-1]
        if codes.shape[0] != 8 or activity.shape != (length,):
            raise ValueError("Phrase codes and activity must align")
        start, end = position, position + length
        if index % 2 == 0:
            user[:, start:end] = codes
            user_activity[start:end] = activity
        else:
            target[:, start:end] = codes
            output_activity[start:end] = activity
        position = end
        if index + 1 < len(phrases):
            position += pause
    return {
        "user": user,
        "target": target,
        "user_activity": user_activity,
        "activity": output_activity,
    }


def slice_stream(
    stream: dict, chunk: int, chunks: int, chapter: str, speaker: str
) -> list[dict]:
    """Cut one chapter into non-overlapping ``chunks``-window streams."""
    if chunk < 1 or chunks < 1:
        raise ValueError("Chunk and chunk count must be positive")
    span = chunk * chunks
    length = stream["user"].shape[-1]
    if length < span:
        return []
    result = []
    for start in range(0, length - span + 1, span):
        result.append(
            {
                "user": stream["user"][:, start : start + span].clone(),
                "target": stream["target"][:, start : start + span].clone(),
                "user_activity": stream["user_activity"][start : start + span].clone(),
                "activity": stream["activity"][start : start + span].clone(),
                "chapter": chapter,
                "speaker": speaker,
            }
        )
    return result


def build_split(
    root: Path,
    silence: torch.Tensor,
    *,
    chunk: int = 256,
    chunks: int = 8,
    limit: int | None = None,
) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (utterance_number(str(row["id"])), str(row["id"])))
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        identity = str(row["id"])
        cached = torch.load(
            root / str(row["tokens_path"]), map_location="cpu", weights_only=True
        )
        codes, activity = _real(cached, str(row.get("source", "1a_on_B")))
        if activity.shape != (codes.shape[-1],):
            raise ValueError(f"Activity does not match frames: {identity}")
        by_chapter[chapter_of(identity)].append({"codes": codes, "activity": activity})
    streams = []
    for chapter in sorted(by_chapter):
        assembled = assemble_chapter(by_chapter[chapter], silence)
        for stream in slice_stream(
            assembled, chunk, chunks, chapter, speaker_of(chapter)
        ):
            streams.append(stream)
            if limit is not None and len(streams) >= limit:
                return streams
    return streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.chunk < 1 or args.chunks < 1:
        raise ValueError("Chunk and chunk count must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("The stream limit must be positive")
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    if silence.shape != (8,):
        raise ValueError("Silence codes must be eight Mimi indices")
    args.output.mkdir(parents=True, exist_ok=True)
    total_streams = 0
    total_frames = 0
    by_split = {}
    for root in args.inputs:
        streams = build_split(
            root,
            silence,
            chunk=args.chunk,
            chunks=args.chunks,
            limit=args.limit,
        )
        split = root.name or "split"
        torch.save(streams, args.output / f"{split}.pt")
        frames = sum(int(stream["user"].shape[-1]) for stream in streams)
        by_split[split] = {
            "streams": len(streams),
            "frames": frames,
            "hours": round(frames / FRAME_RATE / 3600, 4),
        }
        total_streams += len(streams)
        total_frames += frames
    summary = {
        "streams": total_streams,
        "frames": total_frames,
        "hours": round(total_frames / FRAME_RATE / 3600, 4),
        "chunk": args.chunk,
        "chunks": args.chunks,
        "pause_frames": PAUSE_FRAMES,
        "by_split": by_split,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
