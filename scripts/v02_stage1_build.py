"""Build Stage 1 A→B token caches and a manifest from encoded single-stream speech.

Input is the output of ``scripts/v02_encode_librispeech.py``: one token cache
per utterance with the real speech on the channel named by ``source``. The
script finds a pause, puts everything before it on channel A and everything
after it on channel B, then writes a new immutable cache and manifest.
"""

import argparse
import json
import math
import os
import zlib
from collections import Counter
from pathlib import Path

import torch

from dauys.v02_data import sha256
from dauys.v02_stage1 import FRAME_RATE, ab_layout, find_cut


def _real(cached: dict, source: str) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "1a_on_A":
        return cached["user"].long(), cached["user_activity"].long()
    if source == "1a_on_B":
        return cached["target"].long(), cached["activity"].long()
    raise ValueError(f"Stage 1 expects single-stream speech, not {source}")


def _atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Encoded roots, each containing manifest.jsonl and tokens/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--min-before-seconds", type=float, default=3.0)
    parser.add_argument("--min-after-seconds", type=float, default=2.0)
    parser.add_argument("--min-pause-frames", type=int, default=3)
    parser.add_argument(
        "--max-pause-frames",
        type=int,
        default=32,
        help="Keep the reply within ~2.5 s of the cut so crops contain B speech",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    silence = torch.tensor(
        json.loads(args.silence_codes.read_text(encoding="utf-8")), dtype=torch.long
    )
    if silence.shape != (8,):
        raise ValueError("Silence codes must be eight Mimi indices")
    min_before = math.ceil(args.min_before_seconds * FRAME_RATE)
    min_after = math.ceil(args.min_after_seconds * FRAME_RATE)
    if min(args.min_pause_frames, min_before, min_after) < 1:
        raise ValueError("Cut requirements must be positive")

    output = args.output
    tokens_dir = output / "tokens"
    items_dir = output / "items"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    skipped_by_input: Counter = Counter()
    seen: set[str] = set()
    codec_sha: str | None = None

    for root in args.inputs:
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            if args.limit is not None and stats["built"] >= args.limit:
                break
            identity = str(row["id"])
            if identity in seen:
                continue
            seen.add(identity)
            item_path = items_dir / f"{identity}.json"
            if item_path.is_file():
                stats["kept"] += 1
                continue
            row_codec = row.get("codec_sha256")
            if row_codec:
                if codec_sha is None:
                    codec_sha = str(row_codec)
                elif str(row_codec) != codec_sha:
                    raise ValueError("All inputs must come from one Mimi encoder")
            cached = torch.load(
                root / str(row["tokens_path"]), map_location="cpu", weights_only=True
            )
            codes, activity = _real(cached, str(row["source"]))
            frames = codes.shape[-1]
            if activity.shape != (frames,):
                raise ValueError(f"Activity does not match frames: {identity}")
            cut = find_cut(
                activity.tolist(),
                min_before=min_before,
                min_after=min_after,
                min_pause=args.min_pause_frames,
                max_pause=args.max_pause_frames,
                seed=zlib.crc32(identity.encode()),
            )
            if cut is None:
                stats["skipped"] += 1
                skipped_by_input[root.name] += 1
                continue
            layout = ab_layout(codes, activity, cut, silence)
            cache = {
                "user": layout.user.short(),
                "target": layout.target.short(),
                "user_activity": layout.user_activity.to(torch.uint8),
                "activity": layout.target_activity.to(torch.uint8),
            }
            token_path = tokens_dir / f"{identity}.pt"
            _atomic_save(token_path, cache)
            item = {
                "manifest": {
                    "id": identity,
                    "speaker": str(row["user_voice"]),
                    "chapter": (
                        identity.rsplit("-", 1)[0] if "-" in identity else identity
                    ),
                    "frames": frames,
                    "cut_frame": cut,
                    "tokens_path": f"tokens/{identity}.pt",
                    "sha256": sha256(token_path),
                    "source_root": str(root),
                    "source_dataset": row.get("dataset"),
                    "source_revision": row.get("dataset_revision"),
                    "source_split": row.get("dataset_split"),
                    "codec_sha256": row.get("codec_sha256"),
                }
            }
            item_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            stats["built"] += 1
            if stats["built"] % 1000 == 0:
                print(json.dumps(stats), flush=True)
        if args.limit is not None and stats["built"] >= args.limit:
            break

    items = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(items_dir.glob("*.json"))
    ]
    if not items:
        raise ValueError("No Stage 1 items were built")
    manifest = "\n".join(json.dumps(item["manifest"]) for item in items) + "\n"
    (output / "manifest.jsonl").write_text(manifest, encoding="utf-8")
    summary = {
        "items": len(items),
        "built": stats["built"],
        "skipped": stats["skipped"],
        "kept": stats["kept"],
        "min_before_frames": min_before,
        "min_after_frames": min_after,
        "min_pause_frames": args.min_pause_frames,
        "max_pause_frames": args.max_pause_frames,
        "skipped_by_input": dict(skipped_by_input),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
