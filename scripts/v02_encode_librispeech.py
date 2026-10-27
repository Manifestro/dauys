"""Resumeable, batched Mimi encoding of LibriSpeech single-stream speech."""

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, load_dataset
from huggingface_hub import HfApi

from dauys.codec import encode, load_mimi
from dauys.v02_data import sha256

FRAME_SAMPLES = 1920
DATASET = "openslr/librispeech_asr"


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _audio(row: dict[str, Any]) -> tuple[str, str, np.ndarray]:
    identity = re.sub(r"[^A-Za-z0-9_-]", "_", str(row["id"]))
    if not identity:
        raise ValueError("LibriSpeech item has no ID")
    payload = row["audio"]
    source = payload.get("bytes")
    if source is None:
        path = payload.get("path")
        if not path:
            raise ValueError(f"LibriSpeech audio is missing: {identity}")
        source = Path(path).read_bytes()
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=source,
        capture_output=True,
        check=True,
    )
    pcm = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(pcm) < FRAME_SAMPLES or not np.isfinite(pcm).all():
        raise ValueError(f"Invalid LibriSpeech audio: {identity}")
    return identity, str(row["speaker_id"]), pcm


def _activity(audio: np.ndarray, frames: int) -> list[int]:
    padded = np.pad(audio, (0, frames * FRAME_SAMPLES - len(audio)))
    rms = np.sqrt(np.mean(padded.reshape(frames, FRAME_SAMPLES) ** 2, axis=1))
    active = (rms > 10 ** (-45 / 20)).astype(np.uint8)
    if not active.any():
        raise ValueError("LibriSpeech clip has no active speech frames")
    return active.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-items", type=int, default=16)
    parser.add_argument("--max-padded-frames", type=int, default=2048)
    parser.add_argument("--decode-workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-vram-fraction", type=float, default=0.85)
    parser.add_argument(
        "--local-root",
        type=Path,
        help="Extracted OpenSLR SLR12 split (e.g. LibriSpeech/train-clean-360) "
        "or a downloaded Hugging Face parquet split (e.g. clean/train.360) "
        "instead of streaming from Hugging Face",
    )
    args = parser.parse_args()
    # cuDNN TF32 convolutions make batched Mimi codes differ from single-item
    # codes in about 1% of acoustic tokens; full precision makes them identical.
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("LibriSpeech Mimi encoding requires CUDA")
    # Varying batch lengths grow PyTorch's cache to the whole card; on WSL the
    # driver then pages VRAM into host memory and encoding slows about 7x.
    torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction)
    if min(args.batch_items, args.max_padded_frames, args.decode_workers) < 1:
        raise ValueError("Batch and worker settings must be positive")
    if args.local_root is not None:
        if not args.local_root.is_dir():
            raise ValueError(f"Local LibriSpeech split is missing: {args.local_root}")
        args.dataset = "openslr/slr12"
        args.split = args.local_root.name
        revision = "local"
    else:
        revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    if not revision:
        raise ValueError("Could not resolve the LibriSpeech revision")
    root = args.output
    items_dir = root / "items"
    tokens_dir = root / "tokens"
    items_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    codec_sha = sha256(args.mimi_checkpoint)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    silence = encode(codec, torch.zeros(1, 1, FRAME_SAMPLES, device="cuda"))[
        0, :, 0
    ].cpu()
    _atomic(root / "silence_codes.json", (json.dumps(silence.tolist()) + "\n").encode())
    stream: Any
    parquet = sorted(args.local_root.glob("*.parquet")) if args.local_root else []
    if parquet:
        args.dataset = DATASET
        stream = load_dataset(
            "parquet",
            data_files=[str(x) for x in parquet],
            split="train",
            streaming=True,
        ).cast_column("audio", Audio(decode=False))
    elif args.local_root is not None:
        stream = (
            {
                "id": path.stem,
                "speaker_id": path.parent.parent.name,
                "audio": {"path": str(path)},
            }
            for path in sorted(args.local_root.glob("*/*/*.flac"))
        )
    else:
        stream = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
            revision=revision,
            streaming=True,
        )
        if isinstance(stream.features["audio"], Audio):
            stream = stream.cast_column("audio", Audio(decode=False))
    pending: list[tuple[str, str, np.ndarray]] = []
    seen: set[str] = set()
    processed = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        longest = max(math.ceil(len(audio) / FRAME_SAMPLES) for _, _, audio in pending)
        pcm = np.stack(
            [
                np.pad(audio, (0, longest * FRAME_SAMPLES - len(audio)))
                for _, _, audio in pending
            ]
        )
        codes = (
            encode(codec, torch.from_numpy(pcm.copy())[:, None].cuda()).short().cpu()
        )
        if codes.shape != (len(pending), 8, longest):
            raise ValueError("Mimi returned the wrong LibriSpeech batch geometry")
        for index, (identity, speaker, audio) in enumerate(pending):
            frames = math.ceil(len(audio) / FRAME_SAMPLES)
            active = _activity(audio, frames)
            on_a = hashlib.sha256(identity.encode()).digest()[0] % 2 == 0
            source = "1a_on_A" if on_a else "1a_on_B"
            real_codes = codes[index, :, :frames].clone()
            silent_codes = silence[:, None].expand(8, frames).short().clone()
            cache = {
                "user": real_codes if on_a else silent_codes,
                "target": silent_codes if on_a else real_codes,
                "user_activity": torch.tensor(
                    active if on_a else [0] * frames, dtype=torch.uint8
                ),
                "activity": torch.tensor(
                    [0] * frames if on_a else active, dtype=torch.uint8
                ),
            }
            token_path = tokens_dir / f"{identity}.pt"
            temporary = token_path.with_suffix(".tmp")
            torch.save(cache, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(token_path)
            manifest = {
                "id": identity,
                "source": source,
                "skill": "single_speech",
                "family": f"librispeech/{speaker}",
                "user_voice": speaker,
                "frames": frames,
                "tokens_path": f"tokens/{identity}.pt",
                "sha256": sha256(token_path),
                "dataset": args.dataset,
                "dataset_revision": revision,
                "dataset_config": args.config,
                "dataset_split": args.split,
                "codec_sha256": codec_sha,
            }
            companion = {
                "id": identity,
                "activity_a": cache["user_activity"].tolist(),
                "activity_b": cache["activity"].tolist(),
                "event_frames": [],
            }
            _atomic(
                items_dir / f"{identity}.json",
                (
                    json.dumps({"manifest": manifest, "companion": companion}) + "\n"
                ).encode(),
            )
        print(json.dumps({"encoded": len(pending), "processed": processed}), flush=True)
        pending = []

    rows_iterator = iter(stream)
    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        while True:
            rows = list(itertools.islice(rows_iterator, args.decode_workers * 4))
            if not rows:
                break
            decoded = executor.map(_audio, rows)
            for identity, speaker, audio in decoded:
                if args.limit is not None and processed >= args.limit:
                    break
                if identity in seen:
                    raise ValueError(f"Duplicate LibriSpeech ID: {identity}")
                seen.add(identity)
                processed += 1
                item_path = items_dir / f"{identity}.json"
                if item_path.is_file():
                    saved = json.loads(item_path.read_text())
                    manifest = saved["manifest"]
                    if (
                        manifest["dataset_revision"] != revision
                        or manifest["dataset_config"] != args.config
                        or manifest["dataset_split"] != args.split
                        or manifest["codec_sha256"] != codec_sha
                        or sha256(root / manifest["tokens_path"]) != manifest["sha256"]
                    ):
                        raise ValueError(
                            f"LibriSpeech cache resume mismatch: {identity}"
                        )
                    continue
                frames = math.ceil(len(audio) / FRAME_SAMPLES)
                proposed = max(
                    [frames] + [math.ceil(len(x[2]) / FRAME_SAMPLES) for x in pending]
                )
                if pending and (
                    len(pending) >= args.batch_items
                    or (len(pending) + 1) * proposed > args.max_padded_frames
                ):
                    flush()
                pending.append((identity, speaker, audio))
            if args.limit is not None and processed >= args.limit:
                break
    flush()
    items = [json.loads(path.read_text()) for path in sorted(items_dir.glob("*.json"))]
    if not items:
        raise ValueError("No LibriSpeech items were encoded")
    _atomic(
        root / "manifest.jsonl",
        ("\n".join(json.dumps(x["manifest"]) for x in items) + "\n").encode(),
    )
    _atomic(
        root / "companion.jsonl",
        ("\n".join(json.dumps(x["companion"]) for x in items) + "\n").encode(),
    )
    print(json.dumps({"items": len(items), "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
