"""Resumeable, batched Mimi cache builder for synchronized hohere audio."""

import argparse
import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset

from dauys.codec import encode, load_mimi
from dauys.v02_data import sha256
from dauys.v02_encode import (
    AudioConversation,
    companion_from_events,
    conversation_from_row,
    encode_conversations,
    encode_long_conversation,
)
from dauys.v02_hohere import cache_with_label, companion_from_label

DEFAULT_DATASET = "manifestro/hohere-1m"
DEFAULT_REVISION = "2e3880f30e8725ed242d9d2f71e0794d62ed1795"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_tokens(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _source_rows(dataset: str, revision: str, splits: list[str]):
    for split in splits:
        stream = load_dataset(dataset, split=split, revision=revision, streaming=True)
        for column in ("channel_a", "channel_b"):
            if isinstance(stream.features[column], Audio):
                stream = stream.cast_column(column, Audio(decode=False))
        yield from stream


def _decode_block(
    rows: list[dict[str, Any]], executor: ThreadPoolExecutor
) -> list[AudioConversation]:
    return list(executor.map(conversation_from_row, rows))


def _family(scenario: str) -> str:
    parts = scenario.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Scenario must contain a family component")
    return parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-conversations", type=int, default=8)
    parser.add_argument("--max-padded-frames", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--labels-parquet", type=Path)
    labels.add_argument(
        "--labels-revision", help="Pinned revision of the Hohere labels dataset"
    )
    parser.add_argument("--allow-derived-labels", action="store_true")
    args = parser.parse_args()
    # cuDNN TF32 convolutions make batched Mimi codes differ from single-item
    # codes in about 1% of acoustic tokens; full precision makes them identical.
    torch.backends.cudnn.allow_tf32 = False
    if not (args.labels_parquet or args.labels_revision or args.allow_derived_labels):
        raise ValueError(
            "Official Hohere labels are required for a research token cache"
        )
    if args.allow_derived_labels and (args.labels_parquet or args.labels_revision):
        raise ValueError("Derived and official labels cannot be selected together")
    if not torch.cuda.is_available():
        raise RuntimeError("Batched Mimi preprocessing requires CUDA")
    if min(args.batch_conversations, args.max_padded_frames, args.decode_workers) < 1:
        raise ValueError("Batch and worker settings must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Conversation limit must be positive")
    root = args.output
    labels_path = args.labels_parquet
    if args.labels_revision:
        from huggingface_hub import hf_hub_download

        labels_path = Path(
            hf_hub_download(
                repo_id="manifestro/hohere-dialogues",
                repo_type="dataset",
                filename="labels/hohere-1m/labels-00000-of-00001.parquet",
                revision=args.labels_revision,
                token=True,
            )
        )
    label_rows = None
    label_digest = None
    if labels_path is not None:
        import pyarrow.parquet as pq

        label_digest = sha256(labels_path)
        rows = pq.read_table(labels_path).to_pylist()
        label_rows = {str(row["id"]): row for row in rows}
        if len(label_rows) != len(rows):
            raise ValueError("Hohere labels contain duplicate conversation IDs")
    items_dir = root / "items"
    tokens_dir = root / "tokens"
    items_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    codec_sha = sha256(args.mimi_checkpoint)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    silence = encode(codec, torch.zeros(1, 1, 1920, device="cuda"))[0, :, 0].cpu()
    _save_tokens(root / "silence_codes.pt", {"codes": silence})
    batch: list[AudioConversation] = []
    checked_batch = False
    seen: set[str] = set()
    count = 0

    def save_cache(
        conversation: AudioConversation, cache: dict[str, torch.Tensor]
    ) -> None:
        token_path = tokens_dir / f"{conversation.id}.pt"
        if label_rows is not None:
            if conversation.id not in label_rows:
                raise ValueError(f"Missing Hohere labels: {conversation.id}")
            companion = companion_from_label(
                label_rows[conversation.id], conversation.frames
            )
            cache = cache_with_label(cache, companion)
        else:
            companion = companion_from_events(conversation)
        _save_tokens(token_path, cache)
        manifest = {
            "id": conversation.id,
            "source": "hohere",
            "skill": conversation.skill,
            "family": _family(conversation.scenario),
            "user_voice": conversation.voice_a,
            "frames": conversation.frames,
            "tokens_path": f"tokens/{conversation.id}.pt",
            "sha256": sha256(token_path),
            "dataset": args.dataset,
            "dataset_revision": args.revision,
            "codec_sha256": codec_sha,
        }
        if label_digest is not None:
            manifest["labels_sha256"] = label_digest
        _atomic_bytes(
            items_dir / f"{conversation.id}.json",
            (
                json.dumps({"manifest": manifest, "companion": companion}) + "\n"
            ).encode(),
        )

    def flush() -> None:
        nonlocal batch, checked_batch
        if not batch:
            return
        caches = encode_conversations(
            codec,
            batch,
            torch.device("cuda"),
            check_first_against_single=not checked_batch,
        )
        checked_batch = True
        for conversation, cache in zip(batch, caches):
            save_cache(conversation, cache)
        print(json.dumps({"encoded": len(batch), "total": count}), flush=True)
        batch = []

    rows = _source_rows(args.dataset, args.revision, args.splits)
    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        while True:
            block = list(itertools.islice(rows, args.decode_workers * 4))
            if not block:
                break
            fresh = []
            for row in block:
                identity = str(row["id"])
                if not re.fullmatch(r"[A-Za-z0-9_-]+", identity) or identity in seen:
                    raise ValueError("Unsafe or duplicate conversation ID")
                seen.add(identity)
                if args.limit is not None and count >= args.limit:
                    break
                count += 1
                item_path = items_dir / f"{identity}.json"
                if item_path.is_file():
                    saved = json.loads(item_path.read_text())
                    manifest = saved["manifest"]
                    if (
                        manifest["dataset"] != args.dataset
                        or manifest["dataset_revision"] != args.revision
                        or manifest["codec_sha256"] != codec_sha
                        or sha256(root / manifest["tokens_path"]) != manifest["sha256"]
                    ):
                        raise ValueError(f"Resume cache mismatch: {identity}")
                    if (
                        label_digest is not None
                        and manifest.get("labels_sha256") != label_digest
                    ):
                        if identity not in label_rows:
                            raise ValueError(f"Missing Hohere labels: {identity}")
                        companion = companion_from_label(
                            label_rows[identity], int(manifest["frames"])
                        )
                        token_path = root / manifest["tokens_path"]
                        cached = torch.load(
                            token_path, map_location="cpu", weights_only=True
                        )
                        _save_tokens(token_path, cache_with_label(cached, companion))
                        manifest["sha256"] = sha256(token_path)
                        manifest["labels_sha256"] = label_digest
                        _atomic_bytes(
                            item_path,
                            (
                                json.dumps(
                                    {"manifest": manifest, "companion": companion}
                                )
                                + "\n"
                            ).encode(),
                        )
                    continue
                fresh.append(row)
            for conversation in _decode_block(fresh, executor):
                if 2 * conversation.frames > args.max_padded_frames:
                    flush()
                    cache = encode_long_conversation(
                        codec,
                        conversation,
                        torch.device("cuda"),
                        args.max_padded_frames // 2,
                    )
                    save_cache(conversation, cache)
                    continue
                proposed_frames = max(
                    [item.frames for item in batch] + [conversation.frames]
                )
                if batch and (
                    len(batch) >= args.batch_conversations
                    or 2 * (len(batch) + 1) * proposed_frames > args.max_padded_frames
                ):
                    flush()
                batch.append(conversation)
            if args.limit is not None and count >= args.limit:
                break
    flush()
    items = [json.loads(path.read_text()) for path in sorted(items_dir.glob("*.json"))]
    if not items:
        raise ValueError("No encoded conversations were found")
    if any(
        item["manifest"]["dataset"] != args.dataset
        or item["manifest"]["dataset_revision"] != args.revision
        or item["manifest"]["codec_sha256"] != codec_sha
        or (
            label_digest is not None
            and item["manifest"].get("labels_sha256") != label_digest
        )
        for item in items
    ):
        raise ValueError("Output directory contains another dataset or codec revision")
    _atomic_bytes(
        root / "manifest.jsonl",
        ("\n".join(json.dumps(item["manifest"]) for item in items) + "\n").encode(),
    )
    _atomic_bytes(
        root / "companion.jsonl",
        ("\n".join(json.dumps(item["companion"]) for item in items) + "\n").encode(),
    )
    print(
        json.dumps(
            {"conversations": len(items), "manifest": str(root / "manifest.jsonl")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
