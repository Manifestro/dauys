"""Immutable corpus manifests, companion events, and frame-padded batches."""

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .v02_training import Event, Source, Stage, V02Batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative_file(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its corpus root")
    return candidate


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    source: Source
    skill: str
    family: str
    user_voice: str
    frames: int
    token_path: Path
    token_sha256: str
    activity_a: tuple[int, ...]
    activity_b: tuple[int, ...]
    events: tuple[Event, ...]
    negative_id: str | None = None
    negative_path: Path | None = None
    negative_sha256: str | None = None


def load_corpus(
    root: str | Path,
    manifest: str | Path,
    companion: str | Path,
    *,
    verify_hashes: bool = True,
) -> list[CorpusRecord]:
    root = Path(root)
    rows = _jsonl(Path(manifest))
    details = {row["id"]: row for row in _jsonl(Path(companion))}
    if not rows or len(details) != len(rows):
        raise ValueError("Manifest and companion must contain the same conversations")
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    for row in rows:
        identity = str(row["id"])
        if identity in seen or identity not in details:
            raise ValueError("Duplicate or missing companion conversation")
        seen.add(identity)
        if row.get("selected_split") in ("validation", "test"):
            raise ValueError("Evaluation splits cannot enter a training corpus")
        source = row["source"]
        if source not in ("1a_on_A", "1a_on_B", "1b", "hohere"):
            raise ValueError("Unknown corpus source")
        frames = int(row["frames"])
        detail = details[identity]
        activity_a = tuple(int(value) for value in detail["activity_a"])
        activity_b = tuple(int(value) for value in detail["activity_b"])
        if frames < 1 or len(activity_a) != frames or len(activity_b) != frames:
            raise ValueError("Companion activity does not match token frames")
        if not set(activity_a + activity_b) <= {0, 1}:
            raise ValueError("Activity must be binary")
        if source == "1a_on_A" and any(activity_b):
            raise ValueError("A-only speech cannot have B activity")
        if source == "1a_on_B" and any(activity_a):
            raise ValueError("B-only speech cannot have A activity")
        events = tuple(Event(**event) for event in detail["event_frames"])
        token_path = _relative_file(root, str(row["tokens_path"]))
        negative = detail.get("negative")
        if negative:
            if negative["user_voice"] != row["user_voice"]:
                raise ValueError("Matched negative must use the same user voice")
            if tuple(negative["activity_a"]) != activity_a:
                raise ValueError(
                    "Matched negative must preserve the user turn timeline"
                )
        negative_path = (
            _relative_file(root, str(negative["tokens_path"])) if negative else None
        )
        record = CorpusRecord(
            id=identity,
            source=source,
            skill=str(row.get("skill", "unspecified")),
            family=str(row["family"]),
            user_voice=str(row["user_voice"]),
            frames=frames,
            token_path=token_path,
            token_sha256=str(row["sha256"]),
            activity_a=activity_a,
            activity_b=activity_b,
            events=events,
            negative_id=str(negative["id"]) if negative else None,
            negative_path=negative_path,
            negative_sha256=str(negative["sha256"]) if negative else None,
        )
        if record.negative_id == record.id:
            raise ValueError("Negative cannot be the same conversation")
        if verify_hashes:
            if sha256(record.token_path) != record.token_sha256:
                raise ValueError(f"Token checksum mismatch: {identity}")
            if (
                record.negative_path is not None
                and sha256(record.negative_path) != record.negative_sha256
            ):
                raise ValueError(f"Negative checksum mismatch: {identity}")
        records.append(record)
    return records


def assert_suite_is_disjoint(
    training: list[CorpusRecord], suite: Sequence[Mapping[str, Any]]
) -> None:
    train_ids = {record.id for record in training}
    families = {record.family for record in training}
    voices = {record.user_voice for record in training}
    for item in suite:
        if item["id"] in train_ids or item["family"] in families:
            raise ValueError(
                "Evaluation suite overlaps a training conversation or family"
            )
        if item.get("unseen_voice", False) and item["user_voice"] in voices:
            raise ValueError("Evaluation suite marks a training voice as unseen")


class CorpusDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[CorpusRecord], crop_frames: int | None = None):
        if not records:
            raise ValueError("At least one training conversation is required")
        if crop_frames is not None and crop_frames < 1:
            raise ValueError("Crop length must be positive")
        self.records = records
        self.crop_frames = crop_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        crop_seed = 0
        if isinstance(index, tuple):
            index, crop_seed = index
        record = self.records[index]
        cached = torch.load(record.token_path, map_location="cpu", weights_only=True)
        user = cached["user"].long()
        output = cached["target"].long()
        if user.shape != output.shape or user.shape != (8, record.frames):
            raise ValueError(f"Invalid token geometry: {record.id}")
        if "activity" in cached and not torch.equal(
            cached["activity"].long(), torch.tensor(record.activity_b)
        ):
            raise ValueError(f"Cached and companion activity differ: {record.id}")
        if "user_activity" in cached and not torch.equal(
            cached["user_activity"].long(), torch.tensor(record.activity_a)
        ):
            raise ValueError(f"Cached and companion user activity differ: {record.id}")
        negative_user = None
        if record.negative_path is not None:
            negative_cached = torch.load(
                record.negative_path, map_location="cpu", weights_only=True
            )
            negative_user = negative_cached["user"].long()
            if negative_user.shape != user.shape:
                raise ValueError(f"Negative token geometry differs: {record.id}")
        initial_output = None
        if self.crop_frames is not None and record.frames > self.crop_frames:
            start = random.Random(crop_seed).randrange(
                record.frames - self.crop_frames + 1
            )
            end = start + self.crop_frames
            initial_output = output[:, start - 1].clone() if start else None
            user = user[:, start:end]
            output = output[:, start:end]
            negative_user = (
                negative_user[:, start:end] if negative_user is not None else None
            )
            events = tuple(
                replace(
                    event,
                    start_frame=max(0, event.start_frame - start),
                    end_frame=min(self.crop_frames, event.end_frame - start),
                )
                for event in record.events
                if event.start_frame < end
                and (event.end_frame > start or event.start_frame >= start)
            )
            record = replace(
                record,
                frames=self.crop_frames,
                activity_a=record.activity_a[start:end],
                activity_b=record.activity_b[start:end],
                events=events,
            )
        return {
            "record": record,
            "user": user,
            "output": output,
            "negative_user": negative_user,
            "initial_output": initial_output,
        }


def collate_cases(cases: list[dict[str, Any]], silence_codes: torch.Tensor) -> V02Batch:
    if not cases or silence_codes.shape != (8,):
        raise ValueError("A nonempty batch and eight silence codes are required")
    batch = len(cases)
    frames = max(case["record"].frames for case in cases)
    user = silence_codes[None, :, None].expand(batch, 8, frames).clone()
    output = user.clone()
    negative = user.clone()
    activity_a = torch.zeros((batch, frames), dtype=torch.long)
    activity_b = torch.zeros((batch, frames), dtype=torch.long)
    valid = torch.zeros((batch, frames), dtype=torch.bool)
    initial_output = silence_codes[None].expand(batch, 8).clone()
    records = [case["record"] for case in cases]
    for index, case in enumerate(cases):
        record = records[index]
        length = record.frames
        user[index, :, :length] = case["user"]
        output[index, :, :length] = case["output"]
        negative[index, :, :length] = (
            case["negative_user"] if case["negative_user"] is not None else case["user"]
        )
        activity_a[index, :length] = torch.tensor(record.activity_a)
        activity_b[index, :length] = torch.tensor(record.activity_b)
        valid[index, :length] = True
        if case["initial_output"] is not None:
            initial_output[index] = case["initial_output"]
    result = V02Batch(
        user_codes=user,
        output_codes=output,
        user_activity=activity_a,
        output_activity=activity_b,
        valid_frames=valid,
        initial_output_codes=initial_output,
        silence_codes=silence_codes.expand(batch, 8).clone(),
        sources=tuple(record.source for record in records),
        conversation_ids=tuple(record.id for record in records),
        events=tuple(record.events for record in records),
        negative_user_codes=negative,
        negative_ids=tuple(record.negative_id or record.id for record in records),
    )
    result.validate()
    return result


def sample_records(
    records: list[CorpusRecord],
    stage: Stage,
    progress: float,
    count: int,
    seed: int,
    replay_fraction: float = 0.15,
    replay_offset: int = 0,
    single_stream_only: bool = False,
) -> list[CorpusRecord]:
    if (
        not 0 <= progress <= 1
        or count < 1
        or not 0 <= replay_fraction <= 1
        or replay_offset < 0
    ):
        raise ValueError("Invalid sampling progress or count")
    rng = random.Random(seed)
    if stage == "pretrain":
        single = [record for record in records if record.source.startswith("1a_")]
        dialogue = [record for record in records if record.source == "1b"]
        if not single or (not single_stream_only and not dialogue):
            raise ValueError(
                "Pre-training requires the configured speech sources"
            )
        if single_stream_only:
            return [rng.choice(single) for _ in range(count)]
        return [
            rng.choice(single if rng.random() < 0.7 else dialogue) for _ in range(count)
        ]
    if stage == "posttrain":
        targeted = [
            record
            for record in records
            if record.source == "hohere"
            and record.skill
            in {
                "yielding",
                "interruption",
                "backchannel",
                "context_memory",
                "smart_home",
            }
        ]
        if not targeted:
            raise ValueError("Post-training requires targeted hohere examples")
        return [rng.choice(targeted) for _ in range(count)]
    if stage != "dialogue":
        raise ValueError("Invalid training stage")
    replay = [record for record in records if record.source != "hohere"]
    hohere = [record for record in records if record.source == "hohere"]
    if not hohere or (replay_fraction > 0 and not replay):
        raise ValueError(
            "Dialogue training requires hohere and requested replay records"
        )
    easy_skills = {"simple_dialogue", "turn_taking"}
    easy = [record for record in hohere if record.skill in easy_skills]
    hard = [record for record in hohere if record.skill not in easy_skills]
    if not easy or not hard:
        raise ValueError("Dialogue curriculum requires easy and hard skills")
    easy_weight = 2 - min(progress / 0.3, 1)
    replay_count = math.floor((replay_offset + count) * replay_fraction) - math.floor(
        replay_offset * replay_fraction
    )
    result = [rng.choice(replay) for _ in range(replay_count)]
    for _ in range(count - replay_count):
        result.append(
            rng.choice(easy if rng.random() < easy_weight / (easy_weight + 1) else hard)
        )
    rng.shuffle(result)
    return result
