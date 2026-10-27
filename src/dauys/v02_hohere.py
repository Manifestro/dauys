"""Adapters for the published Hohere frame labels and evaluation rows."""

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

FRAME_SAMPLES = 1920
SAMPLE_RATE = 24_000
EVAL_SEEN_VOICES = frozenset({"af_sarah", "am_michael", "bf_emma", "bm_george"})
EVAL_UNSEEN_VOICES = frozenset({"af_river", "am_adam", "af_sky", "bf_lily"})


def companion_from_label(row: Mapping[str, Any], frames: int) -> dict[str, Any]:
    """Convert the canonical Hohere labels without re-deriving activity from audio."""
    if (
        int(row["sample_rate"]) != SAMPLE_RATE
        or int(row["frame_samples"]) != FRAME_SAMPLES
    ):
        raise ValueError("Hohere labels must use 24 kHz and 1920-sample frames")
    if int(row["num_frames"]) != frames or frames < 1:
        raise ValueError("Hohere label count differs from Mimi frames")
    activity_a = [int(value) for value in row["activity_a"]]
    activity_b = [int(value) for value in row["activity_b"]]
    if len(activity_a) != frames or len(activity_b) != frames:
        raise ValueError("Hohere activity length differs from Mimi frames")
    if not set(activity_a + activity_b) <= {0, 1}:
        raise ValueError("Hohere activity must be binary")
    events: list[dict[str, Any]] = []
    for original in row["events"]:
        kind = str(original["type"])
        speaker = original.get("speaker")
        start = int(original["start_frame"])
        end = int(original["end_frame"])
        if not 0 <= start <= end <= frames:
            raise ValueError("Hohere event is outside the token sequence")
        mapped: str | None = None
        if (
            kind == "turn_end"
            and speaker == "A"
            and original.get("next_speaker") == "B"
        ):
            mapped, start = "a_turn_end", end
        elif kind == "response_onset" and speaker == "B":
            mapped, end = "response_onset", min(frames, start + 1)
        elif kind == "interruption" and speaker in ("A", "B"):
            mapped = "a_interrupts_b" if speaker == "A" else "b_interrupts_a"
            end = min(frames, start + 1)
        elif kind in ("backchannel", "cutoff") and speaker in ("A", "B"):
            mapped = kind
        if mapped is not None:
            events.append(
                {
                    "type": mapped,
                    "speaker": speaker,
                    "start_frame": start,
                    "end_frame": end,
                    "related": original.get("event_id"),
                }
            )
    events.sort(key=lambda item: (item["start_frame"], item["type"]))
    return {
        "id": str(row["id"]),
        "activity_a": activity_a,
        "activity_b": activity_b,
        "event_frames": events,
    }


def cache_with_label(
    cache: Mapping[str, torch.Tensor], companion: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Keep Mimi codes intact while replacing approximate activity with Hohere labels."""
    user, target = cache["user"], cache["target"]
    frames = len(companion["activity_a"])
    if user.shape != target.shape or user.shape != (8, frames):
        raise ValueError("Cached Mimi codes and Hohere labels have different geometry")
    return {
        **cache,
        "user_activity": torch.tensor(companion["activity_a"], dtype=torch.uint8),
        "activity": torch.tensor(companion["activity_b"], dtype=torch.uint8),
    }


def bench_item(row: Mapping[str, Any], suite: str) -> dict[str, Any]:
    """Keep evaluation metadata and probe fields separate from training targets."""
    if suite not in {f"E{number}" for number in range(1, 7)}:
        raise ValueError("Unknown Hohere suite")
    voice = str(row["voice_a"])
    if "voice_a_seen_in_training" in row:
        unseen = not bool(row["voice_a_seen_in_training"])
    elif suite == "E1" and voice in EVAL_SEEN_VOICES | EVAL_UNSEEN_VOICES:
        unseen = voice in EVAL_UNSEEN_VOICES
    else:
        raise ValueError("Hohere benchmark voice provenance is missing")
    item: dict[str, Any] = {
        "id": str(row["id"]),
        "user_voice": voice,
        "unseen_voice": unseen,
        "family": str(row.get("scenario") or f"hohere-bench/{suite}/{row['id']}"),
    }
    if suite == "E3":
        item["pair_id"] = str(row["pair_id"])
        item["kind"] = str(row["kind"])
        item["category"] = str(row["category"])
    elif suite == "E4":
        item["probe"] = {
            "distance_turns": int(row["distance"]),
            "updates": bool(row["update"]),
            "correct_index": int(row["correct_index"]),
            "stale_index": (
                None if row["stale_index"] is None else int(row["stale_index"])
            ),
        }
    elif suite == "E5":
        item["condition"] = str(row["condition"])
    if "total_samples" in row or "duration_samples" in row:
        samples = int(row.get("total_samples", row.get("duration_samples")))
        frames = math.ceil(samples / FRAME_SAMPLES)
        if (
            samples < FRAME_SAMPLES
            or int(row["sample_rate"]) != SAMPLE_RATE
            or ("num_frames" in row and int(row["num_frames"]) != frames)
        ):
            raise ValueError("Hohere benchmark frame geometry is invalid")
        item["frames"] = frames
    return item


def materialize_bench_row(
    row: Mapping[str, Any], suite: str, root: Path
) -> dict[str, Any]:
    """Save immutable audio beside a JSON-safe suite row, without model-facing text."""
    item = bench_item(row, suite)
    identity = item["id"]
    if not identity or not all(char.isalnum() or char in "_-" for char in identity):
        raise ValueError("Unsafe Hohere benchmark ID")
    audio: dict[str, dict[str, str]] = {}
    analysis: dict[str, Any] = {}
    for key, value in row.items():
        if key in (
            "info_json",
            "metadata_json",
            "events",
            "id",
            "voice_a",
            "voice_a_seen_in_training",
        ):
            continue
        if key in ("channel_a", "channel_b") or key.endswith("_audio"):
            if not isinstance(value, Mapping):
                raise ValueError(f"Missing audio payload: {key}")
            payload = value.get("bytes")
            if payload is None:
                path = value.get("path")
                if not path:
                    raise ValueError(f"Missing audio payload: {key}")
                payload = Path(path).read_bytes()
            if not isinstance(payload, bytes):
                raise ValueError(f"Invalid audio payload: {key}")
            relative = Path(suite) / "audio" / identity / f"{key}.wav"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Existing benchmark audio differs: {destination}")
            else:
                temporary = destination.with_suffix(".wav.tmp")
                with temporary.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
            audio[key] = {"path": str(relative), "sha256": digest}
        elif key in {
            "question_x",
            "question_y",
            "answer_x",
            "answer_y",
            "slot_x",
            "slot_y",
        }:
            analysis[key] = value
        elif key not in item:
            item[key] = value
    item["audio"] = audio
    if row.get("info_json"):
        analysis.update(json.loads(str(row["info_json"])))
    if analysis:
        item["analysis"] = analysis
    return item
