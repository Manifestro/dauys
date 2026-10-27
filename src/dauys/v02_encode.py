"""Batched Mimi tokenization and audio-derived event companions."""

import io
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .codec import encode

FRAME_SAMPLES = 1920
SPEECH_TYPES = {
    "normal_turn",
    "backchannel",
    "overlap",
    "interruption",
    "self_correction",
}


@dataclass(frozen=True)
class AudioConversation:
    id: str
    user: np.ndarray
    target: np.ndarray
    samples: int
    events: tuple[dict[str, Any], ...]
    scenario: str
    skill: str
    voice_a: str

    @property
    def frames(self) -> int:
        return math.ceil(self.samples / FRAME_SAMPLES)


def pcm(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            audio, sample_rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
        elif value.get("path"):
            audio, sample_rate = sf.read(value["path"], dtype="float32")
        elif value.get("array") is not None:
            audio = np.asarray(value["array"], dtype=np.float32)
            sample_rate = value.get("sampling_rate")
        else:
            raise ValueError("Audio row has no data")
    else:
        raise TypeError("Audio field must be a mapping")
    if sample_rate != 24_000 or audio.ndim != 1:
        raise ValueError("Expected mono audio at 24 kHz")
    return audio


def conversation_from_row(row: dict[str, Any]) -> AudioConversation:
    user = pcm(row["channel_a"])
    target = pcm(row["channel_b"])
    samples = int(row["duration_samples"])
    if len(user) != samples or len(target) != samples or samples < FRAME_SAMPLES:
        raise ValueError("Conversation channels are not synchronized")
    return AudioConversation(
        id=str(row["id"]),
        user=user,
        target=target,
        samples=samples,
        events=tuple(row["events"]),
        scenario=str(row["scenario"]),
        skill=str(row["skill"]),
        voice_a=str(row["voice_a"]),
    )


def companion_from_events(conversation: AudioConversation) -> dict[str, Any]:
    frames = conversation.frames
    activity = {
        "A": np.zeros(frames, dtype=np.uint8),
        "B": np.zeros(frames, dtype=np.uint8),
    }
    events: list[dict[str, Any]] = []
    a_ends: list[int] = []
    b_starts: list[int] = []
    for event in conversation.events:
        speaker = event["speaker"]
        kind = event["event_type"]
        start = max(0, int(event["start_sample"]) // FRAME_SAMPLES)
        end = min(frames, math.ceil(int(event["end_sample"]) / FRAME_SAMPLES))
        related = event.get("related_event_id") or event.get("cause_event_id")
        if kind in SPEECH_TYPES:
            if speaker not in ("A", "B") or end <= start:
                raise ValueError("Invalid speech event")
            activity[speaker][start:end] = 1
            if speaker == "A" and kind != "backchannel":
                a_ends.append(end)
            elif kind not in ("backchannel", "interruption"):
                b_starts.append(start)
        if kind == "interruption" and speaker in ("A", "B"):
            events.append(
                {
                    "type": "a_interrupts_b" if speaker == "A" else "b_interrupts_a",
                    "speaker": speaker,
                    "start_frame": start,
                    "end_frame": min(frames, start + 1),
                    "related": related,
                }
            )
        if kind in ("backchannel", "cutoff") and speaker in ("A", "B"):
            events.append(
                {
                    "type": kind,
                    "speaker": speaker,
                    "start_frame": start,
                    "end_frame": end,
                    "related": related,
                }
            )
    for end in a_ends:
        events.append(
            {
                "type": "a_turn_end",
                "speaker": "A",
                "start_frame": end,
                "end_frame": end,
                "related": None,
            }
        )
    for start in b_starts:
        if any(-3 <= start - end <= 25 for end in a_ends):
            events.append(
                {
                    "type": "response_onset",
                    "speaker": "B",
                    "start_frame": start,
                    "end_frame": min(frames, start + 1),
                    "related": None,
                }
            )
    events.sort(key=lambda item: (item["start_frame"], item["type"]))
    return {
        "id": conversation.id,
        "activity_a": activity["A"].tolist(),
        "activity_b": activity["B"].tolist(),
        "event_frames": events,
    }


@torch.inference_mode()
def encode_conversations(
    codec: Any,
    conversations: list[AudioConversation],
    device: torch.device,
    *,
    check_first_against_single: bool = False,
) -> list[dict[str, torch.Tensor]]:
    if not conversations:
        raise ValueError("An encoding batch cannot be empty")
    largest_frames = max(item.frames for item in conversations)
    largest_samples = largest_frames * FRAME_SAMPLES
    channels = []
    for item in conversations:
        for audio in (item.user, item.target):
            channels.append(np.pad(audio, (0, largest_samples - len(audio))))
    pcm_batch = torch.from_numpy(np.stack(channels).copy())[:, None].to(device)
    codes = encode(codec, pcm_batch)
    if codes.shape != (2 * len(conversations), 8, largest_frames):
        raise ValueError("Mimi returned an unexpected token batch shape")
    if check_first_against_single:
        first = conversations[0]
        for channel, audio in enumerate((first.user, first.target)):
            padded = np.pad(audio, (0, first.frames * FRAME_SAMPLES - first.samples))
            reference = encode(
                codec, torch.from_numpy(padded.copy())[None, None].to(device)
            )
            if not torch.equal(codes[channel, :, : first.frames], reference[0]):
                differences = codes[channel, :, : first.frames] != reference[0]
                mismatched = int(differences.sum())
                by_codebook = differences.sum(dim=1).tolist()
                total = reference[0].numel()
                message = (
                    f"Batched Mimi encoding differs from single-item encoding "
                    f"on channel {channel}: {mismatched}/{total} tokens, "
                    f"by codebook {by_codebook}"
                )
                if mismatched / total > 0.01 or by_codebook[0] / first.frames > 0.005:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
    result = []
    for index, item in enumerate(conversations):
        companion = companion_from_events(item)
        result.append(
            {
                "user": codes[2 * index, :, : item.frames].short().cpu(),
                "target": codes[2 * index + 1, :, : item.frames].short().cpu(),
                "user_activity": torch.tensor(
                    companion["activity_a"], dtype=torch.uint8
                ),
                "activity": torch.tensor(companion["activity_b"], dtype=torch.uint8),
            }
        )
    return result


@torch.inference_mode()
def encode_long_conversation(
    codec: Any,
    conversation: AudioConversation,
    device: torch.device,
    chunk_frames: int,
) -> dict[str, torch.Tensor]:
    """Encode long synchronized channels with persistent codec stream state."""
    if chunk_frames < 1:
        raise ValueError("Chunk length must be positive")
    padded_samples = conversation.frames * FRAME_SAMPLES
    audio = np.stack(
        [
            np.pad(conversation.user, (0, padded_samples - conversation.samples)),
            np.pad(conversation.target, (0, padded_samples - conversation.samples)),
        ]
    )
    pcm_batch = torch.from_numpy(audio.copy())[:, None]
    chunks = []
    with codec.streaming(batch_size=2):
        for start in range(0, conversation.frames, chunk_frames):
            stop = min(conversation.frames, start + chunk_frames)
            codes = encode(
                codec,
                pcm_batch[:, :, start * FRAME_SAMPLES : stop * FRAME_SAMPLES].to(
                    device
                ),
            )
            if codes.shape != (2, 8, stop - start):
                raise ValueError("Streaming Mimi returned an unexpected token shape")
            chunks.append(codes.short().cpu())
    joined = torch.cat(chunks, dim=-1)
    companion = companion_from_events(conversation)
    return {
        "user": joined[0],
        "target": joined[1],
        "user_activity": torch.tensor(companion["activity_a"], dtype=torch.uint8),
        "activity": torch.tensor(companion["activity_b"], dtype=torch.uint8),
    }
