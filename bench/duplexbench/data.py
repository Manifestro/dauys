"""Test sets: hohere-bench conversations, ZeroSpeech sLM21 dev, Spoken StoryCloze.

Everything is encoded once with Mimi (8 codebooks) and cached as tensors, so
every model is scored on identical codes.
"""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1920


def load_mimi(device: str = "cuda"):
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(path, device=device)
    mimi.set_num_codebooks(8)
    return mimi.eval()


def read_audio(source, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 audio at ``rate`` from a path or WAV bytes."""
    import soundfile
    from scipy.signal import resample_poly

    data, sr = soundfile.read(
        io.BytesIO(source) if isinstance(source, bytes) else source, dtype="float32"
    )
    if data.ndim == 2:
        data = data.mean(1)
    if sr != rate:
        divisor = np.gcd(sr, rate)
        data = resample_poly(data, rate // divisor, sr // divisor).astype(np.float32)
    return data


@torch.inference_mode()
def encode(mimi, audio: np.ndarray) -> torch.Tensor:
    """[8, T] Mimi codes; audio is padded to whole frames."""
    pad = (-len(audio)) % FRAME_SAMPLES
    wav = torch.from_numpy(np.pad(audio, (0, pad)))[None, None]
    device = next(mimi.parameters()).device
    return mimi.encode(wav.to(device))[0].long().cpu()


@torch.inference_mode()
def decode(mimi, codes: torch.Tensor) -> np.ndarray:
    device = next(mimi.parameters()).device
    return mimi.decode(codes[None].to(device))[0, 0].float().cpu().numpy()


def hohere_conversations(revision: str = "v0.2.0") -> list[dict]:
    """hohere-bench E1: user (A) and assistant (B) channels plus frame activity."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            "manifestro/hohere-bench",
            repo_type="dataset",
            revision=revision,
            allow_patterns=["e1/data/*", "e1/labels/*"],
        )
    )
    labels = {
        row["id"]: row
        for row in pq.read_table(
            next((root / "e1" / "labels").glob("*.parquet"))
        ).to_pylist()
    }
    conversations = []
    for path in sorted((root / "e1" / "data").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            label = labels[row["id"]]
            conversations.append(
                {
                    "id": row["id"],
                    "skill": row["skill"],
                    "voice_a": row["voice_a"],
                    "voice_b": row["voice_b"],
                    "audio_a": read_audio(row["channel_a"]["bytes"]),
                    "audio_b": read_audio(row["channel_b"]["bytes"]),
                    "activity_a": torch.tensor(label["activity_a"], dtype=torch.bool),
                    "activity_b": torch.tensor(label["activity_b"], dtype=torch.bool),
                }
            )
    return conversations


def slm21_pairs(root: Path, task: str) -> list[dict]:
    """sWUGGY (``lexical``) or sBLIMP (``syntactic``) dev pairs.

    Each (id, voice) group holds one correct and one incorrect recording.
    """
    split = root / task / "dev"
    groups: dict[tuple[str, str], dict] = defaultdict(dict)
    with (split / "gold.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["id"], row["voice"])
            side = "good" if row["correct"] == "1" else "bad"
            groups[key][side] = split / f"{row['filename']}.wav"
            groups[key].setdefault("kind", row.get("type") or row.get("frequency"))
    return [
        {"id": f"{key[0]}-{key[1]}", **value}
        for key, value in sorted(groups.items())
        if "good" in value and "bad" in value
    ]


def storycloze_pairs(root: Path) -> list[dict]:
    """tStoryCloze: each story read with its true ending and a random one.

    The release names files ``<story>_correct.wav`` and ``<story>_incorrect.wav``.
    """
    pairs = []
    for good in sorted(root.rglob("*_correct.wav")):
        bad = good.with_name(good.name.replace("_correct.wav", "_incorrect.wav"))
        if bad.exists():
            pairs.append(
                {"id": good.stem[: -len("_correct")], "good": good, "bad": bad}
            )
    return pairs


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=float) + "\n")
