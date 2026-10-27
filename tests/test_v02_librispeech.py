"""LibriSpeech input normalization for single-stream pretraining."""

import io
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/v02_encode_librispeech.py"))
_activity = _script["_activity"]
_audio = _script["_audio"]


def test_librispeech_audio_resamples_to_mimi_rate():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for audio resampling")
    source = np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, source, 16_000, format="FLAC")
    identity, speaker, audio = _audio(
        {"id": "123-45-6", "speaker_id": 123, "audio": {"bytes": buffer.getvalue()}}
    )
    assert (identity, speaker) == ("123-45-6", "123")
    assert len(audio) == 24_000
    assert _activity(audio, 13) == [1] * 13
