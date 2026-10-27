"""Audio probe geometry and free-run boundaries."""

import numpy as np
import pytest
import soundfile as sf
import torch

from dauys.v02_evaluation import Suite
from dauys.v02_hohere_backend import HohereProbeBackend, create_evaluator


def test_candidate_window_covers_unaligned_last_sample(tmp_path, monkeypatch):
    clip = np.full(1920, 0.25, dtype=np.float32)
    for name in ("question_x_audio", "answer_x_audio"):
        sf.write(tmp_path / f"{name}.wav", clip, 24_000)
    backend = HohereProbeBackend(tmp_path, {}, object(), torch.zeros(8))
    monkeypatch.setattr(
        backend,
        "_encode",
        lambda _a, _b: (torch.zeros(8, 3), torch.zeros(8, 3)),
    )
    item = {
        "id": "pair",
        "total_samples": 5760,
        "question_start_sample": 0,
        "answer_start_sample": 2000,
        "audio": {
            name: {"path": f"{name}.wav"}
            for name in ("question_x_audio", "answer_x_audio")
        },
    }
    _, _, start, end = backend._compose(item, "x", "x")
    assert (start, end) == (1, 3)


def test_e2_requires_real_transcription_and_judge(tmp_path):
    backend = HohereProbeBackend(
        tmp_path,
        {"E2": Suite("E2", "1", (), "a" * 64)},
        object(),
        torch.zeros(8),
    )
    with pytest.raises(ValueError, match="ASR and relevance"):
        backend.e2(object(), {}, object())  # type: ignore[arg-type]


def test_evaluator_factory_rejects_missing_external_dependencies(monkeypatch):
    for name in (
        "DAUYS_BENCH_ROOT",
        "DAUYS_MIMI_CHECKPOINT",
        "DAUYS_SILENCE_CODES",
        "DAUYS_ASR_CALLBACK",
        "DAUYS_RELEVANCE_CALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Missing evaluator configuration"):
        create_evaluator(suites={}, records=[])
