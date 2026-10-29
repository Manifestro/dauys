"""Stateful stream assembly and TBPTT chunk-loop checks."""

import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_stateful_build import assemble_chapter, build_split, slice_stream
from v02_stateful_eval import (
    aggregate_gains,
    foreign_stream_index,
    mode_nlls,
    shuffled_order,
    summarize_modes,
)
from v02_stateful_train import run_batch, stack_streams


def _phrase(value: int, frames: int = 2) -> dict:
    return {
        "codes": torch.full((8, frames), value, dtype=torch.long),
        "activity": torch.ones(frames, dtype=torch.long),
    }


def test_assemble_chapter_alternates_channels_and_inserts_silence():
    silence = torch.arange(8, dtype=torch.long)
    first, second = _phrase(11), _phrase(22)
    stream = assemble_chapter([first, second], silence, pause=6)
    assert stream["user"].shape == (8, 10)
    assert torch.equal(stream["user"][:, :2], first["codes"])
    assert torch.equal(stream["target"][:, :2], silence[:, None].expand(8, 2))
    assert stream["user_activity"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert stream["activity"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert torch.equal(stream["user"][:, 2:8], silence[:, None].expand(8, 6))
    assert torch.equal(stream["target"][:, 2:8], silence[:, None].expand(8, 6))
    assert torch.equal(stream["target"][:, 8:], second["codes"])


def test_slice_stream_discards_incomplete_windows_and_streams():
    silence = torch.zeros(8, dtype=torch.long)
    stream = assemble_chapter([_phrase(1), _phrase(2), _phrase(3)], silence)
    assert stream["user"].shape[-1] == 18
    streams = slice_stream(stream, chunk=5, chunks=2, chapter="c", speaker="s")
    assert len(streams) == 1
    assert streams[0]["user"].shape[-1] == 10


def test_build_split_sorts_phrases_and_drops_short_chapters(tmp_path):
    silence = torch.full((8,), 7, dtype=torch.long)
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    rows = []
    for identity, value in (
        ("s1-ch1-0002", 22),
        ("s1-ch1-0001", 11),
        ("s2-ch2-0001", 33),
    ):
        codes = torch.full((8, 2), value, dtype=torch.short)
        activity = torch.ones(2, dtype=torch.uint8)
        path = token_dir / f"{identity}.pt"
        torch.save({"target": codes, "activity": activity}, path)
        rows.append(
            {
                "id": identity,
                "source": "1a_on_B",
                "frames": 2,
                "tokens_path": f"tokens/{identity}.pt",
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    streams = build_split(tmp_path, silence, chunk=5, chunks=2)
    assert len(streams) == 1
    stream = streams[0]
    assert (stream["chapter"], stream["speaker"]) == ("s1-ch1", "s1")
    assert torch.equal(stream["user"][:, :2], torch.full((8, 2), 11))
    assert torch.equal(stream["target"][:, 8:10], torch.full((8, 2), 22))


class _Optimizer:
    def __init__(self):
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        self.steps += 1


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=5)
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, user, target, initial, mamba_states=None):
        self.calls.append(mamba_states)
        if mamba_states is not None:
            mamba_states["seen"] = mamba_states.get("seen", 0) + 1
        batch, frames = target.shape[0], target.shape[-1]
        zeros = lambda *shape: torch.zeros(*shape, requires_grad=True)
        return types.SimpleNamespace(
            cb0_logits=zeros(batch, frames, self.config.vocab_size),
            depth_logits=zeros(batch, frames, 7, self.config.vocab_size),
            activity_logits=zeros(batch, frames, 2),
        )


def _batch(frames: int = 12) -> dict:
    return stack_streams(
        [
            {
                "user": torch.zeros(8, frames, dtype=torch.long),
                "target": torch.zeros(8, frames, dtype=torch.long),
                "user_activity": torch.ones(frames, dtype=torch.long),
                "activity": torch.ones(frames, dtype=torch.long),
            }
        ]
    )


def test_run_batch_carries_states_and_steps_once_per_chunk():
    model = _FakeModel()
    optimizer = _Optimizer()
    logs = run_batch(
        model,
        optimizer,
        _batch(),
        chunk=4,
        stateful=True,
    )
    assert len(logs) == 3
    assert optimizer.steps == 3
    assert all(call is model.calls[0] for call in model.calls)
    assert model.calls[0] == {"seen": 3}


def test_run_batch_resets_states_without_stateful():
    model = _FakeModel()
    optimizer = _Optimizer()
    logs = run_batch(
        model,
        optimizer,
        _batch(),
        chunk=4,
        stateful=False,
    )
    assert len(logs) == 3
    assert optimizer.steps == 3
    assert all(call is None for call in model.calls)


def test_foreign_stream_round_robin_skips_the_same_speaker():
    streams = [
        {"speaker": "a", "user": torch.zeros(8, 8)},
        {"speaker": "a", "user": torch.zeros(8, 8)},
        {"speaker": "b", "user": torch.zeros(8, 8)},
        {"speaker": "a", "user": torch.zeros(8, 8)},
        {"speaker": "b", "user": torch.zeros(8, 8)},
    ]
    assert foreign_stream_index(streams, 0) == 2
    assert foreign_stream_index(streams, 2) == 3
    assert foreign_stream_index(streams, 4) == 0


def test_shuffled_order_is_a_deterministic_permutation():
    assert shuffled_order(0, 0, 0) == []
    assert shuffled_order(1, 0, 0) == [0]
    order = shuffled_order(5, 7, 3)
    assert sorted(order) == list(range(5))
    assert order == shuffled_order(5, 7, 3)


def test_aggregate_gains_and_k2plus_summary():
    constant = aggregate_gains([0.5, 0.5])
    assert constant == {
        "items": 2,
        "mean": 0.5,
        "ci95_low": 0.5,
        "ci95_high": 0.5,
        "positive_rate": 1.0,
    }
    assert aggregate_gains([])["items"] == 0
    nll = {
        "carry": [[0.6, 0.5, 0.4, 0.3], [0.7, 0.6, 0.5, 0.4]],
        "reset": [[0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]],
        "shuffled": [[0.6, 0.5, 0.45, 0.4], [0.7, 0.6, 0.55, 0.5]],
        "foreign": [[0.6, 0.55, 0.5, 0.45], [0.7, 0.65, 0.6, 0.55]],
    }
    summary = summarize_modes(nll)
    assert summary["by_chunk"]["1"]["gain_carry"]["mean"] == 0.2
    assert summary["by_chunk"]["1"]["gain_foreign"]["mean"] == 0.15
    assert summary["by_chunk"]["1"]["order"]["mean"] == 0.0
    assert summary["k2plus"]["gain_carry"]["mean"] == 0.35
    assert summary["k2plus"]["gain_shuffled"]["mean"] == 0.275
    assert summary["k2plus"]["specific"]["mean"] == 0.125


class _ChunkModel(nn.Module):
    def __init__(self, vocab: int = 5):
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=vocab)
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def __call__(self, user, target, initial, mamba_states=None):
        self.calls.append((user.shape, target.shape, initial.shape, mamba_states))
        batch, frames = target.shape[0], target.shape[-1]
        return types.SimpleNamespace(
            cb0_logits=torch.zeros(batch, frames, self.config.vocab_size)
        )


def test_mode_nlls_handles_real_stream_shapes_in_all_modes():
    frames, chunk = 16, 4
    streams = [
        {
            "user": torch.zeros(8, frames, dtype=torch.long),
            "target": torch.zeros(8, frames, dtype=torch.long),
            "user_activity": torch.ones(frames, dtype=torch.long),
            "activity": torch.ones(frames, dtype=torch.long),
            "chapter": "c1",
            "speaker": "s1",
        },
        {
            "user": torch.zeros(8, frames, dtype=torch.long),
            "target": torch.zeros(8, frames, dtype=torch.long),
            "user_activity": torch.ones(frames, dtype=torch.long),
            "activity": torch.ones(frames, dtype=torch.long),
            "chapter": "c2",
            "speaker": "s2",
        },
    ]
    for mode in ("carry", "reset", "shuffled", "foreign"):
        model = _ChunkModel()
        matrix = mode_nlls(model, streams, mode, chunk=chunk, seed=0)
        assert len(matrix) == 2
        assert all(len(row) == frames // chunk for row in matrix)
        assert all(
            user_shape == (1, 8, chunk)
            and target_shape == (1, 8, chunk)
            and initial_shape == (1, 8)
            for user_shape, target_shape, initial_shape, _ in model.calls
        )
        if mode == "reset":
            assert all(states is None for *_, states in model.calls)
        else:
            assert all(isinstance(states, dict) for *_, states in model.calls)
