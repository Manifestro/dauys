"""Stateful stream assembly and TBPTT chunk-loop checks."""

import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_stateful_build import assemble_chapter, build_split, slice_stream
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
