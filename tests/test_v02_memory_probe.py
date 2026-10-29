"""Semantic-memory probe construction and bootstrap checks."""

import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_memory_probe_build import build_probes
from v02_memory_probe_eval import bootstrap_ci, probe_nlls


def _clip(root: Path, name: str, value: int, frames: int = 2) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codes": torch.full((8, frames), value, dtype=torch.short),
            "activity": torch.ones(frames, dtype=torch.uint8),
        },
        path,
    )
    return name


def _fixture(tmp_path: Path):
    child = tmp_path / "child"
    child.mkdir()
    rows = []
    for concept, voice in (("cx", "vp"), ("cx", "vq"), ("cy", "vp")):
        value = {"vp": 11, "vq": 22}.get(voice, 33) + (1 if concept == "cy" else 0)
        clip = _clip(
            child,
            f"clips/{concept}-{voice}.pt",
            value,
            frames=3 if concept == "cy" else 2,
        )
        rows.append(
            {
                "id": f"{concept}-{voice}",
                "speaker": voice,
                "topic": "color",
                "concept": concept,
                "frame": 0,
                "a_clip": clip,
            }
        )
    (child / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    filler = tmp_path / "filler"
    (filler / "tokens").mkdir(parents=True)
    filler_rows = []
    for index, value in enumerate((50, 60)):
        identity = f"filler-{index}"
        path = filler / "tokens" / f"{identity}.pt"
        torch.save(
            {
                "target": torch.full((8, 10), value, dtype=torch.short),
                "activity": torch.ones(10, dtype=torch.uint8),
            },
            path,
        )
        filler_rows.append(
            {
                "id": identity,
                "tokens_path": f"tokens/{identity}.pt",
                "source": "1a_on_B",
                "frames": 10,
            }
        )
    (filler / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in filler_rows)
    )
    return child, filler


def test_memory_probe_builds_controlled_filler_and_score_region(tmp_path):
    child, filler = _fixture(tmp_path)
    silence = torch.full((8,), 7, dtype=torch.long)
    streams = build_probes(
        child,
        filler,
        silence,
        gaps=[0, 5],
        limit=10,
        chunk=4,
        seed=0,
    )
    assert len(streams) == 4
    by_key = {(stream["gap"], stream["kind"]): stream for stream in streams}
    for gap in (0, 5):
        same = by_key[(gap, "same")]
        control = by_key[(gap, "control")]
        assert same["item"] == control["item"]
        assert same["user"].shape[-1] % 4 == 0
        assert torch.equal(same["user"][:, 3:], control["user"][:, 3:])
        assert torch.equal(same["target"][:, 3:], control["target"][:, 3:])
        assert not torch.equal(same["user"][:, :3], control["user"][:, :3])
        start = 3 + 6 + gap + 6
        probe = torch.ones(2, dtype=torch.bool)
        assert same["score"].sum() == probe.sum()
        assert not same["score"][:start].any()
        assert torch.equal(same["score"][start : start + 2], probe)
        filler_start = 3 + 6
        assert same["user"].shape[-1] >= filler_start + gap
        assert torch.equal(
            same["user"][:, filler_start : filler_start + gap],
            control["user"][:, filler_start : filler_start + gap],
        )


def test_bootstrap_ci_is_stable_for_constant_values():
    assert bootstrap_ci([1.0] * 10, samples=100, seed=0) == (1.0, 1.0)
    assert bootstrap_ci([], samples=100, seed=0) == (None, None)


class _ProbeModel(nn.Module):
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


def _probe_stream(frames: int) -> dict:
    return {
        "user": torch.zeros(8, frames, dtype=torch.long),
        "target": torch.zeros(8, frames, dtype=torch.long),
        "user_activity": torch.ones(frames, dtype=torch.long),
        "activity": torch.ones(frames, dtype=torch.long),
        "score": torch.tensor([False] * (frames - 2) + [True, True], dtype=torch.bool),
        "item": "item",
        "gap": 0,
        "kind": "same",
    }


def test_probe_nlls_checks_shapes_initial_and_length_batches():
    streams = [_probe_stream(12), _probe_stream(12), _probe_stream(8)]
    model = _ProbeModel()
    values = probe_nlls(
        model,
        streams,
        chunk=4,
        batch_size=2,
        stateful=True,
    )
    assert len(values) == 3
    assert [call[0][0] for call in model.calls] == [2, 2, 2, 1, 1]
    for user_shape, target_shape, initial_shape, _ in model.calls:
        batch = user_shape[0]
        assert user_shape == (batch, 8, 4)
        assert target_shape == (batch, 8, 4)
        assert initial_shape == (batch, 8)
    assert all(isinstance(states, dict) for *_, states in model.calls)
