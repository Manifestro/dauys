"""Deterministic closed-loop frame contract with a local fake simulator."""

import torch

from dauys.decoding import DecodingConfig
from dauys.v02_simulator import run_closed_loop


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))


class _Session:
    def __init__(self, _model, _codec, batch_size, _decoding, _generator):
        assert batch_size == 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def step(self, user_frame):
        assert user_frame.shape == (1, 1, 1920)
        return user_frame + 1, torch.zeros((1, 2))


class _Simulator:
    def __init__(self, scenario, seed):
        self.scenario = scenario
        self.seed = seed
        self.frames = []
        self.done = False

    def reset(self):
        return torch.zeros((1, 1920))

    def step(self, model_frame):
        self.frames.append(float(model_frame.mean()))
        self.done = len(self.frames) == 3
        return torch.full((1, 1920), float(len(self.frames)))

    def report(self):
        return {
            "seed": self.seed,
            "frames": self.frames,
            "scenario": self.scenario["id"],
        }


def test_same_scenario_and_seed_produce_identical_closed_loop_report(monkeypatch):
    monkeypatch.setattr("dauys.v02_simulator.AudioSession", _Session)
    args = (
        _Model().eval(),
        object(),
        _Simulator,
        {"id": "scenario-1"},
        42,
        DecodingConfig(),
    )
    first = run_closed_loop(*args)
    second = run_closed_loop(*args)
    assert (
        first
        == second
        == {"seed": 42, "frames": [1.0, 2.0, 3.0], "scenario": "scenario-1"}
    )


def test_closed_loop_rejects_training_mode():
    try:
        run_closed_loop(
            _Model(), object(), _Simulator, {}, 42, DecodingConfig()
        )
    except ValueError as error:
        assert "model.eval()" in str(error)
    else:
        raise AssertionError("Training mode must not enter evaluation")
