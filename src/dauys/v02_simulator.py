"""Frame-synchronous client for the external closed-loop conversation simulator."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch

from .decoding import DecodingConfig
from .model import DauysHybrid
from .streaming import AudioSession


class Simulator(Protocol):
    done: bool

    def reset(self) -> torch.Tensor: ...

    def step(self, model_frame: torch.Tensor) -> torch.Tensor: ...

    def report(self) -> Mapping[str, Any]: ...


SimulatorFactory = Callable[[Mapping[str, Any], int], Simulator]


def run_closed_loop(
    model: DauysHybrid,
    codec: Any,
    simulator_factory: SimulatorFactory,
    scenario: Mapping[str, Any],
    seed: int,
    decoding: DecodingConfig,
    max_frames: int = 10_000,
) -> Mapping[str, Any]:
    """Run model output back into a reacting user, one 80 ms frame per step."""
    if max_frames < 1:
        raise ValueError("Maximum frame count must be positive")
    if model.training:
        raise ValueError("Closed-loop evaluation requires model.eval()")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    simulator = simulator_factory(scenario, seed)
    user_frame = simulator.reset()
    if user_frame.shape != (1, 1920):
        raise ValueError("Simulator must return one 24 kHz, 80 ms mono frame")
    count = 0
    with AudioSession(model, codec, 1, decoding, generator) as session:
        while not simulator.done:
            if count >= max_frames:
                raise RuntimeError("Simulator exceeded the configured frame limit")
            output_frame, _activity = session.step(user_frame[None].to(device))
            user_frame = simulator.step(output_frame[0].detach().cpu())
            if user_frame.shape != (1, 1920):
                raise ValueError("Simulator step returned an invalid audio frame")
            count += 1
    report = simulator.report()
    if not isinstance(report, Mapping):
        raise TypeError("Simulator report must be a mapping")
    return report
