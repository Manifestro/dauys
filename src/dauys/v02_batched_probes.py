"""Batched scoring of the frozen Hohere suites.

The per-item probe backend scores one conversation at a time, so a full E1–E6
pass leaves the GPU mostly idle. Here every suite runs as padded batches:
teacher-forced windows are right-padded (the model is causal, so padding after
a window cannot change it), and generation advances all items frame by frame
together, each item switching from its reference to its own output at its own
start frame. Metric definitions are the backend's; only sampling order changes,
so generation-based numbers match the per-item path in distribution, not draw
for draw.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from .codec import decode
from .decoding import (
    ActivityState,
    DecodingConfig,
    activity_decision,
    cb0_decision,
    sample_logits,
)
from .model import DauysHybrid, previous_frames
from .v02_evaluation import word_error_rate

if TYPE_CHECKING:
    from .v02_hohere_backend import HohereProbeBackend

FRAME_BUDGET = 24_000
GENERATION_BATCH = 256
TranscribeMany = Callable[[list[np.ndarray]], list[str]]
JudgeMany = Callable[[list[tuple[str, str]]], list[float]]


def _groups(lengths: Sequence[int], budget: int = FRAME_BUDGET) -> list[list[int]]:
    """Indices grouped by similar length so batch * longest stays under budget."""
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    groups: list[list[int]] = []
    current: list[int] = []
    for index in order:
        longest = max([lengths[i] for i in current] + [lengths[index]])
        if current and longest * (len(current) + 1) > budget:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


class BatchedProbeScorer:
    def __init__(
        self,
        backend: "HohereProbeBackend",
        transcribe_many: TranscribeMany | None,
        judge_many: JudgeMany | None,
    ) -> None:
        self.backend = backend
        self.transcribe_many = transcribe_many
        self.judge_many = judge_many

    def _pad(
        self, streams: Sequence[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        frames = max(stream.shape[1] for stream in streams)
        silence = self.backend.silence_codes[:, None]
        return torch.stack(
            [
                torch.cat((stream, silence.expand(-1, frames - stream.shape[1])), dim=1)
                for stream in streams
            ]
        ).to(device)

    @torch.inference_mode()
    def cb0_nll_many(
        self,
        model: DauysHybrid,
        cases: Sequence[tuple[torch.Tensor, torch.Tensor, int, int]],
        fusion_off: frozenset[int] = frozenset(),
    ) -> list[float]:
        """Teacher-forced CB0 NLL over each case's [start, end) window."""
        device = next(model.parameters()).device
        results = [0.0] * len(cases)
        for group in _groups([case[0].shape[1] for case in cases]):
            a = self._pad([cases[i][0] for i in group], device)
            b = self._pad([cases[i][1] for i in group], device)
            silence = self.backend.silence_codes.to(device).expand(len(group), -1)
            listening = model.encode_listening(
                a, own_previous_codes=previous_frames(b, silence)
            )
            states = model.run_dialogue(
                listening, b, silence, fusion_off_blocks=fusion_off
            )
            logits = model.cb0_head(states)
            for row, index in enumerate(group):
                _, _, start, end = cases[index]
                results[index] = self.backend._window_nll(
                    logits[row], b[row, 0], start, end
                )
        return results

    @torch.inference_mode()
    def e1_many(self, model: DauysHybrid, items: Sequence[Mapping[str, Any]]) -> list:
        device = next(model.parameters()).device
        channels = [self.backend._channels(item) for item in items]
        results: list[dict[str, Any]] = [{} for _ in items]
        for group in _groups(
            [user.shape[1] for user, _ in channels], FRAME_BUDGET // 4
        ):
            a = self._pad([channels[i][0] for i in group], device)
            b = self._pad([channels[i][1] for i in group], device)
            silence = self.backend.silence_codes.to(device).expand(len(group), -1)
            result = model(a, b, silence)
            if result.depth_logits is None:
                raise ValueError("E1 requires depth logits")
            for row, index in enumerate(group):
                frames = channels[index][1].shape[1]
                cb0 = result.cb0_logits[row, :frames]
                target = b[row, :, :frames]
                depth = result.depth_logits[row, :frames]
                results[index] = {
                    "cb0_nll": self.backend._window_nll(cb0, target[0], 0, frames),
                    "cb0_accuracy": float(
                        (cb0.argmax(dim=-1) == target[0]).float().mean()
                    ),
                    "depth_nll": float(
                        F.cross_entropy(
                            depth.float().reshape(-1, depth.shape[-1]),
                            target[1:].transpose(0, 1).reshape(-1),
                        )
                    ),
                }
        cases = [(user, output, 0, output.shape[1]) for user, output in channels]
        ablations = {
            block: self.cb0_nll_many(model, cases, frozenset({block}))
            for block in model.fusion_block_indices()
        }
        for index, result in enumerate(results):
            result["fusion_off_cb0_nll_by_block"] = {
                str(block): values[index] for block, values in ablations.items()
            }
        return results

    @torch.inference_mode()
    def generate_many(
        self,
        model: DauysHybrid,
        requests: Sequence[tuple[torch.Tensor, torch.Tensor, int]],
        decoding: DecodingConfig,
        seed: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Reference frames before each start, then the model's own output.

        Equivalent to prefill_teacher over [0, start) followed by generate_step,
        for every request at once.
        """
        decoding.validate()
        device = next(model.parameters()).device
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.empty(0), torch.empty(0))
        ] * len(requests)
        generator = torch.Generator(device=device).manual_seed(seed)
        order = sorted(range(len(requests)), key=lambda i: requests[i][0].shape[1])
        for chunk_start in range(0, len(order), GENERATION_BATCH):
            group = order[chunk_start : chunk_start + GENERATION_BATCH]
            a = self._pad([requests[i][0] for i in group], device)
            b = self._pad([requests[i][1] for i in group], device)
            starts = torch.tensor([requests[i][2] for i in group], device=device)
            batch, _, frames = a.shape
            silence = self.backend.silence_codes.to(device).expand(batch, -1)
            state = model.new_stream(silence)
            activity_state = ActivityState.silent(batch, device)
            previous = silence
            codes = torch.zeros((batch, 8, frames), dtype=torch.long, device=device)
            speaking_trace = torch.zeros(
                (batch, frames), dtype=torch.bool, device=device
            )
            for frame in range(frames):
                cb0, activity, context = model.stream_step_teacher(
                    a[:, :, frame], previous, state
                )
                live = frame >= starts
                if not live.any():
                    previous = b[:, :, frame]
                    continue
                speaking = activity_decision(
                    activity, decoding, activity_state, generator
                )
                generated = [cb0_decision(cb0, speaking, decoding, generator)]
                for _ in range(1, 8):
                    prefix = torch.stack(generated, dim=-1)[:, None]
                    logits = model.depth(prefix, context)[:, 0, -1]
                    generated.append(
                        sample_logits(
                            logits,
                            decoding.depth_temperature,
                            decoding.depth_top_k,
                            generator,
                        )
                        if decoding.depth_mode == "sample"
                        else logits.argmax(dim=-1)
                    )
                frame_codes = torch.stack(generated, dim=1)
                # Rows still in their reference prefix keep a fresh, silent
                # activity state, exactly as if generation had not started.
                activity_state.speaking = activity_state.speaking & live
                activity_state.speaking_frames = torch.where(
                    live, activity_state.speaking_frames, 0
                )
                activity_state.silent_frames = torch.where(
                    live, activity_state.silent_frames, 1_000_000
                )
                codes[:, :, frame] = frame_codes
                speaking_trace[:, frame] = speaking & live
                previous = torch.where(live[:, None], frame_codes, b[:, :, frame])
            for row, index in enumerate(group):
                start = requests[index][2]
                end = requests[index][0].shape[1]
                outputs[index] = (
                    codes[row, :, start:end].cpu(),
                    speaking_trace[row, start:end].cpu(),
                )
        return outputs

    def _seed(self, offset: int) -> int:
        return 1_000_003 * (offset + 1)

    @torch.inference_mode()
    def e2_many(
        self,
        model: DauysHybrid,
        items: Sequence[Mapping[str, Any]],
        decoding: DecodingConfig,
    ) -> list[dict[str, float]]:
        if self.transcribe_many is None or self.judge_many is None:
            raise ValueError("E2 requires batched ASR and relevance callbacks")
        backend = self.backend
        suite = backend.suites["E2"].items
        channels = [backend._channels(item) for item in items]
        original_cases, matched_cases = [], []
        for item, (user, output) in zip(items, channels, strict=True):
            start = int(item["answer_start_frame"])
            end = min(output.shape[1], start + 25)
            peers = [
                other
                for other in suite
                if other["id"] != item["id"]
                and other["user_voice"] == item["user_voice"]
            ]
            if not peers:
                raise ValueError("E2 has no same-voice matched input")
            alternate, _ = backend._channels(peers[0])
            if alternate.shape[1] < user.shape[1]:
                pad = backend.silence_codes[:, None].expand(
                    -1, user.shape[1] - alternate.shape[1]
                )
                alternate = torch.cat((alternate, pad), dim=1)
            alternate = alternate[:, : user.shape[1]]
            original_cases.append((user, output, start, end))
            matched_cases.append((alternate, output, start, end))
        original = self.cb0_nll_many(model, original_cases)
        matched = self.cb0_nll_many(model, matched_cases)
        per_seed: list[list[dict[str, float]]] = [[] for _ in items]
        for offset in range(backend.seeds):
            early = self.generate_many(
                model,
                [
                    (user, output, int(item["user_start_frame"]))
                    for item, (user, output) in zip(items, channels, strict=True)
                ],
                decoding,
                self._seed(2 * offset),
            )
            replies = self.generate_many(
                model,
                [
                    (user, output, int(item["user_end_frame"]))
                    for item, (user, output) in zip(items, channels, strict=True)
                ],
                decoding,
                self._seed(2 * offset + 1),
            )
            waves = []
            for codes, activity in replies:
                device = next(backend.codec.parameters()).device
                playable = torch.where(
                    activity[None, None, :].to(device),
                    codes[None].to(device),
                    backend.silence_codes[None, :, None].to(device),
                )
                waves.append(
                    decode(backend.codec, playable)[0, 0].float().cpu().numpy()
                )
            transcripts = self.transcribe_many(waves)
            scores = self.judge_many(
                [
                    (item["analysis"]["user_text"], transcript)
                    for item, transcript in zip(items, transcripts, strict=True)
                ]
            )
            for index, item in enumerate(items):
                early_end = max(
                    0, int(item["user_end_frame"]) - int(item["user_start_frame"])
                )
                window = early[index][1][:early_end]
                after = replies[index][1]
                onsets = [
                    i for i in range(max(0, len(after) - 1)) if after[i : i + 2].all()
                ]
                per_seed[index].append(
                    {
                        "false_start": float(
                            any(
                                bool(window[i : i + 6].all())
                                for i in range(max(0, len(window) - 5))
                            )
                        ),
                        "latency": float(onsets[0] * 80 if onsets else 3000),
                        "wer": word_error_rate(
                            item["analysis"]["answer_text"], transcripts[index]
                        ),
                        "relevance": float(scores[index]),
                    }
                )
        return [
            {
                "original_nll": original[index],
                "matched_nll": matched[index],
                "onset_latency_ms": float(np.median([run["latency"] for run in runs])),
                "false_starts": float(np.mean([run["false_start"] for run in runs])),
                "wer": float(np.mean([run["wer"] for run in runs])),
                "relevance": float(np.mean([run["relevance"] for run in runs])),
            }
            for index, runs in enumerate(per_seed)
        ]

    def e3_many(
        self, model: DauysHybrid, items: Sequence[Mapping[str, Any]]
    ) -> list[torch.Tensor]:
        pairs = [("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")]
        cases = [
            self.backend._compose(item, question, answer)
            for item in items
            for question, answer in pairs
        ]
        values = self.cb0_nll_many(model, cases)
        return [
            torch.tensor(
                values[4 * index : 4 * index + 4], dtype=torch.float32
            ).reshape(2, 2)
            for index in range(len(items))
        ]

    def e4_many(
        self, model: DauysHybrid, items: Sequence[Mapping[str, Any]]
    ) -> list[bool]:
        cases = [
            self.backend._compose(item, "base", str(answer))
            for item in items
            for answer in range(4)
        ]
        values = self.cb0_nll_many(model, cases)
        return [
            int(np.argmin(values[4 * index : 4 * index + 4]))
            == int(item["probe"]["correct_index"])
            for index, item in enumerate(items)
        ]

    def _traces(
        self,
        model: DauysHybrid,
        items: Sequence[Mapping[str, Any]],
        start_field: str,
        decoding: DecodingConfig,
        salt: int,
    ) -> list[list[torch.Tensor]]:
        traces: list[list[torch.Tensor]] = [[] for _ in items]
        for offset in range(self.backend.seeds):
            runs = self.generate_many(
                model,
                [
                    (*self.backend._channels(item), int(item[start_field]))
                    for item in items
                ],
                decoding,
                self._seed(100 * salt + offset),
            )
            for index, (_, activity) in enumerate(runs):
                traces[index].append(activity)
        return traces

    def e5_many(
        self,
        model: DauysHybrid,
        items: Sequence[Mapping[str, Any]],
        decoding: DecodingConfig,
    ) -> list[dict[str, float]]:
        traces = self._traces(model, items, "interrupt_start_frame", decoding, 5)
        results = []
        for item, runs in zip(items, traces, strict=True):
            start = int(item["interrupt_start_frame"])
            successes = []
            for activity in runs:
                if item["condition"] == "interrupt":
                    successes.append(
                        float(
                            any(
                                (~activity[i : i + 3]).all()
                                for i in range(min(7, max(0, len(activity) - 2)))
                            )
                        )
                    )
                else:
                    window = activity[
                        : min(25, int(item["assistant_end_frame"]) - start)
                    ]
                    false_yield = any(
                        (~window[i : i + 6]).all()
                        for i in range(max(0, len(window) - 5))
                    )
                    successes.append(float(not false_yield))
            results.append({"yield_within_500ms": float(np.mean(successes))})
        return results

    def e6_many(
        self,
        model: DauysHybrid,
        items: Sequence[Mapping[str, Any]],
        decoding: DecodingConfig,
    ) -> list[dict[str, float]]:
        traces = self._traces(model, items, "turn_start_frame", decoding, 6)
        results = []
        for item, runs in zip(items, traces, strict=True):
            start = int(item["turn_start_frame"])
            end = int(item["turn_end_frame"])
            pauses = list(zip(item["pause_start_frames"], item["pause_end_frames"]))
            rates, placements, takeovers = [], [], []
            for activity in runs:
                trace = activity[: max(0, end - start)].tolist()
                segments, long_segments, frame = [], 0, 0
                while frame < len(trace):
                    if not trace[frame]:
                        frame += 1
                        continue
                    finish = frame + 1
                    while finish < len(trace) and trace[finish]:
                        finish += 1
                    if 2 <= finish - frame <= 15:
                        segments.append(start + frame)
                    elif finish - frame > 15:
                        long_segments += 1
                    frame = finish
                minutes = max((end - start) * 0.08 / 60, 1e-6)
                rates.append(len(segments) / minutes)
                takeovers.append(float(long_segments > 0))
                placements.append(
                    sum(
                        any(a - 2 <= onset <= b + 2 for a, b in pauses)
                        for onset in segments
                    )
                    / max(1, len(segments))
                )
            results.append(
                {
                    "backchannel_rate": float(np.mean(rates)),
                    "backchannel_placement": float(np.mean(placements)),
                    "premature_takeover": float(np.mean(takeovers)),
                }
            )
        return results

    def score_all(
        self, model: DauysHybrid, decoding: DecodingConfig
    ) -> dict[str, list[Any]]:
        suites = self.backend.suites
        return {
            "E1": self.e1_many(model, suites["E1"].items),
            "E2": self.e2_many(model, suites["E2"].items, decoding),
            "E3": self.e3_many(model, suites["E3"].items),
            "E4": [float(v) for v in self.e4_many(model, suites["E4"].items)],
            "E5": self.e5_many(model, suites["E5"].items, decoding),
            "E6": self.e6_many(model, suites["E6"].items, decoding),
        }
