"""Model-side scoring of frozen Hohere audio probes."""

import hashlib
import importlib
import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .codec import decode, encode, load_mimi
from .decoding import DecodingConfig
from .model import DauysHybrid, previous_frames
from .v02_batched_probes import BatchedProbeScorer
from .v02_data import CorpusDataset, CorpusRecord
from .v02_evaluation import Suite, word_error_rate
from .v02_harness import FrozenSuiteHarness

FRAME_SAMPLES = 1920
SAMPLE_RATE = 24_000
Transcript = Callable[[np.ndarray], str]
Relevance = Callable[[str, str], float]


class HohereProbeBackend:
    """Scores audio only; transcript and judge callbacks run after generation."""

    def __init__(
        self,
        root: str | Path,
        suites: Mapping[str, Suite],
        codec: Any,
        silence_codes: torch.Tensor,
        *,
        asr: Transcript | None = None,
        relevance: Relevance | None = None,
        seeds: int = 3,
        token_cache: str | Path | None = None,
    ) -> None:
        if silence_codes.shape != (8,) or seeds < 1:
            raise ValueError(
                "Eight silence codes and a positive seed count are required"
            )
        self.root = Path(root)
        self.suites = suites
        self.codec = codec
        self.silence_codes = silence_codes.long()
        self.asr = asr
        self.relevance = relevance
        self.seeds = seeds
        self._token_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.batched: BatchedProbeScorer | None = None
        self.token_cache_path = Path(token_cache) if token_cache else None
        self._cached_entries = 0
        if self.token_cache_path is not None and self.token_cache_path.is_file():
            payload = torch.load(self.token_cache_path, weights_only=True)
            if payload["suites"] != self._suite_identity():
                raise ValueError("Token cache was built for different suites")
            self._token_cache = {
                key: (user.long(), output.long())
                for key, (user, output) in payload["tokens"].items()
            }
            self._cached_entries = len(self._token_cache)

    def _suite_identity(self) -> dict[str, str]:
        return {
            name: suite.manifest_sha256 for name, suite in sorted(self.suites.items())
        }

    def save_token_cache(self) -> bool:
        """Persist Mimi codes of the frozen probes; they never depend on the model."""
        if (
            self.token_cache_path is None
            or len(self._token_cache) == self._cached_entries
        ):
            return False
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.token_cache_path.with_suffix(".tmp")
        torch.save(
            {
                "suites": self._suite_identity(),
                "tokens": {
                    key: (user.to(torch.int16), output.to(torch.int16))
                    for key, (user, output) in self._token_cache.items()
                },
            },
            temporary,
        )
        temporary.replace(self.token_cache_path)
        self._cached_entries = len(self._token_cache)
        return True

    def _audio(self, item: Mapping[str, Any], field: str) -> np.ndarray:
        asset = item["audio"][field]
        path = (self.root / asset["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Probe audio escapes the benchmark root")
        wave, rate = sf.read(path, dtype="float32")
        if rate != SAMPLE_RATE or wave.ndim != 1:
            raise ValueError("Probe audio must be mono at 24 kHz")
        return wave

    def _encode(
        self, user: np.ndarray, output: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(user) != len(output) or len(user) < FRAME_SAMPLES:
            raise ValueError("Probe channels must have the same nonempty length")
        frames = math.ceil(len(user) / FRAME_SAMPLES)
        padded = np.stack(
            [
                np.pad(user, (0, frames * FRAME_SAMPLES - len(user))),
                np.pad(output, (0, frames * FRAME_SAMPLES - len(output))),
            ]
        )
        device = next(self.codec.parameters()).device
        codes = encode(self.codec, torch.from_numpy(padded.copy())[:, None].to(device))
        if codes.shape != (2, 8, frames):
            raise ValueError("Mimi returned the wrong probe geometry")
        return codes[0].long().cpu(), codes[1].long().cpu()

    def _channels(self, item: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        identity = str(item["id"])
        if identity not in self._token_cache:
            self._token_cache[identity] = self._encode(
                self._audio(item, "channel_a"), self._audio(item, "channel_b")
            )
        return self._token_cache[identity]

    def _compose(
        self,
        item: Mapping[str, Any],
        question: str,
        answer: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        key = f"{item['id']}:{question}:{answer}"
        if key not in self._token_cache:
            samples = int(item["total_samples"])
            user = np.zeros(samples, dtype=np.float32)
            output = np.zeros(samples, dtype=np.float32)
            if question in ("x", "y"):
                clip = self._audio(item, f"question_{question}_audio")
                start = int(item["question_start_sample"])
                user[start : start + len(clip)] = clip
            else:
                user = self._audio(item, "channel_a")
            if answer in ("x", "y"):
                clip = self._audio(item, f"answer_{answer}_audio")
            else:
                clip = self._audio(item, f"candidate_{answer}_audio")
                output = self._audio(item, "channel_b")
            answer_start = int(item["answer_start_sample"])
            if answer_start + len(clip) > samples:
                raise ValueError("Candidate answer exceeds the probe timeline")
            output[answer_start : answer_start + len(clip)] = clip
            self._token_cache[key] = self._encode(user, output)
        start_frame = int(item["answer_start_sample"]) // FRAME_SAMPLES
        clip_field = (
            f"answer_{answer}_audio"
            if answer in ("x", "y")
            else f"candidate_{answer}_audio"
        )
        end_frame = math.ceil(
            (int(item["answer_start_sample"]) + len(self._audio(item, clip_field)))
            / FRAME_SAMPLES
        )
        return (*self._token_cache[key], start_frame, end_frame)

    @staticmethod
    def _window_nll(
        logits: torch.Tensor, target: torch.Tensor, start: int, end: int
    ) -> float:
        if not 0 <= start < end <= target.numel():
            raise ValueError("Score window falls outside the probe")
        return float(F.cross_entropy(logits[start:end].float(), target[start:end]))

    @torch.inference_mode()
    def _cb0_nll(
        self,
        model: DauysHybrid,
        user: torch.Tensor,
        output: torch.Tensor,
        start: int,
        end: int,
        *,
        fusion_off: frozenset[int] = frozenset(),
    ) -> float:
        device = next(model.parameters()).device
        a, b = user[None].to(device), output[None].to(device)
        silence = self.silence_codes[None].to(device)
        listening = model.encode_listening(
            a, own_previous_codes=previous_frames(b, silence)
        )
        states = model.run_dialogue(
            listening,
            b,
            silence,
            fusion_off_blocks=fusion_off,
        )
        logits = model.cb0_head(states)[0]
        return self._window_nll(logits, b[0, 0], start, end)

    @torch.inference_mode()
    def e1(self, model: DauysHybrid, item: Mapping[str, Any]) -> Mapping[str, Any]:
        user, output = self._channels(item)
        device = next(model.parameters()).device
        a, b = user[None].to(device), output[None].to(device)
        result = model(a, b, self.silence_codes[None].to(device))
        cb0 = result.cb0_logits[0]
        depth = result.depth_logits
        if depth is None:
            raise ValueError("E1 requires depth logits")
        ablations = {
            str(block): self._cb0_nll(
                model, user, output, 0, output.shape[1], fusion_off=frozenset({block})
            )
            for block in model.fusion_block_indices()
        }
        return {
            "cb0_nll": self._window_nll(cb0, b[0, 0], 0, output.shape[1]),
            "cb0_accuracy": float((cb0.argmax(dim=-1) == b[0, 0]).float().mean()),
            "depth_nll": float(
                F.cross_entropy(
                    depth[0].float().reshape(-1, depth.shape[-1]),
                    b[0, 1:].transpose(0, 1).reshape(-1),
                )
            ),
            "fusion_off_cb0_nll_by_block": ablations,
        }

    @torch.inference_mode()
    def _generate(
        self,
        model: DauysHybrid,
        user: torch.Tensor,
        output: torch.Tensor,
        start: int,
        decoding: DecodingConfig,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device
        a, b = user[None].to(device), output[None].to(device)
        state = model.prefill_teacher(
            a[:, :, :start], b[:, :, :start], self.silence_codes[None].to(device)
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        generated, activity = [], []
        for frame in range(start, a.shape[-1]):
            codes, speaking = model.generate_step(
                a[:, :, frame], state, decoding, generator
            )
            generated.append(codes[0])
            activity.append(bool(speaking[0]))
        return torch.stack(generated, dim=-1), torch.tensor(activity, dtype=torch.bool)

    def _seeds(self, item: Mapping[str, Any]) -> range:
        digest = hashlib.sha256(str(item["id"]).encode()).digest()
        start = int.from_bytes(digest[:4], "big")
        return range(start, start + self.seeds)

    @torch.inference_mode()
    def e2(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]:
        if self.asr is None or self.relevance is None:
            raise ValueError("E2 requires explicit ASR and relevance judge callbacks")
        user, output = self._channels(item)
        start = int(item["answer_start_frame"])
        end = min(output.shape[1], start + 25)
        original = self._cb0_nll(model, user, output, start, end)
        peers = [
            other
            for other in self.suites["E2"].items
            if other["id"] != item["id"] and other["user_voice"] == item["user_voice"]
        ]
        if not peers:
            raise ValueError("E2 has no same-voice matched input")
        alternate, _ = self._channels(peers[0])
        if alternate.shape[1] < user.shape[1]:
            pad = self.silence_codes[:, None].expand(
                -1, user.shape[1] - alternate.shape[1]
            )
            alternate = torch.cat((alternate, pad), dim=1)
        alternate = alternate[:, : user.shape[1]]
        matched = self._cb0_nll(model, alternate, output, start, end)
        latencies, false_starts, wers, relevance = [], [], [], []
        for seed in self._seeds(item):
            _, early_activity = self._generate(
                model, user, output, int(item["user_start_frame"]), decoding, seed
            )
            early_end = max(
                0, int(item["user_end_frame"]) - int(item["user_start_frame"])
            )
            early = early_activity[:early_end]
            false_starts.append(
                float(
                    any(
                        bool(early[i : i + 6].all())
                        for i in range(max(0, len(early) - 5))
                    )
                )
            )
            codes, activity = self._generate(
                model, user, output, int(item["user_end_frame"]), decoding, seed
            )
            after = activity
            onsets = [
                i for i in range(max(0, len(after) - 1)) if after[i : i + 2].all()
            ]
            latencies.append(float(onsets[0] * 80 if onsets else 3000))
            active = activity[None, None, :].to(codes.device)
            playable = torch.where(
                active, codes[None], self.silence_codes[None, :, None].to(codes.device)
            )
            waveform = decode(self.codec, playable)[0, 0].float().cpu().numpy()
            transcript = self.asr(waveform)
            analysis = item["analysis"]
            wers.append(word_error_rate(analysis["answer_text"], transcript))
            relevance.append(float(self.relevance(analysis["user_text"], transcript)))
        return {
            "original_nll": original,
            "matched_nll": matched,
            "onset_latency_ms": float(np.median(latencies)),
            "false_starts": float(np.mean(false_starts)),
            "wer": float(np.mean(wers)),
            "relevance": float(np.mean(relevance)),
        }

    @torch.inference_mode()
    def e3(self, model: DauysHybrid, item: Mapping[str, Any]) -> torch.Tensor:
        matrix = torch.zeros((2, 2), dtype=torch.float32)
        for q, question in enumerate(("x", "y")):
            for a, answer in enumerate(("x", "y")):
                user, output, start, end = self._compose(item, question, answer)
                matrix[q, a] = self._cb0_nll(model, user, output, start, end)
        return matrix

    @torch.inference_mode()
    def e4(self, model: DauysHybrid, item: Mapping[str, Any]) -> bool:
        scores = []
        for answer in range(4):
            user, output, start, end = self._compose(item, "base", str(answer))
            scores.append(self._cb0_nll(model, user, output, start, end))
        return int(np.argmin(scores)) == int(item["probe"]["correct_index"])

    @torch.inference_mode()
    def e5(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]:
        user, output = self._channels(item)
        start = int(item["interrupt_start_frame"])
        successes = []
        for seed in self._seeds(item):
            _, activity = self._generate(model, user, output, start, decoding, seed)
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
                window = activity[: min(25, int(item["assistant_end_frame"]) - start)]
                false_yield = any(
                    (~window[i : i + 6]).all() for i in range(max(0, len(window) - 5))
                )
                successes.append(float(not false_yield))
        return {"yield_within_500ms": float(np.mean(successes))}

    @torch.inference_mode()
    def e6(
        self, model: DauysHybrid, item: Mapping[str, Any], decoding: DecodingConfig
    ) -> Mapping[str, float]:
        user, output = self._channels(item)
        start = int(item["turn_start_frame"])
        end = int(item["turn_end_frame"])
        rates, placements, takeovers = [], [], []
        pauses = list(zip(item["pause_start_frames"], item["pause_end_frames"]))
        for seed in self._seeds(item):
            _, activity = self._generate(model, user, output, start, decoding, seed)
            trace = activity[: max(0, end - start)].tolist()
            segments = []
            long_segments = 0
            frame = 0
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
        return {
            "backchannel_rate": float(np.mean(rates)),
            "backchannel_placement": float(np.mean(placements)),
            "premature_takeover": float(np.mean(takeovers)),
        }


def _callback(specification: str) -> Callable[..., Any]:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("Callback must be module:function")
    callback = getattr(importlib.import_module(module_name), function_name)
    if not callable(callback):
        raise TypeError("Configured callback is not callable")
    return callback


def create_evaluator(
    *, suites: Mapping[str, Suite], records: list[CorpusRecord]
) -> FrozenSuiteHarness:
    """Training evaluator; all external audio/text dependencies are explicit."""
    required = (
        "DAUYS_BENCH_ROOT",
        "DAUYS_MIMI_CHECKPOINT",
        "DAUYS_SILENCE_CODES",
        "DAUYS_ASR_CALLBACK",
        "DAUYS_RELEVANCE_CALLBACK",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing evaluator configuration: {', '.join(missing)}")
    if not records:
        raise ValueError("Training accuracy requires a training corpus")
    root = Path(os.environ["DAUYS_BENCH_ROOT"])
    if not root.is_dir():
        raise FileNotFoundError(root)
    codes_path = Path(os.environ["DAUYS_SILENCE_CODES"])
    if codes_path.suffix == ".json":
        codes = torch.tensor(json.loads(codes_path.read_text()), dtype=torch.long)
    else:
        saved = torch.load(codes_path, map_location="cpu", weights_only=True)
        codes = (saved["codes"] if isinstance(saved, dict) else saved).long()
    if codes.shape != (8,):
        raise ValueError("Silence codes must have eight entries")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = load_mimi(os.environ["DAUYS_MIMI_CHECKPOINT"], device)
    backend = HohereProbeBackend(
        root,
        suites,
        codec,
        codes,
        asr=_callback(os.environ["DAUYS_ASR_CALLBACK"]),
        relevance=_callback(os.environ["DAUYS_RELEVANCE_CALLBACK"]),
        seeds=int(os.environ.get("DAUYS_PROBE_SEEDS", "3")),
        token_cache=os.environ.get("DAUYS_TOKEN_CACHE") or None,
    )
    if os.environ.get("DAUYS_BATCHED_PROBES", "1") != "0":
        backend.batched = BatchedProbeScorer(
            backend,
            _callback(os.environ["DAUYS_ASR_CALLBACK"] + "_batch"),
            _callback(os.environ["DAUYS_RELEVANCE_CALLBACK"] + "_batch"),
        )
    training_sample = CorpusDataset(records[: min(16, len(records))], crop_frames=256)

    @torch.inference_mode()
    def train_accuracy(model: DauysHybrid) -> float:
        correct = total = 0
        model_device = next(model.parameters()).device
        for index in range(len(training_sample)):
            case = training_sample[index]
            user = case["user"][None].to(model_device)
            output = case["output"][None].to(model_device)
            initial = case["initial_output"]
            if initial is None:
                initial = codes
            prediction = model(user, output, initial[None].to(model_device))
            correct += int((prediction.cb0_logits.argmax(-1) == output[:, 0]).sum())
            total += output.shape[-1]
        return correct / total

    return FrozenSuiteHarness(
        suites, backend, DecodingConfig.v02_default(), train_accuracy
    )
