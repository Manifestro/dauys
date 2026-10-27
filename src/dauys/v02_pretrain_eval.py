"""Reference-free continuation and CB0 generalization checks for pre-training."""

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .codec import decode, load_mimi
from .decoding import DecodingConfig
from .model import DauysHybrid
from .v02_data import CorpusDataset, CorpusRecord, sha256

FRAME_SAMPLES = 1920
SAMPLE_RATE = 24_000
PROMPT_FRAMES = 38
CONTINUATION_FRAMES = 75
ACCURACY_FRAMES = 256
MIN_WORDS = 3
Transcriber = Callable[[list[np.ndarray]], list[str]]


def word_statistics(
    transcripts: Sequence[str], vocabulary: set[str]
) -> dict[str, float]:
    """Empty rate, and the dictionary-word share over nonempty transcripts."""
    if not transcripts:
        raise ValueError("Word statistics require transcripts")
    rates = []
    for text in transcripts:
        words = text.split()
        if len(words) >= MIN_WORDS:
            rates.append(sum(word in vocabulary for word in words) / len(words))
    return {
        "empty_rate": 1 - len(rates) / len(transcripts),
        "word_rate": float(np.mean(rates)) if rates else 0.0,
        "scored": float(len(rates)),
    }


def english_vocabulary() -> set[str]:
    import cmudict

    return {word.lower() for word in cmudict.words()}


class WhisperTranscriber:
    """Whisper with greedy decoding and Whisper's English text normalizer."""

    def __init__(self, model_id: str, device: str, batch_size: int = 16) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.normalize = self.processor.tokenizer.normalize  # type: ignore[attr-defined]
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, waves: list[np.ndarray]) -> list[str]:
        from scipy.signal import resample_poly

        texts: list[str] = []
        for start in range(0, len(waves), self.batch_size):
            chunk = [
                resample_poly(wave, 2, 3).astype(np.float32)
                for wave in waves[start : start + self.batch_size]
            ]
            features = self.processor(
                chunk, sampling_rate=16_000, return_tensors="pt"
            ).input_features.to(self.device, torch.float16)
            tokens = self.model.generate(
                features, language="en", task="transcribe", do_sample=False
            )
            decoded = self.processor.batch_decode(tokens, skip_special_tokens=True)
            texts.extend(self.normalize(text) for text in decoded)
        return texts


def load_pretrain_suite(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    expected = (root / "suite.sha256").read_text().strip()
    if sha256(root / "suite.pt") != expected:
        raise ValueError("Pre-training evaluation suite checksum mismatch")
    suite = torch.load(root / "suite.pt", map_location="cpu", weights_only=False)
    suite["sha256"] = expected
    return suite


def _padded(
    sequences: Sequence[torch.Tensor], silence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = max(item.shape[-1] for item in sequences)
    codes = silence[None, :, None].expand(len(sequences), 8, frames).clone()
    valid = torch.zeros((len(sequences), frames), dtype=torch.bool)
    for index, item in enumerate(sequences):
        codes[index, :, : item.shape[-1]] = item
        valid[index, : item.shape[-1]] = True
    return codes, valid


class PretrainEvaluator:
    """Free continuation on held-out speech plus a teacher-forced CB0 gap."""

    def __init__(
        self,
        suite: dict[str, Any],
        codec: Any,
        silence_codes: torch.Tensor,
        transcribe: Transcriber,
        vocabulary: set[str],
        training_outputs: Sequence[torch.Tensor],
        *,
        audio_dir: str | Path | None = None,
        listen_items: int = 10,
        batch_size: int = 50,
        decoding: DecodingConfig | None = None,
        seed: int = 1234,
    ) -> None:
        if silence_codes.shape != (8,) or not training_outputs:
            raise ValueError("Silence codes and training outputs are required")
        self.items = suite["items"]
        if any(item["codes"].shape[-1] < PROMPT_FRAMES for item in self.items):
            raise ValueError("Every evaluation item must cover the prompt")
        self.ceiling = suite["ceiling"]
        self.codec = codec
        self.silence = silence_codes.long()
        self.transcribe = transcribe
        self.vocabulary = vocabulary
        self.training_outputs = [item.long() for item in training_outputs]
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.listen_items = listen_items
        self.batch_size = batch_size
        self.decoding = decoding or DecodingConfig.v02_default()
        self.decoding.validate()
        self.seed = seed
        self.floor: dict[str, float] | None = None

    @torch.inference_mode()
    def cb0_accuracy(
        self, model: DauysHybrid, outputs: Sequence[torch.Tensor]
    ) -> tuple[float, float]:
        device = next(model.parameters()).device
        correct = total = 0
        nll = 0.0
        for start in range(0, len(outputs), self.batch_size):
            chunk = [
                item[:, :ACCURACY_FRAMES]
                for item in outputs[start : start + self.batch_size]
            ]
            output, valid = _padded(chunk, self.silence)
            output, valid = output.to(device), valid.to(device)
            user = self.silence[None, :, None].expand_as(output).to(device)
            initial = self.silence[None].expand(len(chunk), 8).to(device)
            logits = model(user, output, initial).cb0_logits.float()
            correct += int(((logits.argmax(-1) == output[:, 0]) & valid).sum())
            nll += float(
                F.cross_entropy(logits[valid], output[:, 0][valid], reduction="sum")
            )
            total += int(valid.sum())
        return correct / total, nll / total

    @torch.inference_mode()
    def continue_speech(self, model: DauysHybrid, seed: int) -> list[np.ndarray]:
        device = next(model.parameters()).device
        waves: list[np.ndarray] = []
        generator = torch.Generator(device=device).manual_seed(seed)
        silence = self.silence.to(device)
        for start in range(0, len(self.items), self.batch_size):
            chunk = self.items[start : start + self.batch_size]
            prompt = torch.stack(
                [item["codes"][:, :PROMPT_FRAMES].long() for item in chunk]
            ).to(device)
            batch = prompt.shape[0]
            user = silence[None, :, None].expand(batch, 8, PROMPT_FRAMES)
            state = model.prefill_teacher(user, prompt, silence[None].expand(batch, 8))
            generated, speaking = [], []
            for _ in range(CONTINUATION_FRAMES):
                codes, active = model.generate_step(
                    silence[None].expand(batch, 8), state, self.decoding, generator
                )
                generated.append(codes)
                speaking.append(active)
            codes = torch.stack(generated, dim=-1)
            active = torch.stack(speaking, dim=-1)[:, None]
            playable = torch.where(active, codes, silence[None, :, None])
            pcm = decode(self.codec, torch.cat((prompt, playable), dim=-1).long())
            tail = pcm[:, 0, PROMPT_FRAMES * FRAME_SAMPLES :].float().cpu().numpy()
            waves.extend(list(tail))
        return waves

    def _score(
        self, model: DauysHybrid, seed: int
    ) -> tuple[dict[str, float], list[np.ndarray], list[str]]:
        waves = self.continue_speech(model, seed)
        transcripts = self.transcribe(waves)
        return word_statistics(transcripts, self.vocabulary), waves, transcripts

    def _save_audio(
        self, step: int, waves: list[np.ndarray], transcripts: list[str]
    ) -> None:
        if self.audio_dir is None:
            return
        folder = self.audio_dir / f"step-{step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        listing = []
        for item, wave, text in list(zip(self.items, waves, transcripts))[
            : self.listen_items
        ]:
            sf.write(folder / f"{item['id']}.wav", wave, SAMPLE_RATE)
            listing.append({"id": item["id"], "prompt_text": item["text"], "asr": text})
        (folder / "transcripts.json").write_text(json.dumps(listing, indent=2) + "\n")

    def __call__(self, model: DauysHybrid, step: int) -> dict[str, float]:
        if step < 0 or model.training:
            raise ValueError("Evaluation requires a nonnegative step and eval mode")
        if self.floor is None:
            state = torch.random.get_rng_state()
            torch.manual_seed(self.seed)
            floor_model = type(model)(model.config).to(next(model.parameters()).device)
            torch.random.set_rng_state(state)
            floor_model.eval()
            self.floor, _, _ = self._score(floor_model, self.seed)
            del floor_model
        scores, waves, transcripts = self._score(model, self.seed)
        self._save_audio(step, waves, transcripts)
        train_accuracy, train_nll = self.cb0_accuracy(model, self.training_outputs)
        heldout_accuracy, heldout_nll = self.cb0_accuracy(
            model, [item["codes"].long() for item in self.items]
        )
        span = self.ceiling["word_rate"] - self.floor["word_rate"]
        return {
            "eval/step": float(step),
            "eval/pretrain/cont_word_rate": scores["word_rate"],
            "eval/pretrain/cont_empty_rate": scores["empty_rate"],
            "eval/pretrain/cont_scored": scores["scored"],
            "eval/pretrain/cont_word_rate_floor": self.floor["word_rate"],
            "eval/pretrain/cont_empty_rate_floor": self.floor["empty_rate"],
            "eval/pretrain/cont_word_rate_ceiling": self.ceiling["word_rate"],
            "eval/pretrain/cont_empty_rate_ceiling": self.ceiling["empty_rate"],
            "eval/pretrain/cont_word_rate_score": (
                (scores["word_rate"] - self.floor["word_rate"]) / span
                if abs(span) > 1e-9
                else float("nan")
            ),
            "eval/codebook/train_cb0_accuracy": train_accuracy,
            "eval/codebook/heldout_cb0_accuracy": heldout_accuracy,
            "eval/codebook/train_cb0_nll": train_nll,
            "eval/codebook/heldout_cb0_nll": heldout_nll,
            "eval/codebook/cb0_accuracy_gap": train_accuracy - heldout_accuracy,
        }


def training_outputs(
    records: Sequence[CorpusRecord], count: int = 100
) -> list[torch.Tensor]:
    """Fixed B-channel speech items for the teacher-forced training accuracy."""
    speech = sorted(
        (record for record in records if record.source == "1a_on_B"),
        key=lambda record: record.id,
    )[:count]
    if not speech:
        raise ValueError("Training accuracy requires B-channel speech records")
    dataset = CorpusDataset(list(speech))
    return [dataset[index]["output"] for index in range(len(dataset))]


def create_pretrain_evaluator(
    *, suites: dict[str, Any], records: list[CorpusRecord]
) -> PretrainEvaluator:
    """Training evaluator factory configured through the environment."""
    del suites
    required = ("DAUYS_PRETRAIN_EVAL", "DAUYS_MIMI_CHECKPOINT", "DAUYS_SILENCE_CODES")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing evaluator configuration: {', '.join(missing)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    silence = torch.tensor(
        json.loads(Path(os.environ["DAUYS_SILENCE_CODES"]).read_text()),
        dtype=torch.long,
    )
    return PretrainEvaluator(
        load_pretrain_suite(os.environ["DAUYS_PRETRAIN_EVAL"]),
        load_mimi(os.environ["DAUYS_MIMI_CHECKPOINT"], device),
        silence,
        WhisperTranscriber(
            os.environ.get("DAUYS_ASR_MODEL", "openai/whisper-large-v3"), device
        ),
        english_vocabulary(),
        training_outputs(records),
        audio_dir=os.environ.get("DAUYS_EVAL_AUDIO_DIR"),
    )
