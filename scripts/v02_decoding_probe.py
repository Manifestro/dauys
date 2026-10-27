"""Locate unintelligible E2 replies: decoding modes, oracle CB0, and the codec."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.decoding import DecodingConfig, sample_logits
from dauys.model import DauysHybrid
from dauys.v02_evaluation import load_suite
from dauys.v02_hohere_backend import SAMPLE_RATE, HohereProbeBackend
from dauys.v02_judges import _whisper
from dauys.v02_pretrain_eval import english_vocabulary, word_statistics

SILENT_RMS = 0.003
MODES = {
    "c6_default": DecodingConfig.v02_default(),
    "greedy": DecodingConfig(),
    "sample_all": DecodingConfig(
        cb0_mode="sample", depth_mode="sample", activity_mode="hysteresis"
    ),
}


@torch.inference_mode()
def oracle_cb0(
    model: DauysHybrid,
    user: torch.Tensor,
    output: torch.Tensor,
    start: int,
    silence: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Reference CB0 every frame; the model samples CB1–CB7 as in C6."""
    device = next(model.parameters()).device
    a, b = user[None].to(device), output[None].to(device)
    state = model.prefill_teacher(
        a[:, :, :start], b[:, :, :start], silence[None].to(device)
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    decoding = DecodingConfig.v02_default()
    frames = []
    for frame in range(start, a.shape[-1]):
        assert state.previous_output is not None
        _, _, context = model.stream_step_teacher(
            a[:, :, frame], state.previous_output, state
        )
        generated = [b[:, 0, frame]]
        for _ in range(1, 8):
            prefix = torch.stack(generated, dim=-1)[:, None]
            logits = model.depth(prefix, context)[:, 0, -1]
            generated.append(
                sample_logits(
                    logits, decoding.depth_temperature, decoding.depth_top_k, generator
                )
            )
        codes = torch.stack(generated, dim=1)
        state.previous_output = codes
        frames.append(codes[0])
    return torch.stack(frames, dim=-1)


def _gated(transcripts: list[str], waves: list[np.ndarray]) -> list[str]:
    return [
        "" if np.sqrt(np.mean(wave**2)) < SILENT_RMS else text
        for text, wave in zip(transcripts, waves)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suites-dir", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audio-dir", type=Path, help="Save each reply (model channel only) as wav"
    )
    parser.add_argument("--audio-items", type=int, default=8)
    args = parser.parse_args()
    suite = load_suite(
        args.suites_dir / "E2.json",
        (args.suites_dir / "E2.sha256").read_text().strip(),
    )
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    backend = HohereProbeBackend(args.suites_dir, {"E2": suite}, codec, silence)
    model = DauysHybrid(ModelConfig.from_yaml(args.config)).cuda()
    model.load_state_dict(load_research_checkpoint(args.checkpoint)["model"])
    model.eval()
    asr = _whisper()
    vocabulary = english_vocabulary()
    items = list(suite.items)[:: max(1, len(suite.items) // args.items)][: args.items]
    replies: dict[str, list[np.ndarray]] = {name: [] for name in MODES}
    replies["oracle_cb0"] = []
    replies["reference_mimi"] = []
    active: dict[str, list[float]] = {name: [] for name in MODES}
    for item in items:
        user, output = backend._channels(item)
        start = int(item["answer_start_frame"])
        end = int(item["answer_end_frame"])
        for name, decoding in MODES.items():
            codes, speaking = backend._generate(
                model, user, output, start, decoding, args.seed
            )
            codes, speaking = codes[:, : end - start], speaking[: end - start]
            playable = torch.where(
                speaking[None, None].cuda(),
                codes[None].cuda(),
                silence[None, :, None].cuda(),
            )
            replies[name].append(decode(codec, playable)[0, 0].float().cpu().numpy())
            active[name].append(float(speaking.float().mean()))
        forced = oracle_cb0(model, user, output, start, silence, args.seed)[
            :, : end - start
        ]
        replies["oracle_cb0"].append(
            decode(codec, forced[None].cuda())[0, 0].float().cpu().numpy()
        )
        reference = output[None, :, start:end].cuda()
        replies["reference_mimi"].append(
            decode(codec, reference)[0, 0].float().cpu().numpy()
        )
    if args.audio_dir is not None:
        for name, waves in replies.items():
            (args.audio_dir / name).mkdir(parents=True, exist_ok=True)
            for item, wave in zip(items[: args.audio_items], waves):
                sf.write(args.audio_dir / name / f"{item['id']}.wav", wave, SAMPLE_RATE)
    results = {}
    for name, waves in replies.items():
        transcripts = asr(waves)
        gated = _gated(transcripts, waves)
        results[name] = {
            **{
                f"raw_{k}": v
                for k, v in word_statistics(transcripts, vocabulary).items()
            },
            **word_statistics(gated, vocabulary),
            "rms": float(np.mean([np.sqrt(np.mean(w**2)) for w in waves])),
            "active_fraction": float(np.mean(active[name])) if name in active else None,
            "examples": gated[:6],
        }
        print(json.dumps({name: results[name]}), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
