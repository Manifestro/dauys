"""Reproducible single-conversation overfit probe for the two-phase objective."""

import argparse
import hashlib
import io
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import listening_loss, training_loss
from dauys.model import DauysHybrid

DATASET = "karl4th/hohere-dataset-v0.1"
REVISION = "594fbbdbfdf8b31194c038944745d62b5afcb5ca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/S1-E001"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conversation-id", default="conv_000001")
    parser.add_argument("--self-conditioned-steps", type=int, default=0)
    parser.add_argument("--control-run", default=None)
    return parser.parse_args()


def _pcm(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            signal, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
        elif value.get("path"):
            signal, rate = sf.read(value["path"], dtype="float32")
        elif value.get("array") is not None:
            signal = np.asarray(value["array"], dtype=np.float32)
            rate = value.get("sampling_rate")
        else:
            raise ValueError("Audio row has no bytes, path, or array")
    elif isinstance(value, bytes):
        signal, rate = sf.read(io.BytesIO(value), dtype="float32")
    else:
        raise TypeError(f"Unsupported audio field: {type(value)!r}")
    if rate != 24_000 or signal.ndim != 1:
        raise ValueError(f"Expected mono 24 kHz audio, got {signal.shape} at {rate}")
    return signal


def _load_conversation(conversation_id: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    for row in dataset:
        if row["id"] == conversation_id:
            user, model = _pcm(row["channel_a"]), _pcm(row["channel_b"])
            if user.shape != model.shape:
                raise ValueError("Audio channels have different sample counts")
            return user, model
    raise ValueError(f"Conversation {conversation_id} is absent from train")


def _activity(pcm: np.ndarray, frames: int) -> torch.Tensor:
    block = pcm[: frames * 1920].reshape(frames, 1920)
    rms = np.sqrt(np.mean(np.square(block), axis=1))
    return torch.from_numpy((rms > 1e-4).astype(np.int64))[None].cuda()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_loss_plot(path: Path, values: list[float], boundary: int) -> None:
    width, height, margin = 840, 360, 48
    ceiling = max(values) * 1.05
    floor = min(0.0, min(values) * 0.95)
    points = []
    for index, value in enumerate(values):
        x = margin + index * (width - 2 * margin) / max(len(values) - 1, 1)
        y = (
            height
            - margin
            - (value - floor) * (height - 2 * margin) / (ceiling - floor)
        )
        points.append(f"{x:.1f},{y:.1f}")
    phase_x = margin + boundary * (width - 2 * margin) / max(len(values) - 1, 1)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<line x1="{phase_x:.1f}" y1="{margin}" x2="{phase_x:.1f}" y2="{height - margin}" stroke="#999" stroke-dasharray="5 5"/>
<polyline points="{" ".join(points)}" fill="none" stroke="#164b8a" stroke-width="2"/>
<text x="{margin}" y="30" font-family="sans-serif" font-size="16">Training loss by optimizer step</text>
<text x="{margin}" y="{height - 12}" font-family="sans-serif" font-size="12">Listen</text>
<text x="{phase_x + 8:.1f}" y="{height - 12}" font-family="sans-serif" font-size="12">Respond</text>
<text x="5" y="{margin}" font-family="sans-serif" font-size="11">{ceiling:.2f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA")
    if args.steps < 5 or args.frames < 8 or args.frames > 512:
        raise ValueError("Use at least five steps and 8-512 frames")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    manifest = hf_hub_download(
        DATASET, "manifest.json", repo_type="dataset", revision=REVISION
    )
    manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    user_pcm, model_pcm = _load_conversation(args.conversation_id)
    if len(user_pcm) < args.frames * 1920:
        raise ValueError("Conversation is shorter than the requested frame window")
    user_pcm = user_pcm[: args.frames * 1920]
    model_pcm = model_pcm[: args.frames * 1920]
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with torch.inference_mode():
        user = encode(codec, torch.from_numpy(user_pcm.copy())[None, None].cuda())
        target = encode(codec, torch.from_numpy(model_pcm.copy())[None, None].cuda())
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    user = user.clone()
    target = target.clone()
    silence = silence.clone()
    if user.shape != target.shape or user.shape[-1] != args.frames:
        raise ValueError("Unexpected codec frame alignment")
    activity = _activity(model_pcm, args.frames)
    mask = torch.ones((1, args.frames), device="cuda", dtype=torch.bool)
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    listening_steps = round(args.steps * 0.4)
    if listening_steps < 1 or listening_steps >= args.steps:
        raise ValueError("Both phases need training steps")
    if not 0 <= args.self_conditioned_steps <= args.steps - listening_steps:
        raise ValueError("Self-conditioned steps must fit within the response phase")
    metadata = {
        "experiment_id": output.name,
        "hypothesis": (
            "Self-conditioned response training produces an audible overfit response "
            "from the same dialogue excerpt."
            if args.self_conditioned_steps
            else "A single synchronized dialogue excerpt can be memorized with causal two-phase training."
        ),
        "control_run": args.control_run,
        "code_revision": _git_sha(),
        "code_clean": _git_clean(),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "manifest_sha256": manifest_sha,
        "split": "train",
        "conversation_id": args.conversation_id,
        "frames": args.frames,
        "sample_rate": 24_000,
        "steps": args.steps,
        "listening_steps": listening_steps,
        "response_steps": args.steps - listening_steps,
        "self_conditioned_steps": args.self_conditioned_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "model_parameters": model.trainable_parameters(),
        "model_config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
    _json_write(output / "metadata.json", metadata)
    sf.write(output / "user.wav", user_pcm, 24_000)
    sf.write(output / "target.wav", model_pcm, 24_000)
    started = time.monotonic()
    phase_first: dict[str, float] = {}
    phase_last: dict[str, float] = {}
    loss_curve: list[float] = []
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps):
            phase = "listen" if step < listening_steps else "respond"
            history_codes = None
            if phase == "respond" and step >= args.steps - args.self_conditioned_steps:
                model.eval()
                with torch.inference_mode():
                    stream = model.new_stream(silence)
                    rollout = [
                        model.generate_step(user[:, :, frame], stream)[0]
                        for frame in range(args.frames)
                    ]
                    history_codes = torch.stack(rollout, dim=-1)
                history_codes = history_codes.clone()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                user,
                target,
                silence,
                initial_user_codes=silence if phase == "listen" else None,
                history_codes=history_codes,
            )
            response = training_loss(prediction, target, activity, mask, config)
            user_loss = (
                listening_loss(prediction, user, mask, config)
                if phase == "listen"
                else None
            )
            loss = (
                response.total + user_loss if user_loss is not None else response.total
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            phase_first.setdefault(phase, float(loss.detach()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            current_loss = float(loss.detach())
            phase_last[phase] = current_loss
            loss_curve.append(current_loss)
            record = {
                "step": step + 1,
                "phase": phase,
                "conditioning": "self" if history_codes is not None else "teacher",
                "total_loss": current_loss,
                "response_cb0": float(response.cb0.detach()),
                "response_depth": float(response.depth.detach()),
                "response_activity": float(response.activity.detach()),
                "user_loss": float(user_loss.detach())
                if user_loss is not None
                else None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
                print(json.dumps(record), flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
            "step": args.steps,
        },
        output / "checkpoint.pt",
    )
    model.eval()
    with torch.inference_mode():
        prediction = model(user, target, silence)
        final_response = training_loss(prediction, target, activity, mask, config)
        state = model.new_stream(silence)
        generated = []
        generated_activity = []
        for frame in range(args.frames):
            codes, logits = model.generate_step(user[:, :, frame], state)
            generated.append(codes)
            generated_activity.append(logits.argmax(dim=-1))
        generated_codes = torch.stack(generated, dim=-1)
        generated_activity_tensor = torch.stack(generated_activity, dim=-1)
        generated_pcm = decode(codec, generated_codes)[0, 0].float().cpu().numpy()
        codec_reference_pcm = decode(codec, target)[0, 0].float().cpu().numpy()
    sf.write(output / "generated.wav", generated_pcm, 24_000)
    sf.write(output / "codec_reference.wav", codec_reference_pcm, 24_000)
    _write_loss_plot(output / "loss.svg", loss_curve, listening_steps)
    exact_tokens = generated_codes == target
    exact_frames = exact_tokens.all(dim=1)
    speech_mask = activity.bool()
    summary = {
        "phase_first_loss": phase_first,
        "phase_last_loss": phase_last,
        "final_response_loss": float(final_response.total),
        "duration_seconds": round(time.monotonic() - started, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "generated_frames": args.frames,
        "target_speech_frames": int(activity.sum()),
        "generated_speech_frames": int(generated_activity_tensor.sum()),
        "exact_token_fraction": float(exact_tokens.float().mean()),
        "exact_frames": int(exact_frames.sum()),
        "exact_speech_frames": int(exact_frames[speech_mask].sum()),
        "exact_codec_overfit": bool(exact_tokens.all()),
        "free_running_activity_f1": (
            2
            * int(
                (generated_activity_tensor == activity)
                .logical_and(activity.bool())
                .sum()
            )
            / max(int(activity.sum() + generated_activity_tensor.sum()), 1)
        ),
        "gpu_hours": round((time.monotonic() - started) / 3600, 5),
        "output_files": [
            "metadata.json",
            "metrics.jsonl",
            "checkpoint.pt",
            "user.wav",
            "target.wav",
            "generated.wav",
            "codec_reference.wav",
            "summary.json",
            "report.md",
            "loss.svg",
        ],
    }
    _json_write(output / "summary.json", summary)
    report = f"""# {output.name}: Single-conversation overfit probe

## Hypothesis

{metadata["hypothesis"]}

## Setup

- Code revision: `{metadata["code_revision"]}` (clean: {metadata["code_clean"]})
- Dataset revision: `{REVISION}`; manifest SHA-256: `{manifest_sha}`
- Split and example: `train/{args.conversation_id}`
- Causal window: {args.frames} frames ({args.frames / 12.5:.2f} seconds)
- Seed: {args.seed}; learning rate: {args.learning_rate}
- Listening steps: {listening_steps}; response steps: {args.steps - listening_steps}
- Self-conditioned response steps: {args.self_conditioned_steps}
- Control run: {args.control_run or "none"}
- GPU: {metadata["environment"]["gpu"]}; peak VRAM: {summary["peak_vram_bytes"]} bytes
- Compute: {summary["gpu_hours"]} GPU-hours

## Measured results

- Listening loss: {phase_first["listen"]:.4f} to {phase_last["listen"]:.4f}
- Response loss: {phase_first["respond"]:.4f} to {phase_last["respond"]:.4f}
- Final teacher-forced response loss: {summary["final_response_loss"]:.4f}
- Free-running speech activity F1: {summary["free_running_activity_f1"]:.4f}
- Target / generated speech frames: {summary["target_speech_frames"]} / {summary["generated_speech_frames"]}
- Exact Mimi token fraction: {summary["exact_token_fraction"]:.6f}
- Exact frames: {summary["exact_frames"]} / {args.frames}
- Exact speech frames: {summary["exact_speech_frames"]} / {summary["target_speech_frames"]}
- Exact codec overfit: {summary["exact_codec_overfit"]}
- Frame-level metrics: `metrics.jsonl`
- Loss curve: `loss.svg`
- Fixed audio examples: `user.wav`, `target.wav`, `codec_reference.wav`, `generated.wav`
- Checkpoint: `checkpoint.pt`

## Interpretation

This probe measures memorization of one training excerpt. It does not establish
held-out generalization, speech intelligibility, or semantic understanding.
Listen to the fixed generated sample and compare it with the target before
assigning a qualitative outcome.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
