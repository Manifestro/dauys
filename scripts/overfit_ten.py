"""Memorize ten complete conversations and audit every causal codec token."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from overfit import DATASET, REVISION, _pcm

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, encode, load_mimi
from dauys.config import ModelConfig
from dauys.loss import listening_loss, training_loss
from dauys.model import DauysHybrid


@dataclass
class Conversation:
    identifier: str
    user: torch.Tensor
    target: torch.Tensor
    activity: torch.Tensor
    samples: int
    user_pcm: np.ndarray
    target_pcm: np.ndarray

    @property
    def frames(self) -> int:
        return self.target.shape[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persistent-checkpoint", type=Path, required=True)
    parser.add_argument("--persistent-latest-checkpoint", type=Path)
    parser.add_argument("--persistent-final-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--history-refresh-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_ten(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    if inventory["dataset"] != DATASET or inventory["revision"] != REVISION:
        raise ValueError("Inventory differs from the pinned dataset")
    rows = sorted(
        (row for row in inventory["rows"] if row["split"] == "train"),
        key=lambda row: (row["duration_samples"], row["id"]),
    )[:10]
    if len(rows) != 10 or len({row["id"] for row in rows}) != 10:
        raise ValueError("Ten unique train conversations are required")
    return [
        {
            "id": row["id"],
            "scenario": row["scenario"],
            "source_samples": row["duration_samples"],
            "codec_frames": math.ceil(row["duration_samples"] / 1920),
        }
        for row in rows
    ]


def load_conversations(
    selection: list[dict[str, Any]], codec: Any
) -> list[Conversation]:
    requests = {item["id"]: item for item in selection}
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    for column in ("channel_a", "channel_b"):
        if isinstance(dataset.features[column], Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    found = {}
    for row in dataset:
        identifier = row["id"]
        if identifier not in requests:
            continue
        item = requests[identifier]
        user_pcm, target_pcm = _pcm(row["channel_a"]), _pcm(row["channel_b"])
        samples = item["source_samples"]
        if len(user_pcm) != samples or len(target_pcm) != samples:
            raise ValueError(f"Audio length mismatch for {identifier}")
        frames = item["codec_frames"]
        padded_samples = frames * 1920
        user_pcm = np.pad(user_pcm, (0, padded_samples - samples))
        target_pcm = np.pad(target_pcm, (0, padded_samples - samples))
        with torch.inference_mode():
            user = encode(codec, torch.from_numpy(user_pcm.copy())[None, None].cuda())
            target = encode(
                codec, torch.from_numpy(target_pcm.copy())[None, None].cuda()
            )
        if user.shape != (1, 8, frames) or target.shape != user.shape:
            raise ValueError(f"Codec alignment mismatch for {identifier}")
        blocks = target_pcm.reshape(frames, 1920)
        activity = torch.from_numpy(
            (np.sqrt(np.mean(np.square(blocks), axis=1)) > 1e-4).astype(np.int64)
        )[None]
        found[identifier] = Conversation(
            identifier,
            user.clone(),
            target.clone(),
            activity.clone(),
            samples,
            user_pcm,
            target_pcm,
        )
        if len(found) == 10:
            break
    if set(found) != set(requests):
        raise ValueError(f"Missing conversations: {set(requests) - set(found)}")
    return [found[row["id"]] for row in selection]


def on_device(case: Conversation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return case.user.cuda(), case.target.cuda(), case.activity.cuda()


def rollout(
    model: DauysHybrid, user: torch.Tensor, silence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = model.new_stream(silence)
    frames = []
    activity = []
    for frame in range(user.shape[-1]):
        codes, logits = model.generate_step(user[:, :, frame], state)
        frames.append(codes)
        activity.append(logits.argmax(dim=-1))
    return torch.stack(frames, dim=-1), torch.stack(activity, dim=-1)


def evaluate(
    model: DauysHybrid,
    cases: list[Conversation],
    silence: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    items = []
    exact_tokens = 0
    total_tokens = 0
    with torch.inference_mode():
        for case in cases:
            user, target, activity = on_device(case)
            generated, predicted_activity = rollout(model, user, silence)
            equal = generated == target
            exact = int(equal.sum())
            count = equal.numel()
            exact_tokens += exact
            total_tokens += count
            items.append(
                {
                    "id": case.identifier,
                    "frames": case.frames,
                    "exact_tokens": exact,
                    "total_tokens": count,
                    "exact_frames": int(equal.all(dim=1).sum()),
                    "target_speech_frames": int(activity.sum()),
                    "predicted_speech_frames": int(predicted_activity.sum()),
                }
            )
    return {
        "exact_tokens": exact_tokens,
        "total_tokens": total_tokens,
        "exact_token_fraction": exact_tokens / total_tokens,
        "all_ten_exact": exact_tokens == total_tokens,
        "cases": items,
    }


def export_audio(
    model: DauysHybrid,
    cases: list[Conversation],
    codec: Any,
    silence: torch.Tensor,
    output: Path,
) -> dict[str, int]:
    model.eval()
    audio = output / "audio"
    audio.mkdir(exist_ok=True)
    exact_tokens = {}
    with torch.inference_mode():
        for case in cases:
            user, target, _ = on_device(case)
            generated, _ = rollout(model, user, silence)
            exact_tokens[case.identifier] = int((generated == target).sum())
            wave = decode(codec, generated)[0, 0].float().cpu().numpy()
            reference = decode(codec, target)[0, 0].float().cpu().numpy()
            sf.write(
                audio / f"{case.identifier}_target.wav",
                case.target_pcm[: case.samples],
                24_000,
            )
            sf.write(
                audio / f"{case.identifier}_codec_reference.wav",
                reference[: case.samples],
                24_000,
            )
            sf.write(
                audio / f"{case.identifier}_generated.wav", wave[: case.samples], 24_000
            )
    return exact_tokens


def main() -> None:
    args = parse_args()
    if (
        not torch.cuda.is_available()
        or args.steps < 10
        or args.eval_every < 1
        or args.save_every < 0
        or args.history_refresh_every < 0
        or (args.save_every and args.persistent_latest_checkpoint is None)
    ):
        raise ValueError("CUDA and a valid training schedule are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.persistent_checkpoint.exists() and args.resume is None:
        raise FileExistsError(args.persistent_checkpoint)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    inventory = json.loads(args.inventory.read_text())
    selection = select_ten(inventory)
    args.output.mkdir(parents=True)
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    started = time.monotonic()
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    cases = load_conversations(selection, codec)
    with torch.inference_mode():
        silence = encode(codec, torch.zeros((1, 1, 1920), device="cuda"))[:, :, 0]
    config = ModelConfig.stage1()
    model = DauysHybrid(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    first_step = 1
    best_tokens = -1
    listening_steps = round(args.steps * 0.4)
    if args.resume is not None:
        checkpoint = load_research_checkpoint(args.resume)
        if checkpoint["metadata"]["selection_sha256"] != sha256(
            args.output / "selection.json"
        ):
            raise ValueError("Resume selection mismatch")
        if checkpoint["metadata"]["model_config"] != asdict(config):
            raise ValueError("Resume model configuration mismatch")
        if checkpoint["metadata"]["seed"] != args.seed:
            raise ValueError("Resume seed mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        first_step = int(checkpoint["step"]) + 1
        previous_evaluation = checkpoint.get("evaluation")
        best_tokens = (
            int(previous_evaluation["exact_tokens"])
            if previous_evaluation is not None
            else -1
        )
        listening_steps = int(checkpoint["metadata"]["listening_steps"])
        shutil.copyfile(args.resume, args.output / "checkpoint_best.pt")
        args.persistent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if args.resume != args.persistent_checkpoint:
            shutil.copyfile(args.resume, args.persistent_checkpoint)
    if first_step > args.steps:
        raise ValueError("Resume checkpoint is already at or beyond the final step")
    metadata = {
        "experiment_id": args.output.name,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "manifest_sha256": inventory["manifest_sha256"],
        "inventory_sha256": sha256(args.inventory),
        "selection_sha256": sha256(args.output / "selection.json"),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_clean": not subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "model_config": asdict(config),
        "trainable_parameters": model.trainable_parameters(),
        "seed": args.seed,
        "steps": args.steps,
        "listening_steps": listening_steps,
        "resume_checkpoint_sha256": sha256(args.resume) if args.resume else None,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "history_refresh_every": args.history_refresh_every,
        "sample_count": 10,
        "source_samples": sum(case.samples for case in cases),
        "codec_frames": sum(case.frames for case in cases),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log_path = args.output / "metrics.jsonl"
    last_evaluation = None
    cached_history: dict[str, torch.Tensor] = {}
    with log_path.open("w") as log:
        for step in range(first_step, args.steps + 1):
            case = cases[(step - 1) % len(cases)]
            user, target, activity = on_device(case)
            listen = step <= metadata["listening_steps"]
            history = None
            refreshed = False
            if not listen:
                if args.history_refresh_every:
                    if (
                        not cached_history
                        or (step - listening_steps - 1) % args.history_refresh_every
                        == 0
                    ):
                        model.eval()
                        for source in cases:
                            source_user, _, _ = on_device(source)
                            with torch.inference_mode():
                                generated, _ = rollout(model, source_user, silence)
                            cached_history[source.identifier] = generated.clone().cpu()
                        refreshed = True
                    history = cached_history[case.identifier].cuda().clone()
                else:
                    model.eval()
                    with torch.inference_mode():
                        history, _ = rollout(model, user, silence)
                    history = history.clone()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                user,
                target,
                silence,
                initial_user_codes=silence if listen else None,
                history_codes=history,
            )
            mask = torch.ones_like(activity, dtype=torch.bool)
            response = training_loss(prediction, target, activity, mask, config)
            user_loss = (
                listening_loss(prediction, user, mask, config) if listen else None
            )
            loss = (
                response.total + user_loss if user_loss is not None else response.total
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            record: dict[str, Any] = {
                "step": step,
                "conversation_id": case.identifier,
                "phase": "listen" if listen else "respond",
                "loss": float(loss.detach()),
                "response_cb0": float(response.cb0.detach()),
                "response_depth": float(response.depth.detach()),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if refreshed:
                record["history_refreshed"] = True
            if step % args.eval_every == 0 or step == args.steps:
                last_evaluation = evaluate(model, cases, silence)
                record["evaluation"] = last_evaluation
                if last_evaluation["exact_tokens"] > best_tokens:
                    best_tokens = last_evaluation["exact_tokens"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metadata": metadata,
                        "step": step,
                        "evaluation": last_evaluation,
                    }
                    local_checkpoint = args.output / "checkpoint_best.pt"
                    torch.save(checkpoint, local_checkpoint)
                    args.persistent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local_checkpoint, args.persistent_checkpoint)
                    if sha256(local_checkpoint) != sha256(args.persistent_checkpoint):
                        raise RuntimeError("Persistent checkpoint verification failed")
                    record["best_checkpoint"] = True
            if args.save_every and step % args.save_every == 0 and step < args.steps:
                assert args.persistent_latest_checkpoint is not None
                latest = args.output / "checkpoint_latest.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metadata": metadata,
                        "step": step,
                        "evaluation": last_evaluation,
                    },
                    latest,
                )
                args.persistent_latest_checkpoint.parent.mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copyfile(latest, args.persistent_latest_checkpoint)
                if sha256(latest) != sha256(args.persistent_latest_checkpoint):
                    raise RuntimeError("Latest checkpoint verification failed")
                record["latest_checkpoint"] = True
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or "evaluation" in record:
                print(json.dumps(record), flush=True)
    if last_evaluation is None:
        raise RuntimeError("Final evaluation was not recorded")
    final_checkpoint = args.output / "checkpoint_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
            "step": args.steps,
            "evaluation": last_evaluation,
        },
        final_checkpoint,
    )
    if args.persistent_final_checkpoint is not None:
        args.persistent_final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_checkpoint, args.persistent_final_checkpoint)
        if sha256(final_checkpoint) != sha256(args.persistent_final_checkpoint):
            raise RuntimeError("Final checkpoint verification failed")
    best = load_research_checkpoint(args.output / "checkpoint_best.pt")
    final = load_research_checkpoint(final_checkpoint)
    model.load_state_dict(final["model"], strict=True)
    exported = export_audio(model, cases, codec, silence, args.output)
    expected = {item["id"]: item["exact_tokens"] for item in last_evaluation["cases"]}
    if exported != expected:
        raise RuntimeError("Exported audio codes differ from final evaluation")
    summary = {
        "best_step": best["step"],
        "best_evaluation": best["evaluation"],
        "final_step": args.steps,
        "final_evaluation": last_evaluation,
        "best_checkpoint_sha256": sha256(args.output / "checkpoint_best.pt"),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "persistent_final_checkpoint_sha256": (
            sha256(args.persistent_final_checkpoint)
            if args.persistent_final_checkpoint is not None
            else None
        ),
        "duration_seconds": round(time.monotonic() - started, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
