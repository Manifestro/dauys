"""Does the model say the right reply, not only recognise it?

For every word-meaning question in the held-out voices, the model hears A and
then speaks on its own. Whisper transcribes the model channel for scoring only
(text never enters the model). A reply is ``intelligible`` when Whisper finds
words, and ``correct`` when its words overlap the expected reply more than any
other reply of the same topic.

Decoding variants, so the decoding cannot hide the cause:

- ``default``: the v0.2 default (activity-gated CB0, sampled depth, hysteresis).
- ``greedy``: argmax everywhere.
- ``greedy_ungated``: argmax, and every generated frame is played even when
  the activity head says silence.
- ``sample_low``: CB0 and depth sampled at low temperature.

``seed`` > 0 first teacher-forces that many true frames of the reply after the
pause; by default only the frames generated afterwards are played. With
``include_seed`` the teacher-forced frames are prepended to the audio as well.
If seeding fixes the replies, the model fails at choosing how the reply starts,
not at speaking.
"""

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.decoding import (
    ActivityState,
    DecodingConfig,
    activity_decision,
    cb0_decision,
    sample_logits,
)
from dauys.model import DauysHybrid
from dauys.v02_pretrain_eval import WhisperTranscriber

PAUSE = 6
STOP = {
    "a",
    "an",
    "the",
    "it",
    "is",
    "are",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "we",
    "our",
    "some",
    "here",
    "let's",
    "lets",
    "go",
    "it's",
    "its",
    "says",
    "say",
}
VARIANTS = {
    "default": (DecodingConfig.v02_default(), True),
    "greedy": (DecodingConfig(), True),
    "greedy_ungated": (DecodingConfig(), False),
    "sample_low": (
        DecodingConfig(
            cb0_mode="sample",
            depth_mode="sample",
            cb0_temperature=0.3,
            cb0_top_k=20,
            depth_temperature=0.5,
            depth_top_k=50,
        ),
        True,
    ),
}


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def judge(transcript: str, expected: str, alternatives: list[str]) -> bool:
    heard = content(transcript)
    scores = {reply: len(heard & content(reply)) for reply in alternatives}
    best = max(scores.values())
    return (
        best > 0 and scores[expected] == best and list(scores.values()).count(best) == 1
    )


def combine_guidance(cond_logits, uncond_logits, guidance):
    """Classifier-free guidance in probability space, returned as log-probs."""
    if cond_logits.shape != uncond_logits.shape:
        raise ValueError("Conditional and unconditional logits must align")
    if guidance == 1.0:
        return F.log_softmax(cond_logits, dim=-1)
    if guidance == 0.0:
        return F.log_softmax(uncond_logits, dim=-1)
    cond = F.log_softmax(cond_logits, dim=-1)
    uncond = F.log_softmax(uncond_logits, dim=-1)
    return uncond + guidance * (cond - uncond)


@torch.inference_mode()
def speak_codes(
    model,
    codec,
    silence,
    a_codes,
    b_codes,
    *,
    seed,
    frames,
    decoding,
    gated,
    rng,
):
    """Generate gated model codes [8, frames] on CPU, without seed frames."""
    device = next(model.parameters()).device
    quiet = silence.to(device)[None]
    a = a_codes.long().to(device)[None]
    prefix = a.shape[-1] + PAUSE + seed
    user = quiet[:, :, None].expand(1, 8, prefix).clone()
    user[:, :, : a.shape[-1]] = a
    output = quiet[:, :, None].expand(1, 8, prefix).clone()
    if seed:
        output[:, :, -seed:] = b_codes[:, :seed].long().to(device)[None]
    state = model.prefill_teacher(user, output, quiet)
    codes, active = [], []
    for _ in range(frames):
        step, speaking = model.generate_step(quiet, state, decoding, rng)
        codes.append(step)
        active.append(speaking)
    codes = torch.stack(codes, dim=-1)
    if gated:
        speaking = torch.stack(active, dim=-1)[:, None]
        codes = torch.where(speaking, codes, quiet[:, :, None])
    return codes.long().cpu()


@torch.inference_mode()
def speak(
    model,
    codec,
    silence,
    a_codes,
    b_codes,
    *,
    seed,
    frames,
    decoding,
    gated,
    rng,
    include_seed: bool = False,
):
    device = next(model.parameters()).device
    codes = speak_codes(
        model,
        codec,
        silence,
        a_codes,
        b_codes,
        seed=seed,
        frames=frames,
        decoding=decoding,
        gated=gated,
        rng=rng,
    ).to(device)
    if include_seed and seed:
        seed_codes = b_codes[:, :seed].long().to(device)[None]
        codes = torch.cat((seed_codes, codes), dim=-1)
    return decode(codec, codes.long())[0, 0].float().cpu().numpy()


@torch.inference_mode()
def speak_cfg(
    model,
    codec,
    silence,
    a_codes,
    b_codes,
    *,
    seed,
    frames,
    decoding,
    gated,
    rng,
    guidance,
    include_seed=False,
):
    """Generate with classifier-free guidance over the conditional ear."""
    if guidance == 1.0:
        return speak(
            model,
            codec,
            silence,
            a_codes,
            b_codes,
            seed=seed,
            frames=frames,
            decoding=decoding,
            gated=gated,
            rng=rng,
            include_seed=include_seed,
        )
    device = next(model.parameters()).device
    quiet = silence.to(device)[None]
    a = a_codes.long().to(device)[None]
    prefix = a.shape[-1] + PAUSE + seed
    user = quiet[:, :, None].expand(1, 8, prefix).clone()
    user[:, :, : a.shape[-1]] = a
    silent_user = quiet[:, :, None].expand(1, 8, prefix).clone()
    output = quiet[:, :, None].expand(1, 8, prefix).clone()
    if seed:
        output[:, :, -seed:] = b_codes[:, :seed].long().to(device)[None]
    state_c = model.prefill_teacher(user, output, quiet)
    state_u = model.prefill_teacher(silent_user, output, quiet)
    codes, active = [], []
    for _ in range(frames):
        cb0_c, activity_c, context_c = model.stream_step_teacher(
            quiet, state_c.previous_output, state_c
        )
        cb0_u, _, _ = model.stream_step_teacher(quiet, state_u.previous_output, state_u)
        logits = combine_guidance(cb0_c, cb0_u, guidance)
        if decoding is None:
            generated = [logits.argmax(dim=-1)]
            speaking = activity_c
        else:
            decoding.validate()
            if state_c.activity_state is None:
                state_c.activity_state = ActivityState.silent(
                    cb0_c.shape[0], cb0_c.device
                )
            speaking = activity_decision(
                activity_c, decoding, state_c.activity_state, rng
            )
            generated = [cb0_decision(logits, speaking, decoding, rng)]
        for _target in range(1, 8):
            prefix_codes = torch.stack(generated, dim=-1)[:, None]
            depth_logits = model.depth(prefix_codes, context_c)[:, 0, -1]
            if decoding is not None and decoding.depth_mode == "sample":
                generated.append(
                    sample_logits(
                        depth_logits,
                        decoding.depth_temperature,
                        decoding.depth_top_k,
                        rng,
                    )
                )
            else:
                generated.append(depth_logits.argmax(dim=-1))
        generated_codes = torch.stack(generated, dim=1)
        state_c.previous_output = generated_codes
        state_u.previous_output = generated_codes
        codes.append(generated_codes)
        active.append(speaking)
    codes = torch.stack(codes, dim=-1)
    if gated:
        speaking = torch.stack(active, dim=-1)[:, None]
        codes = torch.where(speaking, codes, quiet[:, :, None])
    if include_seed and seed:
        seed_codes = b_codes[:, :seed].long().to(device)[None]
        codes = torch.cat((seed_codes, codes), dim=-1)
    return decode(codec, codes.long())[0, 0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split", default="test_voice")
    parser.add_argument("--voices", nargs="+", default=["af_nova", "bm_lewis"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--wavs", type=Path, help="Keep a few generated replies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replies: dict[tuple[str, str], str] = {}
    by_topic: dict[str, list[str]] = {}
    for line in args.plan.read_text().splitlines():
        item = json.loads(line)
        if item["b_text"] is None:
            continue
        replies[(item["topic"], item["word"])] = item["b_text"]
        if item["b_text"] not in by_topic.setdefault(item["topic"], []):
            by_topic[item["topic"]].append(item["b_text"])
    root = args.data / args.split
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    chosen, seen = [], set()
    for row in rows:
        key = (row["speaker"], row["topic"], row["word"])
        if row["speaker"] in args.voices and key not in seen:
            seen.add(key)
            chosen.append(row)
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    payload = load_research_checkpoint(args.checkpoint)
    model = DauysHybrid(ModelConfig.from_yaml(args.config))
    model.load_state_dict(payload["model"])
    model = model.cuda().eval()
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    asr = WhisperTranscriber(args.asr_model, "cuda")
    clips = {}
    for row in chosen:
        for name in (row["a_clip"], row["b_clip"]):
            if name not in clips:
                clips[name] = torch.load(root / name, weights_only=True)["codes"]
    results = []
    for variant in args.variants:
        decoding, gated = VARIANTS[variant]
        for seed in args.seeds:
            rng = torch.Generator(device="cuda").manual_seed(0)
            waves = [
                speak(
                    model,
                    codec,
                    silence,
                    clips[row["a_clip"]],
                    clips[row["b_clip"]],
                    seed=seed,
                    frames=args.frames,
                    decoding=dataclasses.replace(decoding),
                    gated=gated,
                    rng=rng,
                )
                for row in chosen
            ]
            texts = asr(waves)
            intelligible = correct = 0
            examples = []
            for row, text in zip(chosen, texts):
                expected = replies[(row["topic"], row["word"])]
                ok = judge(text, expected, by_topic[row["topic"]])
                intelligible += bool(text.split())
                correct += ok
                if len(examples) < 16:
                    examples.append(
                        {
                            "word": row["word"],
                            "expected": expected,
                            "heard": text,
                            "ok": ok,
                        }
                    )
            summary = {
                "variant": variant,
                "seed_frames": seed,
                "items": len(chosen),
                "intelligible": round(intelligible / len(chosen), 3),
                "correct": round(correct / len(chosen), 3),
                "chance_correct": round(
                    float(np.mean([1 / len(by_topic[r["topic"]]) for r in chosen])), 3
                ),
                "examples": examples,
            }
            print(
                json.dumps({k: v for k, v in summary.items() if k != "examples"}),
                flush=True,
            )
            results.append(summary)
            if args.wavs and seed == 0:
                import soundfile

                folder = args.wavs / variant
                folder.mkdir(parents=True, exist_ok=True)
                for row, wave in list(zip(chosen, waves))[:8]:
                    soundfile.write(
                        folder / f"{row['speaker']}-{row['word']}.wav", wave, 24_000
                    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
