"""Listening bench: does a full-duplex model respond to what it hears, or how?

    python -m duplexbench.run prepare --cache CACHE [--slm21 DIR] [--storycloze DIR]
    python -m duplexbench.run score --cache CACHE --model moshi --out OUT
    python -m duplexbench.run score --cache CACHE --model dauys \\
        --dauys-config C.yaml --dauys-checkpoint X.pt --out OUT

Measures (see bench/README.md):
- contingency gap: own-channel NLL with the true partner vs a swapped partner
  (and vs silence), teacher-forced, per conversation;
- echo rate: share of the model's spoken CB0 4-grams found in what the
  partner said, against the same share for a swapped partner;
- voice similarity: speaker-embedding cosine between the model's generated
  speech and its partner's voice, against other partners' voices;
- sWUGGY, sBLIMP, tStoryCloze: pairwise accuracy of CB0 log-likelihood with
  the item spoken on the model's own channel and silence on the partner's.
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from duplexbench import data, metrics
from duplexbench.adapters import DauysAdapter, MoshiAdapter

LEAD_FRAMES = 6  # 0.5 s of silence before a single-stream item
SPEECH_RMS = 0.01  # frame RMS above which decoded model output counts as speech


def prepare(args) -> None:
    mimi = data.load_mimi()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ("hohere", "swuggy", "sblimp", "tstorycloze"))
    conversations = (
        data.hohere_conversations(args.hohere_revision) if "hohere" in only else []
    )
    rows = []
    for item in conversations:
        a, b = data.encode(mimi, item["audio_a"]), data.encode(mimi, item["audio_b"])
        frames = min(a.shape[-1], b.shape[-1], len(item["activity_a"]))
        rows.append(
            {
                **{k: item[k] for k in ("id", "skill", "voice_a", "voice_b")},
                "a": a[:, :frames],
                "b": b[:, :frames],
                "activity_a": item["activity_a"][:frames],
                "activity_b": item["activity_b"][:frames],
            }
        )
    if rows:
        torch.save(rows, cache / "hohere.pt")
    silence = data.encode(mimi, np.zeros(data.FRAME_SAMPLES * 4, dtype=np.float32))
    torch.save(silence[:, -1], cache / "silence.pt")
    print({"hohere": len(rows), "silence": silence[:, -1].tolist()}, flush=True)
    sets = {}
    if args.slm21:
        sets["swuggy"] = data.slm21_pairs(Path(args.slm21), "lexical")
        sets["sblimp"] = data.slm21_pairs(Path(args.slm21), "syntactic")
    if args.storycloze:
        sets["tstorycloze"] = data.storycloze_pairs(Path(args.storycloze))
    for name, pairs in sets.items():
        if name not in only:
            continue
        encoded = [
            {
                "id": pair["id"],
                "kind": pair.get("kind"),
                "good": data.encode(mimi, data.read_audio(str(pair["good"]))),
                "bad": data.encode(mimi, data.read_audio(str(pair["bad"]))),
            }
            for pair in pairs
        ]
        torch.save(encoded, cache / f"{name}.pt")
        print({name: len(encoded)}, flush=True)


def _adapter(args):
    if args.model == "moshi":
        # Moshi learned silence as Mimi's encoding of digital silence.
        silence = torch.load(Path(args.cache) / "silence.pt")
        return MoshiAdapter.load(silence, args.moshi_repo)
    # Dauys pads, starts and mutes with the silence frame it was trained on.
    silence = torch.tensor(args.dauys_silence)
    return DauysAdapter.load(args.dauys_config, args.dauys_checkpoint, silence)


def contingency(adapter, rows: list[dict], max_frames: int, seed: int) -> dict:
    partners = metrics.length_matched_partners([row["a"].shape[-1] for row in rows])
    silence = adapter.silence
    per = []
    for index, row in enumerate(rows):
        frames = min(row["a"].shape[-1], max_frames)
        own = row["b"][:, :frames]
        speech = row["activity_b"][:frames]
        partner = rows[partners[index]]
        timing = row["activity_a"][:frames]
        heard = {
            "true": row["a"][:, :frames],
            "swap": metrics.fit_length(partner["a"], frames, silence),
            "silence": silence[:, None].expand(-1, frames).clone(),
            # Same turn timing, true vs another speaker's words and voice.
            "true_timed": metrics.splice_timed(
                row["a"], row["activity_a"], timing, silence
            ),
            "swap_timed": metrics.splice_timed(
                partner["a"], partner["activity_a"], timing, silence
            ),
        }
        users = torch.stack(list(heard.values()))
        cb0, acoustic = adapter.frame_nll(users, own[None].expand(len(heard), -1, -1))
        result = {
            "id": row["id"],
            "skill": row["skill"],
            "speech_frames": int(speech.sum()),
        }
        for k, name in enumerate(heard):
            result[f"cb0_{name}"] = metrics.masked_mean(cb0[k], speech)
            result[f"acoustic_{name}"] = metrics.masked_mean(acoustic[k], speech)
        per.append(result)
    summary = {}
    for stream in ("cb0", "acoustic"):
        for name in ("true", "swap", "silence", "true_timed", "swap_timed"):
            summary[f"{stream}_{name}"] = metrics.bootstrap_ci(
                [r[f"{stream}_{name}"] for r in per], seed=seed
            )
        for name, base in (
            ("swap", "true"),
            ("silence", "true"),
            ("swap_timed", "true_timed"),
        ):
            summary[f"{stream}_gap_{name}"] = metrics.bootstrap_ci(
                [r[f"{stream}_{name}"] - r[f"{stream}_{base}"] for r in per],
                seed=seed,
            )
    return {"summary": summary, "conversations": per}


def _speech_mask(audio: np.ndarray, frames: int) -> torch.Tensor:
    audio = np.pad(audio, (0, max(0, frames * data.FRAME_SAMPLES - len(audio))))
    chunks = audio[: frames * data.FRAME_SAMPLES].reshape(frames, data.FRAME_SAMPLES)
    return torch.from_numpy(np.sqrt((chunks**2).mean(1)) > SPEECH_RMS)


def _speech_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> list[int]:
    """CB0 tokens of speech frames, with a -1 break between speech runs."""
    out: list[int] = []
    for token, keep in zip(tokens.tolist(), mask.tolist()):
        if keep:
            out.append(token)
        elif out and out[-1] != -1:
            out.append(-1)
    return out


def _echo(spoken: list[int], heard: list[int], n: int) -> float:
    broken = frozenset(
        gram for gram in metrics.ngrams(spoken + [-1] + heard, n) if -1 in gram
    )
    return metrics.echo_rate(spoken, heard, n=n, ignore=broken)


def _segments(audio: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    keep = np.repeat(mask.numpy(), data.FRAME_SAMPLES)[: len(audio)]
    return audio[: len(keep)][keep]


def generation(
    adapter, rows: list[dict], frames: int, batch: int, seed: int, out: Path
) -> dict:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    mimi = data.load_mimi()
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    speaker = (
        WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()
    )

    @torch.inference_mode()
    def embed(audio: np.ndarray) -> torch.Tensor | None:
        if len(audio) < data.SAMPLE_RATE:  # under 1 s of speech: no voice estimate
            return None
        from scipy.signal import resample_poly

        wav = resample_poly(audio, 2, 3).astype(np.float32)
        inputs = extractor(wav, sampling_rate=16_000, return_tensors="pt").to("cuda")
        return speaker(**inputs).embeddings[0].float().cpu()

    partners = metrics.length_matched_partners([row["a"].shape[-1] for row in rows])
    users = [metrics.fit_length(row["a"], frames, adapter.silence) for row in rows]
    generated: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, len(rows), batch):
        generated += list(
            adapter.generate(torch.stack(users[start : start + batch]), seed + start)
        )
        print(
            {"generated": len(generated), "seconds": round(time.time() - started)},
            flush=True,
        )
    torch.save(
        {"ids": [r["id"] for r in rows], "codes": generated}, out / "generated.pt"
    )
    voices = {}
    per = []
    for index, row in enumerate(rows):
        audio = data.decode(mimi, generated[index])
        spoken_mask = _speech_mask(audio, frames)
        heard_mask = metrics.fit_length(
            row["activity_a"][None].long(), frames, torch.zeros(1)
        ).bool()[0]
        partner = rows[partners[index]]
        partner_mask = metrics.fit_length(
            partner["activity_a"][None].long(), frames, torch.zeros(1)
        ).bool()[0]
        spoken = _speech_tokens(generated[index][0], spoken_mask)
        result = {
            "id": row["id"],
            "spoken_seconds": float(spoken_mask.sum()) / 12.5,
            "echo_true": _echo(spoken, _speech_tokens(users[index][0], heard_mask), 4),
            "echo_swap": _echo(
                spoken,
                _speech_tokens(
                    metrics.fit_length(partner["a"], frames, adapter.silence)[0],
                    partner_mask,
                ),
                4,
            ),
        }
        model_voice = embed(_segments(audio, spoken_mask))
        if model_voice is not None:
            for name, (source, mask) in {
                "user": (row, heard_mask),
                "other_user": (partner, partner_mask),
            }.items():
                key = (source["id"], "a")
                if key not in voices:
                    wav = data.decode(
                        mimi, metrics.fit_length(source["a"], frames, adapter.silence)
                    )
                    voices[key] = embed(_segments(wav, mask))
                if voices[key] is not None:
                    result[f"voice_{name}"] = metrics.cosine(model_voice, voices[key])
            reference = data.decode(mimi, row["b"][:, :frames])
            ref_voice = embed(_segments(reference, row["activity_b"][:frames]))
            if ref_voice is not None:
                result["voice_reference_assistant"] = metrics.cosine(
                    model_voice, ref_voice
                )
        per.append(result)
    summary = {
        "spoken_seconds": metrics.bootstrap_ci(
            [r["spoken_seconds"] for r in per], seed=seed
        ),
        "echo_true": metrics.bootstrap_ci([r["echo_true"] for r in per], seed=seed),
        "echo_swap": metrics.bootstrap_ci([r["echo_swap"] for r in per], seed=seed),
        "echo_excess": metrics.bootstrap_ci(
            [r["echo_true"] - r["echo_swap"] for r in per], seed=seed
        ),
    }
    for name in ("voice_user", "voice_other_user", "voice_reference_assistant"):
        summary[name] = metrics.bootstrap_ci(
            [r[name] for r in per if name in r], seed=seed
        )
    summary["voice_copy"] = metrics.bootstrap_ci(
        [
            r["voice_user"] - r["voice_other_user"]
            for r in per
            if "voice_user" in r and "voice_other_user" in r
        ],
        seed=seed,
    )
    return {"summary": summary, "conversations": per}


def single_stream(adapter, pairs: list[dict], batch: int) -> dict:
    """Pairwise accuracy of summed CB0 log-likelihood on the model's own channel."""
    silence = adapter.silence
    items = [
        (index, side, pair[side])
        for index, pair in enumerate(pairs)
        for side in ("good", "bad")
    ]
    items.sort(key=lambda item: item[2].shape[-1])
    scores: dict[tuple[int, str], tuple[float, float]] = {}
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        longest = max(codes.shape[-1] for _, _, codes in chunk) + LEAD_FRAMES
        lead = silence[:, None].expand(-1, LEAD_FRAMES)
        own = torch.stack(
            [
                metrics.fit_length(torch.cat((lead, codes), -1), longest, silence)
                for _, _, codes in chunk
            ]
        )
        user = silence[None, :, None].expand(len(chunk), -1, longest).clone()
        cb0, _ = adapter.frame_nll(user, own)
        for row, (index, side, codes) in enumerate(chunk):
            values = cb0[row, LEAD_FRAMES : LEAD_FRAMES + codes.shape[-1]]
            values = values[~torch.isnan(values)]
            scores[(index, side)] = (-float(values.sum()), -float(values.mean()))
    good_sum = [scores[(i, "good")][0] for i in range(len(pairs))]
    bad_sum = [scores[(i, "bad")][0] for i in range(len(pairs))]
    good_mean = [scores[(i, "good")][1] for i in range(len(pairs))]
    bad_mean = [scores[(i, "bad")][1] for i in range(len(pairs))]
    wins = [1.0 if g > b else 0.5 if g == b else 0.0 for g, b in zip(good_sum, bad_sum)]
    return {
        "pairs": len(pairs),
        "accuracy_sum": metrics.pair_accuracy(good_sum, bad_sum),
        "accuracy_sum_ci": metrics.bootstrap_ci(wins),
        "accuracy_mean": metrics.pair_accuracy(good_mean, bad_mean),
    }


def score(args) -> None:
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = _adapter(args)
    rows = torch.load(cache / "hohere.pt")
    if args.limit:
        rows = rows[: args.limit]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "model": args.model,
        "commit": commit,
        "seed": args.seed,
        "conversations": len(rows),
    }
    if args.model == "dauys":
        meta["checkpoint"] = args.dauys_checkpoint
    else:
        meta["repo"] = args.moshi_repo
    data.save_json(out / "meta.json", meta)
    started = time.time()
    if "contingency" in args.tests:
        result = contingency(adapter, rows, args.max_frames, args.seed)
        data.save_json(out / "contingency.json", result)
        print(
            {
                "contingency": result["summary"]["cb0_gap_swap"],
                "seconds": round(time.time() - started),
            },
            flush=True,
        )
    for name in ("swuggy", "sblimp", "tstorycloze"):
        if name in args.tests and (cache / f"{name}.pt").exists():
            pairs = torch.load(cache / f"{name}.pt")
            if args.limit:
                pairs = pairs[: args.limit * 20]
            result = single_stream(adapter, pairs, args.batch)
            data.save_json(out / f"{name}.json", result)
            print(
                {name: result["accuracy_sum"], "seconds": round(time.time() - started)},
                flush=True,
            )
    if "generation" in args.tests:
        result = generation(
            adapter, rows, args.gen_frames, args.gen_batch, args.seed, out
        )
        data.save_json(out / "generation.json", result)
        print(
            {
                "echo": result["summary"]["echo_excess"],
                "voice": result["summary"]["voice_copy"],
            },
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache", required=True)
    prep.add_argument("--hohere-revision", default="v0.2.0")
    prep.add_argument("--slm21")
    prep.add_argument("--storycloze")
    prep.add_argument(
        "--only", nargs="+", choices=("hohere", "swuggy", "sblimp", "tstorycloze")
    )
    run = sub.add_parser("score")
    run.add_argument("--cache", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model", choices=("moshi", "dauys"), required=True)
    run.add_argument("--moshi-repo", default="kyutai/moshiko-pytorch-bf16")
    run.add_argument("--dauys-config")
    run.add_argument("--dauys-checkpoint")
    run.add_argument(
        "--dauys-silence",
        type=int,
        nargs=8,
        default=[1049, 1700, 1626, 546, 306, 1443, 1871, 2008],
    )
    run.add_argument(
        "--tests",
        nargs="+",
        default=["contingency", "swuggy", "sblimp", "tstorycloze", "generation"],
    )
    run.add_argument("--max-frames", type=int, default=1500)
    run.add_argument("--gen-frames", type=int, default=750)
    run.add_argument("--gen-batch", type=int, default=16)
    run.add_argument("--batch", type=int, default=16)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        if args.model == "dauys" and not (args.dauys_config and args.dauys_checkpoint):
            parser.error("--model dauys needs --dauys-config and --dauys-checkpoint")
        score(args)


if __name__ == "__main__":
    main()
