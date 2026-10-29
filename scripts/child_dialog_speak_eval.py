"""Fast spoken-reply check for the child-dialogue set.

Takes ``--per-group`` held-out questions per family group from voices the model
never trained on. The model hears each question and speaks for ``--frames``
frames with the chosen decoding. Whisper transcribes the model channel for
scoring only.

- ``correct``: the transcript's content words match one answer of the group
  better than any other answer. Concepts that share an answer are one option.
- ``correct_first``: the same, using only the transcript's first six words.
- ``stopped``: the last second of the reply is silent.

``--seeds`` prefixes that many reply frames before free generation, giving one
summary per decoding variant and seed; those seed frames are included in the
audio scored by Whisper. ``--seed-source`` chooses the correct answer, another
answer of the group, or only the correct seed frames without generation.
``--rescore`` reads a previous output and re-scores its transcripts with the
same rules, without loading the model, Mimi or Whisper.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_pretrain_eval import WhisperTranscriber

sys.path.insert(0, str(Path(__file__).parent))
from v02_word_meaning_speak_eval import VARIANTS, content, speak, speak_cfg

QUIET_RMS = 0.01
ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


def _number_words(number: int) -> str:
    if number < 20:
        return ONES[number]
    if number == 100:
        return "one hundred"
    tens, ones = divmod(number, 10)
    return TENS[tens] + (f" {ONES[ones]}" if ones else "")


def words(text: str) -> set[str]:
    """Content words with digits 0..100 spoken as English words."""

    def replace(match: re.Match[str]) -> str:
        number = int(match.group())
        return _number_words(number) if 0 <= number <= 100 else match.group()

    return content(re.sub(r"\b\d+\b", replace, text))


def answer_key(answer: str) -> str:
    """Canonical answer identity, independent of punctuation and word order."""
    return " ".join(sorted(words(answer)))


def answer_groups(concepts: dict[str, dict]) -> dict[str, dict[str, list[str]]]:
    """Group replies by answer, so concepts with the same answer share a vote."""
    groups: dict[str, dict[str, list[str]]] = {}
    for concept in concepts.values():
        answers = groups.setdefault(concept["group"], {})
        replies = answers.setdefault(answer_key(concept["answer"]), [])
        replies.extend(reply for reply in concept["replies"] if reply not in replies)
    return groups


def best_answer(transcript: str, answers: dict[str, list[str]]) -> str | None:
    heard = words(transcript)
    scores = {
        answer: max((len(heard & words(reply)) for reply in replies), default=0)
        for answer, replies in answers.items()
    }
    top = max(scores.values(), default=0)
    winners = [answer for answer, score in scores.items() if score == top]
    return winners[0] if top > 0 and len(winners) == 1 else None


def pick_other_reply(rows_of_group: list[dict], row: dict, rng: random.Random) -> dict:
    """Pick a deterministic reply row whose answer differs from ``row``'s."""
    current = answer_key(row["answer"])
    candidates = [
        other for other in rows_of_group if answer_key(other["answer"]) != current
    ]
    if not candidates:
        raise ValueError(f"No other answer in group {row['topic']!r}")
    return rng.choice(candidates)


def _summary(
    variant: str,
    source: str,
    seed: int,
    guidance: float,
    examples: list[dict],
    by_group: dict[str, dict[str, list[str]]],
    pairs: dict[str, dict],
) -> dict:
    correct = sum(example["ok"] for example in examples)
    first = sum(example["ok_first"] for example in examples)
    stopped = sum(example["stopped"] for example in examples)
    follows = sum(bool(example.get("ok_seed_answer")) for example in examples)
    follows_first = sum(
        bool(example.get("ok_first_seed_answer")) for example in examples
    )
    n = len(examples)
    chance = (
        float(
            np.mean(
                [
                    1 / len(by_group[pairs[example["question"]]["group"]])
                    for example in examples
                ]
            )
        )
        if n
        else 0.0
    )
    return {
        "variant": variant,
        "source": source,
        "seed": seed,
        "guidance": guidance,
        "items": n,
        "correct": round(correct / n, 3) if n else 0.0,
        "correct_first": round(first / n, 3) if n else 0.0,
        "stopped": round(stopped / n, 3) if n else 0.0,
        "chance": round(chance, 3),
        "follows_seed": round(follows / n, 3) if n else 0.0,
        "follows_seed_first": round(follows_first / n, 3) if n else 0.0,
        "examples": examples,
    }


def _print_summary(summary: dict) -> None:
    print(
        json.dumps({k: v for k, v in summary.items() if k != "examples"}),
        flush=True,
    )


def rescore(
    path: Path,
    pairs_path: Path,
    concepts: dict[str, dict],
) -> list[dict]:
    pairs = {
        pair["question"]: pair
        for pair in map(json.loads, pairs_path.read_text().splitlines())
    }
    by_group = answer_groups(concepts)
    results = []
    for previous in json.loads(path.read_text()):
        examples = []
        source = str(previous.get("source", "true"))
        for example in previous["examples"]:
            pair = pairs[example["question"]]
            concept = concepts[pair["concept"]]
            expected = concept["answer"]
            heard = example["heard"]
            answers = by_group[pair["group"]]
            heard_answer = best_answer(heard, answers)
            heard_first = best_answer(" ".join(heard.split()[:6]), answers)
            seed_answer = example.get("seed_answer", expected)
            seed_key = answer_key(seed_answer)
            examples.append(
                {
                    "question": example["question"],
                    "expected": example.get("expected", expected),
                    "heard": heard,
                    "ok": heard_answer == answer_key(expected),
                    "ok_first": heard_first == answer_key(expected),
                    "stopped": bool(example["stopped"]),
                    "seed_answer": seed_answer,
                    "ok_seed_answer": heard_answer == seed_key,
                    "ok_first_seed_answer": heard_first == seed_key,
                }
            )
        summary = _summary(
            previous["variant"],
            source,
            int(previous.get("seed", 0)),
            float(previous.get("guidance", 1.0)),
            examples,
            by_group,
            pairs,
        )
        _print_summary(summary)
        results.append(summary)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mimi-checkpoint", type=Path)
    parser.add_argument("--silence-codes", type=Path)
    parser.add_argument("--data", type=Path, help="encoded split dir")
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--rescore", type=Path)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--variants", nargs="+", default=["greedy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--seed-source",
        choices=("true", "other", "only"),
        default="true",
    )
    parser.add_argument("--guidance", nargs="+", type=float, default=[1.0])
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--wavs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    concepts = {
        c["id"]: c for c in map(json.loads, args.concepts.read_text().splitlines())
    }
    if args.rescore is not None:
        if args.pairs is None:
            parser.error("--pairs is required with --rescore")
        results = rescore(args.rescore, args.pairs, concepts)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        return
    required = (
        "config",
        "checkpoint",
        "mimi_checkpoint",
        "silence_codes",
        "data",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(
            "generation mode requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    if args.seed_source == "only" and any(seed <= 0 for seed in args.seeds):
        parser.error("--seed-source only requires positive --seeds")
    by_group = answer_groups(concepts)
    rows = []
    for line in (args.data / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        row["answer"] = concepts[row["concept"]]["answer"]
        rows.append(row)
    rows_by_group: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_group.setdefault(row["topic"], []).append(row)
    rng = random.Random(0)
    rng.shuffle(rows)
    chosen, count = [], {}
    for row in rows:
        if count.get(row["topic"], 0) < args.per_group:
            count[row["topic"]] = count.get(row["topic"], 0) + 1
            chosen.append(row)
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    payload = load_research_checkpoint(args.checkpoint)
    model = DauysHybrid(ModelConfig.from_yaml(args.config))
    model.load_state_dict(payload["model"])
    model = model.cuda().eval()
    device = next(model.parameters()).device
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    asr = WhisperTranscriber(args.asr_model, "cuda")
    pairs = {
        row["question"]: {"concept": row["concept"], "group": row["topic"]}
        for row in chosen
    }
    others = {}
    if args.seed_source == "other":
        seed_rng = random.Random(0)
        for row in chosen:
            others[row["id"]] = pick_other_reply(
                rows_by_group[row["topic"]], row, seed_rng
            )
    results = []
    for variant in args.variants:
        decoding, gated = VARIANTS[variant]
        for seed in args.seeds:
            for guidance in args.guidance:
                generator = torch.Generator(device="cuda").manual_seed(0)
                waves = []
                for row in chosen:
                    a = torch.load(args.data / row["a_clip"], weights_only=True)[
                        "codes"
                    ]
                    b = torch.load(args.data / row["b_clip"], weights_only=True)[
                        "codes"
                    ]
                    if args.seed_source == "only":
                        codes = b[:, :seed].long().to(device)[None]
                        wave = decode(codec, codes)[0, 0].float().cpu().numpy()
                    else:
                        seed_b = b
                        if args.seed_source == "other":
                            seed_b = torch.load(
                                args.data / others[row["id"]]["b_clip"],
                                weights_only=True,
                            )["codes"]
                        if guidance == 1.0:
                            wave = speak(
                                model,
                                codec,
                                silence,
                                a,
                                seed_b,
                                seed=seed,
                                frames=args.frames,
                                decoding=decoding,
                                gated=gated,
                                rng=generator,
                                include_seed=True,
                            )
                        else:
                            wave = speak_cfg(
                                model,
                                codec,
                                silence,
                                a,
                                seed_b,
                                seed=seed,
                                frames=args.frames,
                                decoding=decoding,
                                gated=gated,
                                rng=generator,
                                guidance=guidance,
                                include_seed=True,
                            )
                    waves.append(wave)
                texts = asr(waves)
                examples = []
                for row, text, wave in zip(chosen, texts, waves):
                    answers = by_group[row["topic"]]
                    expected_answer = concepts[row["concept"]]["answer"]
                    expected = answer_key(expected_answer)
                    seed_answer = (
                        others[row["id"]]["answer"]
                        if args.seed_source == "other"
                        else expected_answer
                    )
                    seed_key = answer_key(seed_answer)
                    heard_answer = best_answer(text, answers)
                    heard_first = best_answer(" ".join(text.split()[:6]), answers)
                    quiet = float(np.sqrt(np.mean(wave[-24_000:] ** 2))) < QUIET_RMS
                    examples.append(
                        {
                            "question": row["question"],
                            "expected": expected_answer,
                            "heard": text,
                            "ok": heard_answer == expected,
                            "ok_first": heard_first == expected,
                            "stopped": quiet,
                            "seed_answer": seed_answer,
                            "ok_seed_answer": heard_answer == seed_key,
                            "ok_first_seed_answer": heard_first == seed_key,
                        }
                    )
                summary = _summary(
                    variant,
                    args.seed_source,
                    seed,
                    guidance,
                    examples,
                    by_group,
                    pairs,
                )
                _print_summary(summary)
                results.append(summary)
                if args.wavs:
                    import soundfile

                    folder = (
                        args.wavs / f"{variant}-s{seed}-{args.seed_source}-g{guidance}"
                    )
                    folder.mkdir(parents=True, exist_ok=True)
                    for row, wave in list(zip(chosen, waves))[:12]:
                        soundfile.write(
                            folder
                            / f"{row['topic']}-{row['word'].replace(' ', '_')}.wav",
                            wave,
                            24_000,
                        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
