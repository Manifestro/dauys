"""Fast spoken-reply check for the child-dialogue set.

Takes ``--per-group`` held-out questions per family group from voices the model
never trained on. The model hears each question and speaks for ``--frames``
frames with the chosen decoding. Whisper transcribes the model channel for
scoring only.

- ``correct``: the transcript's content words match one of the concept's own
  reply wordings better than any wording of another concept in the same group.
- ``correct_first``: the same, using only the transcript's first six words.
- ``stopped``: the last second of the reply is silent.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_pretrain_eval import WhisperTranscriber

sys.path.insert(0, str(Path(__file__).parent))
from v02_word_meaning_speak_eval import VARIANTS, content, speak

QUIET_RMS = 0.01


def best_concept(transcript: str, wordings: dict[str, list[str]]) -> str | None:
    heard = content(transcript)
    scores = {c: max(len(heard & content(w)) for w in ws) for c, ws in wordings.items()}
    top = max(scores.values())
    winners = [c for c, s in scores.items() if s == top]
    return winners[0] if top > 0 and len(winners) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="encoded split dir")
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--variants", nargs="+", default=["greedy"])
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--wavs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    concepts = {
        c["id"]: c for c in map(json.loads, args.concepts.read_text().splitlines())
    }
    by_group: dict[str, dict[str, list[str]]] = {}
    for c in concepts.values():
        by_group.setdefault(c["group"], {})[c["id"]] = c["replies"]
    rows = [
        json.loads(line)
        for line in (args.data / "manifest.jsonl").read_text().splitlines()
    ]
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
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    asr = WhisperTranscriber(args.asr_model, "cuda")
    results = []
    for variant in args.variants:
        decoding, gated = VARIANTS[variant]
        generator = torch.Generator(device="cuda").manual_seed(0)
        waves = []
        for row in chosen:
            a = torch.load(args.data / row["a_clip"], weights_only=True)["codes"]
            b = torch.load(args.data / row["b_clip"], weights_only=True)["codes"]
            waves.append(
                speak(
                    model,
                    codec,
                    silence,
                    a,
                    b,
                    seed=0,
                    frames=args.frames,
                    decoding=decoding,
                    gated=gated,
                    rng=generator,
                )
            )
        texts = asr(waves)
        correct = first = stopped = 0
        examples = []
        for row, text, wave in zip(chosen, texts, waves):
            wordings = by_group[row["topic"]]
            ok = best_concept(text, wordings) == row["concept"]
            ok_first = (
                best_concept(" ".join(text.split()[:6]), wordings) == row["concept"]
            )
            quiet = float(np.sqrt(np.mean(wave[-24_000:] ** 2))) < QUIET_RMS
            correct, first, stopped = correct + ok, first + ok_first, stopped + quiet
            examples.append(
                {
                    "question": row["question"],
                    "expected": concepts[row["concept"]]["answer"],
                    "heard": text,
                    "ok": ok,
                    "ok_first": ok_first,
                    "stopped": quiet,
                }
            )
        n = len(chosen)
        summary = {
            "variant": variant,
            "items": n,
            "correct": round(correct / n, 3),
            "correct_first": round(first / n, 3),
            "stopped": round(stopped / n, 3),
            "chance": round(
                float(np.mean([1 / len(by_group[r["topic"]]) for r in chosen])), 3
            ),
            "examples": examples,
        }
        print(
            json.dumps({k: v for k, v in summary.items() if k != "examples"}),
            flush=True,
        )
        results.append(summary)
        if args.wavs:
            import soundfile

            folder = args.wavs / variant
            folder.mkdir(parents=True, exist_ok=True)
            for row, wave in list(zip(chosen, waves))[:12]:
                soundfile.write(
                    folder / f"{row['topic']}-{row['word'].replace(' ', '_')}.wav",
                    wave,
                    24_000,
                )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
