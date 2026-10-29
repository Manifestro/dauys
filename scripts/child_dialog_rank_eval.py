"""N-way answer choice without collapsing the prior.

For every family group the candidates are all distinct reply clips in the
encoded manifest, not only the correct concept's own replies. The model scores
every candidate reply B after the question A with teacher-forced NLL, and the
cheapest candidate's answer is the prediction.

A prior control repeats the same ranking with A replaced by silence of the same
length. If the model ignores the question, prior top1 should stay at or below
chance and prior agreement should stay high.

Reported: top1, top3 over distinct answers (the correct answer's 0-based rank
is below 3), chance, the mean rank of the correct answer, prior top1, prior
agreement, a by-family split, and a per-question ``items`` list. A
``--per-group`` value of zero uses every question in each group.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.config import ModelConfig
from dauys.model import DauysHybrid

sys.path.insert(0, str(Path(__file__).parent))
from child_dialog_speak_eval import answer_key
from v02_word_meaning_eval import reply_nll


def select_questions(rows: list[dict], per_group: int) -> list[dict]:
    """Shuffle with seed 0 and take up to ``per_group`` rows per topic."""
    if per_group < 0:
        raise ValueError("The per-group count cannot be negative")
    rng = random.Random(0)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if per_group == 0:
        return shuffled
    chosen, count = [], {}
    for row in shuffled:
        topic = row["topic"]
        if count.get(topic, 0) < per_group:
            count[topic] = count.get(topic, 0) + 1
            chosen.append(row)
    return chosen


def collect_candidates(
    rows: list[dict], concepts: dict[str, dict]
) -> dict[str, list[tuple[str, str]]]:
    """Group -> sorted (b_clip, answer key) pairs, one per distinct reply clip."""
    candidates: dict[str, dict[str, str]] = {}
    for row in rows:
        answer = answer_key(concepts[row["concept"]]["answer"])
        group = candidates.setdefault(row["topic"], {})
        previous = group.setdefault(row["b_clip"], answer)
        if previous != answer:
            raise ValueError(f"{row['b_clip']} has answers {previous!r} and {answer!r}")
    return {
        group: sorted(
            ((clip, answer) for clip, answer in replies.items()),
            key=lambda item: item[0],
        )
        for group, replies in candidates.items()
    }


def answer_ranking(nll: list[float], answers: list[str]) -> list[str]:
    """Distinct answers ordered by their cheapest reply's NLL."""
    best: dict[str, float] = {}
    for value, answer in zip(nll, answers, strict=True):
        best[answer] = min(best.get(answer, value), value)
    return sorted(best, key=lambda answer: (best[answer], answer))


def score_matrix(
    nll: list[list[float]],
    answers: list[list[str]],
    expected: list[str],
    families: list[str],
    prior_nll: list[list[float]] | None = None,
    ids: list[str] | None = None,
    questions: list[str] | None = None,
    concepts: list[str] | None = None,
) -> dict:
    """Aggregate ranking metrics from per-question candidate NLL rows."""
    if not (len(nll) == len(answers) == len(expected) == len(families)):
        raise ValueError("NLL rows, answers, expected and families must align")
    if prior_nll is not None and len(prior_nll) != len(nll):
        raise ValueError("Prior NLL rows must align with NLL rows")
    for name, values in (
        ("ids", ids),
        ("questions", questions),
        ("concepts", concepts),
    ):
        if values is not None and len(values) != len(nll):
            raise ValueError(f"{name} must align with NLL rows")
    records = []
    for index, values in enumerate(nll):
        if len(values) != len(answers[index]):
            raise ValueError("Each NLL row must have one value per candidate")
        ranking = answer_ranking(values, answers[index])
        correct = expected[index]
        rank = ranking.index(correct) if correct in ranking else len(ranking)
        best = ranking[0] if ranking else None
        prior_best = None
        prior_top1 = None
        prior_agree = None
        if prior_nll is not None:
            prior_values = prior_nll[index]
            if len(prior_values) != len(answers[index]):
                raise ValueError("Each prior NLL row must match its candidates")
            prior_ranking = answer_ranking(prior_values, answers[index])
            prior_best = prior_ranking[0] if prior_ranking else None
            prior_top1 = prior_best == correct
            prior_agree = prior_best == best
        records.append(
            {
                "id": ids[index] if ids is not None else None,
                "question": questions[index] if questions is not None else None,
                "concept": concepts[index] if concepts is not None else None,
                "family": families[index],
                "expected": correct,
                "predicted": best,
                "top1": best == correct,
                "prior_predicted": prior_best,
                "top3": correct in ranking and rank < 3,
                "chance": 1 / len(set(answers[index])) if answers[index] else 0.0,
                "rank": rank,
                "top1_prior": prior_top1,
                "prior_agree": prior_agree,
            }
        )
    summary = _aggregate(records)
    summary["by_family"] = {
        family: _aggregate([record for record in records if record["family"] == family])
        for family in sorted({record["family"] for record in records})
    }
    summary["items"] = [
        {
            key: record[key]
            for key in (
                "id",
                "question",
                "concept",
                "family",
                "expected",
                "predicted",
                "top1",
                "prior_predicted",
            )
        }
        for record in records
    ]
    return summary


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate(records: list[dict]) -> dict:
    prior = [record["top1_prior"] for record in records]
    agree = [record["prior_agree"] for record in records]
    return {
        "items": len(records),
        "top1": round(_mean([record["top1"] for record in records]), 3),
        "top3": round(_mean([record["top3"] for record in records]), 3),
        "chance": round(_mean([record["chance"] for record in records]), 3),
        "mean_rank": round(_mean([record["rank"] for record in records]), 3),
        "top1_prior": (
            round(_mean([value for value in prior if value is not None]), 3)
            if any(value is not None for value in prior)
            else None
        ),
        "prior_agree": (
            round(_mean([value for value in agree if value is not None]), 3)
            if any(value is not None for value in agree)
            else None
        ),
    }


def _clip(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(root / name, weights_only=True)
    return data["codes"].long(), data["activity"].bool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="encoded split dir")
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--score", choices=("all", "cb0", "depth"), default="all")
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--onset-frames",
        type=int,
        default=0,
        help="Score only the first N speech frames of each reply",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    concepts = {
        c["id"]: c for c in map(json.loads, args.concepts.read_text().splitlines())
    }
    rows = [
        json.loads(line)
        for line in (args.data / "manifest.jsonl").read_text().splitlines()
    ]
    candidate_rows = (
        rows
        if args.candidates is None
        else [
            json.loads(line)
            for line in args.candidates.read_text().splitlines()
            if line.strip()
        ]
    )
    candidates = collect_candidates(candidate_rows, concepts)
    chosen = select_questions(rows, args.per_group)
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    payload = load_research_checkpoint(args.checkpoint)
    model = DauysHybrid(ModelConfig.from_yaml(args.config))
    model.load_state_dict(payload["model"])
    model = model.cuda().eval()
    clips: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def clip(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in clips:
            clips[name] = _clip(args.data, name)
        return clips[name]

    by_group: dict[str, list[dict]] = {}
    for row in chosen:
        by_group.setdefault(row["topic"], []).append(row)
    nll, prior_nll, answers, expected, families = [], [], [], [], []
    ids, question_texts, concept_ids = [], [], []
    for topic, questions in by_group.items():
        options = candidates[topic]
        replies = [clip(name) for name, _ in options]
        option_answers = [answer for _, answer in options]
        normal_pairs, silent_pairs = [], []
        for row in questions:
            a_codes, a_active = clip(row["a_clip"])
            silent = (
                silence[:, None].expand(8, a_codes.shape[-1]).clone(),
                torch.zeros(a_codes.shape[-1], dtype=torch.bool),
            )
            for reply in replies:
                normal_pairs.append(((a_codes, a_active), reply))
                silent_pairs.append((silent, reply))
        normal = reply_nll(
            model,
            silence,
            normal_pairs,
            batch=args.batch,
            onset=args.onset_frames,
            parts=args.score,
        ).reshape(len(questions), len(replies))
        prior = reply_nll(
            model,
            silence,
            silent_pairs,
            batch=args.batch,
            onset=args.onset_frames,
            parts=args.score,
        ).reshape(len(questions), len(replies))
        nll.extend(normal.tolist())
        prior_nll.extend(prior.tolist())
        answers.extend([option_answers] * len(questions))
        expected.extend(
            answer_key(concepts[row["concept"]]["answer"]) for row in questions
        )
        families.extend(
            row.get("family") or concepts[row["concept"]].get("family") or row["topic"]
            for row in questions
        )
        ids.extend(row["id"] for row in questions)
        question_texts.extend(row["question"] for row in questions)
        concept_ids.extend(row["concept"] for row in questions)
    summary = score_matrix(
        nll,
        answers,
        expected,
        families,
        prior_nll,
        ids,
        question_texts,
        concept_ids,
    )
    summary["score"] = args.score
    print(
        json.dumps({k: v for k, v in summary.items() if k != "by_family"}),
        flush=True,
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
