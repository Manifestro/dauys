"""Cut nested child-dialogue training manifests by cumulative audio hours."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

FRAME_RATE = 12.5


def base_speaker(spec: str) -> str:
    """First voice of a blend spec, or the plain voice name."""
    if spec.startswith("blend:"):
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid blend voice spec: {spec}")
        return parts[1]
    return spec


def apply_speaker_key(rows: list[dict], key: str) -> list[dict]:
    """Choose the trainer's speaker grouping while preserving the voice spec."""
    if key not in ("spec", "base"):
        raise ValueError("Speaker key must be spec or base")
    for row in rows:
        spec = str(row.get("voice") or row.get("speaker") or "")
        if not spec:
            continue
        row["voice"] = spec
        row["speaker"] = base_speaker(spec) if key == "base" else spec
    return rows


def _keep_score(identity: str) -> float:
    """A stable number in [0, 1) per row, so balancing is reproducible."""
    return int(hashlib.sha1(identity.encode()).hexdigest()[:12], 16) / 16**12


def balance_answers(
    rows: list[dict], answers: dict[tuple[str, str], str], families: list[str]
) -> list[dict]:
    """Thin each family so every answer is as common as its rarest answer.

    Without this the v2 add family is mostly sums 7-10, and models learn to
    answer "ten" whatever they hear. Rows are dropped by a stable hash of their
    ID, so every cut keeps the same rows.
    """
    family_set = set(families)
    if not family_set:
        return rows

    def answer(row: dict) -> str:
        key = (str(row["family"]), str(row["word"]))
        if key not in answers:
            raise ValueError(f"No answer for {key}")
        return answers[key]

    counts: dict[str, Counter] = {family: Counter() for family in family_set}
    for row in rows:
        if row.get("family") in family_set:
            counts[row["family"]][answer(row)] += 1
    kept = []
    for row in rows:
        family = row.get("family")
        if family in family_set:
            share = min(counts[family].values()) / counts[family][answer(row)]
            if _keep_score(str(row["id"])) >= share:
                continue
        kept.append(row)
    return kept


def load_answers(path: Path) -> dict[tuple[str, str], str]:
    """Map (family, key) to the spoken answer from a concepts JSONL file."""
    answers = {}
    for line in path.read_text().splitlines():
        if line.strip():
            concept = json.loads(line)
            answers[(concept["family"], concept["key"])] = concept["answer"]
    return answers


def select_rows(rows: list[dict], hours: float) -> list[dict]:
    """Take rows in rank order until their audio reaches ``hours``."""
    if hours < 0:
        raise ValueError("The hour target cannot be negative")
    if hours == 0:
        return []
    target = hours * FRAME_RATE * 3600
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["rank"]),
            str(row.get("id", "")),
            str(row.get("_source", "")),
        ),
    )
    selected = []
    frames = 0
    for row in ordered:
        selected.append(dict(row))
        frames += int(row["frames"])
        if frames >= target:
            break
    return selected


def select_cuts(
    manifests: list[tuple[Path, list[dict]]],
    hours: float,
    root: Path | None,
    *,
    speaker_key: str = "base",
    answers: dict[tuple[str, str], str] | None = None,
    balance_families: list[str] = (),
) -> list[dict]:
    """Select a cut and rewrite tokens paths next to their source manifests."""
    rows = []
    for manifest, items in manifests:
        for item in items:
            row = dict(item)
            row["_source"] = str(manifest.parent)
            row["_tokens"] = str((manifest.parent / str(item["tokens_path"])).resolve())
            rows.append(row)
    if balance_families:
        rows = balance_answers(rows, answers or {}, list(balance_families))
    selected = select_rows(rows, hours)
    root_path = None if root is None else Path(root).resolve()
    for row in selected:
        row.pop("_source")
        tokens = row.pop("_tokens")
        row["tokens_path"] = (
            tokens if root_path is None else os.path.relpath(tokens, root_path)
        )
    return apply_speaker_key(selected, speaker_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--hours", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--speaker-key", choices=("spec", "base"), default="base")
    parser.add_argument(
        "--balance-families",
        nargs="*",
        default=["add"],
        help="Families thinned so every answer is equally common (none: pass no names)",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path(__file__).with_name("child_dialog_concepts_v2.jsonl"),
    )
    args = parser.parse_args()
    answers = load_answers(args.concepts) if args.balance_families else None
    manifests = []
    for path in args.manifests:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        manifests.append((path, rows))
    args.output.mkdir(parents=True, exist_ok=True)
    for hours in args.hours:
        rows = select_cuts(
            manifests,
            hours,
            args.root,
            speaker_key=args.speaker_key,
            answers=answers,
            balance_families=args.balance_families,
        )
        path = args.output / f"train-{hours:g}h.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(
            json.dumps(
                {
                    "hours": hours,
                    "rows": len(rows),
                    "frames": sum(int(row["frames"]) for row in rows),
                    "manifest": str(path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
