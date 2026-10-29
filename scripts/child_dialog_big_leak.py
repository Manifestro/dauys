"""Check a big child-dialogue train set for held-out text, voice or frame leaks."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from child_dialog_synth import REPLY_VOICE, TEST_VOICES


def _voice_spec(row: dict) -> str:
    return str(row.get("voice") or row.get("speaker") or "")


def _spec_voices(spec: str) -> list[str]:
    if spec.startswith("blend:"):
        parts = spec.split(":")
        return parts[1:3] if len(parts) == 4 else [spec]
    return [spec] if spec else []


def _frame_groups(rows: list[dict], split: str | None = None) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {}
    for row in rows:
        if split is not None and row.get("split", split) != split:
            continue
        if "frame" not in row:
            continue
        groups.setdefault(str(row.get("group", "")), set()).add(int(row["frame"]))
    return groups


def check_leaks(
    train_rows: list[dict],
    test_rows: list[dict],
    piper_rows: list[dict] | None = None,
) -> dict:
    """Compare train rows with held-out questions, voices, frames and Piper."""
    train_questions = {str(row["question"]) for row in train_rows}
    test_questions = {str(row["question"]) for row in test_rows}
    question_overlap = sorted(train_questions & test_questions)
    test_voice_specs = set()
    af_heart_specs = set()
    piper_voices = {str(row["voice"]) for row in (piper_rows or []) if row.get("voice")}
    piper_voice_specs = set()
    for row in train_rows:
        spec = _voice_spec(row)
        voices = set(_spec_voices(spec))
        if voices & set(TEST_VOICES):
            test_voice_specs.add(spec)
        if REPLY_VOICE in voices:
            af_heart_specs.add(spec)
        if voices & piper_voices:
            piper_voice_specs.add(spec)
    test_frames = _frame_groups(test_rows, "test")
    train_frames = _frame_groups(train_rows)
    frame_overlap = {
        group: sorted(frames & train_frames.get(group, set()))
        for group, frames in test_frames.items()
        if frames & train_frames.get(group, set())
    }
    ok = not (
        question_overlap
        or test_voice_specs
        or af_heart_specs
        or piper_voice_specs
        or frame_overlap
    )
    return {
        "train_rows": len(train_rows),
        "train_questions": len(train_questions),
        "question_overlap": question_overlap,
        "test_voice_specs": sorted(test_voice_specs),
        "af_heart_specs": sorted(af_heart_specs),
        "piper_voice_specs": sorted(piper_voice_specs),
        "frame_overlap": frame_overlap,
        "ok": ok,
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--piper-plan", type=Path)
    args = parser.parse_args()
    train_rows = [row for path in args.train_manifests for row in _rows(path)]
    test_rows = _rows(args.test_manifest)
    piper_rows = [] if args.piper_plan is None else _rows(args.piper_plan)
    result = check_leaks(train_rows, test_rows, piper_rows)
    print(json.dumps(result, sort_keys=True), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
