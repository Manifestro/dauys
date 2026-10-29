"""Re-encode Piper or phone questions into child-dialogue evaluation splits."""

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from dauys.codec import load_mimi

sys.path.insert(0, str(Path(__file__).parent))
from child_dialog_encode import encode_all


def trim_silence(
    codes: torch.Tensor, activity: torch.Tensor, margin: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim leading/trailing silence, keeping a margin on each side."""
    if codes.ndim != 2 or codes.shape[0] != 8:
        raise ValueError("Codes must have shape [8, frames]")
    if activity.shape != (codes.shape[-1],):
        raise ValueError("Activity must match the code frames")
    active = torch.nonzero(activity.bool()).flatten()
    if not len(active):
        return codes[:, :0], activity[:0]
    start = max(0, int(active[0]) - margin)
    end = min(codes.shape[-1], int(active[-1]) + 1 + margin)
    return codes[:, start:end], activity[start:end]


def build_rows(
    original_rows: list[dict],
    plan: list[dict],
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    margin: int = 2,
) -> dict[str, list[tuple[dict, torch.Tensor, torch.Tensor]]]:
    """Build variant -> (new row, question codes, activity)."""
    by_id = {str(row["id"]): row for row in original_rows}
    variants: dict[str, list[tuple[dict, torch.Tensor, torch.Tensor]]] = defaultdict(
        list
    )
    for item in plan:
        identity = str(item["row"])
        voice = str(item["voice"])
        codes, active = encoded[str(item["wav"])]
        codes, active = trim_silence(codes, active, margin)
        row = dict(by_id[identity])
        row.update(
            {
                "id": f"{identity}-{voice}",
                "speaker": voice,
                "a_clip": f"clips/q-{identity}-{voice}.pt",
            }
        )
        variants[str(item["variant"])].append((row, codes, active))
    return dict(variants)


def _prepare_wavs(plan: list[dict], wavs: Path, root: Path) -> list[str]:
    names = []
    for item in plan:
        name = str(Path(item["wav"]))
        audio, rate = soundfile.read(wavs / name, dtype="float32", always_2d=True)
        mono = audio.mean(1)
        if rate != 24_000:
            mono = resample_poly(mono, 24_000, rate)
        audio = mono.astype(np.float32)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(target, audio, 24_000, subtype="FLOAT")
        names.append(name)
    return names


def _copy_candidate_clips(rows: list[dict], source: Path, target: Path) -> None:
    for row in rows:
        relative = Path(str(row["b_clip"]))
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.data / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    plan = [
        json.loads(line) for line in args.plan.read_text().splitlines() if line.strip()
    ]
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    with tempfile.TemporaryDirectory() as folder:
        prepared = Path(folder)
        names = _prepare_wavs(plan, args.wavs, prepared)
        encoded = encode_all(codec, prepared, names, args.batch)
    by_variant = build_rows(rows, plan, encoded)
    for variant, items in sorted(by_variant.items()):
        target = args.output / variant
        (target / "clips").mkdir(parents=True, exist_ok=True)
        _copy_candidate_clips(rows, args.data, target)
        manifest = []
        for row, codes, active in items:
            clip = target / str(row["a_clip"])
            clip.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "codes": codes.short(),
                    "activity": active.to(torch.uint8),
                },
                clip,
            )
            manifest.append(row)
        (target / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest)
        )
        (target / "candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    print(json.dumps({name: len(items) for name, items in by_variant.items()}))


if __name__ == "__main__":
    main()
