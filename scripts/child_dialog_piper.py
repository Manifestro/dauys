"""Re-voice child-dialogue questions with Piper, without importing dauys."""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import v02_word_meaning_piper as reference

VARIANTS = ("clean", "noise", "phone")


def pick_rows(rows: list[dict]) -> list[dict]:
    """First row by id for each distinct question text."""
    picked = {}
    for row in sorted(rows, key=lambda row: str(row["id"])):
        picked.setdefault(str(row["question"]), row)
    return list(picked.values())


def select_rows(rows: list[dict], per_group: int) -> list[dict]:
    """Limit picked rows per topic, then sort the result by id."""
    if per_group < 0:
        raise ValueError("The per-group count cannot be negative")
    rng = random.Random(0)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if per_group == 0:
        chosen = shuffled
    else:
        chosen, count = [], {}
        for row in shuffled:
            topic = row["topic"]
            if count.get(topic, 0) < per_group:
                count[topic] = count.get(topic, 0) + 1
                chosen.append(row)
    return sorted(chosen, key=lambda row: str(row["id"]))


def synthesize(
    text: str,
    model: Path,
    names: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "raw.wav"
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(model), "-f", str(raw)],
            input=text.encode(),
            check=True,
            capture_output=True,
        )
        audio, rate = soundfile.read(raw, dtype="float32", always_2d=True)
    audio = resample_poly(audio.mean(1), 24_000, rate).astype(np.float32)
    versions = reference.variants(audio, rng)
    result = {}
    for name in names:
        version = versions[name]
        peak = max(1.0, float(np.abs(version).max()))
        result[name] = (version / peak).astype(np.float32)
    return result


def run_job(
    output: Path,
    voice: str,
    model: Path,
    row: dict,
    names: list[str],
    job_index: int,
) -> list[dict]:
    rng = np.random.default_rng(job_index)
    versions = synthesize(str(row["question"]), model, names, rng)
    entries = []
    for variant in names:
        relative = Path(variant) / voice / f"{row['id']}.wav"
        target = output / "wavs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(target, versions[variant], 24_000)
        entries.append(
            {
                "row": row["id"],
                "voice": voice,
                "variant": variant,
                "wav": str(relative),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["clean", "phone"]
    )
    parser.add_argument("--voices", type=int, default=len(reference.VOICES))
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.voices <= len(reference.VOICES):
        raise ValueError("Voice count must lie inside the Piper voice list")
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    picked = select_rows(pick_rows(rows), args.per_group)
    voices = [Path(path).name for path in reference.VOICES[: args.voices]]
    models = []
    for path in reference.VOICES[: args.voices]:
        model = hf_hub_download("rhasspy/piper-voices", f"{path}.onnx")
        hf_hub_download("rhasspy/piper-voices", f"{path}.onnx.json")
        models.append(Path(model))
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, voice in zip(models, voices):
        for row in picked:
            jobs.append((voice, model, row, len(jobs)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_job,
                args.output,
                voice,
                model,
                row,
                args.variants,
                job_index,
            )
            for voice, model, row, job_index in jobs
        ]
        results = [future.result() for future in futures]
    plan = [entry for entries in results for entry in entries]
    voice_order = {voice: index for index, voice in enumerate(voices)}
    plan.sort(
        key=lambda item: (
            voice_order[item["voice"]],
            item["row"],
            item["variant"],
        )
    )
    (args.output / "plan.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in plan)
    )
    print(json.dumps({"rows": len(picked), "items": len(plan)}))


if __name__ == "__main__":
    main()
