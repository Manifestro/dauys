"""Align ``hohere label`` output with the listening bench cache for V10 step 1.

    hohere label --in-dir SLM21 --out-dir LABELS
    python scripts/v10_step1_labels.py --test swuggy --stimuli SLM21 \
        --labels LABELS --cache CACHE --packed PACKED

Writes ``CACHE/<test>_m_B.pt`` and ``CACHE/<test>_m_B-sil.pt``: one ``good`` and
``bad`` monologue per cached pair, one token per cached CB0 frame. CB0 stays the
bench's; only m comes from hohere. The bench encodes whole files, so an event
moves by the ``start_s`` that ``label`` trimmed. The pairing copies the bench's
(``bench/duplexbench/data.py``), since the cache keeps no paths, and every pair is
checked against the cache's ``id``.

With ``--label-cache DIR`` it also writes the same pairs on the CB0 that ``label``
encoded through torchaudio, as in training, with trimmed edges. Score that directory
as a secondary line beside the primary one: the difference shows what the bench's
resampler costs in points. It never replaces the primary score.
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from dauys.v10_step1_data import stimulus_m

VARIANTS = ("B", "B-sil")


def slm21_pairs(root: Path, task: str) -> list[dict]:
    split = root / task / "dev"
    groups: dict[tuple[str, ...], dict] = defaultdict(dict)
    with (split / "gold.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["id"], row["voice"], row.get("subtype") or "")
            side = "good" if row["correct"] == "1" else "bad"
            groups[key][side] = split / f"{row['filename']}.wav"
    return [
        {"id": "-".join(part for part in key if part), **value}
        for key, value in sorted(groups.items())
        if "good" in value and "bad" in value
    ]


def storycloze_pairs(root: Path) -> list[dict]:
    sides: dict[str, dict] = defaultdict(dict)
    for path in root.rglob("*.wav"):
        if path.name.startswith("."):
            continue
        _, story, ending = path.stem.split("_", 2)
        if ending in ("correct", "incorrect"):
            sides[story]["good" if ending == "correct" else "bad"] = path
    return [
        {"id": story, **pair}
        for story, pair in sorted(sides.items())
        if "good" in pair and "bad" in pair
    ]


def stimulus_pairs(test: str, root: Path) -> list[dict]:
    if test == "swuggy":
        return slm21_pairs(root, "lexical")
    if test == "sblimp":
        return slm21_pairs(root, "syntactic")
    return storycloze_pairs(root)


def _manifest(labels: Path) -> dict[str, dict]:
    lines: dict[str, dict] = {}
    with (labels / "manifest.jsonl").open(encoding="utf-8") as stream:
        for text in stream:
            line = json.loads(text)
            if line["source"] in lines:
                raise ValueError(f"A stimulus cut into pieces: {line['source']}")
            lines[line["source"]] = line
    return lines


def align_test(
    test: str,
    stimuli: Path,
    labels: Path,
    cache: Path,
    packed: Path,
    label_cache: Path | None = None,
) -> dict:
    """Write both sidecars of one test; returns counts for the run's report."""
    from hohere.records import read_record

    meta = json.loads((packed / "meta.json").read_text(encoding="utf-8"))
    units = int(meta["units"])
    version_key = f"units_k{units}"
    cached = torch.load(cache / f"{test}.pt", map_location="cpu", weights_only=True)
    pairs = stimulus_pairs(test, stimuli)
    if [pair["id"] for pair in pairs] != [entry["id"] for entry in cached]:
        raise ValueError(f"The stimuli and the cache of {test} hold different pairs")
    lines = _manifest(labels)
    sidecars: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    relabelled: list[dict] = []
    without_units = 0
    for pair, entry in zip(pairs, cached):
        own = {"id": entry["id"], "kind": entry.get("kind")}
        row: dict[str, dict[str, torch.Tensor]] = {v: {} for v in VARIANTS}
        for side in ("good", "bad"):
            source = pair[side].relative_to(stimuli).with_suffix("").as_posix()
            line = lines[source]
            if line[version_key] != meta["centroids_sha256"]:
                raise ValueError(f"Other {version_key} centroids in {source}")
            without_units += int(line["n_units"]) == 0
            record = read_record((labels / f"{line['id']}.npz").read_bytes())
            own[side] = torch.from_numpy(record["codes"][:8].astype("int64"))
            frames = int(entry[side].shape[-1])
            for variant in VARIANTS:
                row[variant][side] = stimulus_m(
                    record, float(line["start_s"]), variant, units, frames
                )
        for variant in VARIANTS:
            sidecars[variant].append(row[variant])
        relabelled.append(own)
    for variant, payload in sidecars.items():
        torch.save(payload, cache / f"{test}_m_{variant}.pt")
    if label_cache is not None:
        label_cache.mkdir(parents=True, exist_ok=True)
        torch.save(relabelled, label_cache / f"{test}.pt")
        shutil.copyfile(cache / "silence.pt", label_cache / "silence.pt")
    return {"test": test, "pairs": len(pairs), "stimuli_without_units": without_units}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test", choices=("swuggy", "sblimp", "tstorycloze"), required=True
    )
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--label-cache", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            align_test(
                args.test,
                args.stimuli,
                args.labels,
                args.cache,
                args.packed,
                args.label_cache,
            )
        )
    )


if __name__ == "__main__":
    main()
