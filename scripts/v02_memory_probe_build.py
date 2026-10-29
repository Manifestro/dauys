"""Build semantic-memory probes that interleave filler between cue and question."""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from v02_stateful_build import _real, assemble_chapter

PAUSE = 6


def _clip(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(root / name, map_location="cpu", weights_only=True)
    return data["codes"].long(), data["activity"].long()


def load_filler(root: Path, silence: torch.Tensor) -> dict:
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: str(row["id"]))
    phrases = []
    for row in rows:
        cached = torch.load(
            root / str(row["tokens_path"]), map_location="cpu", weights_only=True
        )
        codes, activity = _real(cached, str(row.get("source", "1a_on_B")))
        phrases.append({"codes": codes, "activity": activity})
    if not phrases:
        raise ValueError("The filler split is empty")
    return assemble_chapter(phrases, silence)


def _question_voices(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (str(row["topic"]), int(row["frame"]), str(row["concept"]))
        grouped[key].setdefault(str(row["speaker"]), row)
    return grouped


def make_items(rows: list[dict], *, limit: int, seed: int) -> list[dict]:
    grouped = _question_voices(rows)
    by_frame: dict[tuple[str, int], list[str]] = defaultdict(list)
    for topic, frame, concept in grouped:
        by_frame[(topic, frame)].append(concept)
    candidates = []
    for (topic, frame), concepts in sorted(by_frame.items()):
        for x, y in itertools.combinations(sorted(concepts), 2):
            voices_x = set(grouped[(topic, frame, x)])
            voices_y = set(grouped[(topic, frame, y)])
            common = sorted(voices_x & voices_y)
            if len(voices_x) < 2 or not common:
                continue
            candidates.append(
                {
                    "topic": topic,
                    "frame": frame,
                    "x": x,
                    "y": y,
                    "seed_voices": common,
                    "probe_voices": sorted(voices_x),
                }
            )
    rng = random_source(seed)
    rng.shuffle(candidates)
    items = []
    for candidate in candidates[:limit]:
        vp = rng.choice(candidate["seed_voices"])
        vq = rng.choice([voice for voice in candidate["probe_voices"] if voice != vp])
        items.append({**candidate, "seed_voice": vp, "probe_voice": vq})
    return items


def random_source(seed: int):
    import random

    return random.Random(seed)


def build_probe(
    *,
    item: str,
    gap: int,
    kind: str,
    seed_codes: torch.Tensor,
    seed_activity: torch.Tensor,
    filler: dict,
    probe_codes: torch.Tensor,
    probe_activity: torch.Tensor,
    silence: torch.Tensor,
    chunk: int,
) -> dict:
    if filler["user"].shape[-1] != gap:
        raise ValueError("Filler length must equal the requested gap")
    seed_frames = seed_codes.shape[-1]
    probe_frames = probe_codes.shape[-1]
    used = seed_frames + PAUSE + gap + PAUSE + probe_frames
    frames = ((used + chunk - 1) // chunk) * chunk
    user = silence[:, None].expand(8, frames).clone()
    target = user.clone()
    user_activity = torch.zeros(frames, dtype=torch.long)
    target_activity = torch.zeros(frames, dtype=torch.long)
    score = torch.zeros(frames, dtype=torch.bool)
    user[:, :seed_frames] = seed_codes
    user_activity[:seed_frames] = seed_activity
    start = seed_frames + PAUSE
    user[:, start : start + gap] = filler["user"]
    target[:, start : start + gap] = filler["target"]
    user_activity[start : start + gap] = filler["user_activity"]
    target_activity[start : start + gap] = filler["activity"]
    start += gap + PAUSE
    target[:, start : start + probe_frames] = probe_codes
    target_activity[start : start + probe_frames] = probe_activity
    score[start : start + probe_frames] = probe_activity.bool()
    return {
        "item": item,
        "gap": gap,
        "kind": kind,
        "user": user,
        "target": target,
        "user_activity": user_activity,
        "activity": target_activity,
        "score": score,
    }


def _pad_seed(
    clip: tuple[torch.Tensor, torch.Tensor],
    silence: torch.Tensor,
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes, activity = clip
    if codes.shape[-1] == frames:
        return clip
    padded = silence[:, None].expand(8, frames).clone()
    padded[:, : codes.shape[-1]] = codes
    padded_activity = torch.zeros(frames, dtype=torch.long)
    padded_activity[: activity.shape[-1]] = activity
    return padded, padded_activity


def build_probes(
    child: Path,
    filler_root: Path,
    silence: torch.Tensor,
    *,
    gaps: list[int],
    limit: int,
    chunk: int,
    seed: int,
) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (child / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    grouped = _question_voices(rows)
    items = make_items(rows, limit=limit, seed=seed)
    filler = load_filler(filler_root, silence)
    if max(gaps, default=0) > filler["user"].shape[-1]:
        raise ValueError("The filler stream is shorter than the largest gap")
    rng = random_source(seed + 1)
    streams = []
    for item in items:
        key_x = (item["topic"], item["frame"], item["x"])
        key_y = (item["topic"], item["frame"], item["y"])
        row_x_seed = grouped[key_x][item["seed_voice"]]
        row_y_seed = grouped[key_y][item["seed_voice"]]
        row_x_probe = grouped[key_x][item["probe_voice"]]
        seed_x = _clip(child, str(row_x_seed["a_clip"]))
        seed_y = _clip(child, str(row_y_seed["a_clip"]))
        seed_frames = max(seed_x[0].shape[-1], seed_y[0].shape[-1])
        seed_x = _pad_seed(seed_x, silence, seed_frames)
        seed_y = _pad_seed(seed_y, silence, seed_frames)
        probe = _clip(child, str(row_x_probe["a_clip"]))
        item_id = f"{item['topic']}-f{item['frame']:02d}-{item['x']}-vs-{item['y']}"
        for gap in gaps:
            offset = (
                0 if gap == 0 else rng.randrange(filler["user"].shape[-1] - gap + 1)
            )
            piece = {
                "user": filler["user"][:, offset : offset + gap],
                "target": filler["target"][:, offset : offset + gap],
                "user_activity": filler["user_activity"][offset : offset + gap],
                "activity": filler["activity"][offset : offset + gap],
            }
            for kind, seed_clip in (("same", seed_x), ("control", seed_y)):
                stream = build_probe(
                    item=item_id,
                    gap=gap,
                    kind=kind,
                    seed_codes=seed_clip[0],
                    seed_activity=seed_clip[1],
                    filler=piece,
                    probe_codes=probe[0],
                    probe_activity=probe[1],
                    silence=silence,
                    chunk=chunk,
                )
                stream.update(
                    {
                        "topic": item["topic"],
                        "frame": item["frame"],
                        "concept_x": item["x"],
                        "concept_y": item["y"],
                        "seed_voice": item["seed_voice"],
                        "probe_voice": item["probe_voice"],
                    }
                )
                streams.append(stream)
    return streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--filler", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--gaps", type=int, nargs="+", default=[0, 256, 768, 1536])
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.items < 1 or args.chunk < 1 or any(gap < 0 for gap in args.gaps):
        raise ValueError("Items, chunk and gaps must be positive")
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    if silence.shape != (8,):
        raise ValueError("Silence codes must be eight Mimi indices")
    streams = build_probes(
        args.child,
        args.filler,
        silence,
        gaps=args.gaps,
        limit=args.items,
        chunk=args.chunk,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(streams, args.output)
    print(json.dumps({"streams": len(streams), "items": args.items, "gaps": args.gaps}))


if __name__ == "__main__":
    main()
