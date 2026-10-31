"""Packed CB0 plus monologue data for the V10 step 1 arms.

``pack_tier`` turns one ready tier of the hohere shards into flat files, once, so a
run maps them instead of opening one archive per recording. Chunks never cross a
recording and depend only on the record lengths and the seed, so every arm of a
seed sees the same frames in the same order and only its ``m`` file differs.

Each arm's ``m`` comes from ``hohere.units.build_m``, not from the B stream: the
frame of a SIL in B depends on the units around it, which must not reach B-sil.
"""

import json
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from .v10_step1 import TRAIN_ARMS, V10Step1Config

MIN_CHUNK_FRAMES = 64  # a shorter tail of a recording is dropped
SHARD_PREFIX = "librilight"
ARM_FILES = {"B-sil": "m-b-sil.u16", "B": "m-b.u16"}  # arm A is PAD, no file needed


def _units_key(units: int) -> str:
    if units not in (4096, 8192):
        raise ValueError(f"hohere labels only 4096 or 8192 units, not {units}")
    return f"units_k{units}"


def read_tier(root: str | Path, tier: str, units: int) -> tuple[dict, list[dict]]:
    """The ``READY`` report of a tier and its manifest lines in hohere's order.

    Each line also gets ``tar``, its shard's path relative to ``root``. Shards are
    not sorted by rank, and shard numbers repeat across ``part-*`` folders.
    """
    root = Path(root)
    key = _units_key(units)
    ready_path = root / SHARD_PREFIX / f"READY-{tier}.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"Tier {tier} is not ready: {ready_path}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not ready["ready"] or len(ready[key]) != 1:
        raise ValueError(f"Tier {tier} has no single set of {key} centroids")
    version = ready[key][0]
    rows = []
    for manifest in sorted((root / SHARD_PREFIX).rglob("shard-*.jsonl")):
        tar = manifest.with_suffix(".tar").relative_to(root).as_posix()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row["order_rank"]) > int(ready["last_order_rank"]):
                continue  # a later tier: the repo may hold more than this one
            if row[key] != version:
                raise ValueError(f"Other {key} centroids in {row['id']}")
            rows.append({**row, "tar": tar})
    identities = {row["id"] for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate step 1 IDs")
    if len(rows) != int(ready["chunks_found"]):
        raise ValueError(
            f"Tier {tier} lists {ready['chunks_found']} chunks, found {len(rows)}"
        )
    rows.sort(key=lambda row: (int(row["order_rank"]), row["id"]))
    return ready, rows


def fetch_tier(repo: str, tier: str, root: str | Path, *, units: int) -> None:
    """Download a tier's report and manifests, then only the shards it needs."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{SHARD_PREFIX}/READY-{tier}.json", f"{SHARD_PREFIX}/*.jsonl"],
    )
    _, rows = read_tier(root, tier, units)
    snapshot_download(
        repo,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=sorted({row["tar"] for row in rows}),
    )


def stimulus_m(
    record: dict, start_s: float, variant: str, units: int, frames: int
) -> torch.Tensor:
    """m of a stimulus on the frames of its whole file, as the bench cache holds it.

    ``hohere label`` trims the edges, so its 50 Hz events count from ``start_s``.
    """
    from hohere.units import build_m_with_stats

    offset = round(start_s * 50)
    shifted = {
        "unit_start": record["unit_start"] + offset,
        f"unit_k{units}": record[f"unit_k{units}"],
        "sil_start": record["sil_start"] + offset,
    }
    m, stats = build_m_with_stats(shifted, variant, units, frames=frames)
    if stats.dropped:
        raise ValueError(f"{stats.dropped} events fall outside {frames} frames")
    return torch.from_numpy(m.astype("int64"))


def _pack_tar(
    root: Path, output: Path, units: int, item: tuple[str, dict[str, tuple[int, int]]]
) -> tuple[int, int]:
    """Write one shard's records at their offsets; returns SIL counts of B-sil."""
    from hohere.records import read_record
    from hohere.units import build_m

    tar_name, wanted = item
    written = sil_total = sil_same = 0
    with (
        tarfile.open(root / tar_name, "r:") as tar,
        (output / "s.u16").open("r+b") as s_file,
        (output / ARM_FILES["B"]).open("r+b") as b_file,
        (output / ARM_FILES["B-sil"]).open("r+b") as sil_file,
    ):
        for member in tar:
            identity = member.name.removesuffix(".npz")
            if identity not in wanted:
                continue
            offset, frames = wanted[identity]
            record = read_record(tar.extractfile(member).read())
            s = record["codes"][0]
            if s.size != frames:
                raise ValueError(f"The manifest and the record differ: {identity}")
            m_b = build_m(record, "B", units)
            m_sil = build_m(record, "B-sil", units)
            for stream, values in ((s_file, s), (b_file, m_b), (sil_file, m_sil)):
                stream.seek(offset * 2)
                stream.write(values.astype("<u2").tobytes())
            sil_total += int((m_sil == units).sum())
            sil_same += int(((m_sil == units) & (m_b == units)).sum())
            written += 1
    if written != len(wanted):
        raise ValueError(f"{tar_name} holds {written} of {len(wanted)} records")
    return sil_total, sil_same


def pack_tier(
    root: str | Path,
    tier: str,
    output: str | Path,
    *,
    units: int,
    workers: int = 8,
) -> dict:
    """Write ``s.u16``, one ``m`` file per arm and ``meta.json`` from local shards.

    ``meta.json`` comes last, so an interrupted pack cannot be opened.
    """
    root, output = Path(root), Path(output)
    ready, rows = read_tier(root, tier, units)
    output.mkdir(parents=True, exist_ok=True)
    frames = [int(row["frames"]) for row in rows]
    for name in ("s.u16", *ARM_FILES.values()):
        with (output / name).open("wb") as stream:
            stream.truncate(sum(frames) * 2)
    shards: dict[str, dict[str, tuple[int, int]]] = {}
    offset = 0
    for row, length in zip(rows, frames):
        shards.setdefault(row["tar"], {})[row["id"]] = (offset, length)
        offset += length
    with ProcessPoolExecutor(workers) as pool:
        counts = list(pool.map(partial(_pack_tar, root, output, units), shards.items()))
    meta = {
        "units": units,
        "tier": tier,
        "centroids_sha256": ready[_units_key(units)][0],
        "last_order_rank": ready["last_order_rank"],
        "missing_files": ready["missing_files"],
        "sil_b_sil": sum(total for total, _ in counts),
        "sil_same_frame_in_b": sum(same for _, same in counts),
        "ids": [row["id"] for row in rows],
        "frames": frames,
    }
    (output / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return meta


@dataclass(frozen=True)
class PackedCorpus:
    s: torch.Tensor | None  # None for arm m-only, whose s is all BOS
    m: torch.Tensor | None  # None for arm A, whose m is all PAD
    frames: torch.Tensor

    @classmethod
    def open(cls, path: str | Path, config: V10Step1Config, arm: str) -> "PackedCorpus":
        if arm not in TRAIN_ARMS:
            raise ValueError(f"Unknown arm: {arm!r}")
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if int(meta["units"]) != config.units:
            raise ValueError("The packed corpus uses another unit vocabulary")
        frames = torch.tensor(meta["frames"], dtype=torch.long)
        size = int(frames.sum())
        m = (
            None
            if arm == "A"
            else torch.from_file(
                str(path / ARM_FILES["B" if arm == "m-only" else arm]),
                size=size,
                dtype=torch.uint16,
            )
        )
        s = (
            None
            if arm == "m-only"
            else torch.from_file(str(path / "s.u16"), size=size, dtype=torch.uint16)
        )
        return cls(
            s,
            m,
            frames,
        )


def _epoch_chunks(
    frames: torch.Tensor, chunk_frames: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """One pass in random order: starts and lengths in the flat files."""
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), frames.cumsum(0)[:-1]))
    shifts = (torch.rand(len(frames), generator=generator) * chunk_frames).long()
    shifts = torch.minimum(shifts, (frames - chunk_frames).clamp_min(0))
    remaining = frames - shifts
    counts = (remaining + chunk_frames - 1) // chunk_frames
    owners = torch.arange(len(frames)).repeat_interleave(counts)
    first = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
    within = torch.arange(len(owners)) - first[owners]
    starts = shifts[owners] + within * chunk_frames
    lengths = torch.minimum(
        frames[owners] - starts, torch.full_like(starts, chunk_frames)
    )
    keep = lengths >= min(MIN_CHUNK_FRAMES, chunk_frames)
    order = torch.randperm(int(keep.sum()), generator=generator)
    return (offsets[owners] + starts)[keep][order], lengths[keep][order]


def _chunks_per_epoch(frames: torch.Tensor, chunk_frames: int) -> int:
    return max(1, int(frames.sum()) // chunk_frames)


def steps_for_epochs(
    frames: torch.Tensor, *, chunk_frames: int, batch_size: int, epochs: int
) -> int:
    """Steps that pass over the corpus the requested number of times."""
    if chunk_frames < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("Epoch sizing needs positive sizes")
    return max(1, _chunks_per_epoch(frames, chunk_frames) // batch_size) * epochs


def chunk_plan(
    frames: torch.Tensor,
    *,
    chunk_frames: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[steps, batch]`` starts and lengths: record lengths and the seed only."""
    if not len(frames) or chunk_frames < 1 or steps < 1 or batch_size < 1:
        raise ValueError("A chunk plan needs records and positive sizes")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + 7)
    needed = steps * batch_size
    starts: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    planned = 0
    while planned < needed:
        epoch_starts, epoch_lengths = _epoch_chunks(frames, chunk_frames, generator)
        if not len(epoch_starts):
            raise ValueError("No recording is long enough for one chunk")
        starts.append(epoch_starts)
        lengths.append(epoch_lengths)
        planned += len(epoch_starts)
    return (
        torch.cat(starts)[:needed].view(steps, batch_size),
        torch.cat(lengths)[:needed].view(steps, batch_size),
    )


class V10BatchSource(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """One batch per step, with the same chunks for all arms of a seed."""

    def __init__(
        self,
        corpus: PackedCorpus,
        config: V10Step1Config,
        *,
        chunk_frames: int,
        steps: int,
        batch_size: int,
        seed: int,
        start: int = 0,
    ) -> None:
        if not 0 <= start <= steps:
            raise ValueError("The start step must lie inside the plan")
        self.corpus = corpus
        self.config = config
        self.steps = steps
        self.start = start
        self.starts, self.lengths = chunk_plan(
            corpus.frames,
            chunk_frames=chunk_frames,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.steps - self.start

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        starts = self.starts[self.start + index]
        lengths = self.lengths[self.start + index]
        longest = int(lengths.max())
        s = torch.full((len(starts), longest), self.config.s_bos_id, dtype=torch.long)
        m = torch.full((len(starts), longest), self.config.pad_id, dtype=torch.long)
        for row, (begin, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            if self.corpus.s is not None:
                s[row, :length] = self.corpus.s[begin : begin + length].long()
            if self.corpus.m is not None:
                m[row, :length] = self.corpus.m[begin : begin + length].long()
        return {"m": m, "s": s, "lengths": lengths}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return (self[index] for index in range(len(self)))
