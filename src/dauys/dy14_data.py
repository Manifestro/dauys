"""Windows, anchors and candidate partner windows for DY-14.

A conversation on the 10 Hz grid has two channels; a sequence takes one as "self" and
the other as "partner". At an anchor frame ``t`` the target is the partner's units that
start in ``[t + GAP, t + GAP + WIN)`` (0.4 to 2.4 s ahead), as an ordered list without
timing or silence. A window is valid with at least ``MIN_UNITS`` units. Hard negatives
are valid windows of the same partner in the same conversation at least ``MIN_DIST``
frames from the anchor: same voice, same people, other words.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CTX = 600  # 60 s
GAP, WIN = 4, 20
MIN_UNITS = 3
MAX_UNITS = WIN  # at most one unit per grid frame
MIN_DIST = 100
NEGATIVES = 15
FIRST_ANCHOR = 100  # training anchors need at least 10 s of context
WHO_AHEAD = 10
OWN_AHEAD = 20


@dataclass
class Conv:
    name: str
    tok: np.ndarray  # [2, T] int64
    pros: np.ndarray  # [2, T, 3] float32
    k: int

    @property
    def length(self) -> int:
        return self.tok.shape[1]

    def __post_init__(self) -> None:
        self.unit = self.tok < self.k  # [2, T]
        self.active = self.tok != self.k + 1
        c = np.concatenate([np.zeros((2, 1), np.int64), np.cumsum(self.unit, 1)], 1)
        t = np.arange(self.length)
        lo = np.clip(t + GAP, 0, self.length)
        hi = np.clip(t + GAP + WIN, 0, self.length)
        self.window_units = c[:, hi] - c[:, lo]  # [2, T]
        self.valid = (self.window_units >= MIN_UNITS) & (t + GAP + WIN <= self.length)[
            None
        ]
        self.starts = [np.flatnonzero(self.valid[c]) for c in range(2)]
        # next own unit within OWN_AHEAD frames, -100 if none
        self.next_unit = np.full((2, self.length), -100, np.int64)
        for ch in range(2):
            pos = np.flatnonzero(self.unit[ch])
            idx = np.searchsorted(pos, t, side="right")
            ok = idx < len(pos)
            nxt = np.where(ok, pos[np.minimum(idx, len(pos) - 1)], 10**9)
            near = ok & (nxt - t <= OWN_AHEAD)
            self.next_unit[ch, near] = self.tok[ch, nxt[near]]

    def window(self, ch: int, t: int) -> np.ndarray:
        seg = self.tok[ch, t + GAP : t + GAP + WIN]
        return seg[seg < self.k]


def load_convs(grid_dir: str | Path, names: list[str], k: int) -> list[Conv]:
    convs = []
    for name in names:
        z = np.load(Path(grid_dir) / f"{name}.npz")
        tok = np.stack([z["tok_a"], z["tok_b"]]).astype(np.int64)
        pros = np.stack([z["pros_a"], z["pros_b"]]).astype(np.float32)
        n = min(tok.shape[1], pros.shape[1])
        if n >= CTX + GAP + WIN + MIN_DIST:
            convs.append(Conv(name, tok[:, :n], pros[:, :n], k))
    return convs


def pad_windows(windows: list[np.ndarray], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit lists to ``[N, MAX_UNITS]`` ids (pad ``k``) and a mask of real units."""
    ids = torch.full((len(windows), MAX_UNITS), k, dtype=torch.long)
    for i, w in enumerate(windows):
        ids[i, : len(w)] = torch.from_numpy(w[:MAX_UNITS])
    return ids, ids != k


def hard_negatives(conv: Conv, ch: int, t: int, rng: np.random.Generator) -> list[int]:
    """NEGATIVES distinct valid window starts of ``ch`` at least MIN_DIST from ``t``.

    Rejection sampling: ``rng.choice`` without replacement permutes every start of the
    conversation for each anchor, which made the sampler the bottleneck.
    """
    starts = conv.starts[ch]
    picked: set[int] = set()
    for _ in range(20 * NEGATIVES):
        s = int(starts[rng.integers(len(starts))]) if len(starts) else t
        if abs(s - t) >= MIN_DIST:
            picked.add(s)
            if len(picked) == NEGATIVES:
                return sorted(picked)
    far = starts[np.abs(starts - t) >= MIN_DIST]
    if len(far) < NEGATIVES:
        return []
    return rng.choice(far, size=NEGATIVES, replace=False).tolist()


def sequence(
    conv: Conv, self_ch: int, s: int, self_from: Conv | None = None
) -> dict[str, torch.Tensor]:
    """Model inputs for ``[s, s + CTX)``; ``self_from`` swaps in another conversation's own channel."""
    src = self_from or conv
    idx = (np.arange(s, s + CTX)) % src.length
    return {
        "tok_self": torch.from_numpy(src.tok[self_ch, idx]),
        "tok_partner": torch.from_numpy(conv.tok[1 - self_ch, s : s + CTX]),
        "pros": torch.from_numpy(
            np.concatenate(
                [src.pros[self_ch, idx], conv.pros[1 - self_ch, s : s + CTX]], -1
            )
        ),
    }


class TrainStream(torch.utils.data.IterableDataset):
    """Endless TrainSampler batches; each DataLoader worker draws with its own seed."""

    def __init__(
        self, convs: list[Conv], k: int, batch: int, anchors: int, seed: int
    ) -> None:
        self.args = (convs, k, batch, anchors, seed)

    def __iter__(self):
        convs, k, batch, anchors, seed = self.args
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        sampler = TrainSampler(convs, k, batch, anchors, seed=seed * 1000 + worker)
        while True:
            yield sampler()


class TrainSampler:
    """Random training batches: windows of CTX frames, up to ``anchors`` anchors each."""

    def __init__(
        self, convs: list[Conv], k: int, batch: int, anchors: int, seed: int
    ) -> None:
        self.convs, self.k, self.batch, self.anchors = convs, k, batch, anchors
        self.rng = np.random.default_rng(seed)
        w = np.array([c.length - CTX - GAP - WIN for c in convs], np.float64)
        self.p = w / w.sum()

    def __call__(self) -> dict[str, torch.Tensor]:
        rows, anchor_rows, anchor_pos, pos_w, neg_w = [], [], [], [], []
        own, who = [], []
        for b in range(self.batch):
            conv = self.convs[self.rng.choice(len(self.convs), p=self.p)]
            ch = int(self.rng.integers(2))
            s = int(self.rng.integers(0, conv.length - CTX - GAP - WIN))
            rows.append(sequence(conv, ch, s))
            t_abs = np.arange(s, s + CTX)
            own.append(torch.from_numpy(conv.next_unit[ch, t_abs]))
            ahead = np.stack(
                [
                    conv.active[
                        c,
                        np.minimum(
                            t_abs[:, None] + np.arange(1, WHO_AHEAD + 1),
                            conv.length - 1,
                        ),
                    ]
                    for c in (ch, 1 - ch)
                ],
                1,
            )  # [CTX, 2, WHO_AHEAD]
            who.append(torch.from_numpy(ahead.reshape(CTX, -1).astype(np.float32)))
            cand = (
                np.flatnonzero(
                    conv.valid[1 - ch, s + FIRST_ANCHOR : s + CTX - GAP - WIN]
                )
                + FIRST_ANCHOR
            )
            self.rng.shuffle(cand)
            for p in cand[: self.anchors]:
                negs = hard_negatives(conv, 1 - ch, s + int(p), self.rng)
                if not negs:
                    continue
                anchor_rows.append(b)
                anchor_pos.append(int(p))
                pos_w.append(conv.window(1 - ch, s + int(p)))
                neg_w += [conv.window(1 - ch, n) for n in negs]
        batch = {key: torch.stack([r[key] for r in rows]) for key in rows[0]}
        batch["own_next"] = torch.stack(own)
        batch["who"] = torch.stack(who)
        batch["anchor_row"] = torch.tensor(anchor_rows, dtype=torch.long)
        batch["anchor_pos"] = torch.tensor(anchor_pos, dtype=torch.long)
        ids, mask = pad_windows(pos_w + neg_w, self.k)
        n = len(pos_w)
        batch["cand_ids"] = torch.cat(
            [ids[:n, None], ids[n:].view(n, NEGATIVES, -1)], 1
        )  # [N, 16, U]
        batch["cand_mask"] = torch.cat(
            [mask[:n, None], mask[n:].view(n, NEGATIVES, -1)], 1
        )
        return batch


def build_eval_anchors(
    convs: list[Conv], seed: int = 1234, per_window: int = 40
) -> dict:
    """Fixed evaluation anchors: windows every 300 frames, anchors in their second half.

    ``all`` anchors are up to ``per_window`` valid frames per window; ``onset`` anchors are
    every frame where the partner was silent for the last 1 s, has a valid window, and
    the self channel spoke in the last 3 s. Negatives are fixed per anchor.
    """
    rng = np.random.default_rng(seed)
    items = []
    for ci, conv in enumerate(convs):
        for ch in range(2):
            for s in range(0, conv.length - CTX - GAP - WIN + 1, CTX // 2):
                lo = s + CTX // 2
                ts = np.arange(lo, s + CTX)
                ts = ts[ts + GAP + WIN <= conv.length]
                valid = ts[conv.valid[1 - ch, ts]]
                quiet = np.array(
                    [not conv.active[1 - ch, t - 10 : t + 1].any() for t in valid], bool
                )
                spoke = np.array(
                    [conv.active[ch, t - 30 : t + 1].any() for t in valid], bool
                )
                onset = set(valid[quiet & spoke].tolist()) if len(valid) else set()
                pick = (
                    set(rng.permutation(valid)[:per_window].tolist())
                    if len(valid)
                    else set()
                )
                for t in sorted(pick | onset):
                    negs = hard_negatives(conv, 1 - ch, int(t), rng)
                    if negs:
                        items.append(
                            {
                                "conv": ci,
                                "ch": ch,
                                "s": s,
                                "t": int(t),
                                "neg": negs,
                                "all": t in pick,
                                "onset": t in onset,
                            }
                        )
    names = [c.name for c in convs]
    blob = json.dumps({"names": names, "items": items}, sort_keys=True).encode()
    return {"names": names, "items": items, "sha256": hashlib.sha256(blob).hexdigest()}
