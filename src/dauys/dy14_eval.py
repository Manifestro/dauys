"""Evaluation of DY-14 on fixed anchors: top-1 among 16 windows of the same partner.

Each window ``(conv, ch, s)`` is run once with its real self channel and once with the
self channel of the next conversation in the list (other people, same offsets); the
partner channel is never changed. Gates use conversation-level bootstrap CIs.
"""

from collections import defaultdict

import numpy as np
import torch

from .dy14_data import NEGATIVES, Conv, pad_windows, sequence
from .dy14_model import Tower


@torch.no_grad()
def correct_per_anchor(
    model: Tower,
    convs: list[Conv],
    anchors: dict,
    swap: bool,
    device: torch.device,
    batch: int = 32,
) -> np.ndarray:
    model.eval()
    items = anchors["items"]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[(it["conv"], it["ch"], it["s"])].append(i)
    keys = list(groups)
    out = np.zeros(len(items), bool)
    k = model.cfg.k
    for g in range(0, len(keys), batch):
        part = keys[g : g + batch]
        rows = []
        for ci, ch, s in part:
            other = convs[(ci + 1) % len(convs)] if swap else None
            rows.append(sequence(convs[ci], ch, s, self_from=other))
        x = {key: torch.stack([r[key] for r in rows]).to(device) for key in rows[0]}
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            h = model.states(x["tok_self"], x["tok_partner"], x["pros"])
            arow, apos, wins, idx = [], [], [], []
            for r, key in enumerate(part):
                ci, ch, s = key
                conv = convs[ci]
                for i in groups[key]:
                    it = items[i]
                    arow.append(r)
                    apos.append(it["t"] - s)
                    wins += [conv.window(1 - ch, it["t"])] + [
                        conv.window(1 - ch, n) for n in it["neg"]
                    ]
                    idx.append(i)
            ids, mask = pad_windows(wins, k)
            n = len(idx)
            hard, _ = model.scores(
                h,
                torch.tensor(arow, device=device),
                torch.tensor(apos, device=device),
                ids.view(n, NEGATIVES + 1, -1).to(device),
                mask.view(n, NEGATIVES + 1, -1).to(device),
            )
        out[idx] = (hard.argmax(1) == 0).cpu().numpy()
    model.train()
    return out


def bootstrap(
    values: np.ndarray, groups: np.ndarray, reps: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% CI, resampling whole conversations."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: values[groups == g] for g in uniq}
    sums = np.array([by[g].sum() for g in uniq])
    counts = np.array([len(by[g]) for g in uniq])
    stats = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[pick].sum() / max(counts[pick].sum(), 1))
    return (
        float(values.mean()),
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def summarize(real: np.ndarray, swapped: np.ndarray, anchors: dict) -> dict:
    items = anchors["items"]
    conv = np.array([it["conv"] for it in items])
    is_all = np.array([it["all"] for it in items], bool)
    is_on = np.array([it["onset"] for it in items], bool)
    res = {
        "n_all": int(is_all.sum()),
        "n_onset": int(is_on.sum()),
        "chance": 1 / (NEGATIVES + 1),
    }
    res["acc_all"], res["acc_all_lo"], res["acc_all_hi"] = bootstrap(
        real[is_all].astype(float), conv[is_all]
    )
    res["acc_onset"], res["acc_onset_lo"], res["acc_onset_hi"] = bootstrap(
        real[is_on].astype(float), conv[is_on]
    )
    res["acc_onset_swap"], _, _ = bootstrap(swapped[is_on].astype(float), conv[is_on])
    diff = real[is_on].astype(float) - swapped[is_on].astype(float)
    res["delta_onset"], res["delta_onset_lo"], res["delta_onset_hi"] = bootstrap(
        diff, conv[is_on]
    )
    res["acc_all_swap"], _, _ = bootstrap(swapped[is_all].astype(float), conv[is_all])
    return res
