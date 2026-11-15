"""DY-14 code checks required before training (Airon rule): run on the instance.

1. leak: split parts disjoint by conversation (and by person when meta.json has ids);
   test anchors only in test conversations; negatives from the anchor's conversation.
2. known answer: synthetic conversations where the partner repeats the self channel's
   last units after a pause; a small tower must find the true window far above chance
   and lose it when the self channel is swapped.
3. one batch: the same batch 300 times; the contrastive loss must fall near zero.
Prints ``CHECK <name> PASS|FAIL`` lines and exits 1 on any FAIL.

    python scripts/dy14_checklist.py --split split.json --root /data/oy16 --anchors /data/dy14/test_anchors.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dauys.dy14_data import CTX, Conv, TrainSampler, build_eval_anchors
from dauys.dy14_eval import correct_per_anchor, summarize
from dauys.dy14_model import Tower, TowerConfig

K = 64


def synthetic(n: int, length: int, seed: int) -> list[Conv]:
    """Self says random units in bursts; after a 1 s pause the partner repeats the burst."""
    rng = np.random.default_rng(seed)
    convs = []
    for i in range(n):
        tok = np.full((2, length), K + 1, np.int64)
        t = 0
        while t + 80 < length:
            burst = rng.integers(0, K, 8)
            frames = t + np.arange(8) * 2
            tok[0, frames] = burst
            tok[0, frames + 1] = K
            reply = t + 30 + np.arange(8) * 2
            tok[1, reply] = burst
            tok[1, reply + 1] = K
            t += 60 + int(rng.integers(0, 20))
        convs.append(Conv(f"syn{i}", tok, np.zeros((2, length, 3), np.float32), K))
    return convs


def check(name: str, ok: bool, info: dict) -> bool:
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {json.dumps(info)}", flush=True)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--root", default="")
    p.add_argument("--anchors", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ok = True

    split = json.loads(Path(args.split).read_text())
    parts = {k: set(v) for k, v in split.items()}
    overlap = {
        f"{a}-{b}": len(parts[a] & parts[b]) for a in parts for b in parts if a < b
    }
    people = {}
    if args.root:
        for part, names in split.items():
            ids = set()
            for name in names:
                meta = Path(args.root) / name / "meta.json"
                if meta.exists():
                    m = json.loads(meta.read_text())
                    for key, v in m.items() if isinstance(m, dict) else []:
                        if "speaker" in key.lower() or "participant" in key.lower():
                            ids.add(json.dumps(v, sort_keys=True))
            people[part] = ids
    person_overlap = (
        {
            f"{a}-{b}": len(people[a] & people[b])
            for a in people
            for b in people
            if a < b
        }
        if people
        else {}
    )
    ok &= check(
        "leak_split",
        all(v == 0 for v in overlap.values())
        and all(v == 0 for v in person_overlap.values()),
        {
            "conversations": overlap,
            "people_fields": person_overlap,
            "people_found": {k: len(v) for k, v in people.items()},
        },
    )
    if args.anchors:
        a = json.loads(Path(args.anchors).read_text())
        in_test = set(a["names"]) <= parts["test"]
        neg_far = all(abs(n - it["t"]) >= 100 for it in a["items"] for n in it["neg"])
        ok &= check(
            "leak_anchors",
            in_test and neg_far,
            {"names": len(a["names"]), "items": len(a["items"])},
        )

    torch.manual_seed(0)
    train, test = synthetic(40, 3000, 1), synthetic(8, 3000, 2)
    cfg = TowerConfig(
        k=K, d=128, layers=2, heads=4, ffn=256, latent=64, target_d=64, target_layers=1
    )
    model = Tower(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = TrainSampler(train, K, 16, 8, seed=0)
    for step in range(600):
        batch = {k_: v.to(device) for k_, v in sampler().items()}
        loss, logs = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    anchors = build_eval_anchors(test, per_window=20)
    # only channel 1 repeats channel 0, so only self = 0 has a knowable partner window
    anchors["items"] = [it for it in anchors["items"] if it["ch"] == 0]
    real = correct_per_anchor(model, test, anchors, swap=False, device=device)
    swapped = correct_per_anchor(model, test, anchors, swap=True, device=device)
    res = summarize(real, swapped, anchors)
    ok &= check(
        "known_answer",
        res["acc_all"] > 0.6 and res["delta_onset"] > 0.3,
        {k_: round(v, 3) for k_, v in res.items() if isinstance(v, float)},
    )

    model = Tower(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        k_: v.to(device) for k_, v in TrainSampler(train, K, 8, 8, seed=5)().items()
    }
    first = None
    for step in range(300):
        loss, logs = model.loss(batch)
        first = first if first is not None else logs["nce"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    ok &= check(
        "one_batch",
        logs["nce"] < 0.05,
        {"nce_first": round(first, 3), "nce_last": round(logs["nce"], 4)},
    )

    with torch.device("meta"):
        big = Tower(TowerConfig())
    print(
        json.dumps(
            {
                "tower_params_full_config": sum(p_.numel() for p_ in big.parameters()),
                "ctx": CTX,
            }
        ),
        flush=True,
    )
    print("CHECKLIST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
