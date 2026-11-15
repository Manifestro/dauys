"""DY-14 test evaluation and gates.

``--build-anchors``: build the fixed test anchors (seed 1234) and print their SHA-256; run
this before training and record the hash. Without it: evaluate ``best.pt`` of each seed on
those anchors, add the guard numbers, apply the gates fixed in Airon and write a JSON.

    python scripts/dy14_eval.py --grid /data/dy14/grid --split split.json --anchors /data/dy14/test_anchors.json --build-anchors
    python scripts/dy14_eval.py --grid ... --split ... --anchors ... --runs /data/dy14/runs/s0,/data/dy14/runs/s1 \
        --stats /data/dy14/stats.json --out /data/dy14/result.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dauys.dy14_data import build_eval_anchors, load_convs
from dauys.dy14_eval import correct_per_anchor, summarize
from dauys.dy14_model import Tower, TowerConfig

G1_MEAN, CHANCE = 0.125, 1 / 16
G2_DELTA = 0.03


def speaker_probe(
    convs, k: int, device: torch.device, tracks: int = 40, seed: int = 0
) -> dict:
    """Linear probe: which track (person in one conversation) a 10 s bag of units came from."""
    xs, ys, half = [], [], []
    for label, (conv, ch) in enumerate(
        [(c, ch) for c in convs[: tracks // 2] for ch in (0, 1)]
    ):
        tok = conv.tok[ch]
        wins = [tok[s : s + 100] for s in range(0, len(tok) - 100, 100)]
        wins = [w[w < k] for w in wins]
        wins = [w for w in wins if len(w) >= 10]
        for i, w in enumerate(wins):
            v = np.bincount(w, minlength=k).astype(np.float32)
            xs.append(v / v.sum())
            ys.append(label)
            half.append(i >= len(wins) // 2)
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = torch.tensor(ys, device=device)
    test = torch.tensor(half, device=device)
    n_cls = int(y.max()) + 1
    torch.manual_seed(seed)
    w = torch.zeros(k, n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        loss = F.cross_entropy((x[~test] * 100) @ w + b, y[~test]) + 1e-4 * (w**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = ((x[test] * 100) @ w + b).argmax(1).eq(y[test]).float().mean().item()
    return {
        "speaker_probe_acc": acc,
        "speaker_probe_chance": 1 / n_cls,
        "speaker_probe_windows": len(ys),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--build-anchors", action="store_true")
    p.add_argument("--runs", default="")
    p.add_argument("--stats", default="")
    p.add_argument("--out", default="")
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--part", default="test")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    split = json.loads(Path(args.split).read_text())
    convs = load_convs(args.grid, split[args.part][: args.limit or None], args.k)
    if args.build_anchors:
        anchors = build_eval_anchors(convs, seed=1234, per_window=40)
        Path(args.anchors).write_text(json.dumps(anchors))
        n_on = sum(it["onset"] for it in anchors["items"])
        print(
            json.dumps(
                {
                    "test_convs": len(convs),
                    "anchors": len(anchors["items"]),
                    "onset": n_on,
                    "sha256": anchors["sha256"],
                }
            ),
            flush=True,
        )
        return
    anchors = json.loads(Path(args.anchors).read_text())
    blob = json.dumps(
        {"names": anchors["names"], "items": anchors["items"]}, sort_keys=True
    ).encode()
    assert hashlib.sha256(blob).hexdigest() == anchors["sha256"], (
        "anchors changed since they were built"
    )
    assert anchors["names"] == [c.name for c in convs], (
        "anchor conversations differ from the test part"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {"anchors_sha256": anchors["sha256"], "seeds": {}}
    for run in [r for r in args.runs.split(",") if r]:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        model = Tower(TowerConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        real = correct_per_anchor(model, convs, anchors, swap=False, device=device)
        swapped = correct_per_anchor(model, convs, anchors, swap=True, device=device)
        res = summarize(real, swapped, anchors)
        res["best_step"] = ck["step"]
        result["seeds"][Path(run).name] = res
        print(json.dumps({Path(run).name: res}), flush=True)

    seeds = list(result["seeds"].values())
    if seeds:
        g1 = (np.mean([s["acc_all"] for s in seeds]) >= G1_MEAN) and all(
            s["acc_all_lo"] > CHANCE for s in seeds
        )
        g2 = all(
            s["delta_onset"] >= G2_DELTA and s["delta_onset_lo"] > 0 for s in seeds
        )
        result["G1"], result["G2"] = bool(g1), bool(g2)
    guards = speaker_probe(convs, args.k, device)
    if args.stats:
        st = json.loads(Path(args.stats).read_text())[args.part]
        guards.update(
            {
                "units_per_s_speech": st["units_per_s_speech"],
                "collision_share": st["collision_share"],
                "bleed_share": st["bleed_share"],
            }
        )
    result["guards"] = guards
    print(
        json.dumps({"G1": result.get("G1"), "G2": result.get("G2"), "guards": guards}),
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
