"""DY-22 checklist before training: split, data ranges, warm start, one-batch overfit.

    python scripts/dy22_checklist.py --units /opt/dlami/nvme/units --init /data/dy21/run/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy22_train import load_grid

from dauys.dy21_lm import LMConfig, UnitLM, make_target
from dauys.dy22_lm import CONT, EOS, F0_BINS, K, MAX_SIL, SIL, VOCAB, GridLM, from_dy21, loss


def check(name: str, ok: bool, info: dict) -> bool:
    print("CHECK", name, "PASS" if ok else "FAIL", json.dumps(info), flush=True)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--init", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    files = sorted(Path(args.units).glob("*.npz"))
    held = [f for f in files if held_out(f.stem)]
    ok = check("split", 15 <= len(held) <= 30, {"files": len(files), "held_out": len(held)})

    tok, pb = load_grid(files[:5])
    t = tok.astype(np.int64)
    sil = t == SIL
    pos = np.arange(len(t))
    sil_runs = np.where(sil, pos - np.maximum.accumulate(np.where(~sil, pos, -1)), 0)
    share = {n: float((t == v).mean()) for n, v in (("cont", CONT), ("sil", SIL), ("eos", EOS))}
    share["unit"] = float((t < K).mean())
    u = t < K
    info = {**share, "max_sil_run": int(sil_runs.max()), "voiced_share": float(pb[u, 2].mean()),
            "f0_bin_mean": float(pb[u & (pb[:, 2] == 1), 0].mean()), "max_tok": int(t.max())}
    ok &= check("data", t.max() < VOCAB and sil_runs.max() <= MAX_SIL and 0.1 < share["unit"] < 0.6
                and 0.3 < info["voiced_share"] < 0.98 and 6 < info["f0_bin_mean"] < F0_BINS - 6, info)

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    old = UnitLM(LMConfig(**ck["cfg"])).to(device).eval()
    old.load_state_dict(ck["model"])
    new = from_dy21(ck["model"], LMConfig(vocab=VOCAB)).to(device).eval()
    x = torch.randint(0, K, (2, 64), device=device)
    with torch.no_grad():
        a = old.logits(old.states(x))[..., :K]
        b = new.logits(new.states(x))[..., :K]
    ok &= check("warm_start", float((a - b).abs().max()) < 1e-3, {"max_diff": float((a - b).abs().max())})
    del old, new

    small = GridLM(LMConfig(vocab=VOCAB, d=256, layers=4, heads=4, ffn=1024, ctx=256)).to(device)
    tgt = make_target(small)
    ids = torch.from_numpy(t[:257][None].repeat(4, 0)).to(device)
    pr = torch.from_numpy(pb[:257][None].repeat(4, 0).astype(np.int64)).to(device)
    opt = torch.optim.AdamW(small.parameters(), lr=1e-3)
    first = None
    for _ in range(300):
        total, part = loss(small, tgt, ids, pr)
        first = first or part
        opt.zero_grad()
        total.backward()
        opt.step()
    ok &= check("one_batch", part["ce"] < 0.3 * first["ce"] and part["f0"] < 0.5 * first["f0"],
                {"ce": [round(first["ce"], 3), round(part["ce"], 3)], "f0": [round(first["f0"], 3), round(part["f0"], 3)]})
    print("CHECKLIST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
