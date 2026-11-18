"""DY-24 training: variant B, C or D (src/dauys/dy24.py) on the DY-23 shards; same size, batch, lr and steps as DY-23.

    python scripts/dy24_train.py --variant B --units /opt/dlami/nvme/units23 --coder coder.pt \
        --init /data/dy22/run/best.pt --out /data/dy24B/run
"""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import airon
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out, lr_at
from dy23_train import load_grid

from dauys import dy21_lm, dy22_lm, dy23_lm
from dauys.dy21_lm import LMConfig, ema_update, make_target
from dauys.dy23_lm import CONT, EOS, SIL, UNIT, from_dy22
from dauys.dy24 import ContentBooks, FutureLM, loss_d, reinit_unit_rows

EXPERIMENT = "DY-23: голосо-независимые коды слогов (8×256) + темп DY-22 — улучшается ли смысл"
K = 8192


def to_b(ev: np.ndarray, cd: np.ndarray, books: ContentBooks, device) -> np.ndarray:
    """DY-22 grid tokens with content ids (< 2048) at syllables."""
    tok = np.full(len(ev), K + 1, np.int16)
    tok[ev == CONT] = K
    tok[ev == EOS] = K + 2
    u = np.flatnonzero(ev == UNIT)
    for i in range(0, len(u), 1 << 24):
        j = u[i : i + (1 << 24)]
        tok[j] = books.content_ids(torch.from_numpy(cd[j].astype(np.int64)).to(device)).cpu().numpy()
    return tok


def to_c(tok_b: np.ndarray) -> np.ndarray:
    """Deduplicated syllable stream: content ids, EOS (= K) between pieces."""
    keep = (tok_b < K) | (tok_b == K + 2)
    s = tok_b[keep].copy()
    s[s == K + 2] = K
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["B", "C", "D"])
    p.add_argument("--units", required=True)
    p.add_argument("--coder", required=True)
    p.add_argument("--init", required=True, help="B, D: DY-22 best.pt; C: DY-21 best.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--small", action="store_true")
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()
    V = args.variant
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    books = ContentBooks(torch.load(args.coder, map_location="cpu"), device)

    files = sorted(Path(args.units).glob("*.npz"))[: args.limit or None]
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)] or files[-1:]
    t0 = time.time()
    data = {}
    for name, fs in (("train", tr_files), ("val", va_files)):
        ev, cd, pb = load_grid(fs)
        if V == "B":
            data[name] = (to_b(ev, cd, books, device), pb)
        elif V == "C":
            data[name] = (to_c(to_b(ev, cd, books, device)),)
        else:
            data[name] = (ev, cd, pb)
    print("loaded", {k: len(v[0]) for k, v in data.items()}, "in", round(time.time() - t0), "s", flush=True)

    if V == "B":
        cfg = LMConfig(vocab=K + 3)
    elif V == "C":
        cfg = LMConfig(vocab=K + 1)
    else:
        cfg = LMConfig(vocab=4)
    if args.small:
        cfg = LMConfig(vocab=cfg.vocab, d=256, layers=4, heads=4, ffn=1024, ctx=256)
    T = cfg.ctx
    ck = None if args.small else torch.load(args.init, map_location="cpu", weights_only=False)
    if V == "B":
        model = dy22_lm.GridLM(cfg)
        if ck:
            model.load_state_dict(ck["model"])
            reinit_unit_rows(model, K)
        lossf = lambda m, t, b: dy22_lm.loss(m, t, b[0], b[1])
    elif V == "C":
        model = dy21_lm.UnitLM(cfg)
        if ck:
            model.load_state_dict(ck["model"])
            reinit_unit_rows(model, K)
        lossf = lambda m, t, b: dy21_lm.loss(m, t, b[0])
    else:
        model = FutureLM(cfg)
        if ck:
            model.load_state_dict(from_dy22(ck["model"], cfg).state_dict(), strict=False)
        lossf = lambda m, t, b: loss_d(m, t, books, b[0], b[1], b[2])
    model = model.to(device)
    tgt = make_target(model)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)

    def batch(arrs, starts):
        return [torch.from_numpy(np.stack([a[s : s + T + 1] for s in starts]).astype(np.int64)).to(device) for a in arrs]

    tr, va = data["train"], data["val"]
    vstarts = np.random.default_rng(1234).integers(0, len(va[0]) - T - 1, size=256)
    vb = batch(va, vstarts)
    meta = {"run_id": str(uuid.uuid4()), "variant": V, "params": sum(q.numel() for q in model.parameters()),
            "config": vars(cfg), "args": vars(args), "train_tokens": int(len(tr[0])), "val_tokens": int(len(va[0])),
            "val_files_names": [f.stem for f in va_files], "git": os.environ.get("DY24_COMMIT", "")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_files_names"}), flush=True)
    run = airon.init(project="dauys", name=f"dy24{V}-s{args.seed}", config=meta, experiment=EXPERIMENT,
                     tags=["dy24", f"variant-{V}"], total_steps=args.steps, run_id=meta["run_id"],
                     mode="disabled" if args.tracking == "off" else None)

    @torch.no_grad()
    def validate():
        model.eval()
        acc = {}
        for i in range(0, 256, 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, part = lossf(model, tgt, [x[i : i + 32] for x in vb])
            for k, v in part.items():
                acc[k] = acc.get(k, 0.0) + v * 32 / 256
        model.train()
        return acc

    print(json.dumps({"step": 0, "val": validate()}), flush=True)
    best, t0 = math.inf, time.time()
    n = math.ceil(args.batch / args.micro)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        b = batch(tr, rng.integers(0, len(tr[0]) - T - 1, size=args.batch))
        opt.zero_grad(set_to_none=True)
        logs = {}
        for j in range(n):
            mb = [x[j * args.micro : (j + 1) * args.micro] for x in b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, part = lossf(model, tgt, mb)
            (total / n).backward()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / n
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        ema_update(tgt, model, args.ema)
        if (step + 1) % 50 == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = 50 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            res = validate()
            run.log({f"val/{k}": v for k, v in res.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, "val": res}), flush=True)
            if res["loss"] < best:
                best = res["loss"]
                torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res,
                            "variant": V}, out / "best.pt")
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


if __name__ == "__main__":
    main()
