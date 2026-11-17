"""DY-22 training: the grid LM (units + timing + pauses + prosody), warm-started from DY-21.

Same shards and held-out split as DY-21 (sha1 % 50 == 0). Each piece's 10 Hz grid with pauses cut to 2 s,
pieces joined with EOS. Prosody bins are stored as uint8 next to the tokens.

    python scripts/dy22_train.py --units /opt/dlami/nvme/units --init /data/dy21/run/best.pt --out /data/dy22/run
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

from dauys.dy21_lm import LMConfig, ema_update, make_target
from dauys.dy22_lm import EOS, K, MAX_SIL, SIL, VOCAB, GridLM, e_bin, f0_bin, from_dy21, loss

EXPERIMENT = "DY-22: живая речь — модель слогов предсказывает темп, паузы и интонацию (тёплый старт от DY-21)"


def cut_pauses(tok: np.ndarray, pros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sil = tok == SIL
    pos = np.arange(len(tok))
    last = np.maximum.accumulate(np.where(~sil, pos, -1))
    keep = ~sil | (pos - last <= MAX_SIL)
    return tok[keep], pros[keep]


def load_grid(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """All pieces' grids joined with EOS: tokens int16 [N] and prosody bins uint8 [N, 3]."""
    toks, pbs = [], []
    for f in files:
        z = np.load(f)
        tok, pros, index = z["tok"].astype(np.int64), z["pros"].astype(np.float32), z["index"]
        for st, n in index:
            t, p = tok[st : st + n], pros[st : st + n]
            if (t < K).sum() < 16:
                continue
            t, p = cut_pauses(t, p)
            pb = np.stack([f0_bin(torch.from_numpy(p[:, 1])).numpy(), e_bin(torch.from_numpy(p[:, 0])).numpy(),
                           (p[:, 2] > 0.5).astype(np.int64)], 1).astype(np.uint8)
            toks += [t.astype(np.int16), np.array([EOS], np.int16)]
            pbs += [pb, np.zeros((1, 3), np.uint8)]
    return np.concatenate(toks), np.concatenate(pbs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--init", default="", help="DY-21 best.pt; empty = from scratch")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--small", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    files = sorted(Path(args.units).glob("*.npz"))[: args.limit or None]
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)] or files[-1:]
    t0 = time.time()
    train, train_p = load_grid(tr_files)
    val, val_p = load_grid(va_files)
    print("loaded", len(train), len(val), "tokens in", round(time.time() - t0), "s", flush=True)
    cfg = LMConfig(vocab=VOCAB, d=256, layers=4, heads=4, ffn=1024, ctx=256) if args.small else LMConfig(vocab=VOCAB)
    T = cfg.ctx
    vstarts = np.random.default_rng(1234).integers(0, len(val) - T - 1, size=256)
    vb = torch.from_numpy(np.stack([val[s : s + T + 1] for s in vstarts]).astype(np.int64))
    vp = torch.from_numpy(np.stack([val_p[s : s + T + 1] for s in vstarts]).astype(np.int64))

    if args.init and not args.small:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model = from_dy21(ck["model"], cfg).to(device)
    else:
        model = GridLM(cfg).to(device)
    tgt = make_target(model)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    meta = {
        "run_id": str(uuid.uuid4()),
        "params": sum(q.numel() for q in model.parameters()),
        "config": vars(cfg),
        "args": vars(args),
        "train_files": len(tr_files),
        "val_files": len(va_files),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "val_files_names": [f.stem for f in va_files],
        "git": os.environ.get("DY22_COMMIT", ""),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_files_names"}), flush=True)
    run = airon.init(project="dauys", name=f"dy22-s{args.seed}", config=meta, experiment=EXPERIMENT,
                     tags=["dy22", f"seed-{args.seed}"], total_steps=args.steps, run_id=meta["run_id"],
                     mode="disabled" if args.tracking == "off" else None)

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        acc = {}
        for i in range(0, len(vb), 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, part = loss(model, tgt, vb[i : i + 32].to(device), vp[i : i + 32].to(device))
            for k, v in part.items():
                acc[k] = acc.get(k, 0.0) + v * 32 / len(vb)
        model.train()
        acc["ppl_unit"] = math.exp(acc["ce_unit"])
        return acc

    res = validate()
    print(json.dumps({"step": 0, "val": res}), flush=True)
    best, t0 = math.inf, time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        starts = rng.integers(0, len(train) - T - 1, size=args.batch)
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in starts]).astype(np.int64)).to(device)
        pr = torch.from_numpy(np.stack([train_p[s : s + T + 1] for s in starts]).astype(np.int64)).to(device)
        opt.zero_grad(set_to_none=True)
        logs = {}
        n = math.ceil(args.batch / args.micro)
        for c, cp in zip(ids.split(args.micro), pr.split(args.micro)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, part = loss(model, tgt, c, cp)
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
                torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step + 1, "val": res},
                           out / "best.pt")
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


if __name__ == "__main__":
    main()
