"""DY-40 training: depth-transformer LM (src/dauys/dy40_lm.py) on Mimi codes (8 x 2048, 12.5 Hz).

``--units DIR`` holds ``<shard>.npz`` from dy40_mimi.py pass (tok int16 [steps, 8], index = runs of contiguous audio of
one episode). Layout: Moshi's acoustic delay (codes 1..7 one frame after code 0, dy40_lm.delay); each run is preceded by
an EOS position; a training window lies inside one EOS + run block (runs shorter than the context are dropped, as DY-33). Held-out (dy21_train.held_out) shards are validation only. Best checkpoint by validation loss
((10 CE code 0 + sum CE codes 1..7) / 17). Stops at --steps, after --patience validations without a new best, or at
--max-hours. Tracked in Airon.

    python scripts/dy40_train.py --units /opt/dlami/nvme/units --out /data/dy40/run
"""

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

import airon
import numpy as np
import torch

from dauys.dy40_lm import DLConfig, DepthLM, delay, loss
from dy21_train import held_out, lr_at

EXPERIMENT = ("Модель b DY-32 на Mimi: продолжает ли модель речь по теме услышанного лучше, чем на канале L6 "
              "(8 кодов, глубинный декодер, без текста)")


def load_runs(files: list[Path], k: int, ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """Each run of n frames -> EOS position + its n + 1 delayed positions (dy40_lm.delay); a window of ctx + 1
    positions lies inside one EOS + run block (it may start at the EOS, as a generation prompt does). Runs whose block
    is shorter than ctx + 1 are dropped. Returns the joined stream and the allowed window starts."""
    parts, starts, pos = [], [], 0
    eos = np.full((1, 8), k, np.int16)
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        for st, n in index:
            blk = n + 2
            if blk >= ctx + 1:
                parts.append(eos)
                parts.append(delay(tok[st : st + n], k + 1))
                starts.append(np.arange(pos, pos + blk - ctx, dtype=np.int64))
                pos += blk
    return np.concatenate(parts), np.concatenate(starts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=12257)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-hours", type=float, default=5.5)
    p.add_argument("--snapshot-min", type=float, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", default="", help="d,layers,heads,ffn,ctx,dd,dlayers,dheads,dffn; default DLConfig()")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0, help="checklist: train this many steps on ONE fixed batch")
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--name", default="dy40-m")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = DLConfig(**dict(zip(["d", "layers", "heads", "ffn", "ctx", "dd", "dlayers", "dheads", "dffn"],
                              map(int, args.cfg.split(","))))) if args.cfg else DLConfig()
    files = sorted(Path(args.units).glob("*.npz"))[: args.limit or None]
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    assert va_files, "no held-out shard among the units"
    assert not {f.stem for f in tr_files} & {f.stem for f in va_files}
    T = cfg.ctx
    train, tr_ok = load_runs(tr_files, cfg.k, T)
    val, va_ok = load_runs(va_files, cfg.k, T)
    vstarts = va_ok[np.random.default_rng(1234).integers(0, len(va_ok), size=128)]
    vbatch = torch.from_numpy(np.stack([val[s : s + T + 1] for s in vstarts]).astype(np.int64))

    model = DepthLM(cfg).to(device)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    meta = {"run_id": str(uuid.uuid4()), "params": model.parameters_count(),
            "params_temporal": sum(q.numel() for n, q in model.named_parameters() if n.startswith(("emb", "blocks", "norm"))),
            "config": {**vars(cfg), "vocab": cfg.vocab}, "args": vars(args),
            "train_files": len(tr_files), "val_files": [f.stem for f in va_files],
            "train_steps_audio": int(len(train)), 
            "train_hours": round(len(train) / 12.5 / 3600, 1),
            "passes_at_max_steps": round(args.steps * args.batch * (T + 1) / max(len(train), 1), 2),
            "units": "kyutai/mimi rev 89091b3e, 8 x 2048, 12.5 Hz (16 kHz DuplexChat raised to 24 kHz); acoustic codes delayed one frame; EOS + run blocks >= ctx + 1",
            "git": os.environ.get("DY_COMMIT", "")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    run = airon.init(project="dauys", name=args.name if not args.overfit else args.name + "-overfit", config=meta,
                     experiment=args.experiment, tags=["dy40", f"seed-{args.seed}"], total_steps=args.steps,
                     run_id=meta["run_id"], mode="disabled" if args.tracking == "off" else None)

    if args.overfit:
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in tr_ok[:args.micro]]).astype(np.int64)).to(device)
        for step in range(args.overfit):
            for g in opt.param_groups:
                g["lr"] = 1e-3 if step >= 20 else 1e-3 * (step + 1) / 20
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot, part = loss(model, ids)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 25 == 0 or step == args.overfit - 1:
                print(json.dumps({"overfit_step": step, **{k: round(v, 4) for k, v in part.items()}}), flush=True)
        run.finish()
        print("OVERFIT DONE", part["ce"], flush=True)
        return

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        acc: dict[str, float] = {}
        nb = 0
        for i in range(0, len(vbatch), 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, part = loss(model, vbatch[i : i + 8].to(device))
            for k, v in part.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
        model.train()
        return {k: v / nb for k, v in acc.items()}

    best, stale = math.inf, 0

    def checkpoint(step: int) -> bool:
        nonlocal best
        res = validate()
        run.log({f"val/{k}": v for k, v in res.items()}, step=step)
        print(json.dumps({"step": step, "val": res, "utc": time.strftime("%H:%M", time.gmtime())}), flush=True)
        if res["ce"] < best:
            best = res["ce"]
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step, "val": res}, out / "best.tmp")
            os.replace(out / "best.tmp", out / "best.pt")
            return True
        return False

    t_begin, t0, snapped = time.time(), time.time(), False
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        starts = tr_ok[rng.integers(0, len(tr_ok), size=args.batch)]
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in starts]).astype(np.int64)).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        chunks = ids.split(args.micro)
        logs: dict[str, float] = {}
        for c in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot, part = loss(model, c)
            (tot / len(chunks)).backward()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / len(chunks)
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if (step + 1) % 50 == 0:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = 50 / (time.time() - t0)
            t0 = time.time()
            run.log({f"train/{k}": v for k, v in logs.items()}, step=step + 1)
            print(json.dumps({"step": step + 1, **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        late = time.time() - t_begin > args.max_hours * 3600
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps or late:
            stale = 0 if checkpoint(step + 1) else stale + 1
            if (not snapped and args.snapshot_min and time.time() - t_begin > args.snapshot_min * 60
                    and (out / "best.pt").exists()):
                import shutil

                shutil.copy(out / "best.pt", out / "snapshot.tmp")
                os.replace(out / "snapshot.tmp", out / "snapshot.pt")
                snapped = True
                print(json.dumps({"snapshot": step + 1}), flush=True)
            if args.patience and stale >= args.patience:
                print(json.dumps({"early_stop": step + 1, "best": best}), flush=True)
                break
        if late:
            print(json.dumps({"max_hours": args.max_hours, "step": step + 1}), flush=True)
            break
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


if __name__ == "__main__":
    main()
