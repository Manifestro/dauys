"""DY-32 track C training: multi-stream delay-pattern LM (src/dauys/dy32c_lm.py) on parallel code streams.

``--units DIR`` holds ``<shard>.npz`` (tok int16 [frames, S]; index). The model streams are the npz columns in
``--order`` (default 0,1,2,3 = residual levels coarse to fine, so each finer level of a frame is predicted after the
coarser ones). Pieces >= 16 frames are joined with an EOS frame, the whole stream is put
in the delay layout once, batches are random windows of ctx + 1 positions. Held out (dy21_train.held_out) shards are
validation only. Best checkpoint by validation CE (mean over streams). ``--stop-utc HH:MM`` ends training at that
time (validation + checkpoint first). Tracked in Airon like dy21_train.

    python scripts/dy32c_train.py --units /opt/dlami/nvme/c-units --out /data/dy32c/run --steps 20000 --stop-utc 07:10
"""

import argparse
import datetime as dt
import json
import math
import os
import time
import uuid
from pathlib import Path

import airon
import numpy as np
import torch

from dauys.dy32c_lm import MSConfig, MultiStreamLM, delay, loss
from dy21_train import held_out, lr_at

EXPERIMENT = "DY-32: канал, через который слова выживают, без текста — три идеи и повтор GSLM параллельными ветками"


def load_runs(files: list[Path], cols: list[int], k: int, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    """DY-33: only runs (contiguous audio of one episode, index rows of the npz) of >= min_run steps, joined with an
    EOS frame, in the delay layout; returns the stream and the allowed window starts (window inside one run)."""
    parts, starts, pos = [], [], 0
    eos = np.full((1, len(cols)), k, np.int16)
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        for st, n in index:
            if n >= min_run:
                parts.append(tok[st : st + n][:, cols])
                starts.append(np.arange(pos, pos + n - min_run + 1, dtype=np.int64))
                parts.append(eos)
                pos += n + 1
    return delay(np.concatenate(parts), k + 1), np.concatenate(starts)


def load_stream(files: list[Path], cols: list[int], k: int) -> np.ndarray:
    parts = []
    eos = np.full((1, len(cols)), k, np.int16)
    for f in files:
        z = np.load(f)
        tok, index = z["tok"], z["index"]
        for st, n in index:
            if n >= 16:
                parts.append(tok[st : st + n][:, cols])
                parts.append(eos)
    return delay(np.concatenate(parts), k + 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order", default="0,1,2,3", help="npz columns in model stream order (rvq: levels 0..3; "
                   "sb six-layer units: 5,4,3,2,1,0 = layers 23..1)")
    p.add_argument("--k", type=int, default=1024, help="codes per stream")
    p.add_argument("--runs", action="store_true", help="DY-33: windows only inside runs >= ctx + 1 steps (npz index = runs)")
    p.add_argument("--rate", type=float, default=50.0, help="model steps per second of audio (for logging hours)")
    p.add_argument("--patience", type=int, default=0, help="stop after this many validations without a new best")
    p.add_argument("--snapshot-min", type=float, default=0, help="copy best.pt to snapshot.pt once, this many minutes in")
    p.add_argument("--units-desc", default="kNN-VC WavLM-Large layer 6, residual k-means 4 x 1024 (track A corpus codebooks), 50 Hz")
    p.add_argument("--cfg", default="", help="d,layers,heads,ffn,ctx; default MSConfig()")
    p.add_argument("--stop-utc", default="", help="HH:MM UTC: validate, checkpoint and stop at this time")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--name", default="dy32-c")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    order = [int(x) for x in args.order.split(",")]
    cols = order
    if args.cfg:
        d, layers, heads, ffn, ctx = map(int, args.cfg.split(","))
        cfg = MSConfig(streams=len(cols), k=args.k, d=d, layers=layers, heads=heads, ffn=ffn, ctx=ctx)
    else:
        cfg = MSConfig(streams=len(cols), k=args.k)
    PAD = cfg.k + 1
    stop_at = None
    if args.stop_utc:
        now = dt.datetime.now(dt.timezone.utc)
        h, m = map(int, args.stop_utc.split(":"))
        stop_at = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop_at < now:
            stop_at += dt.timedelta(days=1)

    files = sorted(Path(args.units).glob("*.npz"))[: args.limit or None]
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)] or files[-1:]
    T = cfg.ctx
    vrng = np.random.default_rng(1234)
    if args.runs:
        (train, tr_ok), (val, va_ok) = (load_runs(x, cols, cfg.k, T + 1) for x in (tr_files, va_files))
        vstarts = va_ok[vrng.integers(0, len(va_ok), size=128)]
    else:
        train, val = load_stream(tr_files, cols, cfg.k), load_stream(va_files, cols, cfg.k)
        tr_ok = None
        vstarts = vrng.integers(0, len(val) - T - 1, size=128)
    vbatch = torch.from_numpy(np.stack([val[s : s + T + 1] for s in vstarts]).astype(np.int64))

    model = MultiStreamLM(cfg).to(device)
    decay = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    meta = {"run_id": str(uuid.uuid4()), "params": model.parameters_count(), "config": {**vars(cfg), "vocab": cfg.vocab},
            "args": vars(args), "stream_columns": order, "train_files": len(tr_files), "val_files": len(va_files),
            "train_positions": int(len(train)), "train_hours": round(len(train) / args.rate / 3600, 1),
            "train_window_starts": int(len(tr_ok)) if tr_ok is not None else None, "steps_per_s_audio": args.rate,
            "val_positions": int(len(val)), "val_files_names": [f.stem for f in va_files],
            "units": args.units_desc,
            "git": os.environ.get("DY21_COMMIT", "")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "val_files_names"}), flush=True)
    run = airon.init(project="dauys", name=args.name, config=meta, experiment=args.experiment,
                     tags=[args.name.split("-")[0], f"seed-{args.seed}"], total_steps=args.steps, run_id=meta["run_id"],
                     mode="disabled" if args.tracking == "off" else None)

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        tot, per = 0.0, np.zeros(cfg.streams)
        nb = 0
        for i in range(0, len(vbatch), 16):
            ids = vbatch[i : i + 16].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, part = loss(model, ids, PAD)
            tot += part["ce"]
            per += np.array([part[f"ce_s{k}"] for k in range(cfg.streams)])
            nb += 1
        model.train()
        return {"ce": tot / nb, **{f"ce_c{order[k]}": float(per[k] / nb) for k in range(cfg.streams)}}

    def checkpoint(step: int) -> bool:
        nonlocal best
        res = validate()
        run.log({f"val/{k}": v for k, v in res.items()}, step=step)
        print(json.dumps({"step": step, "val": res}), flush=True)
        if res["ce"] < best:
            best = res["ce"]
            ck = {"model": model.state_dict(), "cfg": {k: v for k, v in vars(cfg).items()}, "order": order,
                  "step": step, "val": res}
            torch.save(ck, out / "best.tmp")
            os.replace(out / "best.tmp", out / "best.pt")
            return True
        return False

    best, t0 = math.inf, time.time()
    stale, snapped, t_begin = 0, False, time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        starts = (tr_ok[rng.integers(0, len(tr_ok), size=args.batch)] if tr_ok is not None
                  else rng.integers(0, len(train) - T - 1, size=args.batch))
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in starts]).astype(np.int64)).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        chunks = ids.split(args.micro)
        logs = {}
        for c in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, part = loss(model, c, PAD)
            (total / len(chunks)).backward()
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
        late = stop_at is not None and dt.datetime.now(dt.timezone.utc) >= stop_at
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps or late:
            stale = 0 if checkpoint(step + 1) else stale + 1
            if args.snapshot_min and not snapped and time.time() - t_begin > args.snapshot_min * 60 and (out / "best.pt").exists():
                import shutil
                shutil.copy(out / "best.pt", out / "snapshot.pt")
                snapped = True
                print(json.dumps({"snapshot": step + 1}), flush=True)
            if args.patience and stale >= args.patience:
                print(json.dumps({"early_stop": step + 1, "best_ce": best}), flush=True)
                break
        if late:
            print(json.dumps({"stopped_at_utc": args.stop_utc, "step": step + 1}), flush=True)
            break
    (out / "DONE").write_text(f"{best}\n")
    run.finish()
    print("TRAIN DONE", best, flush=True)


if __name__ == "__main__":
    main()
