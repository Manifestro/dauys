"""DY-41 training: the DY-40 depth LM on Mimi codes, all en-16k training shards, until validation stops improving.

Same model, layout, loss and batch as DY-40 (dy40_lm; acoustic codes one frame after code 0; code-0 weight 10:1;
32 x 2049 positions). Schedule (warmup-stable-decay): warmup --warmup steps, then a constant learning rate; validation
every --val-every steps; when --patience validations in a row bring no new best (improvement < --min-delta), or when
the run is --decay-before-h hours from --max-hours, the learning rate falls linearly to --lr-min over
min(max(1000, 10 % of the steps done), --decay-max) steps, then training stops. Best checkpoint (local best.pt) by
validation loss.

Airon job (airon launch): airon.resume() gives the checkpoint to continue from (model, optimiser, data RNG, schedule
state, best weights, elapsed time); run.checkpoint() uploads a full state every --ckpt-min minutes and once at
--first-ckpt steps (so the dry run has one to continue from). Outside a job it runs the same way with a tracked run.

    python scripts/dy41_train.py --units /opt/dlami/nvme/units --out /data/dy41/run
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import airon
import numpy as np
import torch

from dauys.dy40_lm import DLConfig, DepthLM, loss
from dy21_train import held_out
from dy40_train import load_runs

EXPERIMENT = ("Рецепт DY-40 на Mimi на всех ~19 тыс. ч, обучение до остановки по валидации: появляется ли смысл "
              "продолжения от данных")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1_000_000, help="hard cap on the global step")
    p.add_argument("--more", type=int, default=0, help="stop this many steps after the start (the dry run: 60)")
    p.add_argument("--init-from", default="", help="s3:// state to start from when airon.resume() gives none")
    p.add_argument("--min-start", type=int, default=0,
                   help="fail unless training starts at least at this global step (a resume that silently starts over)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.002)
    p.add_argument("--decay-max", type=int, default=2500)
    p.add_argument("--max-hours", type=float, default=38.0)
    p.add_argument("--decay-before-h", type=float, default=2.0)
    p.add_argument("--ckpt-min", type=float, default=30.0)
    p.add_argument("--first-ckpt", type=int, default=25)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tracking", default="on", choices=["on", "off"])
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = DLConfig()
    T = cfg.ctx

    files = sorted(Path(args.units).glob("*.npz"))
    tr_files = [f for f in files if not held_out(f.stem)]
    va_files = [f for f in files if held_out(f.stem)]
    assert va_files and not {f.stem for f in tr_files} & {f.stem for f in va_files}
    t_load = time.time()
    train, tr_ok = load_runs(tr_files, cfg.k, T)
    val, va_ok = load_runs(va_files, cfg.k, T)
    vstarts = va_ok[np.random.default_rng(1234).integers(0, len(va_ok), size=128)]
    vbatch = torch.from_numpy(np.stack([val[s : s + T + 1] for s in vstarts]).astype(np.int64))
    print(json.dumps({"loaded_s": round(time.time() - t_load), "train_positions": int(len(train))}), flush=True)

    model = DepthLM(cfg).to(device)
    wd = [q for n, q in model.named_parameters() if q.dim() >= 2 and "emb" not in n]
    other = [q for n, q in model.named_parameters() if not (q.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": wd, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    rng = np.random.default_rng(args.seed)
    st = {"step": 0, "best": math.inf, "stale": 0, "phase": "stable", "decay_start": None, "decay_len": None,
          "elapsed": 0.0}

    resume_path = airon.resume()
    if not resume_path and args.init_from:  # no checkpoint from Airon yet: the state another job left in S3
        import subprocess

        local = out / "init_state.pt"
        if not local.exists():
            subprocess.run(["aws", "s3", "cp", args.init_from, str(local), "--only-show-errors"], check=True)
        resume_path = str(local)
    if resume_path:
        f = Path(resume_path)
        if f.is_dir():  # a stored checkpoint may come back as a folder holding the file
            f = max(f.rglob("*.pt"), key=lambda x: x.stat().st_size)
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if "opt" not in ck:  # a weights-only file (best.pt): take the full local state instead
            g = out / "last.pt" if (out / "last.pt").exists() else out / "init_state.pt"
            print(json.dumps({"resume_skipped": str(f), "using": str(g)}), flush=True)
            ck = torch.load(g, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        st = ck["state"]
        if ck.get("best_model") is not None:
            torch.save({"model": ck["best_model"], "cfg": vars(cfg), "step": ck["best_step"], "val": ck["best_val"]},
                       out / "best.pt")
        print(json.dumps({"resumed_from": str(resume_path), "step": st["step"]}), flush=True)
    if st["step"] < args.min_start:
        raise RuntimeError(f"training would start at step {st['step']}, expected >= {args.min_start} (resume: {resume_path})")
    stop_at = min(args.steps, st["step"] + args.more) if args.more else args.steps

    meta = {"params": model.parameters_count(), "config": {**vars(cfg), "vocab": cfg.vocab}, "args": vars(args),
            "train_files": len(tr_files), "val_files": [f.stem for f in va_files],
            "train_positions": int(len(train)), "train_hours": round(len(train) / 12.5 / 3600, 1),
            "units": "kyutai/mimi rev 89091b3e, 8 x 2048, 12.5 Hz; acoustic codes one frame after code 0; EOS + run blocks >= ctx + 1",
            "schedule": "warmup-stable-decay, decay on validation plateau", "git": os.environ.get("DY_COMMIT", "")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    mode = "disabled" if args.tracking == "off" else None
    if os.environ.get("AIRON_JOB"):
        run = airon.init(config=meta, mode=mode)
    else:
        run = airon.init(project="dauys", name="dy41", config=meta, experiment=EXPERIMENT, tags=["dy41"], mode=mode)

    def name(k: str) -> str:  # Airon's standard: the optimised total is loss, its parts loss_<part>
        return {"ce": "loss", "ce_mean": "loss_mean"}.get(k, k.replace("ce_l", "loss_l"))

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

    best_model: dict = {"w": None, "step": None, "val": None}
    if (out / "best.pt").exists():
        b = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        best_model = {"w": {k: v.to(torch.bfloat16) for k, v in b["model"].items()}, "step": b["step"], "val": b["val"]}

    def save_state(tag: str) -> None:
        f = out / "last.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                    "state": {**st, "elapsed": st["elapsed"] + time.time() - t_begin}, "best_model": best_model["w"],
                    "best_step": best_model["step"], "best_val": best_model["val"], "cfg": vars(cfg)}, f.with_suffix(".tmp"))
        os.replace(f.with_suffix(".tmp"), f)
        run.checkpoint(str(f), step=st["step"])
        print(json.dumps({"checkpoint": str(f), "step": st["step"]}), flush=True)

    def lr_now(step: int) -> float:
        if st["phase"] == "stable":
            return args.lr * min(1.0, (step + 1) / args.warmup)
        frac = min(1.0, (step - st["decay_start"]) / st["decay_len"])
        return args.lr + (args.lr_min - args.lr) * frac

    def start_decay(step: int, why: str) -> None:
        st["phase"], st["decay_start"] = "decay", step
        st["decay_len"] = int(min(max(1000, 0.1 * step), args.decay_max))
        print(json.dumps({"decay_from": step, "steps": st["decay_len"], "why": why}), flush=True)
        run.log({"schedule/decay_start": step}, step=step)

    def validation(step: int) -> None:
        res = validate()
        run.log({f"eval/{name(k)}": v for k, v in res.items()}, step=step)
        print(json.dumps({"step": step, "val": res, "utc": time.strftime("%H:%M", time.gmtime())}), flush=True)
        if res["ce"] < st["best"] - args.min_delta or (st["phase"] == "decay" and res["ce"] < st["best"]):
            st["best"], st["stale"] = res["ce"], 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step, "val": res}, out / "best.tmp")
            os.replace(out / "best.tmp", out / "best.pt")
            run.checkpoint(str(out / "best.pt"), step=step, label="best")
            best_model.update(w={k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()},
                              step=step, val=res)
        else:
            st["stale"] += 1
        if st["phase"] == "stable" and st["stale"] >= args.patience:
            start_decay(step, f"{args.patience} validations without a new best")

    t_begin = time.time()
    t_ckpt, t_log = time.time(), time.time()
    first = st["step"]
    stop_reason = "step cap"
    while st["step"] < stop_at:
        step = st["step"]
        for g in opt.param_groups:
            g["lr"] = lr_now(step)
        starts = tr_ok[rng.integers(0, len(tr_ok), size=args.batch)]
        ids = torch.from_numpy(np.stack([train[s : s + T + 1] for s in starts]).astype(np.int64)).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logs: dict[str, float] = {}
        chunks = ids.split(args.micro)
        for c in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot, part = loss(model, c)
            (tot / len(chunks)).backward()
            for k, v in part.items():
                logs[k] = logs.get(k, 0.0) + v / len(chunks)
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        st["step"] = step = step + 1
        if not math.isfinite(logs["ce"]):
            raise RuntimeError(f"loss is not finite at step {step}")
        if step % args.log_every == 0 or step == first + 1:
            logs["lr"] = opt.param_groups[0]["lr"]
            logs["steps_per_s"] = args.log_every / max(time.time() - t_log, 1e-9) if step % args.log_every == 0 else 0.0
            t_log = time.time()
            run.log({f"train/{name(k)}": v for k, v in logs.items()}, step=step, tokens=args.batch * T)
            print(json.dumps({"step": step, **{k: round(v, 4) for k, v in logs.items()}}), flush=True)
        hours = (st["elapsed"] + time.time() - t_begin) / 3600
        if st["phase"] == "stable" and hours > args.max_hours - args.decay_before_h:
            start_decay(step, f"{args.max_hours} h limit")
        if step % args.val_every == 0:
            validation(step)
        if st["phase"] == "decay" and step >= st["decay_start"] + st["decay_len"]:
            if step % args.val_every:
                validation(step)
            stop_reason = "decay finished"
            save_state("final")
            break
        if step == first + args.first_ckpt or time.time() - t_ckpt > args.ckpt_min * 60:
            save_state("periodic")
            t_ckpt = time.time()
    if stop_reason == "step cap":
        validation(st["step"])
    (out / "DONE").write_text(json.dumps({"best": st["best"], "stop": stop_reason, "step": st["step"]}) + "\n")
    run.finish()
    print("TRAIN DONE", st["best"], stop_reason, st["step"], flush=True)


if __name__ == "__main__":
    main()
