"""DY-32 track B: one-voice units over DuplexChat en-16k shards (the dy30_units.py pass with the dy32b channel).

Per shard: pieces >= 5 s, each cut into 20 s windows, WavLM-Large layer-6 features per window (batched by length
with a padding mask), codes per window by the canonical (LJ) codebook and the control (corpus) codebook, joined per
piece. ``<shard>.npz``: ``tok`` int16 codes by --cb, ``tok_c`` int16 codes by --cb2, ``index`` [n, 2] (start, len).
The next shard downloads in the background; uploads to S3 run in the background; shards already in --dst are skipped.

    python scripts/dy32b_units.py run --part 0/3 --work /opt/dlami/nvme/w0 --cb /data/b/cb/lj_k2000_r25.pt \
        --cb2 /data/b/cb/corpus_k2000_r25.pt
    python scripts/dy32b_units.py known --shard <s> --work ... --cb ...   # fp16 vs fp32 code agreement
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import SRC, fetch, listing, sh
from dy32b_channel import SR, Channel, Codebook

DST = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32/b/units"
WIN = 20 * SR
CHUNK = 256  # windows converted to float per step


def shard_codes(ch: Channel, cbs: list[Codebook], pieces: list[np.ndarray], half: bool,
                batch: int = 32) -> list[list[np.ndarray]]:
    wins = [(k, i) for k, w in enumerate(pieces) for i in range(0, len(w), WIN)]
    per = [[[] for _ in pieces] for _ in cbs]
    for b in range(0, len(wins), CHUNK):
        part = wins[b : b + CHUNK]
        x = [pieces[k][i : i + WIN].astype(np.float32) / 32768.0 for k, i in part]
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            fs = ch.feats(x, batch=batch)
        for (k, _), f in zip(part, fs):
            for c, cb in enumerate(cbs):
                per[c][k].append(cb.encode(f.float()).cpu().numpy())
    return [[np.concatenate(r) if r else np.zeros(0, np.int64) for r in pc] for pc in per]


def save(path: Path, toks: list[list[np.ndarray]]) -> int:
    index, pos = [], 0
    for t in toks[0]:
        index.append((pos, len(t)))
        pos += len(t)
    cat = lambda ts: np.concatenate([t.astype(np.int16) for t in ts]) if ts else np.zeros(0, np.int16)
    extra = {"tok_c": cat(toks[1])} if len(toks) > 1 else {}
    np.savez(path, tok=cat(toks[0]), index=np.asarray(index, np.int64).reshape(-1, 2), **extra)
    return pos


def load(a) -> tuple[Channel, list[Codebook]]:
    ch = Channel()
    return ch, [Codebook(a.cb)] + ([Codebook(a.cb2)] if a.cb2 else [])


def cmd_run(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ch, cbs = load(a)
    done = set(listing(a.dst, dirs=False))
    if a.shards:
        mine = [s for s in a.shards.split(",") if s not in done]
    else:
        k, n = map(int, a.part.split("/"))
        allsh = [s for s in listing(a.src, dirs=True) if a.with_held or not held_out(s)]
        if a.order_seed:
            allsh = [allsh[i] for i in np.random.default_rng(a.order_seed).permutation(len(allsh))]
        mine = [s for s in allsh[k::n] if s not in done]
    if a.max_shards:
        mine = mine[: a.max_shards]
    print(json.dumps({"part": a.part, "todo": len(mine), "already_in_dst": len(done)}), flush=True)
    dl, up = ThreadPoolExecutor(1), ThreadPoolExecutor(2)
    uploads, failed = [], []
    lock = threading.Lock()

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
            if a.keep:
                f.rename(Path(a.keep) / f.name)
            else:
                f.unlink()
        except Exception as e:
            with lock:
                failed.append((s, str(e)[-200:]))

    if a.keep:
        Path(a.keep).mkdir(parents=True, exist_ok=True)
    nxt = dl.submit(fetch, a.src, mine[0], work) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        if a.stop_at and time.time() > a.stop_at:
            print("STOP TIME REACHED", flush=True)
            break
        t0 = time.time()
        s, pieces, secs = nxt.result()
        wait = time.time() - t0
        nxt = dl.submit(fetch, a.src, mine[j + 1], work) if j + 1 < len(mine) else None
        toks = shard_codes(ch, cbs, pieces, a.half, a.batch)
        out = work / f"{s}.npz"
        n_tok = save(out, toks)
        del pieces, toks
        uploads.append(up.submit(upload, out, s))
        el = time.time() - t0
        h_all += secs / 3600
        print(json.dumps({"part": a.part, "shard": s, "held": held_out(s), "done": j + 1, "of": len(mine),
                          "hours": round(secs / 3600, 2), "codes": n_tok, "x_realtime": round(secs / el, 1),
                          "waited_for_download_s": round(wait, 1), "hours_total": round(h_all, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    for f in uploads:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"B UNITS DONE part {a.part}", round(h_all, 1), "h", flush=True)


def cmd_known(a) -> None:
    """Code agreement of the fp16 pass vs fp32 on the first ~200 windows of a shard (decides --half)."""
    ch, cbs = load(a)
    _, pieces, _ = fetch(a.src, a.shard, Path(a.work))
    use, n = [], 0
    for w in pieces:
        use.append(w)
        n += -(-len(w) // WIN)
        if n >= 200:
            break
    t = time.time()
    full = shard_codes(ch, cbs, use, False)
    t32 = time.time() - t
    t = time.time()
    half = shard_codes(ch, cbs, use, True)
    t16 = time.time() - t
    secs = sum(map(len, use)) / SR
    agree = [float(np.mean(np.concatenate(x) == np.concatenate(y))) for x, y in zip(full, half)]
    print("KNOWN", json.dumps({"windows": n, "agree_fp16_vs_fp32": agree, "x_realtime_fp32": round(secs / t32, 1),
                               "x_realtime_fp16": round(secs / t16, 1)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "known"):
        x = s.add_parser(name)
        x.add_argument("--work", required=True)
        x.add_argument("--cb", required=True)
        x.add_argument("--cb2", default="")
        x.add_argument("--src", default=SRC)
        x.add_argument("--half", action="store_true")
        x.add_argument("--batch", type=int, default=32, help="windows per WavLM forward (lower = less GPU memory)")
    x = s.choices["run"]
    x.add_argument("--part", default="0/1")
    x.add_argument("--shards", default="")
    x.add_argument("--dst", default=DST)
    x.add_argument("--max-shards", type=int, default=0)
    x.add_argument("--with-held", action="store_true", help="also held-out shards (default: training shards only)")
    x.add_argument("--order-seed", type=int, default=0, help="shuffle the shard list (spread over sources)")
    x.add_argument("--stop-at", type=float, default=0, help="unix time: stop starting new shards after it")
    x.add_argument("--keep", default="", help="move uploaded npz here instead of deleting")
    s.choices["known"].add_argument("--shard", required=True)
    a = p.parse_args()
    {"run": cmd_run, "known": cmd_known}[a.cmd](a)


if __name__ == "__main__":
    main()
