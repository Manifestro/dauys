"""DY-30 data: oracle phonemes over every DuplexChat en-16k shard (training and held-out), on several machines.

Tokens are exactly phon_units.py's: the same recogniser (phon_units.Phonemes, unchanged), the same pieces (>= 5 s),
the same 20 s windows in the same order and the same batches of 32 consecutive windows of the shard, so the output
``<shard>.npz`` (``tok`` int16, ``index`` [n, 2] start, len) is what phon_units.py would write. Only the plumbing
differs: windows are converted batch by batch (no float copy of the whole shard), the next shard is downloaded and
read in a background thread while the current one is on the GPU, and uploads to S3 run in the background.

``run --part k/n`` takes every n-th shard of the sorted list from k (fixed assignment, so machines never overlap);
shards already in ``--dst`` are skipped, so a rerun resumes. ``--shards a,b`` processes just these.
``known`` checks the batching against phon_units.Phonemes on the first >= 200 windows of a shard (must be 100 %).

    python scripts/dy30_units.py run --part 0/8 --work /opt/dlami/nvme/w0
    python scripts/dy30_units.py known --shard la-00030 --work /opt/dlami/nvme/known --out known.json
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from phon_units import SR, WIN, Phonemes

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs"
SRC = f"{BUCKET}/duplexchat/en-16k"
DST = f"{BUCKET}/dauys/phon-units"
BATCH = 32  # phon_units.Phonemes.windows default; part of the token definition


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout


def listing(prefix: str, dirs: bool) -> list[str]:
    out = sh(f"aws s3 ls {prefix}/ || true")
    if dirs:
        return sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))
    return sorted(l.split()[-1].removesuffix(".npz") for l in out.splitlines() if l.strip().endswith(".npz"))


def fetch(src: str, s: str, work: Path) -> tuple[str, list[np.ndarray], float]:
    """Download one shard and return its pieces >= 5 s (int16 views) as phon_units.py selects them."""
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin"):
        sh(f"aws s3 cp {src}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    sh(f"rm -rf {d}")
    pieces = [segs[st : st + ln] for st, ln in idx if ln >= 5 * SR]
    return s, pieces, sum(len(w) for w in pieces) / SR


def shard_tokens(ph: Phonemes, pieces: list[np.ndarray]) -> list[np.ndarray]:
    """Phoneme ids per piece; windows and batches exactly as phon_units.main (all windows of the shard in order,
    32 per call), each window converted only when its batch is due."""
    wins = [(k, i) for k, w in enumerate(pieces) for i in range(0, len(w), WIN)]
    per = [[] for _ in pieces]
    for b in range(0, len(wins), BATCH):
        part = wins[b : b + BATCH]
        res = ph.windows([pieces[k][i : i + WIN].astype(np.float32) / 32768.0 for k, i in part], batch=BATCH)
        for (k, _), r in zip(part, res):
            per[k].append(r)
    return [np.concatenate(r) if r else np.zeros(0, np.int64) for r in per]


def save(path: Path, toks: list[np.ndarray]) -> int:
    index, pos = [], 0
    for t in toks:
        index.append((pos, len(t)))
        pos += len(t)
    tok = np.concatenate([t.astype(np.int16) for t in toks]) if toks else np.zeros(0, np.int16)
    np.savez(path, tok=tok, index=np.asarray(index, np.int64).reshape(-1, 2))
    return pos


def cmd_run(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ph = Phonemes(torch.device("cuda"))
    done = set(listing(a.dst, dirs=False))
    if a.shards:
        mine = [s for s in a.shards.split(",") if s not in done]
    else:
        k, n = map(int, a.part.split("/"))
        mine = [s for s in listing(a.src, dirs=True)[k::n] if s not in done]
    if a.max_shards:
        mine = mine[: a.max_shards]
    print(json.dumps({"part": a.part, "todo": len(mine), "already_in_dst": len(done)}), flush=True)
    dl = ThreadPoolExecutor(1)
    up = ThreadPoolExecutor(2)
    uploads, failed = [], []
    lock = threading.Lock()

    def upload(f: Path, s: str) -> None:
        try:
            sh(f"aws s3 cp {f} {a.dst}/{s}.npz --only-show-errors")
            f.unlink()
        except Exception as e:  # reported at the end; the shard is redone on the next run
            with lock:
                failed.append((s, str(e)[-200:]))

    nxt = dl.submit(fetch, a.src, mine[0], work) if mine else None
    t_all, h_all = time.time(), 0.0
    for j in range(len(mine)):
        t0 = time.time()
        s, pieces, secs = nxt.result()
        wait = time.time() - t0
        nxt = dl.submit(fetch, a.src, mine[j + 1], work) if j + 1 < len(mine) else None
        toks = shard_tokens(ph, pieces)
        out = work / f"{s}.npz"
        n_tok = save(out, toks)
        del pieces, toks
        uploads.append(up.submit(upload, out, s))
        el = time.time() - t0
        h_all += secs / 3600
        print(json.dumps({"part": a.part, "shard": s, "held": held_out(s), "done": j + 1, "of": len(mine),
                          "hours": round(secs / 3600, 2), "phonemes": n_tok, "per_s": round(n_tok / max(secs, 1), 1),
                          "x_realtime": round(secs / el, 1), "waited_for_download_s": round(wait, 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    for f in uploads:
        f.result()
    if failed:
        print("UPLOAD FAILED", json.dumps(failed), flush=True)
        sys.exit(1)
    print(f"PHON UNITS DONE part {a.part}", round(h_all, 1), "h", flush=True)


def cmd_known(a) -> None:
    """New pass vs phon_units.Phonemes on the first pieces of a shard holding >= 200 windows: the reference is the
    phon_units way (all windows of those pieces in one list through Phonemes.windows), compared window by window and
    piece by piece."""
    ph = Phonemes(torch.device("cuda"))
    _, pieces, _ = fetch(a.src, a.shard, Path(a.work))
    use, n_win = [], 0
    for w in pieces:
        use.append(w)
        n_win += -(-len(w) // WIN)
        if n_win >= a.windows:
            break
    wins, owner = [], []
    for k, w in enumerate(use):
        x = w.astype(np.float32) / 32768.0
        for i in range(0, len(x), WIN):
            wins.append(x[i : i + WIN])
            owner.append(k)
    ref = ph.windows(wins)
    ref_piece = [np.concatenate([r for o, r in zip(owner, ref) if o == k]) for k in range(len(use))]
    new_piece = shard_tokens(ph, use)
    same_pieces = [bool(np.array_equal(x, y)) for x, y in zip(ref_piece, new_piece)]
    res = {"shard": a.shard, "windows": len(wins), "pieces": len(use), "phonemes": int(sum(len(x) for x in ref_piece)),
           "identical_pieces": round(float(np.mean(same_pieces)), 4), "pass": bool(all(same_pieces) and len(wins) >= a.windows)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("KNOWN TOKENS", json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("run")
    x.add_argument("--part", default="0/1", help="k/n: every n-th shard from k")
    x.add_argument("--shards", default="", help="comma list instead of --part")
    x.add_argument("--work", required=True)
    x.add_argument("--src", default=SRC)
    x.add_argument("--dst", default=DST)
    x.add_argument("--max-shards", type=int, default=0)
    x = s.add_parser("known")
    x.add_argument("--shard", required=True)
    x.add_argument("--work", required=True)
    x.add_argument("--src", default=SRC)
    x.add_argument("--windows", type=int, default=200)
    x.add_argument("--out", required=True)
    a = p.parse_args()
    {"run": cmd_run, "known": cmd_known}[a.cmd](a)


if __name__ == "__main__":
    main()
