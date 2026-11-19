"""DY-33 track S unit pass: frozen slow codec over the stored layer-6 features -> 12.5 Hz units, 4 codes per step.

Per shard (features from dy33s_feats.py): every piece is encoded on its own (features padded to a multiple of 4 by
repeating the last frame), then pieces are joined into RUNS of contiguous audio of one episode: consecutive pieces of
the shard with the same "ep" whose start equals the previous piece's end (|gap| <= --max-gap s). A run is the unit of
training sequences (the trainer samples windows only inside runs long enough for its context).
Output <shard>.npz: tok int16 [steps, 4], index [runs, 2] (start step, steps), runs.json-like array ``meta``
(episode, start s, end s, pieces) stored as a JSON string. Uploaded to dy33/s/units/.

    python scripts/dy33s_units.py --part 0/3 --codec /data/codec/frozen.pt --feats /opt/dlami/nvme/feats --out /opt/dlami/nvme/units
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import sh
from dy33s_codec import load_codec

DST = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy33/s/units"


@torch.no_grad()
def piece_codes(m, f: np.ndarray, dev, chunk: int = 16384) -> np.ndarray:
    """[T, 1024] fp16 -> [ceil(T/4), 4] codes (whole piece; very long pieces in chunks of 16384 frames)."""
    out = []
    for i in range(0, len(f), chunk):
        x = torch.from_numpy(np.asarray(f[i : i + chunk], np.float32)).to(dev)
        pad = (-len(x)) % 4
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, -1)])
        out.append(m.codes(x[None])[0].cpu().numpy().astype(np.int16))
    return np.concatenate(out) if out else np.zeros((0, 4), np.int16)


def runs_of(meta: list[dict], max_gap: float) -> list[list[int]]:
    runs, cur = [], []
    for i, m in enumerate(meta):
        if cur:
            p = meta[cur[-1]]
            if m["ep"] == p["ep"] and abs(float(m["start"]) - float(p["end"])) <= max_gap:
                cur.append(i)
                continue
            runs.append(cur)
        cur = [i]
    if cur:
        runs.append(cur)
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="0/1")
    p.add_argument("--codec", required=True)
    p.add_argument("--feats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-gap", type=float, default=0.05)
    p.add_argument("--expect", type=int, default=152, help="wait until this many feature shards are DONE")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    m, ck = load_codec(Path(a.codec), dev)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.part.split("/"))
    done_here: set[str] = set()
    t_all, steps_all = time.time(), 0
    while True:
        ready = sorted(d.name for d in Path(a.feats).iterdir() if (d / "DONE").exists())
        mine = [s for i, s in enumerate(ready) if hash_part(s, n) == k and s not in done_here and not (out / f"{s}.npz").exists()]
        if not mine:
            if len(ready) >= a.expect:
                break
            time.sleep(30)
            continue
        s = mine[0]
        t0 = time.time()
        d = Path(a.feats) / s
        F = np.load(d / "feats.npy", mmap_mode="r")
        idx = np.load(d / "index.npy")
        meta = [json.loads(l) for l in open(d / "meta.jsonl")]
        codes = [piece_codes(m, F[st : st + ln], dev) for st, ln in idx]
        runs = runs_of(meta, a.max_gap)
        tok, index, rmeta, pos = [], [], [], 0
        for r in runs:
            c = np.concatenate([codes[i] for i in r])
            tok.append(c)
            index.append((pos, len(c)))
            pos += len(c)
            rmeta.append({"ep": meta[r[0]]["ep"], "start": meta[r[0]]["start"], "end": meta[r[-1]]["end"], "pieces": len(r)})
        f = out / f"{s}.npz"
        np.savez(f.with_suffix(".tmp.npz"), tok=np.concatenate(tok), index=np.asarray(index, np.int64),
                 meta=np.asarray(json.dumps(rmeta)))
        f.with_suffix(".tmp.npz").rename(f)
        sh(f"aws s3 cp {f} {DST}/{s}.npz --only-show-errors")
        done_here.add(s)
        steps_all += pos
        lens = np.array([x[1] for x in index])
        print(json.dumps({"shard": s, "held": held_out(s), "steps": pos, "runs": len(runs), "runs_ge_2049": int((lens >= 2049).sum()),
                          "hours_in_runs_ge_2049": round(float(lens[lens >= 2049].sum()) / 12.5 / 3600, 2),
                          "hours": round(pos / 12.5 / 3600, 2), "sec": round(time.time() - t0, 1)}), flush=True)
    print("UNITS DONE", a.part, round(steps_all / 12.5 / 3600, 1), "h", round(time.time() - t_all), "s", flush=True)


def hash_part(s: str, n: int) -> int:
    return int(s.split("-")[-1]) % n


if __name__ == "__main__":
    main()
