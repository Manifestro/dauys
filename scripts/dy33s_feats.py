"""DY-33 track S: WavLM-Large layer-6 features (kNN-VC, 50 Hz, 1024-d) of every piece of DuplexChat shards, to disk.

Same features as DY-32 track C's units (dy32c_units.RVQUnits.feats: raw waveform, 20 s windows + 80 samples
look-ahead, fp16, padding mask), but kept as features so the slow codec can be trained and applied without running
WavLM again. Every piece >= 1 s is kept (not only >= 5 s), with its pieces.jsonl line, so episodes can be rebuilt as
contiguous audio. Output per shard in --out/<shard>/: feats.npy fp16 [frames, 1024], index.npy [pieces, 2]
(start frame, frames), meta.jsonl (the pieces.jsonl line of each kept piece + "piece": its number in the shard).

    python scripts/dy33s_feats.py --part 0/2 --out /opt/dlami/nvme/feats --work /opt/dlami/nvme/fw0
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import BUCKET, SRC, listing, sh
from dy32c_units import HOP, SR, WIN, RVQUnits, piece_windows


def fetch(s: str, work: Path):
    d = work / s
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin", "pieces.jsonl"):
        sh(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors")
    idx = np.load(d / "index.npy")
    segs = np.fromfile(d / "segs.bin", dtype=np.int16)
    meta = [json.loads(l) for l in open(d / "pieces.jsonl")]
    sh(f"rm -rf {d}")
    keep = [(p, segs[st : st + ln], meta[p]) for p, (st, ln) in enumerate(idx) if ln >= SR]
    return s, keep, sum(len(w) for _, w, _ in keep) / SR


@torch.no_grad()
def piece_feats(u: RVQUnits, pieces: list[np.ndarray], batch: int = 24) -> list[np.ndarray]:
    wins = [(k, x) for k, w in enumerate(pieces) for x in piece_windows(w)]
    order = sorted(range(len(wins)), key=lambda i: -len(wins[i][1]))
    out = {}
    for b in range(0, len(order), batch):
        part = order[b : b + batch]
        f, n = u.feats([wins[i][1].astype(np.float32) / 32768.0 for i in part])
        f = f.half().cpu().numpy()
        for j, i in enumerate(part):
            out[i] = f[j, : min(n[j], WIN // HOP)].copy()
    per = [[] for _ in pieces]
    for i, (k, _) in enumerate(wins):
        per[k].append(i)
    res = []
    for idx in per:  # concatenate piece by piece and free the window copies at once (bounded memory)
        res.append(np.concatenate([out.pop(i) for i in idx]) if idx else np.zeros((0, 1024), np.float16))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="0/1")
    p.add_argument("--shards", default="", help="comma list instead of the training shards of DY-32 track C")
    p.add_argument("--n-train", type=int, default=150, help="DY-32 track C: first 50 of each of 3 parts")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    a = p.parse_args()
    out, work = Path(a.out), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.shards:
        todo = a.shards.split(",")
    else:  # the exact DY-32 c training set = the non-held-out shards of its units-rvq4 folder
        todo = sorted(s for s in listing(f"{BUCKET}/dauys/dy32/c/units-rvq4", dirs=False) if not held_out(s))
        assert len(todo) == a.n_train, len(todo)
        k, n = map(int, a.part.split("/"))
        todo = todo[k::n]
    todo = [s for s in todo if not (out / s / "DONE").exists()]
    print(json.dumps({"part": a.part, "todo": len(todo)}), flush=True)
    u = RVQUnits(torch.device("cuda"))
    dl = ThreadPoolExecutor(1)
    nxt = dl.submit(fetch, todo[0], work) if todo else None
    t_all, h_all = time.time(), 0.0
    for j, s in enumerate(todo):
        t0 = time.time()
        _, keep, secs = nxt.result()
        nxt = dl.submit(fetch, todo[j + 1], work) if j + 1 < len(todo) else None
        fs = piece_feats(u, [w for _, w, _ in keep])
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        index, pos = [], 0
        for f in fs:
            index.append((pos, len(f)))
            pos += len(f)
        mm = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16, shape=(pos, 1024))
        for (st, ln), f in zip(index, fs):
            mm[st : st + ln] = f
        mm.flush()
        del mm
        np.save(d / "index.npy", np.asarray(index, np.int64))
        with open(d / "meta.jsonl", "w") as fh:
            for (pn, _, m) in keep:
                fh.write(json.dumps({**m, "piece": pn}) + "\n")
        (d / "DONE").write_text("1")
        h_all += secs / 3600
        print(json.dumps({"shard": s, "held": held_out(s), "done": j + 1, "of": len(todo), "hours": round(secs / 3600, 2),
                          "frames": pos, "x_realtime": round(secs / (time.time() - t0), 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1)}), flush=True)
    print("FEATS DONE", a.part, round(h_all, 1), "h", flush=True)


if __name__ == "__main__":
    main()
