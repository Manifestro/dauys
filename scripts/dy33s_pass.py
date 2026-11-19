"""DY-33 track S: fused unit pass, audio -> layer-6 features -> frozen slow codec -> units, without storing features.

Same numbers as dy33s_feats.py followed by dy33s_units.py (features rounded to fp16 exactly as they were stored there,
the same per-piece codec call, the same runs), but a shard is processed in groups of pieces of at most --group-h hours
of audio, so only one group's features are in memory at a time (the two-step pass held whole-shard features in RAM,
~20 GB per worker). Shards already in the S3 units folder are skipped; each finished shard is uploaded at once.
The worker's RSS is logged with every shard.

    python scripts/dy33s_pass.py --part 0/3 --codec /data/codec/frozen.pt --out /opt/dlami/nvme/units --work /opt/dlami/nvme/w0
"""

import argparse
import json
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy21_train import held_out
from dy30_units import BUCKET, listing, sh
from dy32c_units import SR, RVQUnits
from dy33s_codec import load_codec
from dy33s_feats import fetch, piece_feats
from dy33s_units import DST, piece_codes, runs_of


def rss_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * resource.getpagesize() / 2**30
    except OSError:
        return -1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="0/1")
    p.add_argument("--shards", default="", help="comma list instead of the DY-32 track C training shards + held-out")
    p.add_argument("--codec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--group-h", type=float, default=1.0)
    p.add_argument("--max-gap", type=float, default=0.05)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if a.shards:
        todo = a.shards.split(",")
    else:
        todo = sorted(s for s in listing(f"{BUCKET}/dauys/dy32/c/units-rvq4", dirs=False) if not held_out(s))
        assert len(todo) == 150, len(todo)
        todo = todo + ["la-00030", "la-00071"]
    k, n = map(int, a.part.split("/"))
    todo = todo[k::n]
    have = set(listing(DST, dirs=False))
    todo = [s for s in todo if s not in have and not (out / f"{s}.npz").exists()]
    print(json.dumps({"part": a.part, "todo": len(todo), "shards": todo}), flush=True)
    if not todo:
        print("PASS DONE", a.part, 0, "h", flush=True)
        return
    dev = torch.device("cuda")
    u = RVQUnits(dev)
    m, _ = load_codec(Path(a.codec), dev)
    dl = ThreadPoolExecutor(1)
    nxt = dl.submit(fetch, todo[0], work)
    t_all, h_all = time.time(), 0.0
    for j, s in enumerate(todo):
        t0 = time.time()
        _, keep, secs = nxt.result()
        nxt = dl.submit(fetch, todo[j + 1], work) if j + 1 < len(todo) else None
        codes, grp, gs = [], [], 0
        lim = int(a.group_h * 3600 * SR)
        for i, (_, w, _) in enumerate(keep):
            grp.append(w)
            gs += len(w)
            if gs >= lim or i == len(keep) - 1:
                for f in piece_feats(u, grp):
                    codes.append(piece_codes(m, f, dev))
                grp, gs = [], 0
        meta = [{**mm, "piece": pn} for pn, _, mm in keep]
        del keep
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
        h_all += secs / 3600
        lens = np.array([x[1] for x in index])
        print(json.dumps({"shard": s, "held": held_out(s), "done": j + 1, "of": len(todo), "steps": pos, "runs": len(runs),
                          "runs_ge_2049": int((lens >= 2049).sum()),
                          "hours_in_runs_ge_2049": round(float(lens[lens >= 2049].sum()) / 12.5 / 3600, 2),
                          "hours": round(secs / 3600, 2), "x_realtime": round(secs / (time.time() - t0), 1),
                          "x_realtime_total": round(h_all * 3600 / (time.time() - t_all), 1),
                          "rss_gb": round(rss_gb(), 1), "utc": time.strftime("%H:%M", time.gmtime())}), flush=True)
    print("PASS DONE", a.part, round(h_all, 1), "h", flush=True)


if __name__ == "__main__":
    main()
