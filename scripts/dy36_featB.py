"""DY-36 dictionary candidate B, feature pass: mean WavLM-Large layer-6 feature of every utterance of the table, each
utterance encoded ALONE (+80 zero samples, as dy35_utt.py), fp16 autocast. Per shard -> <out>/featB/<shard>.npy
[utterances of that shard in table order, 1024] float16. Then `reduce`: minus the speaker's mean in the recording,
truncated SVD to 64 dims fitted on the centred training rows (1/3 of the training recordings, seed 0), L2-normalised
-> dict_B.npz (raw = mean feature projected, thought = centred projection normalised).

    python scripts/dy36_featB.py pass --dict /data/dy36/dict --work /data/w
    python scripts/dy36_featB.py reduce --dict /data/dy36/dict
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
from dy36_common import BUCKET, HELD_DIR, SRC_KEY, log
from dy36_dict import centre, load_table


def fetch(shard: str, work: Path):
    import subprocess

    if (HELD_DIR / shard / "segs.bin").exists():
        d = HELD_DIR / shard
    else:
        d = work / shard
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            subprocess.run(f"aws s3 cp s3://{BUCKET}/{SRC_KEY}/{shard}/{fn} {d}/{fn} --only-show-errors", shell=True, check=True)
    return d


def cmd_pass(a) -> None:
    from dy32b_channel import Channel

    T = load_table(Path(a.dict))
    files = json.loads(str(T["files"]))
    out = Path(a.dict) / "featB"
    out.mkdir(exist_ok=True)
    work = Path(a.work)
    todo = [f for f in range(len(files)) if not (out / f"{files[f]}.npy").exists()]
    if a.part:
        k, n = map(int, a.part.split("/"))
        todo = todo[k::n]
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)  # leave room for the recognisers on a 24 GB GPU
    ch = Channel()
    ex = ThreadPoolExecutor(1)
    nxt = ex.submit(fetch, files[todo[0]], work) if todo else None
    t_all = time.time()
    for j, fi in enumerate(todo):
        d = nxt.result()
        nxt = ex.submit(fetch, files[todo[j + 1]], work) if j + 1 < len(todo) else None
        t0 = time.time()
        rows = np.flatnonzero(T["fi"] == fi)
        idx = np.load(d / "index.npy")
        segs = np.memmap(d / "segs.bin", dtype=np.int16, mode="r")
        res = np.zeros((len(rows), 1024), np.float16)
        order = rows[np.argsort(-T["len"][rows])]
        pos = {int(r): i for i, r in enumerate(rows)}
        sec = 0.0
        for b in range(0, len(order), 256):
            part = order[b : b + 256]
            wavs = []
            for r in part:
                st = int(idx[int(T["piece_orig"][r])][0])
                w = np.asarray(segs[st + int(T["s"][r]) * 320 : st + int(T["e"][r]) * 320], np.float32) / 32768.0
                wavs.append(np.concatenate([w, np.zeros(80, np.float32)]))
                sec += len(w) / 16000
            with torch.autocast("cuda", dtype=torch.float16):
                fs = ch.feats(wavs, batch=a.batch)
            for r, f in zip(part, fs):
                res[pos[int(r)]] = f.mean(0).cpu().numpy().astype(np.float16)
        np.save(out / f"{files[fi]}.npy", res)
        del segs
        if d.parent == work:
            import shutil

            shutil.rmtree(d)
        log(json.dumps({"shard": files[fi], "done": j + 1, "of": len(todo), "utt": len(rows), "audio_h": round(sec / 3600, 2),
                        "x_rt": round(sec / (time.time() - t0), 1), "elapsed_min": round((time.time() - t_all) / 60, 1)}))
    log("PASS B DONE")


def cmd_reduce(a) -> None:
    T = load_table(Path(a.dict))
    files = json.loads(str(T["files"]))
    n = len(T["ep"])
    X = np.zeros((n, 1024), np.float32)
    for fi, f in enumerate(files):
        X[np.flatnonzero(T["fi"] == fi)] = np.load(Path(a.dict) / "featB" / f"{f}.npy").astype(np.float32)
    from dy36_dict import group_mean

    mu, _ = group_mean(X, T["spk"])
    R = (X - mu).astype(np.float32)
    rng = np.random.default_rng(0)
    train = ~T["held"]
    eps = np.unique(T["ep"][train])
    sub = np.flatnonzero(train & np.isin(T["ep"], rng.choice(eps, len(eps) // 3, replace=False)))
    _, S, Vt = np.linalg.svd(R[sub], full_matrices=False)
    V = Vt[:64].T
    raw = X @ V
    th, valid = centre(raw, T["spk"])
    np.savez(Path(a.dict) / "dict_B.npz", raw=raw.astype(np.float32), thought=th, valid=valid, V=V, sv=S[:64])
    log("B DONE valid", float(valid.mean()), "top64 share", float((S[:64] ** 2).sum() / (S ** 2).sum()))


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("pass")
    x.add_argument("--dict", required=True)
    x.add_argument("--work", required=True)
    x.add_argument("--part", default="")
    x.add_argument("--batch", type=int, default=8)
    x.add_argument("--mem-frac", type=float, default=0.4)
    x = s.add_parser("reduce")
    x.add_argument("--dict", required=True)
    a = p.parse_args()
    {"pass": cmd_pass, "reduce": cmd_reduce}[a.cmd](a)


if __name__ == "__main__":
    main()
