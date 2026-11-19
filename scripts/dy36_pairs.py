"""DY-36 candidate D, pair pass: the same utterance in another voice, in WavLM feature space (kNN-VC style), no text.

Per audio shard (all its table utterances, each encoded ALONE + 80 zero samples, as dy35_utt.py; fp16 autocast):
  features  WavLM-Large layer 6, 50 Hz [T, 1024];
  pools     speakers of the shard with >= POOL_MIN frames over their utterances; pool = their frames (utterances in a
            seeded order up to POOL_MAX frames); voice vectors: pool mean feature (1024) and pool code histogram (K);
  pairs     every utterance gets NV target voices: speakers of OTHER recordings of the shard (seeded); each source frame
            is replaced by the mean of its k nearest pool frames (cosine), the result is resampled in time by a factor
            uniform in [TMIN, TMAX] (linear interpolation of features), then quantised by the same K2000 codebook
            (cosine argmax, as Codebook.encode).
Also logged: agreement of the re-quantised source features with the stored table codes (sanity of the feature path).
Output <out>/pairs/<shard>.npz: uid [n] (table rows), tgt [n, NV] (speaker ids = table spk), tempo [n, NV],
codes (int16, concatenated in order i * NV + v), off [n * NV + 1]; spk [m], vfeat [m, 1024] f16, vhist [m, K] f16,
vframes [m]. Uploaded to S3 as each shard completes.

    python scripts/dy36_pairs.py --dict /data/dy36/dict --out /data/dy36/d --work /data/w --n-train 250000
"""

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dy36_common import BUCKET, SRC_KEY, log
from dy36_dict import load_table

K, KNN, NV = 2000, 4, 2
POOL_MIN, POOL_MAX = 3000, 7500
TMIN, TMAX = 0.85, 1.15
S3D = f"s3://{BUCKET}/outputs/dauys/dy36/d"


def fetch(shard: str, work: Path) -> Path:
    d = work / shard
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("index.npy", "segs.bin"):
        subprocess.run(f"aws s3 cp s3://{BUCKET}/{SRC_KEY}/{shard}/{fn} {d}/{fn} --only-show-errors", shell=True, check=True)
    return d


def pick_shards(T: dict, files: list[str], n_train: int, seed: int) -> list[int]:
    """First held-out shard (by name), then training shards in a seeded order until n_train utterances."""
    cnt = np.bincount(T["fi"], minlength=len(files))
    held = sorted({int(f) for f in np.unique(T["fi"][T["held"]])}, key=lambda f: files[f])
    train = [f for f in range(len(files)) if f not in held]
    rng = np.random.default_rng(seed)
    out, tot = held[:1], 0
    for f in rng.permutation(train):
        if tot >= n_train:
            break
        out.append(int(f))
        tot += int(cnt[f])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dict", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--cb", default="/data/cb/km-corpus-K2000.npy")
    p.add_argument("--n-train", type=int, default=250000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-sec", type=float, default=240.0)
    p.add_argument("--mem-frac", type=float, default=0.45)
    p.add_argument("--part", default="", help="k/n: this machine takes every n-th shard of the list from k")
    p.add_argument("--skip", type=int, default=0, help="skip this many shards of the seeded list (already done elsewhere)")
    a = p.parse_args()
    from dy32b_channel import Channel

    T = load_table(Path(a.dict))
    files = json.loads(str(T["files"]))
    out = Path(a.out) / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    shards = pick_shards(T, files, a.n_train, a.seed)[a.skip :]
    if a.part:
        k, n = map(int, a.part.split("/"))
        shards = shards[k::n]
    shards = [f for f in shards if not (out / f"{files[f]}.npz").exists()]
    log("shards", len(shards), [files[f] for f in shards][:5])
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    ch = Channel()
    cent = F.normalize(torch.from_numpy(np.load(a.cb).astype(np.float32)).cuda(), dim=-1).half()

    def quant(f: torch.Tensor) -> torch.Tensor:
        x = F.normalize(f.float(), dim=-1).half()
        return torch.cat([(c @ cent.T).argmax(-1) for c in x.split(32768)])

    # bounded prefetch (1 shard ahead) and background uploads
    q: queue.Queue = queue.Queue(maxsize=1)

    def producer() -> None:
        for f in shards:
            q.put((f, fetch(files[f], work)))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    upq: queue.Queue = queue.Queue(maxsize=4)

    def uploader() -> None:
        while (x := upq.get()) is not None:
            subprocess.run(f"aws s3 cp {x} {S3D}/pairs/{x.name} --only-show-errors", shell=True)

    up = threading.Thread(target=uploader)
    up.start()
    t_all, done = time.time(), 0
    co, off = T["codes"], T["code_off"]
    while (item := q.get()) is not None:
        fi, d = item
        t0 = time.time()
        rows = np.flatnonzero(T["fi"] == fi)
        idx = np.load(d / "index.npy")
        segs = np.memmap(d / "segs.bin", dtype=np.int16, mode="r")
        feats: list = [None] * len(rows)
        order = np.argsort(-T["len"][rows])
        agree, tot, sec = 0, 0, 0.0
        b = 0
        while b < len(order):
            L = T["len"][rows[order[b]]] / 50
            nb = max(1, int(a.batch_sec // max(L, 1.0)))
            part = order[b : b + nb]
            b += nb
            wavs = []
            for j in part:
                r = rows[j]
                st = int(idx[int(T["piece_orig"][r])][0])
                w = np.asarray(segs[st + int(T["s"][r]) * 320 : st + int(T["e"][r]) * 320], np.float32) / 32768.0
                wavs.append(np.concatenate([w, np.zeros(80, np.float32)]))
                sec += len(w) / 16000
            with torch.autocast("cuda", dtype=torch.float16):
                fs = ch.feats(wavs, batch=len(wavs))
            for j, f in zip(part, fs):
                r = rows[j]
                c = quant(f).cpu().numpy()
                ref = co[off[r] : off[r + 1]]
                n = min(len(c), len(ref))
                agree += int((c[:n] == ref[:n]).sum())
                tot += max(len(c), len(ref))
                feats[j] = f.half().cpu()
        del segs
        shutil.rmtree(d)
        t_feat = time.time() - t0
        # pools
        rng = np.random.default_rng(a.seed * 100003 + fi)
        spk = T["spk"][rows]
        sp_u = np.unique(spk)
        pools, sp_ok = {}, []
        for s in sp_u:
            js = rng.permutation(np.flatnonzero(spk == s))
            if T["len"][rows[js]].sum() < POOL_MIN:
                continue
            take, n = [], 0
            for j in js:
                if n >= POOL_MAX:
                    break
                take.append(j)
                n += feats[j].shape[0]
            pools[int(s)] = take
            sp_ok.append(int(s))
        sp_ok = np.array(sp_ok, np.int64)
        sp_ep = sp_ok // 64
        vfeat = np.zeros((len(sp_ok), 1024), np.float16)
        vhist = np.zeros((len(sp_ok), K), np.float16)
        vframes = np.zeros(len(sp_ok), np.int64)
        for k, s in enumerate(sp_ok):
            P = torch.cat([feats[j] for j in pools[s]])
            vfeat[k] = P.float().mean(0).numpy().astype(np.float16)
            h = np.bincount(np.concatenate([co[off[rows[j]] : off[rows[j] + 1]] for j in pools[s]]).astype(np.int64), minlength=K)
            vhist[k] = (h / h.sum()).astype(np.float16)
            vframes[k] = len(P)
        # targets: NV distinct speakers of other recordings
        ep = T["ep"][rows]
        tgt = np.full((len(rows), NV), -1, np.int64)
        for j in range(len(rows)):
            el = np.flatnonzero(sp_ep != ep[j])
            if len(el) >= NV:
                tgt[j] = sp_ok[rng.choice(el, NV, replace=False)]
        tempo = rng.uniform(TMIN, TMAX, size=(len(rows), NV)).astype(np.float32)
        conv: list = [None] * (len(rows) * NV)
        for k, s in enumerate(sp_ok):
            jv = np.argwhere(tgt == s)
            if not len(jv):
                continue
            P = torch.cat([feats[j] for j in pools[s]]).cuda()
            Pn = F.normalize(P.float(), dim=-1).half()
            X = torch.cat([feats[j] for j, _ in jv]).cuda()
            lens = [feats[j].shape[0] for j, _ in jv]
            Y = torch.empty_like(X)
            for c0 in range(0, len(X), 8192):
                sim = F.normalize(X[c0 : c0 + 8192].float(), dim=-1).half() @ Pn.T
                nn = sim.topk(KNN, dim=-1).indices
                Y[c0 : c0 + 8192] = P[nn].float().mean(1).half()
            for (j, v), y in zip(jv, Y.split(lens)):
                n2 = max(2, int(round(len(y) * tempo[j, v])))
                y2 = F.interpolate(y.float().T[None], size=n2, mode="linear", align_corners=True)[0].T
                conv[j * NV + v] = quant(y2).cpu().numpy().astype(np.int16)
        ok = (tgt >= 0).all(1)
        uid = rows[ok]
        keep = [i * NV + v for i in np.flatnonzero(ok) for v in range(NV)]
        cl = [conv[i] for i in keep]
        res = {"uid": uid, "tgt": tgt[ok], "tempo": tempo[ok], "codes": np.concatenate(cl),
               "off": np.concatenate([[0], np.cumsum([len(c) for c in cl])]).astype(np.int64),
               "spk": sp_ok, "vfeat": vfeat, "vhist": vhist, "vframes": vframes,
               "info": np.array(json.dumps({"knn": KNN, "nv": NV, "pool_min": POOL_MIN, "pool_max": POOL_MAX,
                                            "tempo": [TMIN, TMAX], "agree_requant": agree / max(tot, 1)}))}
        f = out / f"{files[fi]}.npz"
        np.savez(f.with_suffix(".tmp.npz"), **res)
        f.with_suffix(".tmp.npz").rename(f)
        upq.put(f)
        done += 1
        del feats
        log(json.dumps({"shard": files[fi], "done": done, "of": len(shards), "utt": len(rows), "pairs": int(ok.sum()),
                        "voices": len(sp_ok), "audio_h": round(sec / 3600, 2), "x_rt_feat": round(sec / t_feat, 1),
                        "agree_requant": round(agree / max(tot, 1), 4), "sec": round(time.time() - t0, 1),
                        "elapsed_min": round((time.time() - t_all) / 60, 1)}))
    upq.put(None)
    up.join()
    log("PAIRS DONE")


if __name__ == "__main__":
    main()
