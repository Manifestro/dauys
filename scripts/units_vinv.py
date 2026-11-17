"""Voice-invariant syllable codes (plan B), checked by the sWUGGY-voice ABX before any LM training.

Pairs from podcast audio (DuplexChat training shards): each piece is also played at a changed speed
(factor a in [0.8, 1.25], which moves pitch and formants together, like another vocal tract) plus light noise.
Segments come from the original; in the perturbed copy the same segments are taken at the scaled frames, so
every syllable has two versions with the same content and a different voice. Segment feature = mean
Sylber layer-9 state, L2-normalised (as DY-14).

Model: encoder MLP 768 -> 256 (content code z); decoder (z, voice summary g) -> feature, where g is the mean
segment feature of the whole piece in the target voice. Losses: rebuild the original-voice feature from the
perturbed z with the original g and vice versa (z must carry the content, g carries the voice), plus a cosine
pull between the two z of a syllable. Then k-means (8192 and 2048) on z gives the discrete codes.
ABX (units_diag.py) on z (DTW cosine) and on the codes (edit distance); DY-14 units are 69.9, Sylber 90.0.

    python scripts/units_vinv.py --zr /data/bench/zr --out /data/vinv/out.json
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from bench_slm import to_int16
from dy14_features import segments_of
from dy21_train import held_out
from units_diag import cos_dtw, dedup, edit, kmeans

from dauys.dy14_units import SR, hidden_states, load_sylber

SRC = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/duplexchat/en-16k"


def pool(states: np.ndarray, seg: np.ndarray) -> np.ndarray:
    seg = np.clip(seg, 0, len(states))
    seg[:, 1] = np.maximum(seg[:, 1], seg[:, 0] + 1)
    seg = np.minimum(seg, len(states))
    c = np.concatenate([np.zeros((1, states.shape[1]), np.float32), np.cumsum(states, 0)])
    f = (c[seg[:, 1]] - c[seg[:, 0]]) / np.maximum(seg[:, 1] - seg[:, 0], 1)[:, None]
    return (f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def perturb(wav: np.ndarray, a: float, rng) -> np.ndarray:
    """Play ``a`` times faster: pitch and formants x a, duration / a; plus noise at 25-40 dB SNR."""
    up, down = int(round(1000 / a)), 1000
    x = resample_poly(wav.astype(np.float32), up, down)
    snr = rng.uniform(25, 40)
    x = x + rng.standard_normal(len(x)) * (np.std(x) / (10 ** (snr / 20)))
    return np.clip(x, -32768, 32767).astype(np.int16)


def pairs(syl, n_seg: int, work: Path, seed: int):
    rng = np.random.default_rng(seed)
    out = subprocess.run(f"aws s3 ls {SRC}/", shell=True, capture_output=True, text=True).stdout
    shards = [s for s in sorted(l.split()[-1].rstrip("/") for l in out.splitlines() if l.strip().startswith("PRE"))
              if not held_out(s)]
    fo, fp, go, gp, total = [], [], [], [], 0
    for s in shards[10:]:  # other shards than the residual-code test
        d = work / s
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("index.npy", "segs.bin"):
            subprocess.run(f"aws s3 cp {SRC}/{s}/{fn} {d}/{fn} --only-show-errors", shell=True, check=True)
        idx = np.load(d / "index.npy")
        segs = np.fromfile(d / "segs.bin", dtype=np.int16)
        for st, ln in idx[:150]:
            if ln < 5 * SR:
                continue
            wav = segs[st : st + min(ln, 30 * SR)]
            so = hidden_states(syl, wav)
            seg = segments_of(so)
            if len(seg) < 4:
                continue
            a = float(rng.choice([rng.uniform(0.8, 0.93), rng.uniform(1.07, 1.25)]))
            sp = hidden_states(syl, perturb(wav, a, rng))
            f_o = pool(so, seg.copy())
            f_p = pool(sp, np.round(seg / a).astype(np.int64))
            fo.append(f_o)
            fp.append(f_p)
            go.append(np.repeat(f_o.mean(0, keepdims=True), len(f_o), 0))
            gp.append(np.repeat(f_p.mean(0, keepdims=True), len(f_p), 0))
            total += len(f_o)
            if total >= n_seg:
                break
        subprocess.run(f"rm -rf {d}", shell=True)
        print("shard", s, "pairs", total, flush=True)
        if total >= n_seg:
            break
    cat = lambda x: torch.from_numpy(np.concatenate(x).astype(np.float32))
    return cat(fo), cat(fp), cat(go), cat(gp)


class Coder(nn.Module):
    def __init__(self, d: int = 768, z: int = 256, h: int = 1024) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z + d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def z(self, f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.enc(f), dim=-1)

    def rebuild(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.dec(torch.cat([z, g], -1)), dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=200000)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--inv", type=float, default=1.0, help="weight of the cosine pull between the two z")
    p.add_argument("--ids", type=int, default=400)
    args = p.parse_args()
    device = torch.device("cuda")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    syl = load_sylber(device)
    t0 = time.time()
    cache = out.parent / "pairs.pt"
    if cache.exists():
        fo, fp, go, gp = torch.load(cache)
    else:
        fo, fp, go, gp = pairs(syl, args.pairs, out.parent / "work", 0)
        torch.save((fo, fp, go, gp), cache)
    print("pairs", len(fo), "mean cos orig/perturbed", round(float((fo * fp).sum(1).mean()), 4),
          "in", round(time.time() - t0), "s", flush=True)
    fo, fp, go, gp = (x.to(device) for x in (fo, fp, go, gp))

    torch.manual_seed(0)
    m = Coder().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(args.steps):
        i = torch.randint(0, len(fo), (4096,), device=device)
        zo, zp = m.z(fo[i]), m.z(fp[i])
        rec = (2 - (m.rebuild(zp, go[i]) * fo[i]).sum(-1) - (m.rebuild(zo, gp[i]) * fp[i]).sum(-1)).mean()
        inv = (1 - (zo * zp).sum(-1)).mean()
        loss = rec + args.inv * inv
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(json.dumps({"step": step + 1, "rec": round(rec.item(), 4), "inv": round(inv.item(), 4)}), flush=True)
    m.eval()
    with torch.no_grad():
        Z = torch.cat([m.z(fo[j : j + 65536]) for j in range(0, len(fo), 65536)])
    books = {k: kmeans(Z, k, cosine=True, iters=25) for k in (8192, 2048)}
    torch.save({"coder": m.state_dict(), "books": {k: v.cpu() for k, v in books.items()}}, out.parent / "vinv.pt")

    d = Path(args.zr) / "lexical" / "dev"
    items = defaultdict(lambda: defaultdict(dict))
    for row in csv.DictReader(open(d / "gold.csv")):
        items[row["id"]][int(row["correct"])][row["voice"]] = d / f"{row['filename']}.wav"
    ids = [i for i, g in items.items() if len(g[0]) == 4 and len(g[1]) == 4]
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    ids = ids[: args.ids]
    reps = {}
    with torch.no_grad():
        for i in ids:
            for c in (0, 1):
                for v, path in items[i][c].items():
                    w = to_int16(*sf.read(path))
                    w = np.pad(w, (0, max(0, SR // 5 - len(w))))
                    st = hidden_states(syl, w)
                    seg = segments_of(st)
                    if len(seg) == 0:
                        reps[(i, c, v)] = {"z": np.zeros((0, 256), np.float32),
                                           **{k: np.zeros(0, np.int64) for k in books}}
                        continue
                    z = m.z(torch.from_numpy(pool(st, seg.copy())).to(device))
                    reps[(i, c, v)] = {"z": z.cpu().numpy(),
                                       **{k: dedup((z @ b.T).argmax(1).cpu().numpy()) for k, b in books.items()}}
    voices = ["A", "C", "D", "F"]

    def abx(dist) -> float:
        right = n = 0.0
        for i in ids:
            for c in (0, 1):
                for v1 in voices:
                    for v2 in voices:
                        if v1 != v2:
                            a, b = dist((i, c, v1), (i, c, v2)), dist((i, c, v1), (i, 1 - c, v1))
                            right += 1.0 if a < b else 0.5 if a == b else 0.0
                            n += 1
        return round(right / n, 4)

    res = {"pairs": len(fo), "inv_weight": args.inv, "abx_z": abx(lambda x, y: cos_dtw(reps[x]["z"], reps[y]["z"]))}
    for k in books:
        res[f"abx_codes_k{k}"] = abx(lambda x, y, k=k: edit(reps[x][k], reps[y][k]))
        res[f"same_word_identical_k{k}"] = round(float(np.mean([np.array_equal(reps[(i, 1, a)][k], reps[(i, 1, b)][k])
                                                                for i in ids for a in voices for b in voices if a < b])), 4)
    out.write_text(json.dumps(res, indent=2))
    print("SUMMARY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
