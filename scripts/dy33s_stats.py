"""DY-33 track S: word count and silence / held-sound share of generated continuations vs real continuations.

Inputs: generated variants (dirs with NN.wav / NNN.wav and gen_units.npz) and the real next 10 s of both DY-32 packs
(real_XX.wav, real_XXX.wav). The real audio is coded the same way (layer-6 features -> frozen slow codec -> codes ->
codec decoder -> kNN-VC) so both sides pass the same channel.
Measures per clip:
  words      Whisper large-v3-turbo word count of the clip (measuring only)
  held       share of steps whose 4 codes all equal the previous step's codes; also level-1-only repeat share
  silence    share of 80 ms frames of the decoded audio with RMS below -40 dBFS
Medians / means over clips go to --out (json) and are printed.

    python scripts/dy33s_stats.py --gen /data/dy33s/gen/s-t10,/data/dy33s/gen-x/s-t10-x --codec /data/codec/frozen.pt \
        --out /data/dy33s/stats_t10.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy32c_resynth import whisper
from dy32c_units import RVQUnits, sh
from dy33s_codec import load_codec
from dy33s_gen import S3, SR, encode

FR = SR * 80 // 1000


def silence(w: np.ndarray) -> float:
    n = len(w) // FR
    if n == 0:
        return float("nan")
    r = np.sqrt((w[: n * FR].reshape(n, FR).astype(np.float64) ** 2).mean(1) + 1e-12)
    return float((20 * np.log10(r) < -40).mean())


def held(c: np.ndarray) -> tuple[float, float]:
    same = c[1:] == c[:-1]
    return float(same.all(1).mean()), float(same[:, 0].mean())


def summary(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in ("words", "held_all4", "held_L1", "silence"):
        v = np.array([r[k] for r in rows], float)
        out[k] = {"median": round(float(np.nanmedian(v)), 4), "mean": round(float(np.nanmean(v)), 4)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="comma list of variant dirs")
    p.add_argument("--codec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--s3", default="")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    gen_rows, gw = [], []
    for d in a.gen.split(","):
        d = Path(d)
        units = np.load(d / "gen_units.npz")
        for nm in sorted(units.files):
            w, _ = sf.read(d / f"{nm}.wav", dtype="float32")
            gw.append(w)
            h4, h1 = held(units[nm])
            gen_rows.append({"id": nm, "variant": d.name, "held_all4": h4, "held_L1": h1, "silence": silence(w)})
    real_rows, rw = [], []
    for pk in ("prompts", "prompts_x"):
        for f in sorted(Path(f"/opt/dlami/nvme/{pk}").glob("real_*.wav")):
            w, sr = sf.read(f, dtype="float32")
            rw.append(w if w.ndim == 1 else w.mean(1))
            real_rows.append({"id": f.stem.split("_")[1], "pack": pk})
    codec, _ = load_codec(Path(a.codec), dev)
    u = RVQUnits(dev)
    toks = encode(u, codec, rw, dev)
    rt = []
    for r, c in zip(real_rows, toks):
        h4, h1 = held(c)
        f = codec.decode(torch.from_numpy(c.astype(np.int64)).to(dev)[None])[0]
        y = u.knn.vocode(f[None].float()).reshape(-1).cpu().numpy()
        rt.append(y)
        r.update({"held_all4": h4, "held_L1": h1, "silence": silence(y)})
    del u
    torch.cuda.empty_cache()
    for rows, ws in ((gen_rows, gw), (real_rows, rt)):
        for r, t in zip(rows, whisper(ws, dev)):
            r["words"], r["whisper"] = len(t.split()), t
    for r, t in zip(real_rows, whisper(rw, dev)):
        r["words_original"] = len(t.split())
    wo = [r["words_original"] for r in real_rows]
    res = {"generated": summary(gen_rows), "real_through_codec": summary(real_rows),
           "real_original_words": {"median": float(np.median(wo)), "mean": round(float(np.mean(wo)), 2)},
           "definitions": {"held_all4": "share of 80 ms steps whose 4 codes all equal the previous step's",
                           "held_L1": "share of steps whose level-1 code equals the previous step's",
                           "silence": "share of 80 ms frames of the audio with RMS < -40 dBFS",
                           "words": "Whisper large-v3-turbo word count per clip (about 10 s)"},
           "gen_dirs": a.gen, "rows_generated": gen_rows, "rows_real": real_rows}
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("generated", "real_through_codec", "real_original_words")}), flush=True)
    if a.s3:
        sh(f"aws s3 cp {a.out} {S3}/{a.s3} --only-show-errors")


if __name__ == "__main__":
    main()
