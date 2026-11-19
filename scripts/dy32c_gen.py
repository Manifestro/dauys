"""DY-32 track C continuations: the multi-stream LM continues every prompt of the common pack; codes -> HiFi-GAN.

prompt_XX.wav (15 s) -> four-level RVQ units (dy32c_units) -> an EOS frame + all 750 prompt frames -> 500 new frames
(10 s; temperature 0.8, top-p 0.95, EOS/PAD banned, seed 1234) -> summed RVQ centroids -> kNN-VC HiFi-GAN.
Writes --out/XX.wav (the continuation only, 16 kHz) and info.json; --s3 NAME uploads to gen/NAME/.

    python scripts/dy32c_gen.py --run /data/dy32c/run --out /data/dy32c/gen --s3 c
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dauys.dy32c_lm import MSConfig, MultiStreamLM, generate
from dy32c_resynth import BUCKET, SR, read_pack
from dy32c_units import RVQUnits, piece_codes, sh


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pack", default="/opt/dlami/nvme/pack")
    p.add_argument("--new-s", type=float, default=10.0)
    p.add_argument("--pack-src", default="prompts", help="pack folder under dy32/ in S3 (prompts or prompts_x)")
    p.add_argument("--variants", default="c:0.8:0.95",
                   help="name:temps:top_ps;... temps/top_ps one value or one per level (comma); temp 0 = greedy; "
                        "out/<name>/, uploaded to gen/<name>/ with --upload")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--s3", default="", help="single default variant: upload to gen/<this>/ (old form)")
    a = p.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckf = out / "ckpt.pt"
    shutil.copy(Path(a.run) / "best.pt", ckf)  # the trainer may replace best.pt meanwhile
    ck = torch.load(ckf, map_location="cpu", weights_only=False)
    cfg = MSConfig(**ck["cfg"])
    order = ck["order"]
    cols = order

    names, waves = read_pack(Path(a.pack), "prompt", a.pack_src)
    u = RVQUnits(dev)
    toks = piece_codes(u, [(w * 32768.0).clip(-32768, 32767).astype(np.int16) for w in waves])
    n = min(len(t) for t in toks)
    prompts = np.stack([np.r_[np.full((1, cfg.streams), cfg.k), t[-n:][:, cols]] for t in toks]).astype(np.int64)

    model = MultiStreamLM(cfg).to(dev)
    model.load_state_dict(ck["model"])
    n_new = int(round(a.new_s * 50))
    inv = [order.index(c) for c in sorted(order)]  # model order -> npz column order
    variants = [v.split(":") for v in a.variants.split(";")]
    if a.s3:
        variants, a.upload = [[a.s3, "0.8", "0.95"]], True
    for name, ts, ps in variants:
        temps = [float(x) for x in ts.split(",")]
        tops = [float(x) for x in ps.split(",")]
        temps = temps * cfg.streams if len(temps) == 1 else temps
        tops = tops * cfg.streams if len(tops) == 1 else tops
        vo = out / name if not a.s3 else out
        vo.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        gen = np.concatenate([generate(model, prompts[i : i + a.batch], n_new, temps, tops, a.seed + i, dev)
                              for i in range(0, len(prompts), a.batch)])
        secs = time.time() - t0
        for nm, g in zip(names, gen):
            sf.write(vo / f"{nm}.wav", u.decode(g[:, inv]), SR)
        np.savez(vo / "gen_units.npz", **{nm: g[:, inv] for nm, g in zip(names, gen)})
        info = {"track": "c", "variant": name, "step": ck["step"], "val": ck["val"],
                "params": sum(v.numel() for v in ck["model"].values()), "config": ck["cfg"],
                "stream_columns_model_order": order, "prompts": f"s3 dy32/{a.pack_src}/ ({len(names)})",
                "sampling": {"temperature_per_level": temps, "top_p_per_level": tops, "temperature 0": "greedy",
                             "seed": f"{a.seed} + batch offset", "banned": "EOS, PAD"},
                "prompt_seen": f"all {n} frames ({n / 50:.1f} s) of each 15 s prompt, after one EOS frame",
                "continuation_s": a.new_s, "generation_seconds": round(secs),
                "units": "kNN-VC WavLM-Large layer-6 features (50 Hz) -> residual k-means, 4 levels x 1024 codes (track A "
                         "rvq-corpus-1024x4 codebooks), 4 parallel streams",
                "model": "causal transformer 24x1024 (dy21 blocks), mean of the stream embeddings in, one head per stream, "
                         "delay pattern (level k shifted by k positions), ctx 2048",
                "decoder": "sum of the 4 level centroids -> kNN-VC prematched HiFi-GAN (github.com/bshall/knn-vc), 16 kHz"}
        (vo / "info.json").write_text(json.dumps(info, indent=2))
        print("GEN", name, json.dumps({k: info[k] for k in ("step", "generation_seconds")}), flush=True)
        if a.upload:
            sh(f"aws s3 cp {vo}/ {BUCKET}/gen/{name}/ --recursive --exclude '*' --include '*.wav' --include info.json "
               "--only-show-errors")
            print(f"UPLOADED gen/{name}", flush=True)


if __name__ == "__main__":
    main()
