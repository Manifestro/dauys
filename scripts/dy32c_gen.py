"""DY-32 track C continuations: the multi-stream LM continues every prompt of the common pack; codes -> HiFi-GAN.

prompt_XX.wav (15 s) -> six-stream units (dy32c_units) -> an EOS frame + all 750 prompt frames -> 500 new frames
(10 s; temperature 0.8, top-p 0.95, EOS/PAD banned, seed 1234) -> SpeechBrain discrete HiFi-GAN (all six streams).
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
from dy32c_resynth import BUCKET, SR, Decoder, read_pack
from dy32c_units import LAYERS, Units, piece_codes, sh


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pack", default="/opt/dlami/nvme/pack")
    p.add_argument("--new-s", type=float, default=10.0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--s3", default="", help="upload to gen/<this>/")
    a = p.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckf = out / "ckpt.pt"
    shutil.copy(Path(a.run) / "best.pt", ckf)  # the trainer may replace best.pt meanwhile
    ck = torch.load(ckf, map_location="cpu", weights_only=False)
    cfg = MSConfig(**ck["cfg"])
    order = ck["order"]
    cols = [LAYERS.index(L) for L in order]

    names, waves = read_pack(Path(a.pack), "prompt")
    u = Units(dev)
    toks = piece_codes(u, [(w * 32768.0).clip(-32768, 32767).astype(np.int16) for w in waves])
    del u
    torch.cuda.empty_cache()
    n = min(len(t) for t in toks)
    prompts = np.stack([np.r_[np.full((1, cfg.streams), cfg.k), t[-n:][:, cols]] for t in toks]).astype(np.int64)

    model = MultiStreamLM(cfg).to(dev)
    model.load_state_dict(ck["model"])
    n_new = int(round(a.new_s * 50))
    t0 = time.time()
    gen = np.concatenate([generate(model, prompts[i : i + a.batch], n_new, a.temp, a.top_p, a.seed + i, dev)
                          for i in range(0, len(prompts), a.batch)])
    secs = time.time() - t0
    del model
    torch.cuda.empty_cache()
    dec = Decoder(dev)
    inv = [order.index(L) for L in LAYERS]  # model order -> layer order
    for nm, g in zip(names, gen):
        sf.write(out / f"{nm}.wav", dec(g[:, inv], LAYERS), SR)
    np.savez(out / "gen_units.npz", **{nm: g[:, inv] for nm, g in zip(names, gen)})
    info = {"track": "c", "step": ck["step"], "val": ck["val"], "params": sum(v.numel() for v in ck["model"].values()),
            "config": ck["cfg"], "stream_layers_model_order": order,
            "sampling": {"temperature": a.temp, "top_p": a.top_p, "seed": f"{a.seed} + batch offset", "banned": "EOS, PAD"},
            "prompt_seen": f"all {n} frames ({n / 50:.1f} s) of each 15 s prompt, after one EOS frame",
            "continuation_s": a.new_s, "generation_seconds": round(secs),
            "units": "WavLM-Large layers 1,3,7,12,18,23 -> SpeechBrain LibriSpeech-960 k-means K=1000 each, 50 Hz, 6 parallel streams",
            "model": "causal transformer 24x1024 (dy21 blocks), mean of 6 stream embeddings in, 6 heads out, delay pattern "
                     "(stream k shifted by k positions), ctx 2048",
            "decoder": "speechbrain/hifigan-wavlm-k1000-LibriTTS (discrete unit HiFi-GAN, all six streams), 16 kHz"}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print("GEN", json.dumps({k: info[k] for k in ("step", "val", "generation_seconds")}), flush=True)
    if a.s3:
        for nm in names:
            sh(f"aws s3 cp {out / nm}.wav {BUCKET}/gen/{a.s3}/{nm}.wav --only-show-errors")
        sh(f"aws s3 cp {out / 'info.json'} {BUCKET}/gen/{a.s3}/info.json --only-show-errors")
        print(f"UPLOADED gen/{a.s3}", flush=True)


if __name__ == "__main__":
    main()
