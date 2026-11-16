"""DY-18 listening: moments where the model did reply, as mostly-model audio, with Whisper text.

For each chosen moment two files: ``NN_model.wav`` = 5 s of real context (both channels) then the
model's 10 s reply in the speaker's voice (SPARC embedding from 10 s of their earlier speech) with
the partner's real continuation at 0.3; ``NN_real.wav`` = the same 5 s then what the person really
said, same mix. ``listen.json`` holds Whisper text of the model reply and of the real reply.

    python scripts/dy18_listen.py --oy /data/oy16 --anchors test_anchors.json --moments moments.json \
        --tokens gen_f1.tokens.npy --means unit_means.pt --config sylber_resynthesis.yaml --out /data/dy18/listen
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy16_voice import SR, load_synth, render, unit_frames

PARTNER = 0.3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oy", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--moments", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--means", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--k", type=int, default=8192)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    names = json.loads(Path(args.anchors).read_text())["names"]
    items = json.loads(Path(args.moments).read_text())
    gens = np.load(args.tokens)
    means = torch.load(args.means)["means"].numpy()
    answered = [i for i, g in enumerate(gens) if (g[:30] < args.k).sum() >= 3][: args.n]

    from sparc import load_model
    from transformers import pipeline

    synth = load_synth(args.config, device)
    coder = load_model("en", device="cuda", use_penn=False)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=0,
        torch_dtype=torch.float16,
    )

    def text(wav: np.ndarray) -> str:
        return asr(
            {"raw": wav.astype(np.float32), "sampling_rate": SR},
            generate_kwargs={"language": "en"},
        )["text"].strip()

    rows = []
    for j, i in enumerate(answered):
        it = items[i]
        d = Path(args.oy) / names[it["conv"]]
        own = np.load(d / ("ab"[it["ch"]] + ".npy"), mmap_mode="r")
        other = np.load(d / ("ab"[1 - it["ch"]] + ".npy"), mmap_mode="r")
        s = (it["t"] + 1) * SR // 10
        ref = np.asarray(own[max(0, s - 70 * SR) : max(0, s - 70 * SR) + 10 * SR])
        if len(ref) < 5 * SR:
            ref = np.asarray(own[s + 20 * SR : s + 30 * SR])
        ctx = (np.asarray(own[s - 5 * SR : s], np.float32) + np.asarray(other[s - 5 * SR : s], np.float32)) / 32768.0
        reply = render(synth, coder, unit_frames(gens[i], means, args.k), ref, device)
        n = len(reply)
        partner = np.asarray(other[s : s + n], np.float32) / 32768.0
        real = np.asarray(own[s : s + n], np.float32) / 32768.0
        sf.write(out / f"{j:02d}_model.wav", np.clip(np.concatenate([ctx, reply + PARTNER * partner]), -1, 1), SR)
        sf.write(out / f"{j:02d}_real.wav", np.clip(np.concatenate([ctx, real + PARTNER * partner]), -1, 1), SR)
        rows.append(
            {
                "file": f"{j:02d}",
                "moment": i,
                "start_s": float(np.argmax(gens[i] < args.k)) / 10,
                "model_text": text(reply),
                "real_text": text(real),
                "partner_before": text(np.asarray(other[s - 5 * SR : s], np.float32) / 32768.0),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    (out / "listen.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
