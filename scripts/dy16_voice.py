"""DY-16 voice: gate V on real test turns, and listening files of generated replies.

Gate V: 100 real own turns (6 s after each generation moment) are resynthesised by the
Sylber synthesis model and the SPARC vocoder (a) from Sylber's own segment features and
(b) from our units (the mean train feature of each unit); voice from another 10 s of the
same person. Whisper (a ruler only) transcribes the original, (a) and (b); WER of (a) and
(b) against the original's transcript.

Listening: 10 generated replies rendered the same way as (b), each after 5 s of the real
context, so most of every file is the model's speech. Files stay private (Open Yap DUA).

    python scripts/dy16_voice.py --oy /data/oy16 --anchors test_anchors.json --moments /data/dy16/moments.json \
        --tokens /data/dy16/gen_f0.tokens.npy --centroids /data/dy14/centroids.pt --means /data/dy14/unit_means.pt \
        --config sylber_resynthesis.yaml --out /data/dy16/voice
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import resample_poly

from dauys.dy14_units import GRID, get_segment, hidden_states, load_sylber

SR = 16000
TURN_S = 6


def load_synth(config: str, device):
    from huggingface_hub import hf_hub_download
    from sylber import SegmentSynthesis

    cfg = yaml.safe_load(Path(config).read_text())
    model = SegmentSynthesis(**cfg).to(device).eval()
    ck = torch.load(
        hf_hub_download("cheoljun95/sylber", "synthesis_sylber.ckpt"),
        map_location="cpu",
    )
    state = {k.replace("net.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def render(synth, coder, frames: np.ndarray, ref: np.ndarray, device) -> np.ndarray:
    feats = torch.from_numpy(frames).float().to(device)[None]
    art = (
        synth.resynthesize(features=feats, steps=50, rand_scale=0.0)[0][0].cpu().numpy()
    )
    sf.write("/tmp/dy16_ref.wav", ref, SR)
    sp = coder.encode("/tmp/dy16_ref.wav")
    pitch = np.exp(art[:, 12:13]) * sp["pitch_stats"][0]
    wav = coder.decode(art[:, :12], pitch, art[:, 13:], sp["spk_emb"])
    wav = np.asarray(wav, np.float32).reshape(-1)
    return resample_poly(wav, SR, coder.sr) if coder.sr != SR else wav


def unit_frames(tok: np.ndarray, means: np.ndarray, k: int) -> np.ndarray:
    """10 Hz tokens to 50 Hz Sylber frames: a unit's mean feature from its onset to the next event."""
    out = np.zeros((len(tok) * GRID, means.shape[1]), np.float32)
    cur = None
    for i, t in enumerate(tok):
        if t < k:
            cur = means[t]
        elif t == k + 1:
            cur = None
        if cur is not None:
            out[i * GRID : (i + 1) * GRID] = cur
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oy", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--moments", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--centroids", required=True)
    p.add_argument("--means", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--k", type=int, default=8192)
    args = p.parse_args()
    out = Path(args.out)
    (out / "listen").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    names = json.loads(Path(args.anchors).read_text())["names"]
    items = json.loads(Path(args.moments).read_text())
    gens = np.load(args.tokens)
    centroids = torch.load(args.centroids).to(device)
    means = torch.load(args.means)["means"].numpy()

    from sparc import load_model
    from transformers import pipeline

    sylber = load_sylber(device)
    synth = load_synth(args.config, device)
    coder = load_model("en", device="cuda", use_penn=False)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=0,
        torch_dtype=torch.float16,
    )
    import jiwer

    def text(wav: np.ndarray) -> str:
        return asr(
            {"raw": wav.astype(np.float32), "sampling_rate": SR},
            generate_kwargs={"language": "en"},
        )["text"]

    rows = []
    for i, it in enumerate(items[: args.n]):
        track = np.load(
            Path(args.oy) / names[it["conv"]] / ("ab"[it["ch"]] + ".npy"), mmap_mode="r"
        )
        s = (it["t"] + 1) * SR // 10
        clip = np.asarray(track[s : s + TURN_S * SR])
        ref = np.asarray(track[max(0, s - 70 * SR) : max(0, s - 70 * SR) + 10 * SR])
        if len(ref) < 5 * SR:
            ref = np.asarray(track[s + 20 * SR : s + 30 * SR])
        states = hidden_states(sylber, clip)
        seg = get_segment(states)
        cont = np.zeros_like(states)
        unit = np.zeros_like(states)
        if len(seg):
            feats = np.stack([states[a:b].mean(0) for a, b in seg])
            ids = (
                (F.normalize(torch.from_numpy(feats).to(device), dim=1) @ centroids.T)
                .argmax(1)
                .cpu()
                .numpy()
            )
            for (a, b), f, u in zip(seg, feats, ids):
                cont[a:b] = f
                unit[a:b] = means[u]
        wav_a = render(synth, coder, cont, ref, device)
        wav_b = render(synth, coder, unit, ref, device)
        t0, ta, tb = text(clip.astype(np.float32) / 32768.0), text(wav_a), text(wav_b)
        rows.append({"ref": t0, "a": ta, "b": tb})
        if i < 5:
            sf.write(out / "listen" / f"resynth_{i:02d}_orig.wav", clip, SR)
            sf.write(out / "listen" / f"resynth_{i:02d}_sylber.wav", wav_a, SR)
            sf.write(out / "listen" / f"resynth_{i:02d}_units.wav", wav_b, SR)
    norm = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    keep = [r for r in rows if norm(r["ref"])]
    wer_a = jiwer.wer(
        [norm(r["ref"]) for r in keep], [norm(r["a"]) or "x" for r in keep]
    )
    wer_b = jiwer.wer(
        [norm(r["ref"]) for r in keep], [norm(r["b"]) or "x" for r in keep]
    )

    for i, it in enumerate(items[:10]):
        track = np.load(
            Path(args.oy) / names[it["conv"]] / ("ab"[it["ch"]] + ".npy"), mmap_mode="r"
        )
        other = np.load(
            Path(args.oy) / names[it["conv"]] / ("ab"[1 - it["ch"]] + ".npy"),
            mmap_mode="r",
        )
        s = (it["t"] + 1) * SR // 10
        ref = np.asarray(track[max(0, s - 70 * SR) : max(0, s - 70 * SR) + 10 * SR])
        ctx = (
            np.asarray(track[s - 5 * SR : s], np.float32)
            + np.asarray(other[s - 5 * SR : s], np.float32)
        ) / 32768.0
        reply = render(synth, coder, unit_frames(gens[i], means, args.k), ref, device)
        partner = np.asarray(other[s : s + len(reply)], np.float32) / 32768.0
        mix = np.concatenate([ctx, reply + 0.7 * partner[: len(reply)]])
        sf.write(out / "listen" / f"reply_{i:02d}.wav", np.clip(mix, -1, 1), SR)
    res = {
        "n": len(keep),
        "wer_sylber_features": wer_a,
        "wer_units": wer_b,
        "V": wer_b <= wer_a + 0.15,
        "rows": rows,
    }
    (out / "voice.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
