"""DY-32 track C channel: codes -> audio with SpeechBrain's discrete HiFi-GAN, and the round trip of the prompt pack.

Decoder: speechbrain/hifigan-wavlm-k1000-LibriTTS (scalable unit HiFi-GAN, attention pooling over the six layer
embeddings; layers left out get id 0, as speechbrain DiscreteSSL.decode does). Ids are code + 1000 * layer_slot + 1.

``pack``: real_XX.wav of the prompt pack -> units (dy32c_units.Units) -> decoder, written for all six streams
(``multi``) and for every single stream alone; Whisper (measuring only) reads the originals and each variant of the
first ``--n-asr`` pieces; WER vs Whisper-on-original picks the best single stream. Outputs go to --out/<variant>/XX.wav
and --out/asr.json; ``--s3`` uploads multi to resynth/c/ and resynth/c-multi/, the best single to resynth/c-single/.

    python scripts/dy32c_resynth.py pack --out /data/dy32c/resynth --s3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dy32c_units import LAYERS, VOC_REPO, RVQUnits, Units, piece_codes, sh

BUCKET = "s3://manifestro-dataset-876108973178-us-east-1-an/outputs/dauys/dy32"
SR = 16000
ASR = "openai/whisper-large-v3-turbo"


class Decoder:
    def __init__(self, device) -> None:
        from speechbrain.inference.vocoders import UnitHIFIGAN

        self.device = device
        self.voc = UnitHIFIGAN.from_hparams(source=VOC_REPO, savedir="/opt/dlami/nvme/sb_voc", run_opts={"device": str(device)})
        self.voc.tokenize = False

    @torch.no_grad()
    def __call__(self, tok: np.ndarray, layers=LAYERS) -> np.ndarray:
        """tok [T, len(layers)] codes 0..999 of these layers -> float32 16 kHz wave."""
        full = np.zeros((len(tok), len(LAYERS)), np.int64)
        for i, L in enumerate(layers):
            j = LAYERS.index(L)
            full[:, j] = tok[:, i].astype(np.int64) + 1000 * j + 1
        w = self.voc.decode_batch(torch.from_numpy(full)[None].to(self.device))
        return w.reshape(-1).float().cpu().numpy()


def wer(ref: str, hyp: str) -> float:
    import jiwer

    from dy30_eval import norm_text

    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return float("nan")
    return float(jiwer.wer(r, h))


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Word errors over all pieces / reference words of all pieces (Whisper on the original audio as reference)."""
    import jiwer

    from dy30_eval import norm_text

    pairs = [(norm_text(r), norm_text(h)) for r, h in zip(refs, hyps) if norm_text(r)]
    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def whisper(waves: list[np.ndarray], device) -> list[str]:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=ASR, device=device, torch_dtype=torch.float16)
    res = asr([{"raw": w.astype(np.float32), "sampling_rate": SR} for w in waves], batch_size=16,
              generate_kwargs={"language": "en", "task": "transcribe"})
    return [r["text"].strip() for r in res]


def read_pack(dst: Path, kind: str = "real") -> tuple[list[str], list[np.ndarray]]:
    dst.mkdir(parents=True, exist_ok=True)
    sh(f"aws s3 sync {BUCKET}/prompts/ {dst}/ --only-show-errors")
    names = sorted(p.stem.split("_")[1] for p in dst.glob(f"{kind}_*.wav"))
    waves = []
    for n in names:
        w, sr = sf.read(dst / f"{kind}_{n}.wav", dtype="float32")
        assert sr == SR, sr
        waves.append(w if w.ndim == 1 else w.mean(1))
    return names, waves


def cmd_rvq(a) -> None:
    """Round trip through the four-level layer-6 RVQ channel: rvq4 (all levels) and rvq1 (level 0 alone)."""
    dev = torch.device("cuda")
    out = Path(a.out)
    names, waves = read_pack(Path(a.pack))
    u = RVQUnits(dev)
    toks = piece_codes(u, [(w * 32768.0).clip(-32768, 32767).astype(np.int16) for w in waves])
    res = {}
    for v, L in (("rvq4", 4), ("rvq2", 2), ("rvq1", 1)):
        (out / v).mkdir(parents=True, exist_ok=True)
        res[v] = [u.decode(t[:, :L]) for t in toks]
        for n, w in zip(names, res[v]):
            sf.write(out / v / f"{n}.wav", w, SR)
    np.savez(out / "real_units_rvq.npz", **{n: t for n, t in zip(names, toks)})
    del u
    torch.cuda.empty_cache()
    k = min(a.n_asr, len(names))
    ref = whisper(waves[:k], dev)
    asr = {"pieces": k, "whisper": ASR, "ref": ref, "wer": {}}
    for v in res:
        hyp = whisper(res[v][:k], dev)
        asr["wer"][v] = {"corpus": corpus_wer(ref, hyp), "median": float(np.nanmedian([wer(r, h) for r, h in zip(ref, hyp)])), "hyp": hyp}
        print("ASR", v, "corpus", round(asr["wer"][v]["corpus"], 4), "median", round(asr["wer"][v]["median"], 4), flush=True)
    (out / "asr_rvq.json").write_text(json.dumps(asr, indent=2))
    if a.s3:
        for dst, v in (("c", "rvq4"), ("c-multi", "rvq4"), ("c-single", "rvq1")):
            sh(f"aws s3 rm {BUCKET}/resynth/{dst}/ --recursive --only-show-errors")
            sh(f"aws s3 sync {out / v} {BUCKET}/resynth/{dst}/ --only-show-errors")
        sh(f"aws s3 cp {out / 'asr_rvq.json'} {BUCKET}/c/resynth_asr_rvq.json --only-show-errors")
        print("UPLOADED resynth", flush=True)


def cmd_pack(a) -> None:
    dev = torch.device("cuda")
    out = Path(a.out)
    names, waves = read_pack(Path(a.pack))
    u = Units(dev)
    toks = piece_codes(u, [(w * 32768.0).clip(-32768, 32767).astype(np.int16) for w in waves])
    del u
    torch.cuda.empty_cache()
    dec = Decoder(dev)
    variants = {"multi": LAYERS, **{f"L{L}": (L,) for L in LAYERS}}
    res = {}
    for v, layers in variants.items():
        (out / v).mkdir(parents=True, exist_ok=True)
        res[v] = []
        for n, t in zip(names, toks):
            cols = [LAYERS.index(L) for L in layers]
            w = dec(t[:, cols], layers)
            sf.write(out / v / f"{n}.wav", w, SR)
            res[v].append(w)
    np.savez(out / "real_units.npz", **{n: t for n, t in zip(names, toks)})
    k = min(a.n_asr, len(names))
    ref = whisper(waves[:k], dev)
    asr = {"pieces": k, "whisper": ASR, "ref": ref, "wer": {}}
    for v in variants:
        hyp = whisper(res[v][:k], dev)
        ws = [wer(r, h) for r, h in zip(ref, hyp)]
        asr["wer"][v] = {"corpus": corpus_wer(ref, hyp), "median": float(np.nanmedian(ws)), "per_piece": ws, "hyp": hyp}
        print("ASR", v, "corpus", round(asr["wer"][v]["corpus"], 4), "median", round(asr["wer"][v]["median"], 4), flush=True)
    best_single = min((v for v in variants if v != "multi"), key=lambda v: asr["wer"][v]["corpus"])
    asr["best_single"] = best_single
    (out / "asr.json").write_text(json.dumps(asr, indent=2))
    print("BEST SINGLE", best_single, "multi", asr["wer"]["multi"]["corpus"], "single", asr["wer"][best_single]["corpus"], flush=True)
    if a.s3:
        for dst, v in (("c", "multi"), ("c-multi", "multi"), ("c-single", best_single)):
            sh(f"aws s3 sync {out / v} {BUCKET}/resynth/{dst}/ --only-show-errors")
        sh(f"aws s3 cp {out / 'asr.json'} {BUCKET}/c/resynth_asr.json --only-show-errors")
        print("UPLOADED resynth", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("pack")
    x.add_argument("--pack", default="/opt/dlami/nvme/pack")
    x.add_argument("--out", required=True)
    x.add_argument("--n-asr", type=int, default=50)
    x.add_argument("--s3", action="store_true")
    x = s.add_parser("rvq")
    x.add_argument("--pack", default="/opt/dlami/nvme/pack")
    x.add_argument("--out", required=True)
    x.add_argument("--n-asr", type=int, default=50)
    x.add_argument("--s3", action="store_true")
    a = p.parse_args()
    {"pack": cmd_pack, "rvq": cmd_rvq}[a.cmd](a)


if __name__ == "__main__":
    main()
