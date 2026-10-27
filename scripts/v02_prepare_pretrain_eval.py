"""Freeze the LibriSpeech dev-clean continuation suite and its Mimi ceiling."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from scipy.signal import resample_poly

from dauys.codec import decode, encode, load_mimi
from dauys.v02_data import sha256
from dauys.v02_pretrain_eval import (
    CONTINUATION_FRAMES,
    FRAME_SAMPLES,
    PROMPT_FRAMES,
    SAMPLE_RATE,
    WhisperTranscriber,
    english_vocabulary,
    word_statistics,
)

DATASET = "openslr/librispeech_asr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--listen-items", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Suite preparation requires CUDA")
    revision = HfApi().dataset_info(DATASET).sha
    stream = load_dataset(
        DATASET, "clean", split="validation", revision=revision, streaming=True
    ).cast_column("audio", Audio(decode=False))
    train_voices = {
        json.loads(line)["user_voice"]
        for line in args.train_manifest.read_text().splitlines()
        if line.strip()
    }
    needed = (PROMPT_FRAMES + CONTINUATION_FRAMES) * FRAME_SAMPLES
    candidates = []
    for row in stream:
        wave, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if rate != 16_000 or wave.ndim != 1:
            raise ValueError("LibriSpeech audio must be mono 16 kHz")
        pcm = resample_poly(wave, 3, 2).astype(np.float32)
        if len(pcm) < needed:
            continue
        if str(row["speaker_id"]) in train_voices:
            raise ValueError("A dev-clean speaker appears in the training corpus")
        key = hashlib.sha256(str(row["id"]).encode()).hexdigest()
        candidates.append(
            (key, str(row["id"]), str(row["speaker_id"]), row["text"], pcm)
        )
    candidates.sort()
    chosen = candidates[: args.items]
    if len(chosen) < args.items:
        raise ValueError("Not enough long dev-clean utterances")
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    items = []
    tails = []
    with torch.inference_mode():
        for _, identity, speaker, text, pcm in chosen:
            frames = len(pcm) // FRAME_SAMPLES
            clip = torch.from_numpy(pcm[: frames * FRAME_SAMPLES].copy())[None, None]
            codes = encode(codec, clip.cuda())[0].short().cpu()
            items.append(
                {"id": identity, "speaker": speaker, "text": text, "codes": codes}
            )
            window = codes[None, :, : PROMPT_FRAMES + CONTINUATION_FRAMES].long().cuda()
            audio = decode(codec, window)[0, 0, PROMPT_FRAMES * FRAME_SAMPLES :]
            tails.append(audio.float().cpu().numpy())
    del codec
    torch.cuda.empty_cache()
    transcripts = WhisperTranscriber(args.asr_model, "cuda")(tails)
    ceiling = word_statistics(transcripts, english_vocabulary())
    args.output.mkdir(parents=True, exist_ok=True)
    listen = args.output / "ceiling-audio"
    listen.mkdir(exist_ok=True)
    for item, wave, transcript in list(zip(items, tails, transcripts))[
        : args.listen_items
    ]:
        sf.write(listen / f"{item['id']}.wav", wave, SAMPLE_RATE)
        item["ceiling_asr"] = transcript
    suite = {
        "items": items,
        "ceiling": ceiling,
        "dataset": DATASET,
        "dataset_revision": revision,
        "split": "clean/validation",
        "codec_sha256": sha256(args.mimi_checkpoint),
        "asr_model": args.asr_model,
        "prompt_frames": PROMPT_FRAMES,
        "continuation_frames": CONTINUATION_FRAMES,
    }
    torch.save(suite, args.output / "suite.pt")
    (args.output / "suite.sha256").write_text(sha256(args.output / "suite.pt") + "\n")
    print(json.dumps({"items": len(items), "ceiling": ceiling, "revision": revision}))


if __name__ == "__main__":
    main()
