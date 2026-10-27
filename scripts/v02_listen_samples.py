"""Write stereo E2 replies (A = user, B = model) with transcripts for listening."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.decoding import DecodingConfig
from dauys.model import DauysHybrid
from dauys.v02_evaluation import load_suite
from dauys.v02_hohere_backend import FRAME_SAMPLES, SAMPLE_RATE, HohereProbeBackend
from dauys.v02_judges import _whisper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suites-dir", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--items", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(
        args.suites_dir / "E2.json",
        (args.suites_dir / "E2.sha256").read_text().strip(),
    )
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    backend = HohereProbeBackend(args.suites_dir, {"E2": suite}, codec, silence)
    model = DauysHybrid(ModelConfig.from_yaml(args.config)).cuda()
    model.load_state_dict(load_research_checkpoint(args.checkpoint)["model"])
    model.eval()
    asr = _whisper()
    stride = max(1, len(suite.items) // args.items)
    chosen = list(suite.items)[::stride][: args.items]
    args.output.mkdir(parents=True, exist_ok=True)
    listing = []
    for item in chosen:
        user, output = backend._channels(item)
        start = int(item["user_end_frame"])
        codes, activity = backend._generate(
            model, user, output, start, DecodingConfig.v02_default(), args.seed
        )
        playable = torch.where(
            activity[None, None, :].cuda(),
            codes[None].cuda(),
            silence[None, :, None].cuda(),
        )
        reply = decode(codec, playable)[0, 0].float().cpu().numpy()
        user_audio = backend._audio(item, "channel_a")
        frames = math.ceil(len(user_audio) / FRAME_SAMPLES)
        b = np.zeros(frames * FRAME_SAMPLES, dtype=np.float32)
        b[start * FRAME_SAMPLES : start * FRAME_SAMPLES + len(reply)] = reply[
            : len(b) - start * FRAME_SAMPLES
        ]
        a = np.pad(user_audio, (0, len(b) - len(user_audio)))
        sf.write(args.output / f"{item['id']}.wav", np.stack([a, b], 1), SAMPLE_RATE)
        heard = asr.confident([reply])[0]
        analysis = item["analysis"]
        listing.append(
            {
                "id": item["id"],
                "user_text": analysis["user_text"],
                "reference_answer": analysis["answer_text"],
                "model_asr": heard["text"],
                "asr_filtered_as_hallucination": heard["filtered"],
                "active_frames": int(activity.sum()),
                "reply_frames": int(activity.numel()),
                "reply_rms": float(np.sqrt(np.mean(reply**2))),
            }
        )
        print(json.dumps(listing[-1]), flush=True)
    (args.output / "transcripts.json").write_text(json.dumps(listing, indent=2) + "\n")


if __name__ == "__main__":
    main()
