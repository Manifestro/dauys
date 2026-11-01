"""Decode what a trained E0b model says, for listening.

    python scripts/v10_e0b_samples.py --checkpoint RUN/checkpoint-012000.pt \
        --mimi tokenizer-e351c8d8-checkpoint125.safetensors --output SAMPLES

Each stereo file holds one reply in each split: left the true partner, right the
model's channel. Each file is 3 s of the real lead-in and then 12 s of model speech,
so most of it is the model. The model hears the true partner throughout. It also
writes the data's own decoded channels of the first clip, to check the right channel.
"""

import argparse
import json
from pathlib import Path

import torch

from dauys.codec import decode, load_mimi
from dauys.v10_e0b import E0bConfig, E0bModel
from dauys.v10_e0b_data import (
    CONTEXT_FRAMES,
    TEST_SPLITS,
    TRAIN_SPLIT,
    fetch_dataset,
    find_places,
    load_clips,
)

LEAD_IN_FRAMES = 38  # ~3 s
MODEL_FRAMES = 150  # 12 s
SAMPLE_RATE = 24_000


@torch.no_grad()
def speak(model, clip, place, device, generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's channel ``[8, F]`` over its frames, and the partner's true codes."""
    start = max(0, place.t0 - CONTEXT_FRAMES)
    end = min(clip.frames, place.t0 + MODEL_FRAMES)
    own = clip.codes[place.role][:, start:end].to(device).clone()
    partner = clip.codes[1 - place.role][:, start:end].to(device)
    own_active = clip.active[place.role][start:end][None].to(device)
    partner_active = clip.active[1 - place.role][start:end][None].to(device)
    t0 = place.t0 - start
    for frame in range(t0, own.shape[1]):
        output = model(own[0][None], partner[0][None], own_active, partner_active)
        probabilities = (output.own_logits[0, frame].float() / 0.8).softmax(-1)
        cb0 = torch.multinomial(probabilities, 1, generator=generator)
        hidden = output.hidden[0, frame][None].to(torch.float32)
        own[:, frame] = model.sample_frame_codes(cb0, hidden, generator=generator)[0]
    first = max(0, t0 - LEAD_IN_FRAMES)
    return own[:, first:], partner[:, first:]


def write_stereo(path: Path, left: torch.Tensor, right: torch.Tensor) -> None:
    import soundfile

    audio = torch.stack((left.flatten(), right.flatten()), dim=1).cpu().numpy()
    soundfile.write(path, audio, SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = E0bModel(E0bConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    codec = load_mimi(args.mimi, device)
    clips = load_clips(args.data_root or fetch_dataset())
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    for split in (TRAIN_SPLIT, *TEST_SPLITS):
        places = find_places(clips, split)
        # The longest stretch of own speech makes the fullest sample.
        place = max(
            places,
            key=lambda item: int(
                clips[item.clip]
                .active[item.role][item.t0 : item.t0 + MODEL_FRAMES]
                .sum()
            ),
        )
        clip = clips[place.clip]
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            own, partner = speak(model, clip, place, device, generator)
        path = args.output / f"{split}.wav"
        write_stereo(
            path,
            decode(codec, partner[None]),
            decode(codec, own[None]),
        )
        written.append(
            {
                "file": path.name,
                "split": split,
                "clip": clip.file,
                "role": place.role,
                "t0": place.t0,
                "step": payload["step"],
            }
        )
    (args.output / "samples.json").write_text(
        json.dumps(written, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(written, indent=2), flush=True)


if __name__ == "__main__":
    main()
