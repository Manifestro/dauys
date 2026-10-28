"""Generate the model's own spoken replies to word-meaning questions.

The model hears a question A (a voice it never trained on) and then speaks
freely for a few seconds with the default decoding. Files hold the model
channel only; ``replies.csv`` lists each file's question and the reply the
training data paired with that word.
"""

import argparse
import json
from pathlib import Path

import soundfile
import torch

from dauys.checkpoint import load_research_checkpoint
from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.decoding import DecodingConfig
from dauys.model import DauysHybrid


def pick(root: Path, speaker: str | None, per_topic: int):
    rows = [
        json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    chosen: dict[str, list[dict]] = {}
    for row in rows:
        if speaker and row["speaker"] != speaker:
            continue
        topic = chosen.setdefault(row["topic"], [])
        if len(topic) < per_topic and all(r["word"] != row["word"] for r in topic):
            topic.append(row)
    return [row for rows in chosen.values() for row in rows]


@torch.inference_mode()
def reply(model, codec, silence, user_codes, frames: int, seed: int):
    device = next(model.parameters()).device
    quiet = silence.to(device)[None]
    user = user_codes.long().to(device)[None]
    cut = user.shape[-1]
    state = model.prefill_teacher(user, quiet[:, :, None].expand(1, 8, cut), quiet)
    generator = torch.Generator(device=device).manual_seed(seed)
    config = DecodingConfig.v02_default()
    codes, active = [], []
    for _ in range(frames):
        step_codes, speaking = model.generate_step(quiet, state, config, generator)
        codes.append(step_codes)
        active.append(speaking)
    codes = torch.stack(codes, dim=-1)
    speaking = torch.stack(active, dim=-1)[:, None]
    playable = torch.where(speaking, codes, quiet[:, :, None])
    return decode(codec, playable.long())[0, 0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["test_voice:af_nova:1", "test_voice:bm_lewis:1"],
        help="split:speaker:per_topic",
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = {}
    for line in (args.data.parent / "wm-plan.jsonl").read_text().splitlines():
        item = json.loads(line)
        plan.setdefault((item["topic"], item["word"]), item["b_text"])
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    payload = load_research_checkpoint(args.checkpoint)
    model = DauysHybrid(ModelConfig.from_yaml(args.config))
    model.load_state_dict(payload["model"])
    model = model.cuda().eval()
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["file,voice,question_word,expected_reply"]
    for source in args.sources:
        split, speaker, per_topic = source.split(":")
        for index, row in enumerate(pick(args.data / split, speaker, int(per_topic))):
            clip = torch.load(args.data / split / row["a_clip"], weights_only=True)
            wave = reply(model, codec, silence, clip["codes"], args.frames, index)
            name = f"{split}-{row['topic']}-{row['word']}.wav"
            soundfile.write(args.output / name, wave, 24_000)
            expected = plan.get((row["topic"], row["word"]), "")
            lines.append(f'{name},{speaker},{row["word"]},"{expected}"')
    (args.output / "replies.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
