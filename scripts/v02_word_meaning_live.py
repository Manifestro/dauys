"""Live voices for the word-meaning test.

``list`` writes the phrases to record: four words per topic, each in its
held-out frame, so neither the voice nor the word–frame combination was seen
in training. ``encode`` turns the recordings into a split in the format of
``v02_word_meaning_encode.py`` (A clips from the recordings, B clips copied
from ``test_voice``), which ``v02_word_meaning_eval.py --splits live`` scores.

Recordings: ``<recordings>/<speaker>/<topic>-<word>.wav`` (any rate, mono or
stereo; WAV, FLAC or OGG).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

WORDS_PER_TOPIC = 4


def chosen():
    sys.path.insert(0, str(Path(__file__).parent))
    from v02_word_meaning_text import TOPICS

    for topic, spec in TOPICS.items():
        frames = spec["frames"]
        words = list(spec["words"])
        for index in range(0, len(words), 3)[:WORDS_PER_TOPIC]:
            # The frame the text plan held out for this word.
            yield (
                topic,
                words[index],
                frames[index % len(frames)].format(w=words[index]),
            )


def write_list(path: Path) -> None:
    lines = ["file,phrase"]
    lines += [f'{topic}-{word}.wav,"{text}"' for topic, word, text in chosen()]
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} phrases -> {path}")


def encode_live(args) -> None:
    import numpy as np
    import soundfile
    import torch
    from scipy.signal import resample_poly

    from dauys.codec import encode, load_mimi

    torch.backends.cudnn.allow_tf32 = False
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    replies = {}
    for line in (args.data / "test_voice" / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        replies[(row["topic"], row["word"])] = row["b_clip"]
    out = args.data / "live"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for speaker_dir in sorted(p for p in args.recordings.iterdir() if p.is_dir()):
        for topic, word, _ in chosen():
            matches = [
                p
                for p in speaker_dir.glob(f"{topic}-{word}.*")
                if p.suffix.lower() in (".wav", ".flac", ".ogg")
            ]
            if not matches:
                print(f"missing {speaker_dir.name}/{topic}-{word}")
                continue
            audio, rate = soundfile.read(matches[0], dtype="float32", always_2d=True)
            audio = audio.mean(1)
            if rate != 24_000:
                audio = resample_poly(audio, 24_000, rate).astype(np.float32)
            frames = -(-len(audio) // 1920)
            pcm = np.pad(audio, (0, frames * 1920 - len(audio)))
            with torch.no_grad():
                codes = encode(codec, torch.from_numpy(pcm)[None, None].cuda())[0]
            rms = np.sqrt((pcm.reshape(frames, 1920) ** 2).mean(1))
            name = f"clips/live-{speaker_dir.name}-{topic}-{word}.pt"
            torch.save(
                {
                    "codes": codes.short().cpu(),
                    "activity": torch.from_numpy(
                        (rms > 10 ** (-45 / 20)).astype(np.uint8)
                    ),
                },
                out / name,
            )
            b_clip = replies[(topic, word)]
            if not (out / b_clip).exists():
                shutil.copy(args.data / "test_voice" / b_clip, out / b_clip)
            rows.append(
                {
                    "id": f"live-{speaker_dir.name}-{topic}-{word}",
                    "speaker": speaker_dir.name,
                    "topic": topic,
                    "word": word,
                    "a_clip": name,
                    "b_clip": b_clip,
                }
            )
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"live_items": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--output", type=Path, required=True)
    encoding = sub.add_parser("encode")
    encoding.add_argument("--recordings", type=Path, required=True)
    encoding.add_argument("--data", type=Path, required=True, help="encode output dir")
    encoding.add_argument("--mimi-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        write_list(args.output)
    else:
        encode_live(args)


if __name__ == "__main__":
    main()
