"""Synthesize the word-meaning plan with Kokoro (24 kHz, Mimi's rate).

Runs in its own environment (``pip install kokoro soundfile`` plus the
espeak-ng system package for out-of-lexicon words); it does not import dauys.
Each distinct (text, voice) is synthesized once: user phrases A in their
item's voice, replies B in the fixed reply voice. Output: one float32 WAV per
clip under ``--output/wavs`` and ``clips.jsonl`` mapping (text, voice) to it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE = 24_000


def clip_id(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from kokoro import KPipeline

    items = [json.loads(line) for line in args.plan.read_text().splitlines()]
    wanted = {(i["a_text"], i["voice"]) for i in items}
    wanted |= {(i["b_text"], i["reply_voice"]) for i in items if i["b_text"]}
    pipelines = {
        code: KPipeline(lang_code=code, device=args.device) for code in ("a", "b")
    }
    wavs = args.output / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    for count, (text, voice) in enumerate(sorted(wanted), start=1):
        identity = clip_id(text, voice)
        path = wavs / f"{identity}.wav"
        if not path.exists():
            pieces = [
                np.asarray(result.audio, dtype=np.float32)
                for result in pipelines[voice[0]](text, voice=voice)
                if result.audio is not None
            ]
            if not pieces:
                raise ValueError(f"Kokoro returned no audio for {voice}: {text}")
            soundfile.write(path, np.concatenate(pieces), SAMPLE_RATE, subtype="FLOAT")
        rows.append({"id": identity, "text": text, "voice": voice, "path": path.name})
        if count % 1000 == 0:
            print(json.dumps({"clips": count, "of": len(wanted)}), flush=True)
    (args.output / "clips.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    print(json.dumps({"clips": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
