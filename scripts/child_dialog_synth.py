"""Synthesize the child-dialogue set with Kokoro, several workers per GPU.

Each training question is spoken by ``--train-voices`` voices and each test
question by every held-out voice. Voices are picked per pair by a stable hash,
so reruns give the same plan. Replies are spoken in one or more caregiver
voices, one clip per distinct reply wording and voice; a pair takes one of its
wordings and the reply voice by the same hash, so a concept's reply varies
across its pairs.

Runs in its own environment (``pip install kokoro soundfile``, plus the
espeak-ng system package) and does not import dauys. Writes 24 kHz float WAVs
under ``--output/wavs`` and ``plan.jsonl`` (one row per spoken pair).
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

REPLY_VOICE = "af_heart"
TEST_VOICES = ("af_nova", "am_eric", "bf_lily", "bm_lewis")
TRAIN_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_river",
    "af_sarah", "af_sky", "am_adam", "am_echo", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "bf_alice", "bf_emma", "bf_isabella", "bm_daniel", "bm_fable",
    "bm_george",
)  # fmt: skip


def stable(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


def clip_name(text: str, voice: str) -> str:
    return hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16] + ".wav"


def make_plan(
    pairs: list[dict],
    train_voices: int,
    reply_voices: tuple[str, ...] = (REPLY_VOICE,),
) -> list[dict]:
    plan = []
    for pair in pairs:
        if pair["split"] == "test":
            voices = list(TEST_VOICES)
        else:
            start = stable(pair["id"]) % len(TRAIN_VOICES)
            voices = [
                TRAIN_VOICES[(start + 7 * i) % len(TRAIN_VOICES)]
                for i in range(train_voices)
            ]
        for voice in voices:
            reply = pair["replies"][stable(pair["id"] + voice) % len(pair["replies"])]
            reply_voice = reply_voices[
                stable(pair["id"] + voice + "reply") % len(reply_voices)
            ]
            plan.append(
                {
                    "id": f"{pair['id']}-{voice}",
                    "pair": pair["id"],
                    "split": pair["split"],
                    "concept": pair["concept"],
                    "family": pair["family"],
                    "group": pair["group"],
                    "key": pair["key"],
                    "frame": pair["frame"],
                    "voice": voice,
                    "reply_voice": reply_voice,
                    "question": pair["question"],
                    "reply": reply,
                    "a_wav": clip_name(pair["question"], voice),
                    "b_wav": clip_name(reply, reply_voice),
                }
            )
    return plan


def worker(args):
    jobs, output, device = args
    import numpy as np
    import soundfile
    from kokoro import KPipeline

    pipelines = {}
    done = 0
    for text, voice in jobs:
        path = output / clip_name(text, voice)
        if path.exists():
            continue
        code = voice[0]
        if code not in pipelines:
            pipelines[code] = KPipeline(lang_code=code, device=device)
        pieces = [
            np.asarray(r.audio, dtype=np.float32)
            for r in pipelines[code](text, voice=voice)
            if r.audio is not None
        ]
        if not pieces:
            raise ValueError(f"Kokoro returned no audio for {voice}: {text}")
        temporary = path.with_suffix(".tmp.wav")
        soundfile.write(temporary, np.concatenate(pieces), 24_000, subtype="FLOAT")
        temporary.replace(path)
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-voices", type=int, default=5)
    parser.add_argument("--reply-voices", nargs="+", default=[REPLY_VOICE])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    plan = make_plan(pairs, args.train_voices, args.reply_voices)
    wavs = args.output / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plan))
    wanted = sorted(
        {(p["question"], p["voice"]) for p in plan}
        | {(p["reply"], p["reply_voice"]) for p in plan},
        key=lambda job: job[1],
    )
    # Contiguous chunks keep each worker on a few voices; many chunks balance the load.
    chunks = [wanted[i : i + 200] for i in range(0, len(wanted), 200)]
    started = time.time()
    print(
        json.dumps(
            {"spoken_pairs": len(plan), "clips": len(wanted), "chunks": len(chunks)}
        ),
        flush=True,
    )
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        made = 0
        for index, count in enumerate(
            pool.imap_unordered(worker, [(c, wavs, args.device) for c in chunks]),
            start=1,
        ):
            made += count
            if index % 10 == 0 or index == len(chunks):
                rate = made / max(1.0, time.time() - started)
                print(
                    json.dumps(
                        {
                            "chunks": index,
                            "of": len(chunks),
                            "clips_made": made,
                            "clips_per_s": round(rate, 1),
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps({"done": True, "seconds": round(time.time() - started)}), flush=True
    )


if __name__ == "__main__":
    main()
