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
import random
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


def _blend_spec(rng: random.Random) -> str:
    first = rng.choice(TRAIN_VOICES)
    second = rng.choice(
        [voice for voice in TRAIN_VOICES if voice[0] == first[0] and voice != first]
    )
    weight = round(rng.uniform(0.3, 0.7), 2)
    return f"blend:{first}:{second}:{weight:.2f}"


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


def make_big_plan(
    pairs: list[dict], renditions: int, seed: int = 0, tag: str = ""
) -> list[dict]:
    """Many ranked train renditions with held-in voices and blended speakers.

    ``tag`` only changes the item id, so a second pool cannot collide with the
    first while keeping the same renditions, ranks and audio.
    """
    plan = []
    for pair in pairs:
        if pair["split"] == "test":
            continue
        for index in range(renditions):
            base_identity = f"{pair['id']}-r{index:03d}"
            identity = f"{pair['id']}-{tag}-r{index:03d}" if tag else base_identity
            rng = random.Random(stable(pair["id"] + str(index)) ^ seed)
            if rng.random() < 0.3:
                spec = rng.choice(TRAIN_VOICES)
            else:
                spec = _blend_spec(rng)
            speed = round(rng.uniform(0.85, 1.15), 2)
            reply = pair["replies"][stable(pair["id"] + spec) % len(pair["replies"])]
            plan.append(
                {
                    "id": identity,
                    "pair": pair["id"],
                    "split": "train",
                    "concept": pair["concept"],
                    "family": pair["family"],
                    "group": pair["group"],
                    "key": pair["key"],
                    "frame": pair["frame"],
                    "voice": spec,
                    "speed": speed,
                    "reply_voice": REPLY_VOICE,
                    "question": pair["question"],
                    "reply": reply,
                    "a_wav": clip_name(pair["question"], f"{spec}@{speed}"),
                    "b_wav": clip_name(reply, REPLY_VOICE),
                    "rank": random.Random(stable(base_identity) ^ seed).random(),
                }
            )
    return plan


def shard_plan(plan: list[dict], shard: int, shards: int) -> list[dict]:
    """Contiguous rank blocks, with shard zero holding the lowest ranks."""
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Shard index must lie inside the shard count")
    ordered = sorted(plan, key=lambda item: float(item["rank"]))
    return [
        item
        for position, item in enumerate(ordered)
        if position * shards // len(ordered) == shard
    ]


def worker(args):
    jobs, output, device, pcm16 = args
    import numpy as np
    import soundfile
    from kokoro import KPipeline

    pipelines = {}
    blended = {}
    done = 0
    for text, spec, speed, name in jobs:
        path = output / name
        if path.exists():
            continue
        code = spec.split(":")[1][0] if spec.startswith("blend:") else spec[0]
        if code not in pipelines:
            pipelines[code] = KPipeline(lang_code=code, device=device)
        pipeline = pipelines[code]
        if spec.startswith("blend:"):
            if spec not in blended:
                _, first, second, weight = spec.split(":")
                value = float(weight)
                blended[spec] = value * pipeline.load_voice(first) + (
                    1 - value
                ) * pipeline.load_voice(second)
            voice = blended[spec]
        else:
            voice = spec
        pieces = [
            np.asarray(r.audio, dtype=np.float32)
            for r in pipeline(text, voice=voice, speed=speed)
            if r.audio is not None
        ]
        if not pieces:
            raise ValueError(f"Kokoro returned no audio for {spec}: {text}")
        temporary = path.with_suffix(".tmp.wav")
        soundfile.write(
            temporary,
            np.concatenate(pieces),
            24_000,
            subtype="PCM_16" if pcm16 else "FLOAT",
        )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-tag", default="")
    parser.add_argument("--big-renditions", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--pcm16", action="store_true")
    args = parser.parse_args()
    if args.big_renditions is not None and args.big_renditions < 1:
        raise ValueError("Big renditions must be positive")
    if (args.shard is None) != (args.shards is None):
        parser.error("--shard and --shards must be used together")
    if args.shards is not None and (
        args.shards < 1 or not 0 <= args.shard < args.shards
    ):
        raise ValueError("Shard index must lie inside the shard count")
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    pairs = [
        json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()
    ]
    if args.big_renditions is None:
        plan = make_plan(pairs, args.train_voices, args.reply_voices)
    else:
        plan = make_big_plan(pairs, args.big_renditions, args.seed, args.id_tag)
    if args.shards is not None:
        plan = shard_plan(plan, args.shard, args.shards)
    wavs = args.output / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plan))
    wanted = {}
    for item in plan:
        wanted[item["a_wav"]] = (
            item["question"],
            item["voice"],
            float(item.get("speed", 1.0)),
        )
        wanted[item["b_wav"]] = (
            item["reply"],
            item["reply_voice"],
            float(item.get("reply_speed", 1.0)),
        )
    jobs = [
        (text, spec, speed, name)
        for name, (text, spec, speed) in sorted(
            wanted.items(), key=lambda item: (item[1][1], item[1][0], item[0])
        )
    ]
    # Contiguous chunks keep each worker on a few voices; many chunks balance the load.
    chunks = [jobs[i : i + 200] for i in range(0, len(jobs), 200)]
    started = time.time()
    print(
        json.dumps(
            {"spoken_pairs": len(plan), "clips": len(jobs), "chunks": len(chunks)}
        ),
        flush=True,
    )
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        made = 0
        for index, count in enumerate(
            pool.imap_unordered(
                worker,
                [(c, wavs, args.device, args.pcm16) for c in chunks],
            ),
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
