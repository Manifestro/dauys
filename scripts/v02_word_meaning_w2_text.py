"""Text plan for W2: replies that depend on the whole phrase, not one word.

Same item format and splits as ``v02_word_meaning_text.py`` (training voices
in five of the frames, the held-out frame in four training voices and in the
four held-out voices). Within a topic the keys have different replies, so the
minimal-pair test of ``v02_word_meaning_eval.py`` applies unchanged:

- ``neg-<word>``: "I am hungry" vs "I am not hungry" (keys yes/no).
- ``order-<n>``: "What comes after five?" vs "... before five?".
- ``mix``: "What do you get if you mix red and yellow?"; two words decide
  the reply, in either order.
- ``tone-<i>``: the same words as a statement or as a question.
- ``lag-<object>``: "The ball is red." then two to three seconds of other
  speech, then "What color is the ball?"; the reply needs the colour heard
  seconds earlier.

The chapter is the family (neg, order, mix, tone, lag), so training pools have
enough phrases per voice.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v02_word_meaning_text import (
    REPLY_VOICE,
    TEST_VOICES,
    TOPICS,
    TRAIN_VOICES,
)

FEELING_NO = "Okay, that is good to know."
NEED_NO = "Okay, we can do something else."
NUMBERS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]
MIXES = {
    ("red", "yellow"): "Orange.",
    ("red", "blue"): "Purple.",
    ("blue", "yellow"): "Green.",
    ("red", "white"): "Pink.",
    ("black", "white"): "Gray.",
    ("red", "green"): "Brown.",
}
TONES = (
    "It is raining",
    "You are tired",
    "The door is open",
    "Dinner is ready",
    "You are leaving",
    "The baby is sleeping",
)
STATEMENT_REPLY = "Okay, thank you for telling me."
QUESTION_REPLY = "Yes, I think so."
OBJECTS = ("ball", "car", "hat", "cup", "kite", "bird")
COLORS = ("red", "blue", "green", "yellow", "white", "black")
FILLERS = (
    "Okay, now listen to me carefully, I want to ask you something.",
    "Let me think for a moment, there is something I want to know.",
    "Now wait a little bit, I have one more question for you.",
)


def families():
    """Yield (family, topic, key, frames, reply): frames are texts, aligned
    across the keys of one topic so a held-out index means the same carrier."""
    for word, reply in TOPICS["feeling"]["words"].items():
        yield (
            "neg",
            f"neg-{word}",
            "yes",
            (f"I feel {word}.", f"I am so {word}.", f"Mom, I am {word}."),
            reply,
        )
        yield (
            "neg",
            f"neg-{word}",
            "no",
            (f"I do not feel {word}.", f"I am not {word}.", f"Mom, I am not {word}."),
            FEELING_NO,
        )
    for word, reply in TOPICS["need"]["words"].items():
        yield (
            "neg",
            f"neg-{word}",
            "yes",
            (f"I want to {word}.", f"I really want to {word}.", f"Can I {word} now?"),
            reply,
        )
        yield (
            "neg",
            f"neg-{word}",
            "no",
            (
                f"I do not want to {word}.",
                f"I really do not want to {word}.",
                f"I cannot {word} now.",
            ),
            NEED_NO,
        )
    for index in range(1, len(NUMBERS) - 1):
        n = NUMBERS[index]
        for relation, answer in (("after", index + 1), ("before", index - 1)):
            yield (
                "order",
                f"order-{n}",
                relation,
                (
                    f"What comes {relation} {n}?",
                    f"Which number is {relation} {n}?",
                    f"Tell me the number {relation} {n}.",
                ),
                f"{NUMBERS[answer].capitalize()}.",
            )
    for (a, b), reply in MIXES.items():
        frames = []
        for x, y in ((a, b), (b, a)):
            frames += [
                f"What do you get if you mix {x} and {y}?",
                f"Mix {x} with {y}, what color is it?",
                f"What color comes from {x} and {y}?",
            ]
        yield "mix", "mix", f"{a}+{b}", tuple(frames), reply
    for index, sentence in enumerate(TONES):
        yield (
            "tone",
            f"tone-{index}",
            "statement",
            (f"{sentence}.", f"Mom, {sentence.lower()}.", f"Look, {sentence.lower()}."),
            STATEMENT_REPLY,
        )
        yield (
            "tone",
            f"tone-{index}",
            "question",
            (f"{sentence}?", f"Mom, {sentence.lower()}?", f"Look, {sentence.lower()}?"),
            QUESTION_REPLY,
        )
    for obj in OBJECTS:
        for color in COLORS:
            yield (
                "lag",
                f"lag-{obj}",
                color,
                tuple(
                    f"The {obj} is {color}. {filler} What color is the {obj}?"
                    for filler in FILLERS
                ),
                f"It is {color}.",
            )


def plan() -> list[dict]:
    items = []
    topic_index: dict[str, int] = {}
    for family, topic, key, frames, reply in families():
        position = topic_index.setdefault(topic, len(topic_index))
        held_out = position % len(frames)
        for frame_index, text in enumerate(frames):
            if frame_index == held_out:
                splits = (("test_combo", TRAIN_VOICES[:4]), ("test_voice", TEST_VOICES))
            else:
                splits = (("train", TRAIN_VOICES),)
            for split, voices in splits:
                for voice in voices:
                    items.append(
                        {
                            "id": f"w2-{split}-{topic}-{key}-f{frame_index}-{voice}",
                            "split": split,
                            "family": family,
                            "chapter": family,
                            "topic": topic,
                            "word": key,
                            "frame": frame_index,
                            "voice": voice,
                            "reply_voice": REPLY_VOICE,
                            "a_text": text,
                            "b_text": reply,
                        }
                    )
    return items


def check(items: list[dict]) -> None:
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item ids")
    replies: dict[str, dict[str, str]] = {}
    for item in items:
        seen = replies.setdefault(item["topic"], {})
        if seen.setdefault(item["word"], item["b_text"]) != item["b_text"]:
            raise ValueError(f"Key {item['word']} of {item['topic']} has two replies")
    for topic, keys in replies.items():
        if len(set(keys.values())) != len(keys):
            raise ValueError(f"Two keys of {topic} share a reply")
    train = {
        (i["topic"], i["word"], i["frame"]) for i in items if i["split"] == "train"
    }
    held = {(i["topic"], i["word"], i["frame"]) for i in items if i["split"] != "train"}
    if train & held:
        raise ValueError("A held-out phrase is also in training")
    if {i["voice"] for i in items if i["split"] == "train"} & set(TEST_VOICES):
        raise ValueError("A test voice leaked into training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = plan()
    check(items)
    args.output.write_text("".join(json.dumps(i) + "\n" for i in items))
    counts: dict[str, int] = {}
    for item in items:
        key = f"{item['split']}/{item['family']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"items": len(items), **counts}))


if __name__ == "__main__":
    main()
