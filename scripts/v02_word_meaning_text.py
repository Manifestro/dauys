"""Text plan for the word-meaning set: replies that depend on one word.

Every item is a user phrase A and a reply B. Within a topic all A phrases share
carrier frames ("What color is the {w}?") and only the slot word changes; the
reply is fixed by that word ("It is blue."). A model that predicts B from A
better than chance has to tell the words apart by sound and map each to its
reply. Text is used only to synthesize audio; the model never sees it.

Splits, so a pass cannot come from memorizing whole phrases or voices:

- ``train``: each word in five of its topic's six frames, training voices.
- ``test_combo``: each word in its held-out frame, four training voices.
- ``test_voice``: the held-out frame in four voices never used for training.
- ``control``: nonsense words in the held-out frames, held-out voices.
- ``val``: the training frames in the held-out voices; only the training
  loop's held-out metrics use it (they need eight words per frame and voice).

Minimal pairs for the evaluation are two items of one topic, frame and voice
that differ only in the slot word.
"""

import argparse
import json
from pathlib import Path

REPLY_VOICE = "af_heart"
TEST_VOICES = ("af_nova", "am_eric", "bf_lily", "bm_lewis")
TRAIN_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bm_daniel",
    "bm_fable",
    "bm_george",
)
NONSENSE = ("blick", "dax", "wug", "fep", "zorp", "tiv")

TOPICS = {
    "need": {
        "frames": (
            "I want to {w}.",
            "Can I {w} now?",
            "Mom, I need to {w}.",
            "I really want to {w}.",
            "Let me {w}, please.",
            "It is time to {w}.",
        ),
        "words": {
            "drink": "Here is a glass of water.",
            "eat": "Here is some bread.",
            "sleep": "Okay, go to bed.",
            "play": "Let's play with the ball.",
            "read": "Here is a book.",
            "walk": "Let's go outside.",
            "swim": "Let's go to the pool.",
            "sing": "La la la, let's sing together.",
            "dance": "Let's turn on the music.",
            "draw": "Here are some pencils.",
            "wash": "Here is the soap.",
            "rest": "Sit down on the sofa.",
        },
    },
    "color": {
        "frames": (
            "What color is the {w}?",
            "Tell me the color of the {w}.",
            "What is the color of the {w}?",
            "Do you know what color the {w} is?",
            "The {w}, what color is it?",
            "Which color is the {w}?",
        ),
        "words": {
            "sky": "It is blue.",
            "grass": "It is green.",
            "snow": "It is white.",
            "sun": "It is yellow.",
            "strawberry": "It is red.",
            "coal": "It is black.",
            "carrot": "It is orange.",
            "plum": "It is purple.",
            "chocolate": "It is brown.",
            "elephant": "It is gray.",
            "flamingo": "It is pink.",
            "lemon": "It is bright yellow, like the sun.",
        },
    },
    "animal": {
        "frames": (
            "What does a {w} say?",
            "How does a {w} sound?",
            "What sound does a {w} make?",
            "Make the sound of a {w}.",
            "Show me how a {w} talks.",
            "Do you know what a {w} says?",
        ),
        "words": {
            "cat": "It says meow.",
            "dog": "It says woof woof.",
            "cow": "It says moo.",
            "duck": "It says quack quack.",
            "sheep": "It says baa.",
            "pig": "It says oink oink.",
            "horse": "It says neigh.",
            "owl": "It says hoot hoot.",
            "frog": "It says ribbit.",
            "lion": "It roars, roar!",
            "mouse": "It says squeak squeak.",
            "bee": "It goes buzz.",
        },
    },
    "opposite": {
        "frames": (
            "What is the opposite of {w}?",
            "Tell me the opposite of {w}.",
            "The opposite of {w} is what?",
            "Say the opposite of {w}.",
            "What word means the opposite of {w}?",
            "Do you know the opposite of {w}?",
        ),
        "words": {
            "hot": "Cold.",
            "big": "Small.",
            "up": "Down.",
            "day": "Night.",
            "fast": "Slow.",
            "open": "Closed.",
            "wet": "Dry.",
            "happy": "Sad.",
            "old": "Young.",
            "full": "Empty.",
            "light": "Dark.",
            "loud": "Quiet.",
        },
    },
    "feeling": {
        "frames": (
            "I feel {w}.",
            "I am so {w}.",
            "Today I feel {w}.",
            "Mom, I am {w}.",
            "I think I am {w}.",
            "I am getting {w}.",
        ),
        "words": {
            "cold": "Put on a warm jacket.",
            "warm": "Let's open the window.",
            "tired": "You should take a nap.",
            "sick": "Let's call the doctor.",
            "sad": "Come here, I will hug you.",
            "bored": "Let's play a game.",
            "scared": "Don't worry, I am here.",
            "thirsty": "Here is some juice.",
            "hungry": "Let's make a sandwich.",
            "sleepy": "Let's turn off the light.",
            "happy": "I am glad to hear that.",
            "lonely": "I will stay with you.",
        },
    },
    "number": {
        "frames": (
            "What comes after {w}?",
            "What number is after {w}?",
            "Count one more after {w}.",
            "Which number follows {w}?",
            "After {w} comes what?",
            "Tell me the number after {w}.",
        ),
        "words": {
            "one": "Two.",
            "two": "Three.",
            "three": "Four.",
            "four": "Five.",
            "five": "Six.",
            "six": "Seven.",
            "seven": "Eight.",
            "eight": "Nine.",
            "nine": "Ten.",
            "ten": "Eleven.",
            "eleven": "Twelve.",
            "twelve": "Thirteen.",
        },
    },
    "place": {
        "frames": (
            "Where do we keep the {w}?",
            "Where should I put the {w}?",
            "Where is the {w} kept?",
            "Tell me where the {w} goes.",
            "In which room is the {w}?",
            "Where can I find the {w}?",
        ),
        "words": {
            "milk": "In the fridge.",
            "clothes": "In the wardrobe.",
            "books": "On the shelf.",
            "car": "In the garage.",
            "plates": "In the cupboard.",
            "toys": "In the toy box.",
            "shoes": "By the front door.",
            "towels": "In the bathroom.",
            "pillows": "On the bed.",
            "flowers": "In the vase.",
            "coats": "On the hook.",
            "trash": "In the bin.",
        },
    },
    "tool": {
        "frames": (
            "What do we use a {w} for?",
            "What is a {w} for?",
            "Why do we need a {w}?",
            "What can I do with a {w}?",
            "Tell me what a {w} is used for.",
            "What does a {w} help us do?",
        ),
        "words": {
            "spoon": "To eat soup.",
            "pen": "To write a letter.",
            "key": "To open the door.",
            "towel": "To dry ourselves.",
            "umbrella": "To stay dry in the rain.",
            "phone": "To call our friends.",
            "knife": "To cut the bread.",
            "toothbrush": "To clean our teeth.",
            "scissors": "To cut paper.",
            "cup": "To drink tea.",
            "bed": "To sleep at night.",
            "chair": "To sit down.",
        },
    },
}


def plan() -> list[dict]:
    items = []
    for topic, spec in TOPICS.items():
        frames = spec["frames"]
        for word_index, (word, reply) in enumerate(spec["words"].items()):
            held_out = word_index % len(frames)
            for frame_index, frame in enumerate(frames):
                text = frame.format(w=word)
                if frame_index == held_out:
                    splits = (
                        ("test_combo", TRAIN_VOICES[:4]),
                        ("test_voice", TEST_VOICES),
                    )
                else:
                    splits = (("train", TRAIN_VOICES), ("val", TEST_VOICES))
                for split, voices in splits:
                    for voice in voices:
                        items.append(
                            {
                                "id": f"{split}-{topic}-{word}-f{frame_index}-{voice}",
                                "split": split,
                                "topic": topic,
                                "word": word,
                                "frame": frame_index,
                                "voice": voice,
                                "reply_voice": REPLY_VOICE,
                                "a_text": text,
                                "b_text": reply,
                            }
                        )
        # Controls: nonsense in the held-out frames, held-out voices, no reply.
        for frame_index, frame in enumerate(frames):
            for nonsense in NONSENSE:
                for voice in TEST_VOICES:
                    items.append(
                        {
                            "id": f"control-{topic}-{nonsense}-f{frame_index}-{voice}",
                            "split": "control",
                            "topic": topic,
                            "word": nonsense,
                            "frame": frame_index,
                            "voice": voice,
                            "reply_voice": REPLY_VOICE,
                            "a_text": frame.format(w=nonsense),
                            "b_text": None,
                        }
                    )
    return items


def check(items: list[dict]) -> None:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item ids")
    for topic, spec in TOPICS.items():
        replies = list(spec["words"].values())
        if len(replies) != len(set(replies)):
            raise ValueError(f"Two words of {topic} share a reply")
    train_voices = {i["voice"] for i in items if i["split"] == "train"}
    if train_voices & set(TEST_VOICES):
        raise ValueError("A test voice leaked into training")
    train = {
        (i["topic"], i["word"], i["frame"]) for i in items if i["split"] == "train"
    }
    held = {
        (i["topic"], i["word"], i["frame"])
        for i in items
        if i["split"].startswith("test")
    }
    if train & {h for h in held if h[1] not in NONSENSE}:
        raise ValueError("A held-out word-frame pair is also in training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = plan()
    check(items)
    args.output.write_text("".join(json.dumps(i) + "\n" for i in items))
    counts: dict[str, int] = {}
    for item in items:
        counts[item["split"]] = counts.get(item["split"], 0) + 1
    print(json.dumps({"items": len(items), **counts}))


if __name__ == "__main__":
    main()
