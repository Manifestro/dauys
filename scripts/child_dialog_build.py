"""Build the child-dialogue text set from reviewed concepts and carrier frames.

Every pair is correct by construction: a reviewed concept (key word and
answer, ``scripts/child_dialog_concepts_v1.jsonl``) is placed into
hand-written child phrasings of its family, and the reply comes from the
family's reply templates. Free LLM generation was tried first and rejected:
most generated questions did not match their answer.

Splits: per family, three frames are held out. ``test`` pairs use only those
frames (every concept in a phrasing never trained), ``train`` the rest.
Outputs ``concepts.jsonl``, ``frames.jsonl``, ``pairs.jsonl`` and
``summary.json``.
"""

import argparse
import json
import random
import re
from pathlib import Path

PLURAL = {
    "eyes",
    "ears",
    "hands",
    "feet",
    "teeth",
    "legs",
    "lungs",
    "shoes",
    "toys",
    "books",
    "socks",
    "eggs",
    "forks",
    "clothes",
    "grapes",
    "scissors",
    "glasses",
}
CATEGORY_MASS = {
    "broccoli",
    "lettuce",
    "corn",
    "spinach",
    "celery",
    "bread",
    "soup",
    "cake",
    "grass",
}
PLACE_DROP = {"couch", "bed"}
MASS = {"water", "milk", "juice", "popcorn", "ice cream", "grapes"}
NEED_VERBS = {"drink", "eat", "sleep", "play", "read", "draw", "swim"}
NEED_PLACES = {"park", "potty"}
NEED_DROP = {"music", "song", "sand", "swing", "slide"}
FEELING_HAVE = {"tummy ache", "headache"}
WEATHER = {
    "rain": "it is raining",
    "sun": "the sun is out",
    "wind": "it is so windy",
    "snow": "it is snowing",
    "storm": "there is a storm",
    "mud": "there is mud everywhere",
    "dark": "it is dark outside",
    "ice": "there is ice on the road",
    "thunder": "I hear thunder",
    "hot": "it is so hot outside",
}

FRAMES = {
    "need_verb": [
        "I want to {k}.",
        "Can I {k} now?",
        "Mom, I want to {k}.",
        "I really want to {k}.",
        "Can I {k}, please?",
        "Please, can I {k}?",
        "I need to {k}.",
        "Can we {k} now?",
        "I want to {k} now.",
        "Mom, can I {k}?",
        "Let me {k}, please.",
        "I would like to {k}.",
        "Can I {k} now, please?",
        "I just want to {k}.",
        "Mommy, I want to {k}.",
    ],
    "need_thing": [
        "I want {a} {k}.",
        "Can I have {a} {k}?",
        "Mom, I want {a} {k}.",
        "Please, can I have {a} {k}?",
        "I would like {a} {k}.",
        "Can I get {a} {k}?",
        "Mommy, can I have {a} {k}?",
        "I really want {a} {k}.",
        "Give me {a} {k}, please.",
        "May I have {a} {k}?",
        "I want {a} {k}, please.",
        "Can I have {a} {k} now?",
        "I need {a} {k}.",
        "Mom, can I have {a} {k}?",
        "Daddy, I want {a} {k}.",
    ],
    "need_place": [
        "I want to go to the {k}.",
        "Can we go to the {k}?",
        "I need to go to the {k}.",
        "Mom, can we go to the {k}?",
        "Please, can we go to the {k}?",
        "Let's go to the {k}.",
        "Can I go to the {k} now?",
        "I really want to go to the {k}.",
        "I want to go to the {k} now.",
        "Take me to the {k}, please.",
        "Can we go to the {k} now?",
        "Mom, I want to go to the {k}.",
        "Can we please go to the {k}?",
        "Can you take me to the {k}?",
        "Daddy, can we go to the {k}?",
    ],
    "feeling": [
        "I feel {k}.",
        "I am {k}.",
        "Mom, I am {k}.",
        "I am so {k}.",
        "I feel so {k}.",
        "Mommy, I feel {k}.",
        "I am really {k}.",
        "I think I am {k}.",
        "I am very {k}.",
        "Mom, I feel {k}.",
        "Daddy, I am {k}.",
        "I feel really {k}.",
        "I am getting {k}.",
        "I am so {k} today.",
        "Mom, I feel so {k}.",
    ],
    "feeling_have": [
        "I have a {k}.",
        "Mom, I have a {k}.",
        "I have a bad {k}.",
        "Mommy, I have a {k}.",
        "I think I have a {k}.",
        "I have a big {k}.",
        "Daddy, I have a {k}.",
        "I got a {k}.",
        "Mom, I got a {k}.",
        "I have a {k} now.",
        "Ouch, I have a {k}.",
        "I have a {k} today.",
        "I have a {k}, mom.",
        "I still have a {k}.",
        "Mom, I have a bad {k}.",
    ],
    "weather": [
        "{S}.",
        "Mom, {s}.",
        "Look, {s}.",
        "Oh no, {s}.",
        "Mommy, {s}.",
        "Daddy, {s}.",
        "Look, mom, {s}.",
        "Mom, look, {s}.",
        "Oh, {s}.",
        "{S} today.",
        "Hey, {s}.",
        "Mom, {s} today.",
        "Wow, {s}.",
        "Mommy, look, {s}.",
        "Oh no, mom, {s}.",
    ],
    "color": [
        "What color is the {k}?",
        "What color is the {k} usually?",
        "Mom, what color is the {k}?",
        "Tell me the color of the {k}.",
        "What is the color of the {k}?",
        "Do you know what color the {k} is?",
        "Which color is the {k}?",
        "What color does the {k} have?",
        "Mommy, what color is the {k}?",
        "What color is the {k}, mom?",
        "Can you tell me the color of the {k}?",
        "The {k}, what color is it?",
        "What is the {k} color?",
        "What color is the {k}, daddy?",
        "Daddy, what color is the {k}?",
    ],
    "animal_sound": [
        "What does {a} {k} say?",
        "What sound does {a} {k} make?",
        "How does {a} {k} sound?",
        "What noise does {a} {k} make?",
        "Mom, what does {a} {k} say?",
        "What does the {k} say?",
        "Make the sound of {a} {k}.",
        "How does the {k} go?",
        "What sound does the {k} make?",
        "Mommy, what does {a} {k} say?",
        "Show me how {a} {k} talks.",
        "What does {a} {k} say, mom?",
        "Do you know what {a} {k} says?",
        "Tell me what {a} {k} says.",
        "Daddy, what does {a} {k} say?",
    ],
    "baby_animal": [
        "What is a baby {k} called?",
        "What do we call a baby {k}?",
        "What is a baby {k}?",
        "Mom, what is a baby {k} called?",
        "What is the name of a baby {k}?",
        "Do you know what a baby {k} is called?",
        "What do you call a baby {k}?",
        "Tell me what a baby {k} is called.",
        "Mommy, what is a baby {k}?",
        "What is a little {k} called?",
        "What is the baby of a {k} called?",
        "How do we call a baby {k}?",
        "Daddy, what is a baby {k} called?",
        "What is a baby {k} called, mom?",
        "What name does a baby {k} have?",
    ],
    "animal_home": [
        "Where does {a} {k} live?",
        "Where do {p} live?",
        "Where is the home of {a} {k}?",
        "Mom, where does {a} {k} live?",
        "Where does the {k} live?",
        "Do you know where {a} {k} lives?",
        "Where does {a} {k} sleep?",
        "Where is the home of the {k}?",
        "Mommy, where does {a} {k} live?",
        "Tell me where {a} {k} lives.",
        "Where does {a} {k} make its home?",
        "Where does {a} {k} stay?",
        "Daddy, where does {a} {k} live?",
        "Where does {a} {k} live, mom?",
        "Where is the house of the {k}?",
    ],
    "use": [
        "What is {a} {k} for?",
        "What do we use {a} {k} for?",
        "What can I do with {a} {k}?",
        "Mom, what is {a} {k} for?",
        "Why do we need {a} {k}?",
        "What does {a} {k} do?",
        "What is the {k} for?",
        "Tell me what {a} {k} is for.",
        "What do we do with {a} {k}?",
        "Mommy, what is {a} {k} for?",
        "What is this {k} for?",
        "Why do we have {a} {k}?",
        "What do you do with {a} {k}?",
        "What is {a} {k} for, mom?",
        "Daddy, what is {a} {k} for?",
    ],
    "use_plural": [
        "What are {k} for?",
        "What do we use {k} for?",
        "What can I do with {k}?",
        "Mom, what are {k} for?",
        "Why do we need {k}?",
        "What do {k} do?",
        "What are the {k} for?",
        "Tell me what {k} are for.",
        "What do we do with {k}?",
        "Mommy, what are {k} for?",
        "What are these {k} for?",
        "Why do we have {k}?",
        "What do you do with {k}?",
        "What are {k} for, mom?",
        "Daddy, what are {k} for?",
    ],
    "place": [
        "Where do we keep the {k}?",
        "Where {does} the {k} go?",
        "Where {is} the {k}?",
        "Mom, where do we keep the {k}?",
        "Where should I put the {k}?",
        "Where do I put the {k}?",
        "Where can I find the {k}?",
        "Where {is} the {k} kept?",
        "Mommy, where {is} the {k}?",
        "Tell me where the {k} {goes}.",
        "Where do we put the {k}?",
        "Where should the {k} go?",
        "Daddy, where do we keep the {k}?",
        "Where {is} the {k}, mom?",
        "Where {does} the {k} usually go?",
    ],
    "opposite": [
        "What is the opposite of {k}?",
        "Tell me the opposite of {k}.",
        "What is the opposite of the word {k}?",
        "Mom, what is the opposite of {k}?",
        "Do you know the opposite of {k}?",
        "Say the opposite of {k}.",
        "What word is the opposite of {k}?",
        "What is the other side of {k}?",
        "Mommy, what is the opposite of {k}?",
        "Can you tell me the opposite of {k}?",
        "What means the opposite of {k}?",
        "The opposite of {k} is what?",
        "What is the opposite of {k}, mom?",
        "Daddy, what is the opposite of {k}?",
        "What is the opposite of {k}, daddy?",
    ],
    "body": [
        "What do we do with our {k}?",
        "What {are} our {k} for?",
        "What {do} our {k} do?",
        "Mom, what do we do with our {k}?",
        "Why do we have {k_a}?",
        "What can I do with my {k}?",
        "What {are} my {k} for?",
        "Tell me what our {k} {do}.",
        "Mommy, what {are} our {k} for?",
        "What do I use my {k} for?",
        "What do we use our {k} for?",
        "What {is_are} the {k} for?",
        "Daddy, what do we do with our {k}?",
        "What do you do with your {k}?",
        "What {are} {k_a} for?",
    ],
    "category": [
        "Is {a} {k} {o1} or {o2}?",
        "Is {a} {k} {o2} or {o1}?",
        "Mom, is {a} {k} {o1} or {o2}?",
        "Tell me, is {a} {k} {o1} or {o2}?",
        "Is the {k} {o1} or {o2}?",
        "Is {a} {k} {o2} or is it {o1}?",
        "Do you know if {a} {k} is {o1} or {o2}?",
        "Mommy, is {a} {k} {o2} or {o1}?",
        "What is {a} {k}, {o1} or {o2}?",
        "Is {a} {k} {o1}, or is it {o2}?",
        "Daddy, is {a} {k} {o1} or {o2}?",
        "Is {a} {k} {o2}, or {o1}?",
        "Which one is {a} {k}, {o1} or {o2}?",
        "What is {a} {k}, {o2} or {o1}?",
        "Is {a} {k} {o1} or {o2}, mom?",
    ],
    "number_next": [
        "What comes after {k}?",
        "What number comes after {k}?",
        "What is after {k}?",
        "Mom, what comes after {k}?",
        "Which number is after {k}?",
        "What number follows {k}?",
        "Tell me the number after {k}.",
        "After {k}, what comes next?",
        "What is the next number after {k}?",
        "Mommy, what comes after {k}?",
        "What comes next after {k}?",
        "Count one more after {k}.",
        "What number is next after {k}?",
        "Daddy, what comes after {k}?",
        "After {k} comes what?",
    ],
    "number_before": [
        "What comes before {k}?",
        "What number comes before {k}?",
        "What is before {k}?",
        "Mom, what comes before {k}?",
        "Which number is before {k}?",
        "What number is just before {k}?",
        "Tell me the number before {k}.",
        "Before {k}, what number is there?",
        "What is the number before {k}?",
        "Mommy, what comes before {k}?",
        "What comes just before {k}?",
        "Count back one from {k}.",
        "What number is right before {k}?",
        "Daddy, what comes before {k}?",
        "Before {k} comes what?",
    ],
    "add": [
        "What is {x} plus {y}?",
        "How much is {x} and {y}?",
        "What is {x} and {y} together?",
        "Mom, what is {x} plus {y}?",
        "If I have {x} and get {y} more, how many?",
        "What do {x} and {y} make?",
        "Tell me {x} plus {y}.",
        "How many is {x} plus {y}?",
        "Mommy, what is {x} and {y}?",
        "Add {x} and {y}, please.",
        "What is {x} add {y}?",
        "{X} plus {y} is what?",
        "Daddy, what is {x} plus {y}?",
        "How much is {x} plus {y}?",
        "Can you add {x} and {y}?",
    ],
}
REPLIES = {
    "color": ["It is {w}.", "{W}.", "The {k} is {w}.", "It's {w}."],
    "animal_sound": ["It says {w}.", "{W}!", "{A} {k} says {w}."],
    "baby_animal": ["{ans}", "It is called {ans_l}", "A baby {k} is {ans_l}"],
    "animal_home": ["{ans}", "It lives {ans_l}", "{A} {k} lives {ans_l}"],
    "use": ["{ans}", "We use it {ans_l}"],
    "use_plural": ["{ans}", "We use them {ans_l}"],
    "place": ["{ans}", "We keep {it} {ans_l}"],
    "opposite": ["{ans}", "The opposite is {w}.", "It is {w}."],
    "number_next": ["{ans}", "{W} comes after {k}.", "After {k} comes {w}."],
    "number_before": ["{ans}", "{W} comes before {k}.", "Before {k} comes {w}."],
    "add": ["{ans}", "{X} plus {y} is {w}.", "That makes {w}."],
    "category": ["{ans}", "It is {ans_l}", "{A} {k} is {ans_l}"],
}
HELD_OUT = 3


def article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def plural(word: str) -> str:
    irregular = {
        "mouse": "mice",
        "sheep": "sheep",
        "fish": "fish",
        "deer": "deer",
        "wolf": "wolves",
        "goose": "geese",
        "polar bear": "polar bears",
        "octopus": "octopuses",
        "fox": "foxes",
    }
    return irregular.get(word, word + "s")


def group(concept: dict) -> str | None:
    family, key = concept["family"], concept["key"]
    if family == "need":
        if key in NEED_DROP:
            return None
        return (
            "need_verb"
            if key in NEED_VERBS
            else "need_place"
            if key in NEED_PLACES
            else "need_thing"
        )
    if family == "place" and key in PLACE_DROP:
        return None
    if family == "feeling" and key in FEELING_HAVE:
        return "feeling_have"
    if family == "use" and key in PLURAL:
        return "use_plural"
    return family


def answer_word(answer: str) -> str:
    """The defining last word(s) of a short answer: 'It is blue.' -> 'blue'."""
    text = answer.rstrip(".!").strip()
    for prefix in ("It is ", "It says "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text.lower()


def fields(concept: dict) -> dict:
    key = concept["key"]
    many = key in PLURAL
    ans = concept["answer"]
    w = answer_word(ans)
    out = {
        "k": key,
        "a": "" if many else article(key),
        "A": "" if many else article(key).capitalize(),
        "p": plural(key),
        "w": w,
        "W": w[:1].upper() + w[1:],
        "ans": ans,
        "ans_l": ans[:1].lower() + ans[1:],
        "is": "are" if many else "is",
        "does": "do" if many else "does",
        "goes": "go" if many else "goes",
        "it": "them" if many else "it",
        "are": "are" if many else "is",
        "do": "do" if many else "does",
        "is_are": "are" if many else "is",
        "k_a": key if many else f"{article(key)} {key}",
    }
    if concept["family"] == "need" and key in MASS:
        out |= {"a": "some", "A": "Some"}
    if concept["family"] == "category" and key in CATEGORY_MASS:
        out |= {"a": "", "A": ""}
    if concept["family"] == "category":
        o1, o2 = concept["options"].split(" or a ")
        out |= {"o1": f"{article(o1)} {o1}", "o2": f"{article(o2)} {o2}"}
    if concept["family"] == "add":
        x, y = key.split()
        out |= {"x": x, "y": y, "X": x.capitalize()}
    if concept["family"] == "weather":
        statement = WEATHER[key]
        out |= {"s": statement, "S": statement[:1].upper() + statement[1:]}
    return out


def render(template: str, values: dict) -> str:
    text = template.format(**values)
    text = (
        re.sub(r"\s+", " ", text)
        .strip()
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace(" .", ".")
    )
    return text[:1].upper() + text[1:]


def build(concepts: list[dict], seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    held = {g: set(rng.sample(range(len(f)), HELD_OUT)) for g, f in FRAMES.items()}
    frames = [
        {
            "group": g,
            "index": i,
            "template": t,
            "split": "test" if i in held[g] else "train",
        }
        for g, ts in FRAMES.items()
        for i, t in enumerate(ts)
    ]
    kept, pairs = [], []
    for concept in concepts:
        g = group(concept)
        if g is None:
            continue
        values = fields(concept)
        reply_templates = REPLIES.get(g) or REPLIES.get(concept["family"]) or ["{ans}"]
        replies = list(dict.fromkeys(render(t, values) for t in reply_templates))
        kept.append({**concept, "group": g, "replies": replies})
        for index, template in enumerate(FRAMES[g]):
            question = render(template, values)
            pairs.append(
                {
                    "id": f"{concept['id']}-f{index:02d}",
                    "concept": concept["id"],
                    "family": concept["family"],
                    "group": g,
                    "key": concept["key"],
                    "frame": index,
                    "question": question,
                    "answer": concept["answer"],
                    "replies": replies,
                    "split": "test" if index in held[g] else "train",
                }
            )
    return kept, frames, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path(__file__).with_name("child_dialog_concepts_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    concepts = [json.loads(line) for line in args.concepts.read_text().splitlines()]
    kept, frames, pairs = build(concepts, args.seed)
    questions = [p["question"] for p in pairs]
    if len(questions) != len({q.lower() for q in questions}):
        raise ValueError("Two pairs share a question with different concepts")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("concepts", kept), ("frames", frames), ("pairs", pairs)):
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    summary = {
        "concepts": len(kept),
        "frames": len(frames),
        "pairs": len(pairs),
        "test_pairs": sum(p["split"] == "test" for p in pairs),
        "by_family": {
            f: sum(c["family"] == f for c in kept)
            for f in sorted({c["family"] for c in kept})
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
