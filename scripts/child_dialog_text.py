"""Build the child-dialogue text set: questions and replies for synthesis.

Every pair is a short child-directed exchange in which the reply follows from
the meaning of the question: what the child wants, how they feel, what colour
a thing is, what an animal says, and so on. Text only exists to be synthesised;
the model never sees it.

Stages (each reads the previous stage's file and writes its own):

- ``concepts``: an LLM lists items per family with their answer; a second
  pass judges each fact, and only facts judged correct, common knowledge and
  unambiguous are kept. Numbers and sums are computed, not generated.
- ``expand``: per concept, 40 ways a child could say it (all containing the
  key word) and 4 ways to reply. ``--concepts`` takes a reviewed concept list
  instead of the generated one (``scripts/child_dialog_concepts_v1.jsonl``).
- ``validate``: rule checks (key word present, answer not given away, length,
  characters, duplicates), then an LLM judge drops unnatural or ambiguous
  lines and replies that do not answer the question.
- ``export``: ``concepts.jsonl``, ``pairs.jsonl`` and a summary, with three
  questions per concept held out as ``test`` phrasing.

The LLM is an OpenAI-compatible server (vLLM) at ``--base-url``.
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

FAMILIES = {
    "need": (
        (
            "things a small child wants to do or have right now (drink, eat, sleep, "
            "play, read, draw, swim, a hug, a banana...)"
        ),
        "a short, kind caregiver reply that gives it or starts doing it",
        "I want to drink.",
        "Here is some water.",
    ),
    "feeling": (
        (
            "how a small child feels in body or mood (cold, hot, tired, sick, sad, "
            "scared, bored, hungry, thirsty, happy...)"
        ),
        "a short, kind caregiver reply that helps with that feeling",
        "I feel cold.",
        "Put on your warm jacket.",
    ),
    "color": (
        (
            "everyday things whose usual colour every child knows (sky, grass, "
            "snow, banana, tomato, coal, milk...); skip things with no single colour"
        ),
        "the colour",
        "What color is the grass?",
        "It is green.",
    ),
    "animal_sound": (
        "animals whose sound children learn (cat, dog, cow, duck, sheep, owl, bee...)",
        "the sound it makes",
        "What does a cow say?",
        "It says moo.",
    ),
    "use": (
        "household objects a child knows (spoon, key, towel, broom, cup, pillow...)",
        "what we use it for",
        "What is a towel for?",
        "To dry yourself.",
    ),
    "place": (
        (
            "things at home a child sees every day (milk, shoes, toothbrush, "
            "plates, toys, books...)"
        ),
        "where we usually keep or find it",
        "Where do we keep the milk?",
        "In the fridge.",
    ),
    "opposite": (
        (
            "simple words with a clear opposite (hot, big, up, open, wet, fast, "
            "full, light...)"
        ),
        "the opposite word",
        "What is the opposite of hot?",
        "Cold.",
    ),
    "body": (
        "body parts a child can name (eyes, ears, nose, mouth, hands, feet, teeth...)",
        "what we do with it",
        "What do we do with our ears?",
        "We hear with them.",
    ),
    "baby_animal": (
        (
            "animals whose baby has its own name that children learn (dog, cat, "
            "cow, sheep, horse, duck, frog...)"
        ),
        "the name of the baby",
        "What is a baby dog called?",
        "A puppy.",
    ),
    "animal_home": (
        (
            "animals whose home has a name children learn (bird, bee, spider, "
            "fish, rabbit, horse...)"
        ),
        "where it lives",
        "Where does a bird live?",
        "In a nest.",
    ),
    "weather": (
        (
            "weather and situations a child notices outside (rain, snow, sun, "
            "wind, dark, mud...)"
        ),
        "a short caregiver reply about what to wear or do",
        "It is raining outside.",
        "Take your umbrella.",
    ),
    "category": (
        (
            "everyday things that clearly belong to one of two simple groups a "
            "child learns (fruit or vegetable, animal or plant, food or toy...)"
        ),
        "the right group",
        "Is a carrot a fruit or a vegetable?",
        "A vegetable.",
    ),
}
# Families whose question names the options, so the answer word may appear in it,
# and caregiver-action families, whose reply is free and cannot be given away.
OPTION_FAMILIES = {"category"}
FREE_REPLY_FAMILIES = {"need", "feeling", "weather"}
STOP = {
    "here",
    "some",
    "the",
    "you",
    "your",
    "with",
    "them",
    "and",
    "for",
    "are",
    "its",
    "will",
    "can",
    "let",
    "lets",
    "get",
    "put",
    "take",
    "have",
    "this",
    "that",
    "there",
    "yes",
    "okay",
    "good",
    "now",
    "out",
    "all",
    "our",
    "very",
    "too",
    "then",
}
NUMBERS = [
    "zero",
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
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
LINE = re.compile(r"^[A-Za-z][A-Za-z ,.'?!-]*$")
QUESTIONS, REPLIES = 40, 4
HELD_OUT = 3


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def stems(text: str) -> set[str]:
    return {w.rstrip("s") for w in words(text) if len(w) > 2 and w not in STOP}


async def chat(client, model, prompt, schema, *, temperature=0.7, retries=3):
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=3000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": schema},
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2)


def list_schema(item: dict) -> dict:
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
    }


PAIR = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["key", "answer"],
}
VERDICT = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "common_knowledge": {"type": "boolean"},
        "unambiguous": {"type": "boolean"},
    },
    "required": ["correct", "common_knowledge", "unambiguous"],
}


async def build_concepts(client, model, per_family: int, rounds: int) -> list[dict]:
    async def propose(family, round_index):
        what, answer, _, _ = FAMILIES[family]
        prompt = (
            f"List {per_family} different {what}. For each give the key word "
            f"(one or two plain English words, singular where natural) and "
            f"{answer}, as a 4-year-old would learn it. Use the simplest common "
            f"words. Batch {round_index + 1}: prefer items you have not listed "
            f"before, but only very common ones."
        )
        out = await chat(client, model, prompt, list_schema(PAIR), temperature=0.9)
        return [
            (family, i["key"].strip().lower(), i["answer"].strip())
            for i in out["items"]
        ]

    batches = await asyncio.gather(
        *(propose(f, r) for f in FAMILIES for r in range(rounds))
    )
    seen, candidates = set(), []
    for family, key, answer in (x for b in batches for x in b):
        if (family, key) in seen or not LINE.match(key) or not answer:
            continue
        seen.add((family, key))
        candidates.append({"family": family, "key": key, "answer": answer})

    async def judge(concept):
        what, answer, _, _ = FAMILIES[concept["family"]]
        prompt = (
            f"Family: {what}. Expected: {answer}.\nKey: {concept['key']}\n"
            f"Answer: {concept['answer']}\nIs the answer correct, common "
            f"knowledge for a 4-year-old, and the one answer most people would "
            f"give? Be strict."
        )
        verdict = await chat(client, model, prompt, VERDICT, temperature=0.0)
        return all(verdict.values())

    keep = await asyncio.gather(*(judge(c) for c in candidates))
    concepts = [c for c, ok in zip(candidates, keep) if ok]
    for n in range(1, 20):
        concepts.append(
            {
                "family": "number_next",
                "key": NUMBERS[n],
                "answer": NUMBERS[n + 1].capitalize() + ".",
            }
        )
        concepts.append(
            {
                "family": "number_before",
                "key": NUMBERS[n],
                "answer": NUMBERS[n - 1].capitalize() + ".",
            }
        )
    for a in range(1, 6):
        for b in range(1, 6):
            concepts.append(
                {
                    "family": "add",
                    "key": f"{NUMBERS[a]} {NUMBERS[b]}",
                    "answer": NUMBERS[a + b].capitalize() + ".",
                }
            )
    for index, concept in enumerate(concepts):
        concept["id"] = f"c{index:05d}"
    return concepts


EXPAND = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
        "replies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions", "replies"],
}
GUIDE = {
    "number_next": ("ask which number comes after {key}", "the next number"),
    "number_before": ("ask which number comes before {key}", "the number before"),
    "add": ("ask how much {key} make together (say both numbers as words)", "the sum"),
}


async def expand(client, model, concept) -> dict:
    family = concept["family"]
    if family in GUIDE:
        ask, reply = GUIDE[family]
        ask = ask.format(key=concept["key"])
        example_q, example_a = "", ""
    else:
        _, reply, example_q, example_a = FAMILIES[family]
        ask = f"say or ask something whose answer is: {concept['answer']}"
        if concept.get("options"):
            ask = f"ask whether a {concept['key']} is a {concept['options']}"
    prompt = (
        f"A small child talks to a caregiver. Write {QUESTIONS} different short "
        f"things the child could say to {ask}. Every line must contain the exact "
        f"words '{concept['key']}', be at most 12 words, use only very simple "
        f"everyday English, and vary the wording (questions, requests, "
        f"statements, with and without 'mom', 'please', 'I think').\n"
        f"Then write {REPLIES} different short caregiver replies (at most 8 "
        f"words) that all give the same answer: {concept['answer']} ({reply}).\n"
        + (f"Example: '{example_q}' -> '{example_a}'.\n" if example_q else "")
        + "Write numbers as words. No names, no brands."
    )
    out = await chat(client, model, prompt, EXPAND, temperature=0.9)
    return {**concept, "questions": out["questions"], "replies": out["replies"]}


def rule_filter(item: dict) -> dict:
    key_words = words(item["key"])
    answer_stems = stems(item["answer"]) - stems(item["key"])

    def ok_question(q):
        q = q.strip()
        found = words(q)
        has_key = all(w in found for w in key_words)
        checked = item["family"] not in OPTION_FAMILIES | FREE_REPLY_FAMILIES
        gives_away = checked and bool(answer_stems & stems(q))
        return LINE.match(q) and 2 <= len(found) <= 12 and has_key and not gives_away

    def ok_reply(r):
        r = r.strip()
        return LINE.match(r) and 1 <= len(words(r)) <= 8

    questions = list(
        dict.fromkeys(q.strip() for q in item["questions"] if ok_question(q))
    )
    replies = list(dict.fromkeys(r.strip() for r in item["replies"] if ok_reply(r)))
    return {**item, "questions": questions, "replies": replies}


JUDGE = {
    "type": "object",
    "properties": {
        "bad_questions": {"type": "array", "items": {"type": "integer"}},
        "bad_replies": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["bad_questions", "bad_replies"],
}


async def judge_lines(client, model, item) -> dict:
    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(item["questions"]))
    replies = "\n".join(f"{i}. {r}" for i, r in enumerate(item["replies"]))
    prompt = (
        f"Correct answer: {item['answer']}\nChild lines:\n{questions}\n"
        f"Caregiver replies:\n{replies}\n"
        "List the numbers of child lines that are unnatural, not simple "
        "child-level English, or that a caregiver could reasonably answer "
        "differently than the correct answer. List the numbers of replies that "
        "do not clearly give the correct answer or are not simple. Be strict."
    )
    verdict = await chat(client, model, prompt, JUDGE, temperature=0.0)
    bad_q, bad_r = set(verdict["bad_questions"]), set(verdict["bad_replies"])
    return {
        **item,
        "questions": [q for i, q in enumerate(item["questions"]) if i not in bad_q],
        "replies": [r for i, r in enumerate(item["replies"]) if i not in bad_r],
    }


def export(items: list[dict], output: Path, seed: int) -> dict:
    rng = random.Random(seed)
    seen_questions: dict[str, str] = {}
    concepts, pairs = [], []
    dropped = {"few_questions": 0, "few_replies": 0, "shared_question": 0}
    for item in items:
        # A line that appears under two concepts would have two right answers.
        for q in item["questions"]:
            seen_questions.setdefault(q.lower(), item["id"])
    for item in items:
        questions = [
            q for q in item["questions"] if seen_questions[q.lower()] == item["id"]
        ]
        dropped["shared_question"] += len(item["questions"]) - len(questions)
        if len(questions) < 10:
            dropped["few_questions"] += 1
            continue
        if len(item["replies"]) < 2:
            dropped["few_replies"] += 1
            continue
        rng.shuffle(questions)
        concepts.append(
            {
                k: item[k]
                for k in ("id", "family", "key", "answer", "options")
                if k in item
            }
            | {"replies": item["replies"]}
        )
        for index, question in enumerate(questions):
            pairs.append(
                {
                    "id": f"{item['id']}-q{index:02d}",
                    "concept": item["id"],
                    "family": item["family"],
                    "key": item["key"],
                    "question": question,
                    "replies": item["replies"],
                    "split": "test" if index < HELD_OUT else "train",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "concepts.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in concepts)
    )
    (output / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    summary = {
        "concepts": len(concepts),
        "pairs": len(pairs),
        "test_pairs": sum(p["split"] == "test" for p in pairs),
        "by_family": {
            f: sum(c["family"] == f for c in concepts)
            for f in sorted({c["family"] for c in concepts})
        },
        "dropped": dropped,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    limit = asyncio.Semaphore(args.concurrency)

    async def bounded(coro):
        async with limit:
            return await coro

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    if args.stage == "concepts":
        concepts = await build_concepts(
            client, args.model, args.per_family, args.rounds
        )
        (work / "concepts-raw.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in concepts)
        )
        print(json.dumps({"concepts": len(concepts)}))
    elif args.stage == "expand":
        source = args.concepts or work / "concepts-raw.jsonl"
        concepts = [json.loads(line) for line in source.read_text().splitlines()]
        items = await asyncio.gather(
            *(bounded(expand(client, args.model, c)) for c in concepts)
        )
        (work / "expanded.jsonl").write_text(
            "".join(json.dumps(i) + "\n" for i in items)
        )
        print(json.dumps({"expanded": len(items)}))
    elif args.stage == "validate":
        items = [
            json.loads(line)
            for line in (work / "expanded.jsonl").read_text().splitlines()
        ]
        before = sum(len(i["questions"]) for i in items)
        items = [rule_filter(i) for i in items]
        after_rules = sum(len(i["questions"]) for i in items)
        items = await asyncio.gather(
            *(bounded(judge_lines(client, args.model, i)) for i in items)
        )
        after_judge = sum(len(i["questions"]) for i in items)
        (work / "validated.jsonl").write_text(
            "".join(json.dumps(i) + "\n" for i in items)
        )
        print(
            json.dumps(
                {
                    "questions": before,
                    "after_rules": after_rules,
                    "after_judge": after_judge,
                }
            )
        )
    elif args.stage == "export":
        items = [
            json.loads(line)
            for line in (work / "validated.jsonl").read_text().splitlines()
        ]
        print(json.dumps(export(items, args.output, args.seed)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["concepts", "expand", "validate", "export"])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concepts", type=Path, help="Reviewed concept list")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
