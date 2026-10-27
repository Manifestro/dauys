"""Count-based CB0 reference points: unigram and bigram next-frame predictors."""

import argparse
import json
from pathlib import Path

import torch

from dauys.v02_pretrain_eval import load_pretrain_suite

VOCAB = 2048


def _speech(cache: dict, source: str) -> torch.Tensor:
    return (cache["user"] if source == "1a_on_A" else cache["target"])[0].long()


def _sequences(root: Path, manifest: Path, limit: int | None) -> list[torch.Tensor]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    sequences = []
    for row in rows[:limit]:
        cache = torch.load(
            root / row["tokens_path"], map_location="cpu", weights_only=True
        )
        sequences.append(_speech(cache, row["source"]))
    return sequences


def _fit(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    unigram = torch.ones(VOCAB, dtype=torch.float64)
    bigram = torch.full((VOCAB, VOCAB), 1e-3, dtype=torch.float64)
    for codes in sequences:
        unigram += torch.bincount(codes, minlength=VOCAB)
        pairs = codes[:-1] * VOCAB + codes[1:]
        bigram += torch.bincount(pairs, minlength=VOCAB * VOCAB).view(VOCAB, VOCAB)
    return unigram / unigram.sum(), bigram / bigram.sum(1, keepdim=True)


def _score(
    sequences: list[torch.Tensor], unigram: torch.Tensor, bigram: torch.Tensor
) -> dict[str, float]:
    frames = correct_uni = correct_bi = 0
    nll_uni = nll_bi = 0.0
    top = int(unigram.argmax())
    following = bigram.argmax(1)
    for codes in sequences:
        target = codes[1:]
        previous = codes[:-1]
        frames += target.numel()
        correct_uni += int((target == top).sum())
        correct_bi += int((following[previous] == target).sum())
        nll_uni -= float(unigram[target].log().sum())
        nll_bi -= float(bigram[previous, target].log().sum())
    return {
        "unigram_accuracy": correct_uni / frames,
        "unigram_nll": nll_uni / frames,
        "bigram_accuracy": correct_bi / frames,
        "bigram_nll": nll_bi / frames,
        "frames": float(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls100-root", type=Path, required=True)
    parser.add_argument("--pretrain-suite", type=Path, required=True)
    parser.add_argument("--hohere-root", type=Path, required=True)
    parser.add_argument("--hohere-train", type=Path, required=True)
    parser.add_argument("--hohere-heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ls100 = _sequences(args.ls100_root, args.ls100_root / "manifest.jsonl", None)
    suite = load_pretrain_suite(args.pretrain_suite)
    dev_clean = [item["codes"][0].long() for item in suite["items"]]
    hohere_rows = [
        json.loads(line) for line in args.hohere_train.read_text().splitlines() if line
    ]
    hohere_train = [
        _speech(
            torch.load(args.hohere_root / row["tokens_path"], weights_only=True),
            row["source"],
        )
        for row in hohere_rows
        if row["source"] == "hohere"
    ]
    hohere_heldout = _sequences(args.hohere_root, args.hohere_heldout, None)
    results = {}
    unigram, bigram = _fit(ls100)
    results["librispeech_dev_clean"] = _score(dev_clean, unigram, bigram)
    unigram, bigram = _fit(hohere_train)
    results["hohere_heldout"] = _score(hohere_heldout, unigram, bigram)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
