"""Re-score kept pre-training checkpoints with Whisper's hallucination filter."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dauys.codec import decode, load_mimi
from dauys.config import ModelConfig
from dauys.model import DauysHybrid
from dauys.v02_pretrain_eval import (
    CONTINUATION_FRAMES,
    FRAME_SAMPLES,
    PROMPT_FRAMES,
    PretrainEvaluator,
    WhisperTranscriber,
    english_vocabulary,
    load_pretrain_suite,
    word_statistics,
)


def _scores(results: list[dict], waves: list[np.ndarray], vocabulary: set[str]) -> dict:
    raw = word_statistics([item["text"] for item in results], vocabulary)
    kept = word_statistics(
        ["" if item["filtered"] else item["text"] for item in results], vocabulary
    )
    return {
        "word_rate": raw["word_rate"],
        "empty_rate": raw["empty_rate"],
        "filtered_word_rate": kept["word_rate"],
        "filtered_empty_rate": kept["empty_rate"],
        "filtered_scored": kept["scored"],
        "hallucination_filtered": float(
            np.mean([item["filtered"] for item in results])
        ),
        "rms": float(np.mean([np.sqrt(np.mean(wave**2)) for wave in waves])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--mimi-checkpoint", type=Path, required=True)
    parser.add_argument("--silence-codes", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = load_pretrain_suite(args.suite)
    codec = load_mimi(args.mimi_checkpoint, "cuda")
    silence = torch.tensor(json.loads(args.silence_codes.read_text()), dtype=torch.long)
    asr = WhisperTranscriber(args.asr_model, "cuda")
    vocabulary = english_vocabulary()
    evaluator = PretrainEvaluator(
        suite, codec, silence, asr, vocabulary, [suite["items"][0]["codes"]]
    )
    rows: dict[str, dict] = {}
    with torch.inference_mode():
        tails = []
        for item in suite["items"]:
            window = item["codes"][None, :, : PROMPT_FRAMES + CONTINUATION_FRAMES]
            audio = decode(codec, window.long().cuda())[0, 0]
            tails.append(audio[PROMPT_FRAMES * FRAME_SAMPLES :].float().cpu().numpy())
    rows["ceiling"] = _scores(asr.confident(tails), tails, vocabulary)
    config = ModelConfig.from_yaml(args.config)
    torch.manual_seed(evaluator.seed)
    candidates: list[tuple[str, DauysHybrid]] = [
        ("floor", DauysHybrid(config).cuda().eval())
    ]
    for path in args.checkpoints:
        model = DauysHybrid(config).cuda()
        saved = torch.load(path, map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        candidates.append((f"step-{int(saved['step'])}", model.eval()))
    for name, model in candidates:
        seeds = []
        for offset in range(args.seeds):
            waves = evaluator.continue_speech(model, evaluator.seed + offset)
            seeds.append(_scores(asr.confident(waves), waves, vocabulary))
        rows[name] = {key: float(np.mean([s[key] for s in seeds])) for key in seeds[0]}
        accuracy, nll = evaluator.cb0_accuracy(
            model, [item["codes"].long() for item in suite["items"]]
        )
        rows[name]["heldout_cb0_accuracy"] = accuracy
        rows[name]["heldout_cb0_nll"] = nll
        print(json.dumps({name: rows[name]}), flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
