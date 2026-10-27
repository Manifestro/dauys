"""Output-side ASR and relevance callbacks for the frozen-suite harness."""

import os
import re
from functools import cache
from typing import Any

import numpy as np
import torch

from .v02_pretrain_eval import WhisperTranscriber

RELEVANCE_PROMPT = """You judge a spoken assistant reply from its ASR transcript.
User said: "{user}"
Assistant said: "{reply}"
Score how relevant and appropriate the assistant reply is to what the user said:
0 = empty, unintelligible, or unrelated
1 = loosely related or generic filler
2 = relevant but incomplete or partly wrong
3 = relevant, coherent and appropriate
Answer with one digit only."""


@cache
def _whisper() -> WhisperTranscriber:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperTranscriber(
        os.environ.get("DAUYS_ASR_MODEL", "openai/whisper-large-v3"), device
    )


def whisper_asr(waveform: np.ndarray) -> str:
    """Greedy Whisper transcript; Whisper hallucinations on silence count as empty."""
    return whisper_asr_batch([waveform])[0]


def whisper_asr_batch(waveforms: list[np.ndarray]) -> list[str]:
    results = _whisper().confident(
        [np.asarray(waveform, dtype=np.float32) for waveform in waveforms]
    )
    return ["" if result["filtered"] else str(result["text"]) for result in results]


@cache
def _judge() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.environ.get("DAUYS_JUDGE_MODEL", "Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[arg-type]
    return tokenizer, model.eval()


def llm_relevance(user_text: str, transcript: str) -> float:
    """0–3 relevance from a fixed local LLM with greedy decoding."""
    return llm_relevance_batch([(user_text, transcript)])[0]


@torch.inference_mode()
def llm_relevance_batch(
    pairs: list[tuple[str, str]], batch_size: int = 32
) -> list[float]:
    """Batched form of llm_relevance; empty transcripts score 0 without the judge."""
    scores = [0.0] * len(pairs)
    pending = [i for i, (_, reply) in enumerate(pairs) if reply.split()]
    if not pending:
        return scores
    tokenizer, model = _judge()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": RELEVANCE_PROMPT.format(
                            user=pairs[i][0], reply=pairs[i][1]
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for i in chunk
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        answers = tokenizer.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for index, answer in zip(chunk, answers, strict=True):
            match = re.search(r"[0-3]", answer)
            if match is None:
                raise ValueError(f"Relevance judge returned no score: {answer!r}")
            scores[index] = float(match.group())
    return scores
