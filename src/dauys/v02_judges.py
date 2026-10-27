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
    result = _whisper().confident([np.asarray(waveform, dtype=np.float32)])[0]
    return "" if result["filtered"] else str(result["text"])


@cache
def _judge() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.environ.get("DAUYS_JUDGE_MODEL", "Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return tokenizer, model.eval()


@torch.inference_mode()
def llm_relevance(user_text: str, transcript: str) -> float:
    """0–3 relevance from a fixed local LLM with greedy decoding."""
    if len(transcript.split()) == 0:
        return 0.0
    tokenizer, model = _judge()
    messages = [
        {
            "role": "user",
            "content": RELEVANCE_PROMPT.format(user=user_text, reply=transcript),
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    match = re.search(r"[0-3]", answer)
    if match is None:
        raise ValueError(f"Relevance judge returned no score: {answer!r}")
    return float(match.group())
