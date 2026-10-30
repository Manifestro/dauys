# Listening bench

Does a full-duplex speech model answer **what** it hears, or only **how** it sounds?

Existing full-duplex benchmarks (Full-Duplex-Bench v1/v1.5/v2, MTR-DuplexBench) score behaviour: pauses, backchannels, turn-taking, interruptions, overlap, context over turns. This bench measures how much a model's own speech depends on the *content* of its partner's speech. Every number comes from audio only; no text enters any measure.

## Measures

| Measure | What it asks | How |
| --- | --- | --- |
| Contingency gap | Does the reply depend on what this partner said? | Teacher-forced NLL of the model's own channel (CB0, and the mean of CB1–7) with the true partner, a swapped partner from another conversation, and silence. Gap = NLL(swap) − NLL(true), per conversation. A plain swap also breaks turn timing, so a **timed swap** places the other partner's speech exactly in the true partner's speaking frames (silence elsewhere), against the true partner spliced the same way: that gap is content and voice only. |
| Echo rate | Does the model repeat its partner's sounds? | The model hears the partner channel and speaks freely. Share of its spoken CB0 4-grams that also occur in what the partner said, against the same share for a swapped partner. Excess echo = true − swap. |
| Voice similarity | Does the model copy its partner's voice? | Speaker-embedding cosine (WavLM-base-plus-sv) between the model's generated speech and its partner's voice, against other partners' voices. Voice copy = user − other user. |
| sWUGGY, sBLIMP, tStoryCloze | Does the semantic stream know words, grammar, story sense? | The item is spoken on the model's own channel after 0.5 s of silence, with silence on the partner channel. Pairwise accuracy of summed CB0 log-likelihood (mean-normalized as a second view). |

All intervals are 95% bootstrap intervals over conversations or pairs.

## Data

- **hohere-bench v0.2.0, E1** (`manifestro/hohere-bench`): 267 synthetic two-channel conversations, user (A) and assistant (B), with per-frame activity labels.
- **ZeroSpeech 2021 sLM21 dev**: sWUGGY (`lexical/dev`) and sBLIMP (`syntactic/dev`), from `https://download.zerospeech.com/datasets/sLM21.dataset.zip`.
- **tStoryCloze** (Hassid et al. 2023, slp-rl/SpokenStoryCloze).

Everything is encoded once with Mimi (8 codebooks, 12.5 Hz) and cached, so every model is scored on identical codes.

## Models

- **Moshi** (`kyutai/moshiko-pytorch-bf16`). Moshi also writes an inner-monologue text stream. For teacher forcing it is set to the model's own greedy text: one pass with padding text predicts it, a second pass scores the audio. Generation uses `LMGen` with Kyutai's default sampling.
- **Dauys** (the research branch's `dauys` package, v0.2 and later).

## Run

The bench runs inside an environment with `torch`, `moshi`, `transformers`, `soundfile`, `scipy`, `pyarrow` and `huggingface_hub`; for Dauys also the research-branch `dauys` package with `mamba-ssm`.

```bash
cd bench
python -m duplexbench.run prepare --cache CACHE --slm21 SLM21_DIR --storycloze TSC_DIR
python -m duplexbench.run score --cache CACHE --model moshi --out results/moshi
python -m duplexbench.run score --cache CACHE --model dauys \
  --dauys-config CONFIG.yaml --dauys-checkpoint CHECKPOINT.pt --out results/dauys
```

`--tests` picks a subset (`contingency swuggy sblimp tstorycloze generation`); `--limit N` runs a quick check on the first N conversations.

## Tests

```bash
cd bench && python -m pytest tests
```
