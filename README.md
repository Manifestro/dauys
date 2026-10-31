# Dauys

"Dauys" means "voice" in Kazakh. It is a research project by Manifestro on compact, audio-native, full-duplex spoken dialogue: a model that listens to one audio stream while generating a second, synchronized one, through speech, silence and overlap.

**The question:** how compact can a model be, and how little data does it need, to understand speech and respond to it sensibly? We measure abilities that are usually mixed together separately: recognizing words, understanding a question, choosing the right answer, saying it clearly, and actually listening to the other speaker.

This is research code. No release model exists yet, and natural full-duplex conversation is a goal, not a result.

## Status

| Line | Branch | State |
| --- | --- | --- |
| v0.1, baseline | `main` | Complete. Hybrid architecture, streaming and training tools. |
| v0.2, speech understanding on synthetic speech | `research/hybrid-v0.2` | Final run in progress. About 101M parameters. |
| v1.0, real human speech | in development | Not published yet. |
| Listening bench | `main`, [`bench/`](bench/README.md) | Usable. Results so far: Moshi, in branch `bench-results`. |

The rest of this file describes `main`. Code for v0.2, including its model configuration, training and evaluation scripts, is on `research/hybrid-v0.2` and is not merged here.

## Listening bench

[`bench/`](bench/README.md) measures how much a full-duplex model's own speech depends on what its partner says, from audio alone: contingency gap, echo rate, voice copying, and sWUGGY, sBLIMP and tStoryCloze on the semantic stream. It has adapters for Moshi and Dauys. The bench README lists the data and how to run it.

## Principles

- No text enters or leaves the model. Text is allowed only as an external yardstick in tests. v0.1 starts from random dialogue weights; v0.2 initializes its Mamba layers from a text-trained model (`state-spaces/mamba2-130m`), without text embeddings or a text head, so its inputs and outputs are audio only but part of its initial knowledge comes from text.
- A pretrained, frozen Mimi codec.
- Each idea is tested on its own, with a criterion fixed in advance and with controls. Negative results count as results. See the [research protocol](RESEARCH.md) and the [evaluation protocol](docs/evaluation.md).

## v0.1 on `main`

The candidate model is specified in [Dauys Hybrid v0.1](docs/architecture.md) and [its packaged configuration](src/dauys/configs/dauys-hybrid-v0.1.yaml): a Mamba-2 listening tower, a Mamba-2 plus causal Transformer dialogue tower, causal cross-attention from listening to dialogue, Mimi token prediction and speech activity prediction.

Dataset generation lives outside this repository. Dauys consumes prepared, synchronized two-channel conversations through the [data interface](docs/data-interface.md). Channel 1 is always the user and channel 2 the model-side speaker. Roles do not swap within a recording, and silence and overlap stay on the shared timeline.

## Setup

Use Python 3.11 or 3.12 and uv:

```bash
uv sync --extra test
uv run pytest -q
uv run ruff check .
```

The model integration checks need the `gpu` extra (Mamba-2 acceleration), and the codec check needs a local Mimi checkpoint.

Datasets, checkpoints, generated audio and run records are stored outside Git. Experiment records identify the exact code revision, configuration, dataset version and artifact location.

## License

The repository code is licensed under [Apache 2.0](LICENSE). Third-party codecs, weights and datasets keep their own terms and are not distributed here.
