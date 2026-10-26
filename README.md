# Dauys

Dauys is an experimental research project for compact, audio-native, full-duplex spoken dialogue. Its intended model listens to one audio stream while generating a second, synchronized stream. Both streams remain active through speech, silence, and overlap.

## Research target

- At most 65 million trainable parameters, with 40–60 million preferred.
- A pretrained, frozen Mimi encoder and decoder.
- A lightweight Mamba-2 listening tower and a Mamba-2 plus causal Transformer dialogue tower.
- Causal cross-attention from listening states to dialogue generation.
- Mimi token prediction and speech activity prediction.
- Audio-only dialogue training, with no transcripts or text tokens.
- Native streaming inference with simultaneous listening and speech generation.
- English for the initial experiments.

The dialogue model starts from random weights. Dataset generation lives outside this repository. Dauys consumes prepared, synchronized two-channel conversations through the [data interface](docs/data-interface.md).

## Channel convention

Channel 1 is always the user. Channel 2 is always the model-side speaker. Roles do not swap within a recording. Silence and overlapping speech remain on the shared timeline.

## Research stages

1. **Proof of concept:** train on 20–50 hours and establish audible streaming output, stable learning, and measurable dependence on incoming audio.
2. **Baseline comparison:** compare a dual-stream Transformer with the hybrid architecture on the same data splits and comparable budgets, then scale to roughly 200–1,000 hours.
3. **Improvements:** investigate architecture and training changes only if the baseline comparison warrants them.

See [research protocol](RESEARCH.md) and [evaluation protocol](docs/evaluation.md).

The first candidate model is specified in [Dauys Hybrid v0.1](docs/architecture.md) and [its packaged configuration](src/dauys/configs/dauys-hybrid-v0.1.yaml).

## Repository status

Hybrid v0.1 model code, a frame-synchronous audio interface, and pre-training checks are present. They have not yet been run against a working Mimi checkpoint or the target accelerator. Training runs, checkpoints, benchmarks, and audio demonstrations have not been produced.

Use Python 3.11 or 3.12 and uv for the package environment. The local version file selects Python 3.11:

```bash
uv sync --extra test
```

The model integration checks require an environment with Mamba-2 acceleration. Install the `gpu` extra there. The codec check additionally requires a local Mimi checkpoint:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
uv build
```

Large datasets, audio outputs, and checkpoints are excluded from Git. Experiment records should identify the exact code revision, configuration, dataset version, and external artifact location.

## License

The repository code is licensed under [Apache 2.0](LICENSE). Third-party codecs, weights, and datasets retain their own terms and are not distributed here.
