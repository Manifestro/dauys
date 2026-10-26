# Research protocol

## Question

Can a compact, audio-native model learn useful spoken interaction from synchronized conversations without text supervision?

## Stage 1: proof of concept

Train the hybrid model from random dialogue weights on 20–50 hours of prepared two-channel conversation. Keep the pretrained Mimi codec frozen.

Evaluate:

- Training stability and held-out loss.
- Intelligibility of generated speech.
- Dependence of predicted responses on the correct incoming audio.
- Speech activity, turn-taking, overlap, and interruption behavior.
- Concurrent listening and generation during streaming inference.

A working prototype requires audible output and measurable audio conditioning. Meaningful dialogue is desirable but is not a prerequisite for completing this technical stage.

## Stage 2: comparative baseline

Compare a dual-stream causal Transformer and the Dauys hybrid using the same train, validation, and test splits; compatible objectives; and comparable parameter and compute budgets. Expand to approximately 200–1,000 hours if Stage 1 supports it.

Report quality, interaction behavior, generalization, parameter count, compute, and streaming latency. Evaluate held-out dialogues, speakers, and scenarios. Include shuffled-input and silent-input controls.

## Stage 3: improvements

Pursue architecture, hierarchical prediction, training, interactive learning, or continuous latent processing only when Stage 2 provides evidence for a useful direction. Compare each change with the established baselines under matched conditions.

## Evidence standard

Record the code revision, immutable dataset identifier, manifest checksum, split IDs, configuration, seed, environment, compute usage, metrics, and fixed audio examples for every reported experiment. Separate measured results from interpretation. Preserve unsuccessful experiments in the evaluation record.

Working software and a scientific claim are separate outcomes. A decreasing loss or audible speech alone does not establish semantic understanding.
