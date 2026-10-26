# Evaluation protocol

## Audio conditioning

Evaluate the likelihood of a held-out model-side response under its correct user audio, a shuffled user input, and silent user input. Report:

```text
delta_nll = NLL(response | shuffled input) - NLL(response | correct input)
```

A positive difference indicates that the correct input better predicts the response. This alone does not establish semantic understanding. Use contrastive questions, matched speaker and duration controls, and generated responses to test alternative explanations.

## Full-duplex behavior

Evaluate continuous input and output with pauses, overlap, and interruptions. Report speech activity F1, overlap detection, reaction latency to interruption, inappropriate interruptions, and measured streaming latency. Specify the measurement points used for each latency figure.

## Generated speech

Use a fixed, held-out set of prompts and publish the selection rule for audio examples. Report intelligibility and acoustic quality. Keep failed and low-quality samples represented in aggregate results.

## Listener evaluation

For an initial blind assessment, use 30–50 short examples and at least three independent listeners when feasible. Rate intelligibility, relevance, naturalness, and timing on a five-point scale. Keep model identity hidden from listeners. Subjective ratings are not required for initial technical smoke tests.

## Comparisons

Use the same evaluation set, decoding settings, and measurement procedure for each compared model. Report uncertainty and repeated runs for claims based on small differences.
