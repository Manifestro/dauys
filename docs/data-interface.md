# Audio data interface

Dauys consumes prepared conversations. Dataset creation and speech synthesis are outside this repository.

## Audio

- Each example contains two synchronized audio channels on one continuous timeline.
- Each channel is mono at 24 kHz before Mimi encoding.
- Channel 1 always contains the user; channel 2 always contains the model-side speaker.
- Keep actual silence, pauses, interruptions, and simultaneous speech. Do not concatenate isolated turns into an artificial alternating sequence.
- Preserve the full duration of both channels. Both channels must share the same sample rate, start time, and frame count.
- Do not include transcripts, text tokens, or text-derived training targets in the model input.
- Audio must be decodable and compatible with the frozen codec after a documented preprocessing step. Mimi produces eight codebooks of 2,048 values at 12.5 frames per second.

## Dataset identity

Each delivered dataset needs an immutable version identifier and a manifest checksum. The manifest must provide stable example IDs, audio locations, durations, sample rates, channel layout, split IDs, and source/provenance identifiers. The train, validation, and test split assignments must be fixed before evaluation.

Split at the conversation or source group level to prevent segments of the same recording from appearing in multiple splits. Record speaker and scenario group IDs where available so held-out generalization can be measured without exposing personal identifiers.

## Validation

Before training, validate channel order, equal channel lengths, decode success, duration, clipping, unexpected empty channels, and split leakage. Preserve the original audio and record every resampling or codec preprocessing step.

