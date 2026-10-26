"""Deterministic training history schedule."""


def own_history_step(step: int, conversation_count: int) -> bool:
    """Alternate history mode for every conversation across adjacent epochs."""
    if step < 1 or conversation_count < 1:
        raise ValueError("Step and conversation count must be positive")
    position = (step - 1) % conversation_count
    epoch = (step - 1) // conversation_count
    return (position + epoch) % 2 == 1


def e005_phase(epoch: int, step_in_epoch: int, epoch_steps: int) -> tuple[int, int]:
    """Return the attention window and rollout length for the six-epoch plan."""
    if not 0 <= epoch < 6 or not 0 <= step_in_epoch < epoch_steps:
        raise ValueError("Invalid E005 epoch position")
    start = epoch + step_in_epoch / epoch_steps
    end = epoch + (step_in_epoch + 1) / epoch_steps
    if start < 2:
        return 256, 0
    if start < 4:
        return 256, 32 + int((start - 2) * 16)
    return min(512, 256 + int((end - 4) * 128)), min(128, 64 + int((end - 4) * 32))
