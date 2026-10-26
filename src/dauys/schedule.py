"""Deterministic training history schedule."""


def own_history_step(step: int, conversation_count: int) -> bool:
    """Alternate history mode for every conversation across adjacent epochs."""
    if step < 1 or conversation_count < 1:
        raise ValueError("Step and conversation count must be positive")
    position = (step - 1) % conversation_count
    epoch = (step - 1) // conversation_count
    return (position + epoch) % 2 == 1
