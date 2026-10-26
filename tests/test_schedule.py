"""Training history scheduling invariants."""

import pytest

from dauys.schedule import e005_phase, own_history_step


@pytest.mark.parametrize("conversation_count", [68, 69, 70])
def test_each_conversation_alternates_history_by_epoch(conversation_count: int):
    for position in range(conversation_count):
        first = own_history_step(position + 1, conversation_count)
        second = own_history_step(position + 1 + conversation_count, conversation_count)
        assert first != second


def test_history_modes_are_balanced_over_two_epochs():
    count = 69
    assert (
        sum(own_history_step(step, count) for step in range(1, 2 * count + 1)) == count
    )


def test_e005_phase_boundaries_and_final_context():
    assert e005_phase(1, 9, 10) == (256, 0)
    assert e005_phase(2, 0, 10) == (256, 32)
    assert e005_phase(4, 0, 10)[0] > 256
    assert e005_phase(5, 9, 10) == (512, 128)
    with pytest.raises(ValueError):
        e005_phase(6, 0, 10)
