"""Training history scheduling invariants."""

import pytest

from dauys.schedule import own_history_step


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
