"""Checks for the word-meaning NLL part selector."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from v02_word_meaning_eval import combine_parts


def test_combine_parts_selects_cb0_depth_or_both():
    cb0 = torch.tensor([1.0, 2.0])
    depth = torch.tensor([3.0, 4.0])
    assert torch.equal(combine_parts(cb0, depth, "all"), cb0 + depth)
    assert torch.equal(combine_parts(cb0, depth, "cb0"), cb0)
    assert torch.equal(combine_parts(cb0, depth, "depth"), depth)
    with pytest.raises(ValueError, match="Unknown"):
        combine_parts(cb0, depth, "other")
