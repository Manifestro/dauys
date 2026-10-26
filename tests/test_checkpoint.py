"""Checkpoint compatibility checks for research runs."""

import torch

from dauys.checkpoint import load_research_checkpoint


def test_older_checkpoint_metadata_loads_with_weights_only(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": {"weight": torch.ones(2)}, "metadata": {"torch": torch.__version__}},
        path,
    )
    loaded = load_research_checkpoint(path)
    torch.testing.assert_close(loaded["model"]["weight"], torch.ones(2))
    assert str(loaded["metadata"]["torch"]) == str(torch.__version__)
