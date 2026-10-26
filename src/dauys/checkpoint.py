"""Safe loading for experiment checkpoints produced by this package."""

from pathlib import Path
from typing import Any

import torch


def load_research_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load tensor weights and metadata from a trusted local research run."""
    # Older runs stored torch.__version__ as its TorchVersion subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)
