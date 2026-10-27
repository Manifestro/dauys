"""Copy Mamba-2 backbone layers from a pretrained language model into the dialogue tower.

Only residual blocks move (pre-norm weight and the Mamba-2 mixer). Token embeddings,
the final norm and the text output head are never read, so the model still sees
and produces audio tokens only.
"""

import torch

from dauys.model import DauysHybrid, MambaBlock

LAYER_PREFIX = "backbone.layers."


def backbone_layers(
    state: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith(LAYER_PREFIX):
            continue
        index, name = key[len(LAYER_PREFIX) :].split(".", 1)
        layers.setdefault(int(index), {})[name] = value
    return layers


def transfer_mamba2_backbone(
    model: DauysHybrid,
    state: dict[str, torch.Tensor],
    source_layers: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Load language-model layers into dialogue Mamba blocks, in order.

    Returns (source layer, dialogue block index) pairs. By default the first N
    language-model layers fill the N dialogue Mamba blocks.
    """
    layers = backbone_layers(state)
    targets = [
        index
        for index, block in enumerate(model.dialogue)
        if isinstance(block, MambaBlock)
    ]
    sources = list(range(len(targets))) if source_layers is None else source_layers
    if len(sources) != len(targets):
        raise ValueError("Source layers must match the dialogue Mamba blocks")
    pairs = []
    for source, target in zip(sources, targets, strict=True):
        if source not in layers:
            raise ValueError(f"Language model has no backbone layer {source}")
        block = model.dialogue[target]
        expected = block.state_dict()
        weights = layers[source]
        if set(weights) != set(expected):
            difference = sorted(set(weights) ^ set(expected))
            raise ValueError(f"Layer {source} parameters differ: {difference}")
        for name, tensor in weights.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(
                    f"Layer {source} {name}: {tuple(tensor.shape)} "
                    f"!= {tuple(expected[name].shape)}"
                )
        block.load_state_dict(
            {name: tensor.to(expected[name].dtype) for name, tensor in weights.items()}
        )
        pairs.append((source, target))
    return pairs


def load_language_model_state(repo: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, "pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=True)
