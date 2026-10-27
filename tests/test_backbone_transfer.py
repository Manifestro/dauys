"""Backbone transfer from a Mamba-2 language model into the dialogue tower."""

import sys
import types
from importlib.resources import files

import pytest
import torch
from torch import nn

from dauys.backbone_transfer import transfer_mamba2_backbone
from dauys.config import ModelConfig
from dauys.model import DauysHybrid, DialogueBlock, MambaBlock

SCREEN = files("dauys").joinpath("configs/dauys-hybrid-v0.2-screen.yaml")


class _Mixer(nn.Module):
    def __init__(self, d_model, d_state=64, layer_idx=None, **_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.A_log = nn.Parameter(torch.zeros(d_model // 64))
        self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x, inference_params=None):
        return self.out_proj(self.in_proj(x)[..., : 2 * x.shape[-1]]).cumsum(dim=1)


@pytest.fixture
def stand_in_mamba():
    module = types.ModuleType("mamba_ssm")
    module.Mamba2 = _Mixer
    old = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = module
    try:
        yield
    finally:
        if old is None:
            del sys.modules["mamba_ssm"]
        else:
            sys.modules["mamba_ssm"] = old


def _language_model_state(model: DauysHybrid, layers: int) -> dict[str, torch.Tensor]:
    template = next(b for b in model.dialogue if isinstance(b, MambaBlock))
    state = {
        "backbone.embedding.weight": torch.randn(50288, 768),
        "backbone.norm_f.weight": torch.randn(768),
        "lm_head.weight": torch.randn(50288, 768),
    }
    for layer in range(layers):
        for name, tensor in template.state_dict().items():
            state[f"backbone.layers.{layer}.{name}"] = torch.full_like(
                tensor, float(layer + 1)
            )
    return state


def test_screen_config_matches_mamba2_130m_geometry():
    config = ModelConfig.from_yaml(str(SCREEN))
    assert config.proxy
    assert (config.dialogue_dim, config.dialogue_d_state) == (768, 128)
    assert config.listening_d_state == 64
    assert config.mamba_norm_eps == pytest.approx(1e-5)
    assert config.zero_init_fusion_outputs
    assert (config.depth_layers, config.depth_dim, config.depth_heads) == (4, 512, 8)


def test_full_config_keeps_its_fixed_geometry():
    config = ModelConfig.v02()
    assert (config.dialogue_d_state, config.listening_d_state) == (64, 64)
    assert config.mamba_norm_eps is None
    assert not config.zero_init_fusion_outputs


def test_fusion_blocks_start_as_identity(stand_in_mamba):
    model = DauysHybrid(ModelConfig.from_yaml(str(SCREEN)))
    blocks = [b for b in model.dialogue if isinstance(b, DialogueBlock)]
    assert len(blocks) == 4
    for block in blocks:
        for layer in (
            block.self_attention.output,
            block.cross_attention.output,
            block.ffn.down,
        ):
            assert not layer.weight.any() and not layer.bias.any()
        assert block.self_norm.eps == block.cross_norm.eps


def test_transfer_fills_dialogue_mamba_blocks_in_order(stand_in_mamba):
    torch.manual_seed(0)
    model = DauysHybrid(ModelConfig.from_yaml(str(SCREEN)))
    embeddings = [e.weight.clone() for e in model.embeddings]
    head = model.cb0_head.weight.clone()
    pairs = transfer_mamba2_backbone(model, _language_model_state(model, 24))
    assert pairs == [(0, 0), (1, 1), (2, 3), (3, 4), (4, 6), (5, 8)]
    for source, target in pairs:
        for tensor in model.dialogue[target].state_dict().values():
            assert torch.all(tensor == source + 1)
    assert all(
        torch.equal(before, after.weight)
        for before, after in zip(embeddings, model.embeddings, strict=True)
    )
    assert torch.equal(head, model.cb0_head.weight)
    assert all(
        block.norm.eps == pytest.approx(1e-5)
        for block in model.dialogue
        if isinstance(block, MambaBlock)
    )


def test_transfer_rejects_mismatched_layers(stand_in_mamba):
    model = DauysHybrid(ModelConfig.from_yaml(str(SCREEN)))
    state = _language_model_state(model, 24)
    with pytest.raises(ValueError, match="no backbone layer"):
        transfer_mamba2_backbone(model, _language_model_state(model, 3))
    state["backbone.layers.0.norm.weight"] = torch.ones(640)
    with pytest.raises(ValueError, match="norm.weight"):
        transfer_mamba2_backbone(model, state)


def test_transferred_blocks_reproduce_the_language_model():
    """GPU only: our blocks must compute exactly what the source layers compute."""
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("Mamba-2 kernels need CUDA")
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    torch.manual_seed(0)
    language_model = MambaLMHeadModel(
        MambaConfig(
            d_model=768,
            n_layer=6,
            ssm_cfg={"layer": "Mamba2"},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        ),
        device="cuda",
    ).eval()
    model = DauysHybrid(ModelConfig.from_yaml(str(SCREEN)))
    transfer_mamba2_backbone(model, language_model.state_dict())
    model = model.cuda().eval()
    x = torch.randn(2, 48, 768, device="cuda")
    expected, residual = x, None
    with torch.no_grad():
        for layer in language_model.backbone.layers:
            expected, residual = layer(expected, residual)
        expected = expected + residual
        ours = x
        for index in (0, 1, 3, 4, 6, 8):
            ours = model.dialogue[index](ours)
    torch.testing.assert_close(ours, expected, rtol=2e-3, atol=2e-3)
