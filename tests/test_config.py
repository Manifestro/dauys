"""Locally runnable package and configuration checks."""

from dauys.config import ModelConfig


def test_packaged_stage1_configuration():
    config = ModelConfig.stage1()
    assert config.dialogue_blocks.count("mamba2") == 6
    assert config.dialogue_blocks.count("transformer_cross") == 4
    assert config.attention_window == 512
    assert config.max_trainable_parameters == 65_000_000


def test_packaged_v02_configurations():
    full = ModelConfig.v02()
    proxy = ModelConfig.v02_proxy()
    assert full.name == "dauys-hybrid-v0.2"
    assert full.listening_prediction
    assert full.activity_ahead_horizon == 5
    assert full.dropout == 0.1
    assert proxy.proxy
    assert proxy.dialogue_dim < full.dialogue_dim
