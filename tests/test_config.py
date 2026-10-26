"""Locally runnable package and configuration checks."""

from dauys.config import ModelConfig


def test_packaged_stage1_configuration():
    config = ModelConfig.stage1()
    assert config.dialogue_blocks.count("mamba2") == 6
    assert config.dialogue_blocks.count("transformer_cross") == 4
    assert config.attention_window == 512
    assert config.max_trainable_parameters == 65_000_000
