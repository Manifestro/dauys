"""Validated configuration for the Stage 1 model."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_trainable_parameters: int
    sample_rate: int
    frame_rate: float
    codebooks: int
    vocab_size: int
    embedding_dim: int
    listening_dim: int
    listening_layers: int
    dialogue_dim: int
    dialogue_blocks: tuple[str, ...]
    attention_heads: int
    ffn_dim: int
    depth_dim: int
    depth_layers: int
    depth_heads: int
    attention_window: int
    input_delay_frames: int
    cb0_weight: float
    depth_weight: float
    activity_weight: float

    @classmethod
    def stage1(cls) -> "ModelConfig":
        resource = files("dauys").joinpath("configs/dauys-hybrid-v0.1.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls._from_mapping(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls._from_mapping(data)

    @classmethod
    def _from_mapping(cls, data: dict) -> "ModelConfig":
        config = cls(
            name=data["model"]["name"],
            max_trainable_parameters=data["model"]["max_trainable_parameters"],
            sample_rate=data["codec"]["sample_rate"],
            frame_rate=data["codec"]["frame_rate"],
            codebooks=data["codec"]["codebooks"],
            vocab_size=data["codec"]["vocab_size"],
            embedding_dim=data["embedding"]["dimension"],
            listening_dim=data["listening"]["dimension"],
            listening_layers=data["listening"]["layers"],
            dialogue_dim=data["dialogue"]["dimension"],
            dialogue_blocks=tuple(data["dialogue"]["blocks"]),
            attention_heads=data["dialogue"]["attention_heads"],
            ffn_dim=data["dialogue"]["ffn_dimension"],
            depth_dim=data["depth_decoder"]["dimension"],
            depth_layers=data["depth_decoder"]["layers"],
            depth_heads=data["depth_decoder"]["heads"],
            attention_window=data["streaming"]["attention_window_frames"],
            input_delay_frames=data["streaming"]["input_delay_frames"],
            cb0_weight=data["loss"]["cb0"],
            depth_weight=data["loss"]["depth"],
            activity_weight=data["loss"]["activity"],
        )
        config.validate(data)
        return config

    def validate(self, data: dict) -> None:
        if self.name != "dauys-hybrid-v0.1":
            raise ValueError("Unsupported model configuration")
        if data["codec"]["name"] != "mimi" or not data["codec"]["frozen"]:
            raise ValueError("The Stage 1 codec must be frozen Mimi")
        if (self.sample_rate, self.frame_rate, self.codebooks, self.vocab_size) != (
            24000, 12.5, 8, 2048
        ):
            raise ValueError("Unexpected Mimi token geometry")
        if data["embedding"]["share_between_streams"] is not True:
            raise ValueError("Embeddings must be shared between streams")
        if (self.embedding_dim, self.listening_dim, self.listening_layers) != (
            256, 384, 4
        ):
            raise ValueError("Unexpected listening tower geometry")
        if self.dialogue_dim != 640 or self.dialogue_blocks != (
            "mamba2", "mamba2", "transformer_cross", "mamba2", "mamba2",
            "transformer_cross", "mamba2", "transformer_cross", "mamba2",
            "transformer_cross",
        ):
            raise ValueError("Unexpected dialogue tower schedule")
        if (self.attention_heads, self.ffn_dim) != (10, 1728):
            raise ValueError("Unexpected dialogue attention geometry")
        if (self.depth_dim, self.depth_layers, self.depth_heads) != (384, 2, 6):
            raise ValueError("Unexpected depth decoder geometry")
        if data["depth_decoder"]["codebooks"] != list(range(1, 8)):
            raise ValueError("Depth decoder must predict codebooks 1 through 7")
        for section in ("listening", "dialogue"):
            layer = data[section] if section == "listening" else data[section]["mamba"]
            settings = ("d_state", "d_conv", "expand", "headdim", "ngroups")
            if tuple(layer[key] for key in settings) != (64, 4, 2, 64, 1):
                raise ValueError(f"Unexpected {section} Mamba-2 settings")
        if self.attention_window != 512 or self.input_delay_frames < 0:
            raise ValueError("Invalid streaming window or input delay")
        streaming = data["streaming"]
        if streaming["causal"] is not True or streaming["native_full_duplex"] is not True:
            raise ValueError("Streaming must remain causal and full duplex")
        if self.max_trainable_parameters != 65_000_000:
            raise ValueError("Unexpected parameter cap")
