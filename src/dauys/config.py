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
    dropout: float = 0.0
    attention_dropout: float = 0.0
    listening_prediction: bool = False
    activity_ahead_horizon: int = 0
    proxy: bool = False
    listening_d_state: int = 64
    dialogue_d_state: int = 64
    mamba_norm_eps: float | None = None
    zero_init_fusion_outputs: bool = False

    @classmethod
    def stage1(cls) -> "ModelConfig":
        resource = files("dauys").joinpath("configs/dauys-hybrid-v0.1.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls._from_mapping(data)

    @classmethod
    def v02(cls) -> "ModelConfig":
        resource = files("dauys").joinpath("configs/dauys-hybrid-v0.2.yaml")
        with resource.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls._from_mapping(data)

    @classmethod
    def v02_proxy(cls) -> "ModelConfig":
        resource = files("dauys").joinpath("configs/dauys-hybrid-v0.2-proxy.yaml")
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
            dropout=float(data.get("regularization", {}).get("dropout", 0.0)),
            attention_dropout=float(
                data.get("regularization", {}).get("attention_dropout", 0.0)
            ),
            listening_prediction=bool(data.get("heads", {}).get("listening", False)),
            activity_ahead_horizon=int(data.get("heads", {}).get("activity_ahead", 0)),
            proxy=bool(data["model"].get("proxy", False)),
            listening_d_state=int(data["listening"]["d_state"]),
            dialogue_d_state=int(data["dialogue"]["mamba"]["d_state"]),
            mamba_norm_eps=data["model"].get("mamba_norm_eps"),
            zero_init_fusion_outputs=bool(
                data["model"].get("zero_init_fusion_outputs", False)
            ),
        )
        config.validate(data)
        return config

    def validate(self, data: dict) -> None:
        if self.name not in ("dauys-hybrid-v0.1", "dauys-hybrid-v0.2"):
            raise ValueError("Unsupported model configuration")
        if data["codec"]["name"] != "mimi" or not data["codec"]["frozen"]:
            raise ValueError("The Stage 1 codec must be frozen Mimi")
        if (self.sample_rate, self.frame_rate, self.codebooks, self.vocab_size) != (
            24000,
            12.5,
            8,
            2048,
        ):
            raise ValueError("Unexpected Mimi token geometry")
        if data["embedding"]["share_between_streams"] is not True:
            raise ValueError("Embeddings must be shared between streams")
        if not self.proxy and (
            self.embedding_dim,
            self.listening_dim,
            self.listening_layers,
        ) != (256, 384, 4):
            raise ValueError("Unexpected listening tower geometry")
        if not self.proxy and (
            self.dialogue_dim != 640
            or self.dialogue_blocks
            != (
                "mamba2",
                "mamba2",
                "transformer_cross",
                "mamba2",
                "mamba2",
                "transformer_cross",
                "mamba2",
                "transformer_cross",
                "mamba2",
                "transformer_cross",
            )
        ):
            raise ValueError("Unexpected dialogue tower schedule")
        if not self.proxy and (self.attention_heads, self.ffn_dim) != (10, 1728):
            raise ValueError("Unexpected dialogue attention geometry")
        if not self.proxy and (self.depth_dim, self.depth_layers, self.depth_heads) != (
            384,
            2,
            6,
        ):
            raise ValueError("Unexpected depth decoder geometry")
        if data["depth_decoder"]["codebooks"] != list(range(1, 8)):
            raise ValueError("Depth decoder must predict codebooks 1 through 7")
        for section in ("listening", "dialogue"):
            layer = data[section] if section == "listening" else data[section]["mamba"]
            settings = ("d_conv", "expand", "headdim", "ngroups")
            if tuple(layer[key] for key in settings) != (4, 2, 64, 1):
                raise ValueError(f"Unexpected {section} Mamba-2 settings")
            allowed_states = (64, 128) if self.proxy else (64,)
            if layer["d_state"] not in allowed_states:
                raise ValueError(f"Unexpected {section} Mamba-2 state size")
        if self.attention_window != 512 or self.input_delay_frames < 0:
            raise ValueError("Invalid streaming window or input delay")
        streaming = data["streaming"]
        if (
            streaming["causal"] is not True
            or streaming["native_full_duplex"] is not True
        ):
            raise ValueError("Streaming must remain causal and full duplex")
        if self.max_trainable_parameters != 65_000_000 and not self.proxy:
            raise ValueError("Unexpected parameter cap")
        if self.max_trainable_parameters < 1:
            raise ValueError("Invalid parameter cap")
        if (self.mamba_norm_eps is not None or self.zero_init_fusion_outputs) and (
            not self.proxy
        ):
            raise ValueError("Initialization changes are limited to proxy configs")
        if self.name == "dauys-hybrid-v0.1" and (
            self.dropout
            or self.attention_dropout
            or self.listening_prediction
            or self.activity_ahead_horizon
            or self.proxy
        ):
            raise ValueError("v0.1 settings cannot enable v0.2 changes")
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            raise ValueError("Invalid dropout probability")
        if self.activity_ahead_horizon < 0:
            raise ValueError("Invalid activity prediction horizon")
        if self.proxy:
            if self.name != "dauys-hybrid-v0.2":
                raise ValueError("Only v0.2 supports proxy geometry")
            if self.listening_dim % 64 or self.dialogue_dim % 64:
                raise ValueError("Proxy Mamba widths must be divisible by 64")
            if (
                self.dialogue_dim % self.attention_heads
                or self.depth_dim % self.depth_heads
            ):
                raise ValueError("Proxy attention widths must divide by heads")
            if not self.dialogue_blocks or not set(self.dialogue_blocks) <= {
                "mamba2",
                "transformer_cross",
            }:
                raise ValueError("Invalid proxy dialogue schedule")
