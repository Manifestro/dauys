"""Model adapters: one interface for every full-duplex model on the bench.

Both models read and write Mimi codes (8 codebooks, 12.5 Hz), so the same
encoded conversations and the same semantic stream (codebook 0) serve both.

Interface:
- ``frame_nll(user, own)``: teacher-forced per-frame NLL of the model's own
  channel given both channels, as ``(cb0, acoustic)`` of shape [B, T]
  (acoustic = mean over codebooks 1-7). NaN where a frame has no prediction.
- ``generate(user, seed)``: the model's own channel, free-running, while it
  hears ``user`` from the first frame, as [B, 8, T].
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class Adapter(Protocol):
    """Silence is the Mimi code frame of digital silence, encoded once."""

    name: str
    silence: torch.Tensor

    def frame_nll(
        self, user: torch.Tensor, own: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def generate(self, user: torch.Tensor, seed: int) -> torch.Tensor: ...


def _token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """NLL per position from [..., V] logits; NaN logits give NaN."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    valid = ~torch.isnan(flat).any(-1)
    out = torch.full((flat.shape[0],), float("nan"), device=logits.device)
    if valid.any():
        out[valid] = F.cross_entropy(
            flat[valid], targets.reshape(-1)[valid], reduction="none"
        )
    return out.reshape(targets.shape)


@dataclass
class MoshiAdapter:
    """Kyutai Moshi (moshiko): 7B temporal transformer plus depformer.

    Moshi also writes an inner-monologue text stream. Teacher forcing needs a
    value for it, and a full-duplex model has no reference text for its own
    reply, so the text stream is the model's own greedy choice: a first pass
    with padding text predicts it, a second pass scores the audio with it.
    """

    lm: object
    silence: torch.Tensor
    name: str = "moshi"
    text_passes: int = 2

    @classmethod
    def load(
        cls,
        silence: torch.Tensor,
        repo: str = "kyutai/moshiko-pytorch-bf16",
        device: str = "cuda",
    ) -> "MoshiAdapter":
        from moshi.models import loaders

        info = loaders.CheckpointInfo.from_hf_repo(repo)
        lm = info.get_moshi(device=device, dtype=torch.bfloat16).eval()
        return cls(lm, silence.long())

    @property
    def device(self) -> torch.device:
        return next(self.lm.parameters()).device  # type: ignore[attr-defined]

    @torch.inference_mode()
    def frame_nll(
        self, user: torch.Tensor, own: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lm = self.lm
        batch, _, frames = own.shape
        text = torch.full(
            (batch, 1, frames),
            lm.existing_text_padding_id,  # type: ignore[attr-defined]
            dtype=torch.long,
            device=self.device,
        )
        own = own.to(self.device)
        user = user.to(self.device)
        for _ in range(self.text_passes):
            out = lm(torch.cat((text, own, user), dim=1))  # type: ignore[operator]
            greedy = out.text_logits[:, 0].float().nan_to_num(-1e9).argmax(-1)
            text = greedy[:, None]
        nll = _token_nll(out.logits, own)
        return nll[:, 0].cpu(), nll[:, 1:].mean(1).cpu()

    @torch.inference_mode()
    def generate(self, user: torch.Tensor, seed: int) -> torch.Tensor:
        from moshi.models import LMGen

        torch.manual_seed(seed)
        batch, _, frames = user.shape
        generator = LMGen(self.lm)
        outputs: list[torch.Tensor] = []
        user = user.to(self.device)
        silence = self.silence.to(self.device)[None, :, None].expand(batch, -1, -1)
        with generator.streaming(batch):
            for frame in range(frames + generator.max_delay):
                heard = user[:, :, frame : frame + 1] if frame < frames else silence
                tokens = generator.step(heard)
                if tokens is not None:
                    outputs.append(tokens[:, 1:9, 0])
        return torch.stack(outputs[:frames], dim=-1).cpu()


@dataclass
class DauysAdapter:
    """Dauys hybrid (research branch ``dauys`` package, v0.2 and later)."""

    model: object
    silence: torch.Tensor
    name: str = "dauys"

    @classmethod
    def load(
        cls, config: str, checkpoint: str, silence: torch.Tensor, device: str = "cuda"
    ) -> "DauysAdapter":
        from dauys.checkpoint import load_research_checkpoint
        from dauys.config import ModelConfig
        from dauys.model import DauysHybrid

        model = DauysHybrid(ModelConfig.from_yaml(config))
        payload = load_research_checkpoint(checkpoint)
        model.load_state_dict(payload["model"])
        model = model.to(device).eval()
        return cls(model, silence.long())

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device  # type: ignore[attr-defined]

    @torch.inference_mode()
    def frame_nll(
        self, user: torch.Tensor, own: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user, own = user.to(self.device), own.to(self.device)
        initial = self.silence.to(self.device)[None].expand(own.shape[0], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(user, own, initial)  # type: ignore[operator]
        cb0 = _token_nll(out.cb0_logits, own[:, 0])
        depth = _token_nll(out.depth_logits, own[:, 1:].transpose(1, 2))
        return cb0.cpu(), depth.mean(-1).cpu()

    @torch.inference_mode()
    def generate(self, user: torch.Tensor, seed: int) -> torch.Tensor:
        from dauys.decoding import DecodingConfig

        model = self.model
        batch, _, frames = user.shape
        user = user.to(self.device)
        silence = self.silence.to(self.device)[None].expand(batch, -1)
        state = model.new_stream(silence)  # type: ignore[attr-defined]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        decoding = DecodingConfig()
        frames_out = []
        for frame in range(frames):
            codes, _ = model.generate_step(  # type: ignore[attr-defined]
                user[:, :, frame], state, decoding, generator
            )
            frames_out.append(codes)
        return torch.stack(frames_out, dim=-1).cpu()
