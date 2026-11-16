"""DY-19: the DY-16 speaker plus an onset head that decides when to start speaking.

The onset head reads the state at frame ``t`` and, on frames where the own channel is silent,
predicts whether the own channel becomes active within the next 10 frames (1 s). Its target is
already in every training batch: the own half of the who-and-when target. All DY-16 losses stay.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .dy14_data import WHO_AHEAD
from .dy14_model import TowerConfig
from .dy16_model import Speaker

ONSET_WEIGHT = 1.0


def onset_target(batch: dict[str, torch.Tensor], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(silent mask [B, T], target [B, T]): own silent at t, own active somewhere in t+1..t+10."""
    silent = batch["tok_self"] == k + 1
    target = (batch["who"][..., :WHO_AHEAD] > 0.5).any(-1).float()
    return silent, target


class Onsetter(Speaker):
    def __init__(self, cfg: TowerConfig) -> None:
        super().__init__(cfg)
        self.onset = nn.Linear(cfg.d, 1)

    def onset_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.onset(h).float().squeeze(-1)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.states(batch["tok_self"], batch["tok_partner"], batch["pros"])
        total, logs = self._tower_loss(h, batch)
        speech = F.cross_entropy(
            self.speech_logits(h[:, :-1]).flatten(0, 1), batch["tok_self"][:, 1:].flatten()
        )
        silent, target = onset_target(batch, self.cfg.k)
        onset = F.binary_cross_entropy_with_logits(self.onset_logits(h)[silent], target[silent])
        total = total + speech + ONSET_WEIGHT * onset
        logs.update({"speech": speech.item(), "onset": onset.item(), "loss": total.item()})
        return total, logs


def load_speaker(path: str, device) -> Speaker:
    """DY-16 Speaker or DY-19 Onsetter, by what the checkpoint holds."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cls = Onsetter if "onset.weight" in ck["model"] else Speaker
    model = cls(TowerConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    return model
