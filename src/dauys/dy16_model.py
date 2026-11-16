"""DY-16: the DY-14 tower with a speech head that writes the own channel frame by frame.

The head reads the state at frame ``t`` and predicts the own token at ``t + 1`` among
K + 2 classes (a unit starts / speech continues / silence). Sampling it frame by frame
generates the own channel; the DY-14 losses stay on so listening is kept.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .dy14_model import Tower, TowerConfig

SPEECH_WEIGHT = 1.0


class Speaker(Tower):
    def __init__(self, cfg: TowerConfig) -> None:
        super().__init__(cfg)
        self.speech = nn.Linear(cfg.d, cfg.k + 2)

    def speech_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.speech(h).float()

    def loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        h = self.states(batch["tok_self"], batch["tok_partner"], batch["pros"])
        total, logs = self._tower_loss(h, batch)
        speech = F.cross_entropy(
            self.speech_logits(h[:, :-1]).flatten(0, 1),
            batch["tok_self"][:, 1:].flatten(),
        )
        total = total + SPEECH_WEIGHT * speech
        logs.update({"speech": speech.item(), "loss": total.item()})
        return total, logs

    def _tower_loss(
        self, h: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        hard, inbatch = self.scores(
            h,
            batch["anchor_row"],
            batch["anchor_pos"],
            batch["cand_ids"],
            batch["cand_mask"],
        )
        scale = self.log_scale.clamp(max=4.6052).exp()
        eye = torch.eye(len(hard), device=hard.device, dtype=torch.bool)
        logits = torch.cat(
            [hard * scale, (inbatch * scale).masked_fill(eye, float("-inf"))], 1
        )
        nce = F.cross_entropy(
            logits, torch.zeros(len(hard), dtype=torch.long, device=hard.device)
        )
        own = F.cross_entropy(
            self.own_next(h).flatten(0, 1).float(),
            batch["own_next"].flatten(),
            ignore_index=-100,
        )
        who = F.binary_cross_entropy_with_logits(self.who(h).float(), batch["who"])
        total = nce + 0.5 * own + 0.2 * who
        return total, {"nce": nce.item(), "own": own.item(), "who": who.item()}


@torch.no_grad()
def sample_next(
    logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator
) -> torch.Tensor:
    """One token per row from ``[B, V]`` logits with temperature and nucleus sampling."""
    probs = torch.softmax(logits / temperature, -1)
    sorted_p, idx = probs.sort(-1, descending=True)
    keep = sorted_p.cumsum(-1) - sorted_p < top_p
    sorted_p = sorted_p * keep
    pick = torch.multinomial(
        sorted_p / sorted_p.sum(-1, keepdim=True), 1, generator=gen
    )
    return idx.gather(-1, pick)[:, 0]
