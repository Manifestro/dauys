"""Audio-token preference objectives and simulator report scoring."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RewardWeights:
    onset: float = 1.0
    yield_latency: float = 1.0
    backchannel: float = 1.0
    relevance: float = 1.0
    task_success: float = 1.0
    false_start: float = 1.0
    talk_over: float = 1.0


def reward_from_report(report: dict[str, float], weights: RewardWeights) -> float:
    """Score a simulator report after all components have been measured."""
    required = {
        "onset_human_range",
        "yield_within_500ms",
        "backchannel_appropriate",
        "relevance_0_to_3",
        "task_success",
        "false_starts",
        "talk_over_frames",
        "response_opportunities",
        "duration_frames",
    }
    if not required <= report.keys():
        raise ValueError(f"Missing reward components: {sorted(required - report.keys())}")
    if report["response_opportunities"] <= 0 or report["duration_frames"] <= 0:
        raise ValueError("Reward denominators must be positive")
    return (
        weights.onset * report["onset_human_range"]
        + weights.yield_latency * report["yield_within_500ms"]
        + weights.backchannel * report["backchannel_appropriate"]
        + weights.relevance * report["relevance_0_to_3"] / 3
        + weights.task_success * report["task_success"]
        - weights.false_start
        * report["false_starts"]
        / report["response_opportunities"]
        - weights.talk_over * report["talk_over_frames"] / report["duration_frames"]
    )


def best_of_n(
    candidates: list[tuple[str, str, dict[str, float]]],
    weights: RewardWeights,
) -> dict[str, str]:
    """Choose one audio candidate per context for winner-only SFT."""
    winners: dict[str, tuple[str, float]] = {}
    for context_id, candidate_id, report in candidates:
        score = reward_from_report(report, weights)
        if context_id not in winners or score > winners[context_id][1]:
            winners[context_id] = (candidate_id, score)
    return {context: winner[0] for context, winner in winners.items()}


def sequence_log_probability(
    logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Sum causal audio-token log probabilities over valid frames."""
    if logits.ndim != 3 or tokens.shape != logits.shape[:2] or valid.shape != tokens.shape:
        raise ValueError("Expected logits [batch, frames, vocabulary] and frame masks")
    values = F.log_softmax(logits.float(), dim=-1).gather(
        -1, tokens.unsqueeze(-1)
    ).squeeze(-1)
    return (values * valid.to(values.dtype)).sum(dim=1)


def audio_dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    chosen_context_ids: tuple[str, ...],
    rejected_context_ids: tuple[str, ...],
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO on response sequences paired under identical audio contexts."""
    if beta <= 0 or chosen_context_ids != rejected_context_ids:
        raise ValueError("DPO requires positive beta and identical contexts")
    tensors = (
        chosen_logp,
        rejected_logp,
        reference_chosen_logp,
        reference_rejected_logp,
    )
    if not tensors or any(tensor.shape != (len(chosen_context_ids),) for tensor in tensors):
        raise ValueError("DPO inputs must contain one sequence score per pair")
    advantage = (
        chosen_logp - rejected_logp
        - (reference_chosen_logp - reference_rejected_logp).detach()
    )
    return -F.logsigmoid(beta * advantage).mean()


def clipped_policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    valid: torch.Tensor,
    *,
    clip: float = 0.2,
) -> torch.Tensor:
    """Frame-wise PPO surrogate; rewards and advantages come from the simulator."""
    if logp.shape != old_logp.shape or logp.shape != advantages.shape or logp.shape != valid.shape:
        raise ValueError("Policy tensors and valid mask must have equal shape")
    if logp.ndim != 2 or not 0 < clip < 1 or not valid.any():
        raise ValueError("Expected nonempty [batch, frames] data and valid clip")
    ratio = (logp - old_logp.detach()).clamp(-20, 20).exp()
    surrogate = torch.minimum(
        ratio * advantages.detach(),
        ratio.clamp(1 - clip, 1 + clip) * advantages.detach(),
    )
    return -(surrogate * valid.to(surrogate.dtype)).sum() / valid.sum()


def group_relative_advantages(
    rewards: torch.Tensor, context_ids: tuple[str, ...]
) -> torch.Tensor:
    """Center and scale simulator rewards within each sampled context group."""
    if rewards.ndim != 1 or rewards.numel() != len(context_ids):
        raise ValueError("One reward and context ID are required per rollout")
    result = torch.zeros_like(rewards)
    for context in set(context_ids):
        indices = [index for index, value in enumerate(context_ids) if value == context]
        if len(indices) < 2:
            raise ValueError("Group-relative updates require at least two rollouts")
        selected = rewards[indices]
        result[indices] = (selected - selected.mean()) / selected.std(
            unbiased=False
        ).clamp_min(1e-6)
    return result
