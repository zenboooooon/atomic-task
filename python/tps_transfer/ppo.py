"""PPO/GAE primitives with explicit terminal and interruption semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GaeResult:
    advantages: Tensor
    returns: Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GaeResult:
    """Compute one agent's chronological rollout segment.

    A true terminal has no value bootstrap. An external truncation uses its
    *final* observation's value but the GAE recursion stops at that boundary.
    The last item of the segment also stops recursion while still bootstrapping
    if it is an unfinished episode.
    """
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("expected a nonempty one-dimensional rollout")
    for name, item in (
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if item.shape != rewards.shape:
            raise ValueError(f"{name} shape mismatch")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("boundary flags must be boolean")
    if torch.any(terminated & truncated):
        raise ValueError("a step cannot be both terminated and truncated")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")

    advantages = torch.empty_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.numel() - 1, -1, -1):
        bootstrap = (~terminated[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        continue_episode = (~(terminated[index] | truncated[index])).to(rewards.dtype)
        carry = delta + gamma * gae_lambda * continue_episode * carry
        advantages[index] = carry
    return GaeResult(advantages=advantages, returns=advantages + values)


def normalize_advantages(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("advantages must be a nonempty vector")
    return (values - values.mean()) / (values.std(unbiased=False) + epsilon)


def balanced_minibatches(
    task_sizes: Mapping[str, int],
    per_task: int,
    generator: torch.Generator,
) -> Iterator[dict[str, Tensor]]:
    """Independently shuffle equal task rollouts and yield equal task slices."""
    if not task_sizes or per_task < 1:
        raise ValueError("task sizes and per_task must be positive")
    sizes = set(task_sizes.values())
    if len(sizes) != 1:
        raise ValueError("all tasks must contribute the same number of transitions")
    size = sizes.pop()
    if size < per_task or size % per_task:
        raise ValueError("task size must be divisible by per_task")
    permutations = {
        task: torch.randperm(size, generator=generator) for task in task_sizes
    }
    for offset in range(0, size, per_task):
        yield {
            task: permutation[offset : offset + per_task]
            for task, permutation in permutations.items()
        }


@dataclass(frozen=True)
class PpoLosses:
    actor: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def ppo_losses(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    entropy_estimate: Tensor,
    clip_epsilon: float = 0.2,
) -> PpoLosses:
    shape = new_log_prob.shape
    if new_log_prob.ndim != 1:
        raise ValueError("PPO expects one-dimensional minibatch tensors")
    if any(item.shape != shape for item in
           (old_log_prob, advantages, values, returns, entropy_estimate)):
        raise ValueError("PPO minibatch shape mismatch")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must lie in (0, 1)")
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return PpoLosses(
        actor=-torch.minimum(unclipped, clipped).mean(),
        value=(values - returns).square().mean(),
        entropy=entropy_estimate.mean(),
        approximate_kl=((ratio - 1) - log_ratio).mean(),
        clip_fraction=(torch.abs(ratio - 1) > clip_epsilon).float().mean(),
    )


def pretraining_optimizer(
    shared: nn.Module,
    actors: Mapping[str, nn.Module],
    critics: Mapping[str, nn.Module],
    shared_lr: float = 1e-4,
    head_lr: float = 3e-4,
) -> torch.optim.Optimizer:
    if set(actors) != set(critics) or not actors:
        raise ValueError("each task needs exactly one actor and one critic")
    shared_params = list(shared.parameters())
    head_params = [
        parameter
        for task in sorted(actors)
        for module in (actors[task], critics[task])
        for parameter in module.parameters()
    ]
    if len({id(p) for p in shared_params + head_params}) != len(shared_params + head_params):
        raise ValueError("optimizer parameters overlap")
    return torch.optim.Adam(
        [
            {"params": shared_params, "lr": shared_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )


def downstream_optimizer(
    shared: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    train_shared: bool,
    shared_lr: float = 1e-4,
    head_lr: float = 3e-4,
) -> torch.optim.Optimizer:
    head_params = list(actor.parameters()) + list(critic.parameters())
    if train_shared:
        shared.requires_grad_(True)
        return torch.optim.Adam(
            [
                {"params": list(shared.parameters()), "lr": shared_lr},
                {"params": head_params, "lr": head_lr},
            ]
        )
    shared.requires_grad_(False)
    shared.eval()
    return torch.optim.Adam(head_params, lr=head_lr)
