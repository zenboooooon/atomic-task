"""Single-task PPO pilot against the rendered Unity task player."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from torch import Tensor

from .episode import AgentKey, EpisodeTracker, Transition
from .model import ActorHead, CriticHead, SharedEncoder, TanhGaussian
from .observation import FrameProcessor, history_batch
from .ppo import compute_gae, normalize_advantages, ppo_losses
from .smoke import _steps


@dataclass(frozen=True)
class Sample:
    transition: Transition
    pre_tanh: np.ndarray
    old_log_prob: float
    old_value: float


@dataclass(frozen=True)
class Rollout:
    samples: list[Sample]
    advantages: Tensor
    returns: Tensor
    terminal_successes: int
    terminal_failures: int
    interruptions: int
    surplus_transitions_discarded: int = 0


class PilotCollector:
    def __init__(
        self, executable: Path, task: str, seed: int, processor: FrameProcessor,
        shared: SharedEncoder, actor: ActorHead, critic: CriticHead,
        device: torch.device, worker_id: int = 0,
    ) -> None:
        self.processor = processor
        self.shared = shared
        self.actor = actor
        self.critic = critic
        self.device = device
        self.environment = UnityEnvironment(
            file_name=str(executable), worker_id=worker_id, seed=seed, no_graphics=False,
            additional_args=[f"--tps-task={task}", "--tps-split=train"],
        )
        self.tracker = EpisodeTracker(worker_id=worker_id)
        self.pending: dict[AgentKey, tuple[np.ndarray, float, float] | None] = {}
        self.environment.reset()
        if len(self.environment.behavior_specs) != 1:
            raise RuntimeError("single-task pilot expects one behavior")
        self.behavior = next(iter(self.environment.behavior_specs))
        self.decisions, terminals = self.environment.get_steps(self.behavior)
        self.tracker.observe(self.behavior, _steps(self.decisions, False), _steps(terminals, True))

    def close(self) -> None:
        self.environment.close()

    def collect(self, count: int, gamma: float, gae_lambda: float) -> Rollout:
        if count < 1:
            raise ValueError("rollout count must be positive")
        self.shared.eval()
        self.actor.eval()
        self.critic.eval()
        samples: list[Sample] = []
        successes = failures = interruptions = surplus = 0
        idle_steps = 0
        while len(samples) < count:
            action_count = len(self.decisions)
            if action_count:
                ids = [int(agent_id) for agent_id in self.decisions.agent_id]
                histories = [self.tracker.history_for(self.behavior, agent_id) for agent_id in ids]
                images, actions, valid = history_batch(histories, self.processor, self.device)
                with torch.inference_mode():
                    features = self.shared(images, actions, valid)
                    mean, log_std = self.actor(features)
                    distribution = TanhGaussian(mean, log_std)
                    selected, raw = distribution.sample()
                    log_prob = distribution.log_prob(raw)
                    value = self.critic(features)
                selected_np = selected.cpu().numpy().astype(np.float32)
                raw_np = raw.cpu().numpy().astype(np.float32)
                for row, agent_id in enumerate(ids):
                    key = self.tracker.key_for(self.behavior, agent_id)
                    self.tracker.record_action(self.behavior, agent_id, selected_np[row])
                    if key in self.pending:
                        raise RuntimeError("duplicate pending pilot action")
                    self.pending[key] = (
                        raw_np[row], float(log_prob[row].item()), float(value[row].item())
                    )
                self.environment.set_actions(self.behavior, ActionTuple(continuous=selected_np))
            self.environment.step()
            self.decisions, terminals = self.environment.get_steps(self.behavior)
            transitions = self.tracker.observe(
                self.behavior, _steps(self.decisions, False), _steps(terminals, True)
            )
            for transition in transitions:
                metadata = self.pending.pop(transition.key)
                if metadata is None:
                    raise RuntimeError("discard-only action appeared during collection")
                raw_action, old_log_prob, old_value = metadata
                if len(samples) >= count:
                    surplus += 1
                    continue
                samples.append(Sample(transition, raw_action, old_log_prob, old_value))
                if transition.terminated:
                    if transition.reward > 0.5:
                        successes += 1
                    else:
                        failures += 1
                elif transition.truncated:
                    interruptions += 1
            idle_steps = 0 if action_count or transitions else idle_steps + 1
            if idle_steps > 10:
                raise RuntimeError("Unity stopped returning decisions or transitions")
        # Resolve actions sent under the old policy before any PPO parameter
        # update. Zero actions used only for drainage are excluded from training.
        drain_steps = 0
        while self.pending:
            if drain_steps >= 10:
                raise RuntimeError("pending Unity actions did not drain at rollout boundary")
            if len(self.decisions):
                ids = [int(agent_id) for agent_id in self.decisions.agent_id]
                zeros = np.zeros((len(ids), 5), dtype=np.float32)
                for row, agent_id in enumerate(ids):
                    key = self.tracker.key_for(self.behavior, agent_id)
                    self.tracker.record_action(self.behavior, agent_id, zeros[row])
                    if key in self.pending:
                        raise RuntimeError("duplicate pending drain action")
                    self.pending[key] = None
                self.environment.set_actions(self.behavior, ActionTuple(continuous=zeros))
            self.environment.step()
            self.decisions, terminals = self.environment.get_steps(self.behavior)
            transitions = self.tracker.observe(
                self.behavior, _steps(self.decisions, False), _steps(terminals, True)
            )
            for transition in transitions:
                self.pending.pop(transition.key)
                surplus += 1
            drain_steps += 1
        rewards = torch.tensor([sample.transition.reward for sample in samples], dtype=torch.float32)
        values = torch.tensor([sample.old_value for sample in samples], dtype=torch.float32)
        next_values = torch.zeros_like(values)
        with torch.inference_mode():
            live_indices = [index for index, sample in enumerate(samples)
                            if not sample.transition.terminated]
            for start in range(0, len(live_indices), 16):
                indices = live_indices[start:start + 16]
                next_histories = [samples[index].transition.next_history for index in indices]
                images, actions, valid = history_batch(next_histories, self.processor, self.device)
                estimate = self.critic(self.shared(images, actions, valid)).cpu()
                next_values[indices] = estimate
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        episode_indices: dict[AgentKey, list[int]] = {}
        for index, sample in enumerate(samples):
            episode_indices.setdefault(sample.transition.key, []).append(index)
        for indices in episode_indices.values():
            result = compute_gae(
                rewards[indices], values[indices], next_values[indices],
                torch.tensor([samples[index].transition.terminated for index in indices]),
                torch.tensor([samples[index].transition.truncated for index in indices]),
                gamma=gamma, gae_lambda=gae_lambda,
            )
            advantages[indices] = result.advantages
            returns[indices] = result.returns
        return Rollout(samples, normalize_advantages(advantages), returns,
                       successes, failures, interruptions, surplus)


def update_pilot(
    rollout: Rollout, processor: FrameProcessor, shared: SharedEncoder,
    actor: ActorHead, critic: CriticHead, optimizer: torch.optim.Optimizer,
    device: torch.device, epochs: int = 4, minibatch: int = 8,
) -> dict[str, float]:
    if any(parameter.requires_grad for parameter in shared.parameters()):
        shared.train()
    else:
        shared.eval()
    actor.train()
    critic.train()
    old_log_prob = torch.tensor([sample.old_log_prob for sample in rollout.samples], device=device)
    raw = torch.from_numpy(np.stack([sample.pre_tanh for sample in rollout.samples])).to(device)
    advantages = rollout.advantages.to(device)
    returns = rollout.returns.to(device)
    rng = torch.randperm
    metrics: list[tuple[float, float, float, float, float]] = []
    initial_log_prob_max_error = None
    epochs_completed = 0
    for epoch in range(epochs):
        epoch_start = len(metrics)
        for index in rng(len(rollout.samples)).split(minibatch):
            selected = [rollout.samples[int(item)] for item in index]
            images, actions, valid = history_batch(
                [sample.transition.history for sample in selected], processor, device
            )
            features = shared(images, actions, valid)
            mean, log_std = actor(features)
            distribution = TanhGaussian(mean, log_std)
            new_log_prob = distribution.log_prob(raw[index])
            if initial_log_prob_max_error is None:
                initial_log_prob_max_error = float(
                    (new_log_prob - old_log_prob[index]).abs().max().detach().cpu()
                )
            value = critic(features)
            losses = ppo_losses(
                new_log_prob, old_log_prob[index], advantages[index], value,
                returns[index], distribution.entropy_estimate(),
            )
            loss = losses.actor + 0.5 * losses.value - 0.01 * losses.entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(shared.parameters()) + list(actor.parameters()) + list(critic.parameters()), 0.5
            )
            optimizer.step()
            metrics.append(tuple(float(item.detach().cpu()) for item in (
                losses.actor, losses.value, losses.entropy,
                losses.approximate_kl, losses.clip_fraction,
            )))
        epochs_completed += 1
        if np.asarray(metrics[epoch_start:])[:, 3].mean() > 0.02:
            break
    average = np.asarray(metrics).mean(axis=0)
    return {
        "actor_loss": float(average[0]),
        "value_loss": float(average[1]),
        "entropy_estimate": float(average[2]),
        "approximate_kl": float(average[3]),
        "clip_fraction": float(average[4]),
        "initial_log_prob_max_error": float(initial_log_prob_max_error),
        "epochs_completed": epochs_completed,
        "kl_early_stopped": epochs_completed < epochs,
    }
