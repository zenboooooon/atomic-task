"""Balanced four-task PPO pilot on one rendered 16-arena Unity player."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from torch import Tensor

from .episode import AgentKey, EpisodeTracker
from .model import ActorHead, CriticHead, SharedEncoder, TanhGaussian
from .observation import FrameProcessor, history_batch
from .pilot import Rollout, Sample
from .ppo import balanced_minibatches, compute_gae, normalize_advantages, ppo_losses
from .smoke import _steps
from .primitive_variants import COLOR_TASKS

TASKS = ("look", "approach", "stop", "avoid", "jump")
TREASURE_SOURCE_TASKS = ("look", "approach", "avoid")
GAZE_SOURCE_TASKS = ("look", "approach")


class MultiTaskCollector:
    def __init__(
        self, executable: str, seed: int, processor: FrameProcessor,
        shared: SharedEncoder, actors: Mapping[str, ActorHead],
        critics: Mapping[str, CriticHead], device: torch.device,
        tasks: tuple[str, ...] = TASKS,
        worker_id: int = 0,
    ) -> None:
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASKS + COLOR_TASKS):
            raise ValueError("invalid primitive task set")
        if set(actors) != set(tasks) or set(critics) != set(tasks):
            raise ValueError("an actor and critic head are required for each task")
        self.tasks = tasks
        self.processor = processor
        self.shared = shared
        self.actors = actors
        self.critics = critics
        self.device = device
        self.environment = UnityEnvironment(
            file_name=executable, worker_id=worker_id, seed=seed, no_graphics=False,
            additional_args=["--tps-task=pretrain", "--tps-split=train"],
        )
        self.tracker = EpisodeTracker(worker_id=worker_id)
        self.pending: dict[AgentKey, tuple[np.ndarray, float, float] | None] = {}
        self.environment.reset()
        self.behaviors: dict[str, str] = {}
        for behavior, spec in self.environment.behavior_specs.items():
            task = behavior.split("?", 1)[0].removeprefix("TPS_")
            if task not in self.tasks or task in self.behaviors:
                raise RuntimeError(f"unexpected or duplicate task behavior: {behavior}")
            if spec.action_spec.continuous_size != 5 or spec.action_spec.discrete_size:
                raise RuntimeError(f"unexpected action specification: {behavior}")
            self.behaviors[task] = behavior
        if set(self.behaviors) != set(self.tasks):
            raise RuntimeError("expected task behaviors did not appear")
        self.decisions = {}
        for task, behavior in self.behaviors.items():
            decisions, terminals = self.environment.get_steps(behavior)
            self.tracker.observe(behavior, _steps(decisions, False), _steps(terminals, True))
            self.decisions[task] = decisions

    def close(self) -> None:
        self.environment.close()

    def collect(self, per_task: int, gamma: float = 0.99,
                gae_lambda: float = 0.95) -> tuple[dict[str, Rollout], dict[str, int]]:
        if per_task < 1:
            raise ValueError("per-task rollout count must be positive")
        self.shared.eval()
        for module in list(self.actors.values()) + list(self.critics.values()):
            module.eval()
        samples: dict[str, list[Sample]] = {task: [] for task in self.tasks}
        observed = {task: 0 for task in self.tasks}
        outcomes = {task: [0, 0, 0] for task in self.tasks}
        idle = 0
        while any(len(samples[task]) < per_task for task in self.tasks):
            actions_sent = 0
            for task in self.tasks:
                behavior = self.behaviors[task]
                decisions = self.decisions[task]
                if not len(decisions):
                    continue
                ids = [int(agent_id) for agent_id in decisions.agent_id]
                histories = [self.tracker.history_for(behavior, agent_id) for agent_id in ids]
                images, actions, valid = history_batch(histories, self.processor, self.device)
                with torch.inference_mode():
                    features = self.shared(images, actions, valid)
                    mean, log_std = self.actors[task](features)
                    distribution = TanhGaussian(mean, log_std)
                    selected, raw = distribution.sample()
                    log_prob = distribution.log_prob(raw)
                    values = self.critics[task](features)
                selected_np = selected.cpu().numpy().astype(np.float32)
                raw_np = raw.cpu().numpy().astype(np.float32)
                for row, agent_id in enumerate(ids):
                    key = self.tracker.key_for(behavior, agent_id)
                    self.tracker.record_action(behavior, agent_id, selected_np[row])
                    if key in self.pending:
                        raise RuntimeError("duplicate pending multitask action")
                    self.pending[key] = (
                        raw_np[row], float(log_prob[row].item()), float(values[row].item())
                    )
                self.environment.set_actions(behavior, ActionTuple(continuous=selected_np))
                actions_sent += len(ids)
            self.environment.step()
            for task in self.tasks:
                behavior = self.behaviors[task]
                decisions, terminals = self.environment.get_steps(behavior)
                transitions = self.tracker.observe(
                    behavior, _steps(decisions, False), _steps(terminals, True)
                )
                self.decisions[task] = decisions
                for transition in transitions:
                    metadata = self.pending.pop(transition.key)
                    if metadata is None:
                        raise RuntimeError("discard-only action appeared during collection")
                    raw, old_log_prob, old_value = metadata
                    observed[task] += 1
                    if len(samples[task]) < per_task:
                        samples[task].append(Sample(transition, raw, old_log_prob, old_value))
                        if transition.terminated:
                            outcomes[task][0 if transition.reward > 0.5 else 1] += 1
                        elif transition.truncated:
                            outcomes[task][2] += 1
            idle = 0 if actions_sent else idle + 1
            if idle > 10:
                raise RuntimeError("multitask Unity player stopped returning decisions")
        drain_steps = 0
        while self.pending:
            if drain_steps >= 10:
                raise RuntimeError("pending multitask actions did not drain")
            for task in self.tasks:
                behavior = self.behaviors[task]
                decisions = self.decisions[task]
                if not len(decisions):
                    continue
                ids = [int(agent_id) for agent_id in decisions.agent_id]
                zeros = np.zeros((len(ids), 5), dtype=np.float32)
                for row, agent_id in enumerate(ids):
                    key = self.tracker.key_for(behavior, agent_id)
                    self.tracker.record_action(behavior, agent_id, zeros[row])
                    if key in self.pending:
                        raise RuntimeError("duplicate pending multitask drain action")
                    self.pending[key] = None
                self.environment.set_actions(behavior, ActionTuple(continuous=zeros))
            self.environment.step()
            for task in self.tasks:
                behavior = self.behaviors[task]
                decisions, terminals = self.environment.get_steps(behavior)
                transitions = self.tracker.observe(
                    behavior, _steps(decisions, False), _steps(terminals, True)
                )
                self.decisions[task] = decisions
                for transition in transitions:
                    self.pending.pop(transition.key)
                    observed[task] += 1
            drain_steps += 1
        rollouts = {}
        for task in self.tasks:
            rollouts[task] = self._finish_task(samples[task], task, outcomes[task],
                                               gamma, gae_lambda)
        return rollouts, {task: observed[task] - per_task for task in self.tasks}

    def _finish_task(self, samples: list[Sample], task: str, outcomes: list[int],
                     gamma: float, gae_lambda: float) -> Rollout:
        rewards = torch.tensor([sample.transition.reward for sample in samples], dtype=torch.float32)
        values = torch.tensor([sample.old_value for sample in samples], dtype=torch.float32)
        next_values = torch.zeros_like(values)
        live = [index for index, sample in enumerate(samples)
                if not sample.transition.terminated]
        with torch.inference_mode():
            for start in range(0, len(live), 16):
                indices = live[start:start + 16]
                images, actions, valid = history_batch(
                    [samples[index].transition.next_history for index in indices],
                    self.processor, self.device,
                )
                next_values[indices] = self.critics[task](
                    self.shared(images, actions, valid)
                ).cpu()
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        groups: dict[AgentKey, list[int]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(sample.transition.key, []).append(index)
        for indices in groups.values():
            gae = compute_gae(
                rewards[indices], values[indices], next_values[indices],
                torch.tensor([samples[index].transition.terminated for index in indices]),
                torch.tensor([samples[index].transition.truncated for index in indices]),
                gamma, gae_lambda,
            )
            advantages[indices] = gae.advantages
            returns[indices] = gae.returns
        return Rollout(samples, normalize_advantages(advantages), returns,
                       outcomes[0], outcomes[1], outcomes[2])


def update_balanced(
    rollouts: Mapping[str, Rollout], processor: FrameProcessor,
    shared: SharedEncoder, actors: Mapping[str, ActorHead],
    critics: Mapping[str, CriticHead], optimizer: torch.optim.Optimizer,
    device: torch.device, epochs: int, per_task_minibatch: int,
    generator: torch.Generator,
    tasks: tuple[str, ...] = TASKS,
) -> dict[str, dict[str, float]]:
    if set(rollouts) != set(tasks) or set(actors) != set(tasks) or set(critics) != set(tasks):
        raise ValueError("rollouts and heads must match the task set")
    shared.train()
    for module in list(actors.values()) + list(critics.values()):
        module.train()
    counts = {task: len(rollouts[task].samples) for task in tasks}
    data = {}
    for task in tasks:
        rollout = rollouts[task]
        data[task] = (
            torch.tensor([sample.old_log_prob for sample in rollout.samples], device=device),
            torch.from_numpy(np.stack([sample.pre_tanh for sample in rollout.samples])).to(device),
            rollout.advantages.to(device), rollout.returns.to(device),
        )
    records: dict[str, list[list[float]]] = {task: [] for task in tasks}
    epochs_completed = 0
    for _ in range(epochs):
        epoch_start = {task: len(records[task]) for task in tasks}
        for selected in balanced_minibatches(counts, per_task_minibatch, generator):
            optimizer.zero_grad(set_to_none=True)
            combined_loss: Tensor | None = None
            for task in tasks:
                indices = selected[task]
                rollout = rollouts[task]
                old_log_prob, raw, advantages, returns = data[task]
                images, actions, valid = history_batch(
                    [rollout.samples[int(item)].transition.history for item in indices],
                    processor, device,
                )
                features = shared(images, actions, valid)
                mean, log_std = actors[task](features)
                distribution = TanhGaussian(mean, log_std)
                losses = ppo_losses(
                    distribution.log_prob(raw[indices]), old_log_prob[indices],
                    advantages[indices], critics[task](features), returns[indices],
                    distribution.entropy_estimate(),
                )
                task_loss = losses.actor + 0.5 * losses.value - 0.01 * losses.entropy
                combined_loss = task_loss if combined_loss is None else combined_loss + task_loss
                records[task].append([
                    float(losses.actor.detach().cpu()),
                    float(losses.value.detach().cpu()),
                    float(losses.approximate_kl.detach().cpu()),
                    float(losses.clip_fraction.detach().cpu()),
                ])
            assert combined_loss is not None
            (combined_loss / len(tasks)).backward()
            parameters = list(shared.parameters())
            for task in tasks:
                parameters += list(actors[task].parameters()) + list(critics[task].parameters())
            # A shared head occurs under three rollout-group names. Clip it once.
            parameters = list({id(parameter): parameter for parameter in parameters}.values())
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
        epochs_completed += 1
        if any(np.asarray(records[task][epoch_start[task]:])[:, 2].mean() > 0.02
               for task in tasks):
            break
    result = {
        task: dict(zip(("actor_loss", "value_loss", "approximate_kl", "clip_fraction"),
                       np.asarray(records[task]).mean(axis=0).tolist()))
        for task in tasks
    }
    result["_summary"] = {
        "epochs_completed": epochs_completed,
        "kl_early_stopped": epochs_completed < epochs,
    }
    return result
