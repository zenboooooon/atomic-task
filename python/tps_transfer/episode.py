"""Match ML-Agents decisions to transitions by stable agent identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .history import HistoryWindow


@dataclass(frozen=True)
class Step:
    agent_id: int
    rgb: np.ndarray
    reward: float
    interrupted: bool = False


@dataclass(frozen=True)
class AgentKey:
    worker_id: int
    behavior: str
    agent_id: int
    episode_id: int


@dataclass(frozen=True)
class Transition:
    key: AgentKey
    decision_index: int
    history: HistoryWindow
    action: np.ndarray
    reward: float
    next_history: HistoryWindow
    terminated: bool
    truncated: bool


@dataclass
class _Active:
    key: AgentKey
    history: HistoryWindow
    decision_index: int = 0
    pending_action: np.ndarray | None = None


class EpisodeTracker:
    """Consume observations, then record actions, once per ML-Agents step.

    A terminal observation closes the pending action before any reset observation
    for the same agent ID is accepted. An interrupted transition carries its final
    observation for value bootstrap; both terminal types clear the episode history.
    """

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self._active: dict[tuple[str, int], _Active] = {}
        self._next_episode: dict[tuple[str, int], int] = {}

    def observe(
        self,
        behavior: str,
        decisions: list[Step],
        terminals: list[Step],
    ) -> list[Transition]:
        transitions: list[Transition] = []
        terminal_ids = [step.agent_id for step in terminals]
        decision_ids = [step.agent_id for step in decisions]
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("duplicate terminal agent ID")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("duplicate decision agent ID")

        for step in terminals:
            identity = (behavior, step.agent_id)
            active = self._active.get(identity)
            if active is None or active.pending_action is None:
                raise ValueError(f"terminal without pending action: {identity}")
            self._active.pop(identity)
            next_history = active.history.advance(active.pending_action, step.rgb)
            transitions.append(
                Transition(
                    active.key,
                    active.decision_index,
                    active.history,
                    active.pending_action,
                    float(step.reward),
                    next_history,
                    terminated=not step.interrupted,
                    truncated=step.interrupted,
                )
            )
            self._next_episode[identity] = active.key.episode_id + 1

        for step in decisions:
            identity = (behavior, step.agent_id)
            active = self._active.get(identity)
            if active is None:
                if step.reward != 0:
                    raise ValueError(f"new episode has nonzero reward: {identity}")
                episode_id = self._next_episode.get(identity, 0)
                key = AgentKey(self.worker_id, behavior, step.agent_id, episode_id)
                self._active[identity] = _Active(key, HistoryWindow.initial(step.rgb))
                continue
            if active.pending_action is None:
                raise ValueError(f"decision without pending action: {identity}")
            next_history = active.history.advance(active.pending_action, step.rgb)
            transitions.append(
                Transition(
                    active.key,
                    active.decision_index,
                    active.history,
                    active.pending_action,
                    float(step.reward),
                    next_history,
                    terminated=False,
                    truncated=False,
                )
            )
            active.history = next_history
            active.pending_action = None
            active.decision_index += 1
        return transitions

    def key_for(self, behavior: str, agent_id: int) -> AgentKey:
        return self._active[(behavior, agent_id)].key

    def history_for(self, behavior: str, agent_id: int) -> HistoryWindow:
        return self._active[(behavior, agent_id)].history

    def record_action(self, behavior: str, agent_id: int, action: np.ndarray) -> None:
        active = self._active[(behavior, agent_id)]
        if active.pending_action is not None:
            raise ValueError("agent already has a pending action")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 5-vector")
        if (np.abs(action) > 1).any():
            raise ValueError("action must lie in [-1, 1]")
        active.pending_action = action.copy()

