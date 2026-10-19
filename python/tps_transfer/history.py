"""Four-observation, three-action causal history without frame copies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenHistory:
    """Inputs ordered as image, action, image, action, image, action, image."""

    images: tuple[np.ndarray | None, ...]
    actions: tuple[np.ndarray | None, ...]
    valid: tuple[bool, ...]


@dataclass(frozen=True)
class HistoryWindow:
    """The last observations and the actions *between* them.

    Arrays are held by reference. Rolling the window only creates short tuples.
    """

    observations: tuple[np.ndarray, ...]
    actions: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.observations) <= 4:
            raise ValueError("history must contain 1 to 4 observations")
        if len(self.actions) != len(self.observations) - 1:
            raise ValueError("each action must connect two observations")

    @classmethod
    def initial(cls, observation: np.ndarray) -> HistoryWindow:
        return cls((observation,), ())

    def advance(self, action: np.ndarray, observation: np.ndarray) -> HistoryWindow:
        return HistoryWindow(
            (self.observations + (observation,))[-4:],
            (self.actions + (action,))[-3:],
        )

    def tokens(self) -> TokenHistory:
        pad = 4 - len(self.observations)
        images: tuple[np.ndarray | None, ...] = (None,) * pad + self.observations
        actions: tuple[np.ndarray | None, ...] = (None,) * pad + self.actions
        valid = tuple(
            item is not None
            for pair in zip(images[:3], actions)
            for item in pair
        ) + (True,)
        return TokenHistory(images, actions, valid)

