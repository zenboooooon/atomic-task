"""Fixed-budget learning-curve metrics for the transfer comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearningCurve:
    """Success counts at a fixed, complete grid of training decisions."""

    steps: np.ndarray
    successes: np.ndarray
    episodes: np.ndarray
    planned_budget: int
    evaluation_interval: int

    def __post_init__(self) -> None:
        steps = np.asarray(self.steps, dtype=np.int64)
        successes = np.asarray(self.successes, dtype=np.int64)
        episodes = np.asarray(self.episodes, dtype=np.int64)
        if steps.ndim != 1 or len(steps) < 2:
            raise ValueError("need at least two evaluation points")
        if successes.shape != steps.shape or episodes.shape != steps.shape:
            raise ValueError("success and episode counts must match steps")
        if (self.planned_budget <= 0 or self.evaluation_interval <= 0
                or self.planned_budget % self.evaluation_interval):
            raise ValueError("budget must be a positive multiple of the evaluation interval")
        expected = np.arange(
            0, self.planned_budget + 1, self.evaluation_interval, dtype=np.int64
        )
        if not np.array_equal(steps, expected):
            raise ValueError("evaluations must cover the complete fixed grid")
        if np.any(episodes <= 0) or np.any(successes < 0) or np.any(successes > episodes):
            raise ValueError("invalid success counts")
        object.__setattr__(self, "steps", steps.copy())
        object.__setattr__(self, "successes", successes.copy())
        object.__setattr__(self, "episodes", episodes.copy())

    @property
    def budget(self) -> int:
        return self.planned_budget

    @property
    def rates(self) -> np.ndarray:
        return self.successes / self.episodes

    @property
    def nauc(self) -> float:
        return float(np.trapz(self.rates, self.steps) / self.budget)

    def t80(self, threshold: float = 0.8, consecutive: int = 3) -> int | None:
        if not 0 < threshold <= 1 or consecutive < 1:
            raise ValueError("invalid sustained-success rule")
        attained = self.rates >= threshold
        for index in range(consecutive - 1, len(attained)):
            if np.all(attained[index - consecutive + 1 : index + 1]):
                return int(self.steps[index])
        return None


@dataclass(frozen=True)
class PairedInterval:
    mean_difference: float
    lower: float
    upper: float
    pairs: int
    confidence: float


def paired_bootstrap(
    treatment: np.ndarray,
    comparator: np.ndarray,
    *,
    draws: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedInterval:
    """Percentile interval from paired, independent run-level units."""
    treatment = np.asarray(treatment, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    if treatment.ndim != 1 or comparator.shape != treatment.shape or not len(treatment):
        raise ValueError("paired arrays must be equal nonempty vectors")
    if not np.isfinite(treatment).all() or not np.isfinite(comparator).all():
        raise ValueError("paired values must be finite")
    if draws < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    differences = treatment - comparator
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    resampled_means = differences[indices].mean(axis=1)
    alpha = 1 - confidence
    lower, upper = np.quantile(resampled_means, (alpha / 2, 1 - alpha / 2))
    return PairedInterval(
        mean_difference=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        pairs=len(differences),
        confidence=confidence,
    )
