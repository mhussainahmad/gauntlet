"""Uniform-random baseline policy."""

from __future__ import annotations

import numpy as np

from gauntlet.policy.base import Action, Observation

__all__ = ["RandomPolicy"]


class RandomPolicy:
    """Uniform-random baseline.

    Samples each action coordinate i.i.d. from ``U[action_low, action_high]``.
    Fully deterministic given a seed — re-running with the same seed produces
    identical actions (spec §6: reproducibility).
    """

    def __init__(
        self,
        action_dim: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_low >= action_high:
            raise ValueError(
                f"action_low ({action_low}) must be strictly less than action_high ({action_high})"
            )
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Action:
        del obs  # random policy ignores observations by design
        sample = self._rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=self.action_dim,
        )
        return np.asarray(sample, dtype=np.float64)

    def reset(self, rng: np.random.Generator) -> None:
        """Adopt the runner's episode RNG so rollouts are reproducible."""
        self._rng = rng
