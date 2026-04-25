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
        """Bind a uniform sampler over ``[action_low, action_high]^action_dim``.

        ``seed`` initialises an internal :class:`numpy.random.Generator`.
        The Runner's per-episode RNG replaces it via :meth:`reset`, so
        ``seed`` only affects callers that drive the policy outside the
        Runner.
        """
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
        """Draw one ``(action_dim,) float64`` action from the configured uniform.

        Observations are ignored by design — the random baseline is the
        zero-information reference point against which a learned policy's
        success rate is judged. Determinism is anchored on :attr:`_rng`,
        which :meth:`reset` rebinds at the start of every episode.
        """
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
