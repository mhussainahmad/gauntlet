"""Action-entropy metric — RFC §5.

Torch-free. Pure numpy. The metric is a proxy: per-dim standard
deviation of the action vector across the steps of one episode, with
the scalar "entropy" reported as the mean over dims. See RFC §5 for
rationale (captures the two observable failure modes, works identically
for deterministic and stochastic policies, no reference distribution
required).

The helper is called directly from :mod:`gauntlet.monitor.score` and
reused by the torch-free smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ActionEntropyStats", "action_entropy"]


# The tabletop env has a 7-D action vector. We do not hard-code this
# check — the monitor module is supposed to be env-agnostic — but the
# tests exercise the 7-D case because that is the only env the harness
# ships in Phase 1/2.


@dataclass(frozen=True)
class ActionEntropyStats:
    """Output of :func:`action_entropy`.

    Attributes:
        per_dim_std: ``(D,)`` float64 array — population standard
            deviation of each action dim across the episode's steps.
        scalar: Scalar float — mean over ``per_dim_std``. This is the
            "entropy" value plotted on the drift panel.
    """

    per_dim_std: NDArray[np.float64]
    scalar: float


def action_entropy(actions: NDArray[np.float64]) -> ActionEntropyStats:
    """Compute per-dim std + scalar mean on a ``(T, D)`` action trajectory.

    Args:
        actions: Float64 ndarray of shape ``(T, D)`` — one episode's
            action trajectory. ``T`` is the number of env steps,
            ``D`` the action dimension.

    Returns:
        :class:`ActionEntropyStats` — per-dim and scalar values.

    Raises:
        ValueError: If ``actions`` is not 2-D, if ``T < 1`` (we need at
            least one step so the std has a defined value), or if
            ``actions`` has a non-float dtype.

    Note:
        ``np.std`` uses population std (``ddof=0``) — a single-step
        trajectory returns an all-zero std vector and a zero scalar.
        That matches the RFC §5 contract exactly (test case 2 in §11).
    """
    if not isinstance(actions, np.ndarray):
        raise ValueError(
            f"actions must be an ndarray; got {type(actions).__name__}",
        )
    if actions.ndim != 2:
        raise ValueError(
            f"actions must be 2-D (T, D); got shape {actions.shape}",
        )
    if actions.shape[0] < 1:
        raise ValueError(
            f"actions must have at least one step; got T={actions.shape[0]}",
        )
    if not np.issubdtype(actions.dtype, np.floating):
        raise ValueError(
            f"actions must be a float dtype; got {actions.dtype}",
        )

    per_dim_std = np.std(actions, axis=0, ddof=0).astype(np.float64, copy=False)
    scalar = float(per_dim_std.mean())
    return ActionEntropyStats(per_dim_std=per_dim_std, scalar=scalar)
