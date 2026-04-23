"""Action-entropy metric — torch-free unit tests.

Exercises :func:`gauntlet.monitor.entropy.action_entropy` against
synthetic trajectories with known properties. Lives outside the
``monitor`` marker because it has no torch dependency — runs in the
default CI job.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauntlet.monitor import action_entropy
from gauntlet.monitor.entropy import ActionEntropyStats


def test_constant_actions_give_zero_entropy() -> None:
    """``A = np.ones((50, 7))`` -> all-zero std, zero scalar (RFC §11 case 3).

    This guards against the classic bug of computing std across dims
    (axis=1) instead of across timesteps (axis=0).
    """
    actions = np.ones((50, 7), dtype=np.float64)
    stats = action_entropy(actions)
    assert isinstance(stats, ActionEntropyStats)
    assert stats.per_dim_std.shape == (7,)
    np.testing.assert_allclose(stats.per_dim_std, 0.0, atol=1e-12)
    assert stats.scalar == 0.0


def test_known_std_round_trips() -> None:
    """Construct a trajectory with dim-specific stds; assert they match."""
    rng = np.random.default_rng(seed=17)
    t_steps = 200
    actions = np.zeros((t_steps, 7), dtype=np.float64)
    # Each dim gets a known-variance noise stream.
    target_stds = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5], dtype=np.float64)
    for i, target in enumerate(target_stds):
        actions[:, i] = rng.normal(loc=0.0, scale=target, size=t_steps)

    stats = action_entropy(actions)
    # Population std over 200 samples with a known scale converges to
    # the target within ~10% in practice; 15% is a generous budget.
    np.testing.assert_allclose(stats.per_dim_std, target_stds, rtol=0.15)
    assert stats.scalar == pytest.approx(float(stats.per_dim_std.mean()))


def test_single_step_returns_zero_std() -> None:
    """T=1 -> population std is zero everywhere (RFC §5 contract)."""
    actions = np.array([[0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -1.0]], dtype=np.float64)
    stats = action_entropy(actions)
    np.testing.assert_allclose(stats.per_dim_std, 0.0, atol=1e-12)
    assert stats.scalar == 0.0


def test_zero_step_actions_raise_value_error() -> None:
    """T=0 is undefined — action_entropy refuses with a clear message."""
    actions = np.zeros((0, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="at least one step"):
        action_entropy(actions)


def test_wrong_shape_rejected() -> None:
    """1-D and 3-D inputs are rejected explicitly."""
    with pytest.raises(ValueError, match="2-D"):
        action_entropy(np.zeros(7, dtype=np.float64))
    with pytest.raises(ValueError, match="2-D"):
        action_entropy(np.zeros((3, 4, 5), dtype=np.float64))


def test_integer_dtype_rejected() -> None:
    """Integer arrays are rejected — ``std`` would upcast silently otherwise."""
    with pytest.raises(ValueError, match="float dtype"):
        action_entropy(np.zeros((10, 7), dtype=np.int64))
