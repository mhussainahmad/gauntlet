"""Fuzz tests for env action-space clipping contract.

Phase 2.5 Task 13 — every concrete env in the registry exposes a Box
``action_space`` and clips out-of-bounds actions silently (see
``TabletopEnv.step``: ``np.clip(a, -1.0, 1.0)``). The contract under
fuzz: an action outside the declared envelope MUST NOT crash the env;
the returned observation, reward, and termination flags remain in-spec.

Cost control: each example loads a fresh :class:`TabletopEnv` (MuJoCo
model load is ~50ms cold). We cap at ``max_examples=10`` and reuse the
env across the inner loop. Wall-time well under 5s.

We do NOT fuzz NaN / inf actions: ``np.clip(NaN, -1, 1)`` returns NaN,
which the env propagates without raising. Asserting that NaN propagates
silently is a behaviour the env does not promise; the production
contract starts at *finite* float64 actions.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.env.tabletop import TabletopEnv

# Action dim mirrors ``TabletopEnv.action_space`` (Box low=-1, high=1, shape=(7,)).
_ACTION_DIM = 7


@pytest.fixture(scope="module")
def env() -> Iterator[TabletopEnv]:
    """One env per module — cuts MuJoCo load cost from ~50ms x N to ~50ms once."""
    e = TabletopEnv(max_steps=4)
    try:
        yield e
    finally:
        e.close()


# ----- finite out-of-bounds -- must clip silently ---------------------------


@given(
    action=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=_ACTION_DIM,
        max_size=_ACTION_DIM,
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=10,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_out_of_bounds_action_does_not_crash_env(
    env: TabletopEnv, action: list[float], seed: int
) -> None:
    """For any finite-but-arbitrary action vector (including values
    far outside the ``[-1, 1]`` declared envelope), :meth:`step`
    completes cleanly: the contract is silent clipping, never an
    uncaught :class:`ValueError` from MuJoCo's control buffer."""
    env.reset(seed=seed)
    a = np.asarray(action, dtype=np.float64)
    obs, reward, terminated, truncated, _info = env.step(a)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool | np.bool_)
    assert isinstance(truncated, bool | np.bool_)
    # Observation has the documented Dict shape with finite entries
    # for the proprio keys (image is uint8 when present, also finite).
    for key in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
        assert key in obs, f"missing obs key {key}"
        arr = np.asarray(obs[key], dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"non-finite values in obs[{key}]: {arr}"


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=10,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_in_bounds_action_step_invariants(env: TabletopEnv, seed: int) -> None:
    """A within-envelope action must produce the same well-formed
    output as the out-of-bounds case — the property is invariant in
    action value, only varying in the env's internal state."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    action = rng.uniform(-1.0, 1.0, size=_ACTION_DIM).astype(np.float64)
    obs, reward, _terminated, _truncated, _info = env.step(action)
    assert isinstance(reward, float)
    assert obs["cube_pos"].shape == (3,)
    assert obs["cube_quat"].shape == (4,)
    assert obs["ee_pos"].shape == (3,)
    assert obs["gripper"].shape == (1,)


# ----- shape mismatches ------------------------------------------------------


@given(
    actual_dim=st.integers(min_value=1, max_value=20).filter(lambda d: d != _ACTION_DIM),
)
@settings(
    max_examples=10,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_wrong_shape_action_raises_typed_error(env: TabletopEnv, actual_dim: int) -> None:
    """Actions of the wrong shape must raise a typed exception, not
    crash MuJoCo with a segfault. The Mujoco bindings raise
    :class:`ValueError` (or numpy raises on the broadcast); both are
    acceptable. An uncaught :class:`SystemError` / segfault is not."""
    env.reset(seed=0)
    bad_action = np.zeros(actual_dim, dtype=np.float64)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        env.step(bad_action)


# ----- action_space metadata invariants --------------------------------------


def test_action_space_low_lt_high() -> None:
    """The Box action_space must have low strictly less than high
    along every coordinate (the contract :class:`RandomPolicy` keys off)."""
    e = TabletopEnv(max_steps=1)
    try:
        low = np.asarray(e.action_space.low, dtype=np.float64)
        high = np.asarray(e.action_space.high, dtype=np.float64)
        assert low.shape == high.shape == (_ACTION_DIM,)
        assert np.all(low < high)
    finally:
        e.close()
