"""Property tests for the ``reset -> step -> reset`` ordering invariant.

Phase 2.5 Task 13 — covers a determinism contract not directly
exercised by ``tests/test_determinism_mujoco.py``: that ``reset(seed)``
fully wipes any per-step state from a prior rollout, regardless of
what actions were applied between the two resets.

The contract under fuzz: starting ``obs`` from
``env.reset(seed=S)`` MUST equal the starting ``obs`` from a second
``env.reset(seed=S)`` even after an arbitrary sequence of
``env.step(a_k)`` calls in between, when no perturbation is queued. A
regression that leaks per-step mocap state past the reset boundary
would produce divergent ``obs`` and fail the property.

The existing ``test_determinism_mujoco.test_restore_baseline_then_reset_matches_clean_reset``
covers the perturbation-then-reset case (apply axis, reset, second
reset is clean); the property here is orthogonal and covers the
*step-history*-then-reset case (no perturbation, but several arbitrary
``step`` calls in between).

Hypothesis budget: ``max_examples=20`` per test (the plan asks for
"100 random seeds"; 200 default examples would saturate the env-step
cost). The conftest profile's ``max_examples=200`` only governs the
no-env tests — these env-touching tests cap themselves explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.env.tabletop import TabletopEnv

pytestmark = pytest.mark.hypothesis_property

# Mirrors the env's action_space shape.
_ACTION_DIM = 7

# Obs keys whose finite mocap-derived contents are expected to be
# bit-equal across two resets at the same seed. ``image`` is excluded
# because the env defaults to ``render_in_obs=False`` so the key is
# absent; the test below targets the state obs surface.
_STATE_OBS_KEYS = ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos")


@pytest.fixture(scope="module")
def env() -> Iterator[TabletopEnv]:
    """Module-scoped env — one MuJoCo model load amortised over all
    examples in this file. ``reset()`` is sub-millisecond after the
    first; ``step()`` is the hypothesis budget driver."""
    e = TabletopEnv(max_steps=64)
    try:
        yield e
    finally:
        e.close()


def _obs_snapshot(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Defensive deep copy of obs arrays — the env may recycle the
    underlying buffers on the next ``step`` / ``reset`` call."""
    return {k: np.asarray(obs[k], dtype=np.float64).copy() for k in _STATE_OBS_KEYS if k in obs}


def _assert_state_obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Bit-equal comparison on the state obs surface."""
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), (
            f"obs key {k!r} differs between resets: a={a[k]!r} b={b[k]!r}"
        )


# ----- core ordering invariant ----------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_steps=st.integers(min_value=0, max_value=8),
    action_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_reset_step_reset_yields_identical_starting_obs(
    env: TabletopEnv, seed: int, n_steps: int, action_seed: int
) -> None:
    """``reset(seed=S) -> obs_a`` and ``reset(seed=S) -> step* -> reset(seed=S) -> obs_b``
    produce bit-identical starting obs.

    Regression target: a leak of per-step mocap state past the reset
    boundary (e.g. a refactor that forgets to call
    :meth:`TabletopEnv.restore_baseline` from inside ``reset``) would
    produce divergent ``obs_b``. The invariant is asserted on the
    state-obs subset (``cube_pos``, ``cube_quat``, ``ee_pos``,
    ``gripper``, ``target_pos``) because the default ``render_in_obs=False``
    omits ``image``; the state surface is the bit-exact contract
    every later test in the codebase keys off (see
    ``tests/test_determinism_mujoco.py``).
    """
    # Round 1: clean reset.
    obs_a, _ = env.reset(seed=seed)
    snap_a = _obs_snapshot(obs_a)

    # Round 2: clean reset, then a sequence of arbitrary in-bound
    # actions, then a second reset at the same seed.
    env.reset(seed=seed)
    rng = np.random.default_rng(action_seed)
    for _ in range(n_steps):
        action = rng.uniform(-1.0, 1.0, size=_ACTION_DIM).astype(np.float64)
        env.step(action)
    obs_b, _ = env.reset(seed=seed)
    snap_b = _obs_snapshot(obs_b)

    _assert_state_obs_equal(snap_a, snap_b)


# ----- variant: actions outside the [-1, 1] envelope still survive reset ---


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_steps=st.integers(min_value=1, max_value=4),
)
@settings(
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_reset_after_extreme_action_step_yields_clean_obs(
    env: TabletopEnv, seed: int, n_steps: int
) -> None:
    """Variant of the property above where the intervening ``step``
    calls feed in finite-but-extreme actions (saturated to ``+1e6``).
    The env's silent-clip contract ensures the actions are bounded
    inside ``[-1, 1]`` before integration, but the resulting mocap
    state may have moved far from its initial pose; the property
    is that ``reset(seed)`` still wipes that state cleanly."""
    obs_a, _ = env.reset(seed=seed)
    snap_a = _obs_snapshot(obs_a)

    env.reset(seed=seed)
    extreme_action = np.full(_ACTION_DIM, 1e6, dtype=np.float64)
    for _ in range(n_steps):
        env.step(extreme_action)
    obs_b, _ = env.reset(seed=seed)
    snap_b = _obs_snapshot(obs_b)

    _assert_state_obs_equal(snap_a, snap_b)


# ----- variant: reset under a different seed branches cleanly --------------


@given(
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
    n_steps=st.integers(min_value=0, max_value=4),
    action_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_reset_with_different_seed_after_steps_matches_clean_reset(
    env: TabletopEnv, seed_a: int, seed_b: int, n_steps: int, action_seed: int
) -> None:
    """``reset(A) -> step* -> reset(B)`` matches the obs of a fresh
    env's ``reset(B)``. Distinct from the previous tests in that the
    second reset uses a *different* seed; the invariant is that
    ``reset(seed=B)`` reaches a state purely determined by ``B`` and
    the underlying MJCF, not by any prior in-process history.

    Skips the trivial ``seed_a == seed_b`` case (covered by the
    primary ordering test above)."""
    if seed_a == seed_b:
        return

    # Reference: a fresh env reset at seed B. The shared module-scoped
    # ``env`` fixture is byte-equivalent to a fresh env on the public
    # state surface (see the across-independent-envs determinism test
    # in ``tests/test_determinism_mujoco.py``) so we can reuse it for
    # the reference too.
    obs_ref, _ = env.reset(seed=seed_b)
    snap_ref = _obs_snapshot(obs_ref)

    # Test: reset at A, step around, reset at B.
    env.reset(seed=seed_a)
    rng = np.random.default_rng(action_seed)
    for _ in range(n_steps):
        action = rng.uniform(-1.0, 1.0, size=_ACTION_DIM).astype(np.float64)
        env.step(action)
    obs_under_test, _ = env.reset(seed=seed_b)
    snap_under_test = _obs_snapshot(obs_under_test)

    _assert_state_obs_equal(snap_ref, snap_under_test)
