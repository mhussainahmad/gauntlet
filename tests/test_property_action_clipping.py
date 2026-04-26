"""Property tests for the env action-clipping invariant.

Phase 2.5 Task 13 — covers two gaps left by
``tests/test_fuzz_action_space.py``:

1. **Post-clip range invariant.** The existing fuzz test asserts that
   the env's *observation* dict stays finite under arbitrary in-shape
   actions. It does NOT assert that the *delivered* per-step action
   (the actual control signal that ``np.clip(a, -1, 1)`` produces) is
   itself within ``[-1, 1]``. We pin that here by snapshotting the
   mocap pose before / after the step and bounding the per-axis delta
   by the env's ``MAX_LINEAR_STEP`` / ``MAX_ANGULAR_STEP`` constants —
   a clipped action of magnitude > 1 would visibly exceed those bounds
   in the snapshot delta, but the env's silent-clip contract keeps the
   delta inside the envelope.

2. **NaN / Inf input behaviour as a spec.** ``np.clip(NaN, -1, 1)``
   returns NaN; the env propagates that NaN into mocap_pos and the
   downstream observation. The existing fuzz test explicitly excludes
   NaN/Inf inputs *and* documents the env's silent propagation. We
   re-pin the gap here as ``pytest.xfail(strict=False)`` properties
   — they serve as a spec for the future hardening that should sanitise
   NaN/Inf inputs at the env boundary (tracked separately; T13 is the
   *measurement* task, not the *fix*). ``strict=False`` lets a future
   sanitiser flip these into XPASS without failing the suite.

The hypothesis budget for the env-touching properties is capped via
per-test ``@settings(max_examples=15)`` because each example pays the
~50 ms MuJoCo step cost; the conftest profile's ``max_examples=200``
default would make the run ~10s otherwise. The conftest profile still
governs the no-env property tests in this file (e.g. action-shape).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from gauntlet.env.tabletop import TabletopEnv

pytestmark = pytest.mark.hypothesis_property

# Mirrors ``TabletopEnv.action_space`` (Box low=-1, high=1, shape=(7,)).
_ACTION_DIM = 7


@pytest.fixture(scope="module")
def env() -> Iterator[TabletopEnv]:
    """One env per module — the MuJoCo model load is the bottleneck;
    ``restore_baseline()`` between examples is cheap (sub-millisecond)."""
    e = TabletopEnv(max_steps=4)
    try:
        yield e
    finally:
        e.close()


# ----- post-clip range invariant (finite inputs only) -----------------------


@given(
    action=st.lists(
        # Wider than 1e6 risks float overflow inside the affine MAX_*_STEP
        # multiply; 1e9 is six orders of magnitude past anything a real
        # policy would emit and fully exercises the silent-clip path.
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=_ACTION_DIM,
        max_size=_ACTION_DIM,
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_step_with_extreme_finite_action_keeps_mocap_delta_bounded(
    env: TabletopEnv, action: list[float], seed: int
) -> None:
    """For any finite-but-arbitrary action vector, the delivered
    per-step mocap-pos delta is bounded by the env's
    ``MAX_LINEAR_STEP`` (with a small slack for float round-off).

    A regression that disabled the clip — say, a refactor that dropped
    the ``np.clip(a, -1.0, 1.0)`` line on the assumption that the
    policy is well-behaved — would let an action of magnitude 1e9
    teleport the mocap by 5e7 m, blowing past the per-step envelope.
    Pinned here so any such regression surfaces immediately.
    """
    obs_pre, _ = env.reset(seed=seed)
    # ``obs["ee_pos"]`` is the public mirror of the env's mocap pos
    # (see ``TabletopEnv._build_obs``), so we can compute the per-step
    # delta without poking private MuJoCo state.
    pre_pos = np.asarray(obs_pre["ee_pos"], dtype=np.float64).copy()

    a = np.asarray(action, dtype=np.float64)
    obs, _reward, _terminated, _truncated, _info = env.step(a)

    post_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
    delta = np.abs(post_pos - pre_pos)
    # MAX_LINEAR_STEP is the per-axis cap. Add a 1e-9 slack for the
    # float-cast comparison. A regression that removed the clip would
    # produce a delta ~5e7 (action 1e9 * 0.05) and miss the envelope
    # by twelve orders of magnitude — well outside any plausible slack.
    bound = TabletopEnv.MAX_LINEAR_STEP + 1e-9
    assert np.all(delta <= bound), (
        f"per-axis mocap delta {delta} exceeds MAX_LINEAR_STEP={bound}; action={action[:3]}"
    )

    # Belt-and-braces: the existing observation-finiteness contract
    # from ``test_fuzz_action_space.py`` should hold here too. Pinned
    # so a refactor that re-introduces NaN propagation on finite
    # inputs is caught by *this* file rather than only the fuzz one.
    for key in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
        arr = np.asarray(obs[key], dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"finite-input action produced non-finite obs[{key}]"


# ----- shape-mismatch property ----------------------------------------------


@given(
    actual_dim=st.integers(min_value=1, max_value=20).filter(lambda d: d != _ACTION_DIM),
    fill_value=st.floats(
        min_value=-1e9,
        max_value=1e9,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_wrong_shape_action_with_arbitrary_fill_raises_value_error(
    actual_dim: int, fill_value: float
) -> None:
    """Wrong-shape action raises a typed exception regardless of the
    fill value. Extends ``test_fuzz_action_space::test_wrong_shape_action_raises_typed_error``
    by also fuzzing the fill (which previously only used zeros). Uses a
    fresh env per example because the wrong-shape branch raises BEFORE
    any mocap update, so the env state is unchanged but the per-example
    cost is still dominated by env construction; this test caps at
    hypothesis-default settings rather than the slow env-step path."""
    e = TabletopEnv(max_steps=1)
    try:
        e.reset(seed=0)
        bad_action = np.full(actual_dim, fill_value, dtype=np.float64)
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            e.step(bad_action)
    finally:
        e.close()


# ----- NaN / Inf input behaviour as a spec (xfail) --------------------------


def _make_action_with_non_finite(fill_strategy: str, dim: int = _ACTION_DIM) -> NDArray[np.float64]:
    """Build a 7-vector containing exactly one non-finite entry.

    The single-entry shape is the most aggressive test (a fully NaN
    vector would also produce NaN obs trivially); we target a single
    axis so a future "sanitise per-axis" fix is correctly exercised.
    """
    a = np.zeros(dim, dtype=np.float64)
    a[0] = {"nan": np.nan, "inf": np.inf, "neg_inf": -np.inf}[fill_strategy]
    return a


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Spec for future hardening: ``np.clip(NaN, -1, 1)`` returns NaN, "
        "and the env propagates the NaN into mocap_pos and the downstream "
        "observation. T13 is the measurement task; the fix (sanitise "
        "non-finite inputs at the env boundary, raise ValueError) lands "
        "in a separate follow-up PR. ``strict=False`` so a future "
        "sanitiser produces XPASS without breaking the suite."
    ),
)
@pytest.mark.parametrize("fill_strategy", ["nan", "inf", "neg_inf"])
def test_step_rejects_non_finite_action_input(fill_strategy: str) -> None:
    """Spec: an action containing NaN / +Inf / -Inf should raise
    :class:`ValueError` at the env boundary (mirroring the wrong-shape
    branch's typed-rejection contract). Currently the env silently
    propagates non-finite inputs through ``np.clip`` into the mocap
    pose, and the resulting observation contains NaN — the test below
    ``test_step_with_non_finite_action_currently_propagates_to_obs``
    pins that current behaviour so a future sanitiser fix surfaces as
    a single-test diff rather than two correlated regressions."""
    e = TabletopEnv(max_steps=1)
    try:
        e.reset(seed=0)
        bad_action = _make_action_with_non_finite(fill_strategy)
        with pytest.raises(ValueError, match=r"non-finite|finite|NaN|Inf"):
            e.step(bad_action)
    finally:
        e.close()


@pytest.mark.parametrize("fill_strategy", ["nan", "inf", "neg_inf"])
def test_step_with_non_finite_action_currently_propagates_to_obs(fill_strategy: str) -> None:
    """Pins the *current* behaviour: a non-finite action propagates
    through ``np.clip`` (which preserves NaN and saturates +/- Inf) and
    appears as non-finite mocap_pos / ee_pos in the next observation.
    This is the gap the xfail spec above documents; pinned here so a
    future sanitiser change knows to flip both tests in lockstep —
    the xfail becomes XPASS, this test starts failing, and the diff
    ratchets the contract honestly."""
    e = TabletopEnv(max_steps=1)
    try:
        e.reset(seed=0)
        bad_action = _make_action_with_non_finite(fill_strategy)
        obs, _reward, _terminated, _truncated, _info = e.step(bad_action)
        ee_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        # NaN saturates `np.clip` to NaN; +/-Inf saturates to the bound,
        # so for the Inf cases the propagated value is a finite +/- 1.0
        # which arrives at the mocap as a delta of MAX_LINEAR_STEP. The
        # NaN case is the one that produces a non-finite obs; the Inf
        # cases produce a finite-but-saturated obs. Document both.
        if fill_strategy == "nan":
            assert not np.all(np.isfinite(ee_pos)), (
                "regression: NaN action no longer propagates to obs — "
                "if intended, flip the xfail above to XPASS in the same diff."
            )
        else:
            # +/-Inf clips to +/-1; the mocap delta is bounded by
            # MAX_LINEAR_STEP. The obs is finite — pin it.
            assert np.all(np.isfinite(ee_pos))
    finally:
        e.close()
