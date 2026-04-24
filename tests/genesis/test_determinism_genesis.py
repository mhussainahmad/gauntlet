"""Genesis backend determinism audit — Phase 2.5 Task 17 cases 1, 4, 6.

Pins the determinism contract RFC-007 §7.1 promises (same-process
bit-determinism on state-only obs):

* **Case 1 — within-backend determinism.** Two ``reset(seed=S)`` calls
  on two independent ``GenesisTabletopEnv`` instances produce
  byte-identical state-only obs. Plus a 20-step rollout
  ``(obs, reward, terminated, truncated)`` tuple is byte-identical
  step-by-step.
* **Case 4 — perturbation determinism.** For every supported axis
  (all 7 — ``VISUAL_ONLY_AXES`` is empty post-RFC-008), two
  independent envs queued with the same ``(name, value)`` produce
  byte-identical post-reset state-only obs.
* **Case 6 — restore-baseline idempotency.** ``set_perturbation``
  then ``reset`` (apply + drain) then a SECOND ``reset`` at the
  same seed must match a clean env's first ``reset`` at that seed
  byte-for-byte — perturbations from one episode must not leak.

The existing ``tests/genesis/test_render_genesis.py`` covers the
*rendered* determinism for the ``render_in_obs=True`` path; this
file is the state-only counterpart and runs the cheaper code path
(no Rasterizer compile).

All tests are marked ``@pytest.mark.genesis`` and only run under
``uv sync --extra genesis --group genesis-dev``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.genesis


# Same canonical (axis, value) set as the MuJoCo and PyBullet audits.
# Genesis declares every axis and ``VISUAL_ONLY_AXES = frozenset()``,
# so no axis is skipped here. The cosmetic axes only mutate shadow
# attrs at ``render_in_obs=False`` (RFC-007 §6 honesty caveat) but
# the equality check still locks the contract.
_AXIS_CASES: tuple[tuple[str, float], ...] = (
    ("lighting_intensity", 0.3),
    ("camera_offset_x", 0.05),
    ("camera_offset_y", -0.05),
    ("object_texture", 1.0),
    ("object_initial_pose_x", 0.07),
    ("object_initial_pose_y", -0.06),
    ("distractor_count", 3.0),
)


def _obs_snapshot(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Defensive deep copy for cross-instance comparison."""
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}


def _assert_obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert set(a) == set(b), f"obs key sets differ: {set(a)} vs {set(b)}"
    for k in a:
        assert np.array_equal(a[k], b[k]), f"key {k!r}: {a[k]!r} != {b[k]!r}"


# ---------------------------------------------------------------- case 1


def test_reset_seed_byte_identical_state_obs_across_independent_envs() -> None:
    """RFC-007 §7.1 — same seed, two independent envs, identical
    state-only obs dict byte-for-byte.

    State-only path (``render_in_obs=False``) so this test does NOT
    pay the Rasterizer cost — the matching rendered case lives in
    ``test_render_genesis.py``.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    a = GenesisTabletopEnv()
    b = GenesisTabletopEnv()
    try:
        obs_a, _ = a.reset(seed=42)
        obs_b, _ = b.reset(seed=42)
        _assert_obs_equal(_obs_snapshot(obs_a), _obs_snapshot(obs_b))
    finally:
        a.close()
        b.close()


def test_extended_state_only_rollout_byte_identical() -> None:
    """Step 7 — fixed 20-step state-only rollout produces byte-identical
    ``(obs, reward, terminated, truncated)`` tuples across two
    independent envs at the same seed.

    The rendered counterpart is :func:`test_post_step_determinism_across_instances`
    in ``test_render_genesis.py``; this file runs at ``render_in_obs=False``
    and so exercises the physics-side determinism cleanly without the
    rendering subsystem.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    def _rollout() -> list[tuple[dict[str, np.ndarray], float, bool, bool]]:
        env = GenesisTabletopEnv(max_steps=25)
        try:
            env.reset(seed=2024)
            rng = np.random.default_rng(999)
            steps: list[tuple[dict[str, np.ndarray], float, bool, bool]] = []
            for _ in range(20):
                action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float64)
                obs, reward, terminated, truncated, _ = env.step(action)
                steps.append(
                    (
                        _obs_snapshot(obs),
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                    )
                )
            return steps
        finally:
            env.close()

    a = _rollout()
    b = _rollout()
    assert len(a) == len(b) == 20
    for i, ((oa, ra, ta, tra), (ob, rb, tb, trb)) in enumerate(zip(a, b, strict=True)):
        _assert_obs_equal(oa, ob)
        assert ra == rb, f"step {i}: reward {ra} != {rb}"
        assert ta == tb, f"step {i}: terminated {ta} != {tb}"
        assert tra == trb, f"step {i}: truncated {tra} != {trb}"
        for k, v in oa.items():
            assert np.all(np.isfinite(v)), f"step {i} key {k!r} not finite"


# ---------------------------------------------------------------- case 4


def test_perturbation_byte_identical_across_independent_envs() -> None:
    """For every axis, two independent envs queued with the same
    ``(name, value)`` perturbation and reset at the same seed
    produce byte-identical state-only obs.

    Cosmetic axes at ``render_in_obs=False`` only touch shadow attrs
    (RFC-007 §6); the test still locks the contract that two envs
    under the same axis configuration agree.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    for axis, value in _AXIS_CASES:
        a = GenesisTabletopEnv()
        b = GenesisTabletopEnv()
        try:
            a.set_perturbation(axis, value)
            b.set_perturbation(axis, value)
            obs_a, _ = a.reset(seed=7)
            obs_b, _ = b.reset(seed=7)
            _assert_obs_equal(_obs_snapshot(obs_a), _obs_snapshot(obs_b))
        finally:
            a.close()
            b.close()


# ---------------------------------------------------------------- case 6


def test_restore_baseline_then_reset_matches_clean_reset() -> None:
    """Episode N: queue an axis, reset (apply + drain queue).
    Episode N+1: nothing queued, reset at the same seed must
    produce the unperturbed seed-driven obs byte-for-byte.

    The Genesis adapter's ``restore_baseline`` re-parks distractors
    + cube alias and resets cosmetic shadows; the second ``reset``
    re-runs the seed RNG on top. Equality with a clean env's first
    ``reset(seed=S)`` is the contract.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    for axis, value in _AXIS_CASES:
        ref = GenesisTabletopEnv()
        try:
            ref_obs, _ = ref.reset(seed=99)
            ref_snap = _obs_snapshot(ref_obs)
        finally:
            ref.close()

        env = GenesisTabletopEnv()
        try:
            env.set_perturbation(axis, value)
            env.reset(seed=99)
            obs2, _ = env.reset(seed=99)
            _assert_obs_equal(ref_snap, _obs_snapshot(obs2))
        finally:
            env.close()
