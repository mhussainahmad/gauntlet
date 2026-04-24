"""PyBullet backend perturbation + restore-baseline determinism audit.

Phase 2.5 Task 17 cases 4, 6, and step 7 (extended trajectory).
The base reset / step / step-sequence determinism cases (case 1)
already live in :mod:`tests.pybullet.test_determinism_pybullet`
landed in Phase 2 Task 9. This file extends that coverage to:

* **Case 4 — perturbation determinism.** Two independent envs queued
  with the same axis perturbation at the same seed produce
  byte-identical state-obs after reset, for every axis the PyBullet
  adapter declares (all 7 — ``VISUAL_ONLY_AXES`` is empty post-RFC-006
  because rendering covers cosmetic axes via ``render_in_obs=True``;
  state-only obs do not surface the cosmetic ones, so the test still
  passes trivially on those axes — that is documented behaviour
  matching MuJoCo, RFC-005 §6.2).
* **Case 6 — restore-baseline idempotency.** ``set_perturbation`` then
  ``reset`` (which drains the queue) then a SECOND ``reset`` at the
  same seed must match a clean env's first ``reset`` at the same
  seed byte-for-byte — i.e. perturbations from one episode must not
  leak into the next.
* **Step 7 — extended rollout determinism.** Fixed 20-step
  ``(obs, reward, terminated, truncated)`` rollout is byte-identical
  across two independent envs at the same seed. Composes the base
  ``test_step_sequence_bit_identical_for_same_seed`` (which compared
  obs-only) with the reward + flag tuple.

All tests are marked ``@pytest.mark.pybullet`` and only run under
``uv sync --extra pybullet --group pybullet-dev``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.pybullet


# Canonical (axis, value) pairs covering every axis the PyBullet
# adapter declares. Same set as the MuJoCo audit — values are inside
# each axis's validated range and chosen far from baseline so a
# silent no-op apply path would not pass the test.
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
    """Defensive deep copy for cross-instance / cross-episode comparison."""
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}


def _assert_obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert set(a) == set(b), f"obs key sets differ: {set(a)} vs {set(b)}"
    for k in a:
        assert np.array_equal(a[k], b[k]), f"key {k!r}: {a[k]!r} != {b[k]!r}"


# ---------------------------------------------------------------- case 4


def test_perturbation_byte_identical_across_independent_envs() -> None:
    """For every axis, two independent PyBullet envs queued with the
    same ``(name, value)`` perturbation and reset at the same seed
    produce byte-identical state-obs.

    Cosmetic axes (``lighting_intensity``, ``camera_offset_*``,
    ``object_texture``) at ``render_in_obs=False`` are observably
    no-ops on the state obs by design (RFC-005 §6.2); the equality
    still holds because both envs follow the same path. The test's
    job is to lock the contract, not to prove sensitivity.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    for axis, value in _AXIS_CASES:
        a = PyBulletTabletopEnv()
        b = PyBulletTabletopEnv()
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
    Episode N+1: nothing queued, reset at the same seed -> obs MUST
    match a freshly-constructed env's first reset at that seed.

    This exercises the full restore-baseline path the PyBullet
    adapter implements: ``reset -> restore_baseline -> _build_scene
    (full rebuild)``. State-only obs comparison is the right
    granularity here — pixel rendering deltas live in the rendering
    test file.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    for axis, value in _AXIS_CASES:
        # Reference: clean env, no perturbation history.
        ref = PyBulletTabletopEnv()
        try:
            ref_obs, _ = ref.reset(seed=99)
            ref_snap = _obs_snapshot(ref_obs)
        finally:
            ref.close()

        env = PyBulletTabletopEnv()
        try:
            env.set_perturbation(axis, value)
            env.reset(seed=99)
            # Pending queue is now empty; reset again should produce
            # the unperturbed seed-99 obs byte-for-byte.
            obs2, _ = env.reset(seed=99)
            _assert_obs_equal(ref_snap, _obs_snapshot(obs2))
        finally:
            env.close()


# ---------------------------------------------------------------- step 7


def test_extended_rollout_byte_identical_with_reward_flags() -> None:
    """Fixed 20-step rollout under the same seed is byte-identical on
    two independent envs across the FULL ``(obs, reward, terminated,
    truncated)`` tuple. The existing ``test_step_sequence_bit_identical_for_same_seed``
    in this directory compared obs only; this test extends to the
    reward + flag side, which is what the Runner records on
    ``Episode.total_reward`` / ``terminated`` / ``truncated``.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    def _rollout() -> list[tuple[dict[str, np.ndarray], float, bool, bool]]:
        env = PyBulletTabletopEnv()
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
