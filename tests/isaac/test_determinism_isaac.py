"""Isaac Sim adapter determinism audit — Phase 2.5 Task 17 cases 1, 4, 6.

Mock-driven (the autouse ``conftest.py`` in this directory injects a
fake ``isaacsim`` / ``omni.isaac.core`` namespace into ``sys.modules``
before each test). Runs in the default torch-free CI job; the
``isaac`` pytest marker is reserved for future real-runtime tests
gated to a GPU runner.

The ``IsaacSimTabletopEnv`` is a state-only first cut (RFC-009 §6.6)
so ``VISUAL_ONLY_AXES`` lists four cosmetic axes that store on
shadow attributes and have no observable state delta. Per Phase 2.5
Task 17's "subject to that backend's ``VISUAL_ONLY_AXES`` skip", we
exercise:

* **Case 1 — within-backend determinism.** Two ``reset(seed=S)`` on
  two independent envs produce byte-identical state obs. Plus a
  fixed 20-step rollout produces byte-identical
  ``(obs, reward, terminated, truncated)`` tuples. Under the mock
  ``_FakeWorld.step`` is a no-op, so the rollout exercises the
  Python-side state machine (gripper, grasp flag, EE teleport)
  without any physics — exactly what RFC-009 §8 designed the mock
  to test.
* **Case 4 — perturbation determinism.** Iterates the three
  state-affecting axes (``object_initial_pose_x``,
  ``object_initial_pose_y``, ``distractor_count``); the four
  cosmetic axes in ``VISUAL_ONLY_AXES`` are skipped per task spec.
* **Case 6 — restore-baseline idempotency.** Iterates ALL 7 axes —
  including the cosmetic ones — because the queue-drain + shadow-
  attribute reset behaviour applies to every axis. Same pattern
  PyBullet uses.
* **State-shape parity vs MuJoCo.** Lives here (not in
  ``tests/test_determinism_cross_backend.py``) because the fake
  isaacsim namespace is autouse-installed per directory; the
  cross-backend file in ``tests/`` cannot import
  ``IsaacSimTabletopEnv`` without that fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict

# Three state-affecting axes (RFC-009 §6.4-§6.5).
_STATE_AFFECTING_AXIS_CASES: tuple[tuple[str, float], ...] = (
    ("object_initial_pose_x", 0.07),
    ("object_initial_pose_y", -0.06),
    ("distractor_count", 3.0),
)

# All 7 axes (the four cosmetic + the three state-affecting). Restore-
# baseline + queue-drain semantics apply to every axis.
_ALL_AXIS_CASES: tuple[tuple[str, float], ...] = (
    ("lighting_intensity", 0.3),
    ("camera_offset_x", 0.05),
    ("camera_offset_y", -0.05),
    ("object_texture", 1.0),
    *_STATE_AFFECTING_AXIS_CASES,
)


def _obs_snapshot(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Defensive deep copy for cross-instance / cross-episode comparison."""
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}


def _assert_obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert set(a) == set(b), f"obs key sets differ: {set(a)} vs {set(b)}"
    for k in a:
        assert np.array_equal(a[k], b[k]), f"key {k!r}: {a[k]!r} != {b[k]!r}"


# ---------------------------------------------------------------- case 1


def test_reset_seed_byte_identical_state_obs_across_independent_envs() -> None:
    """Two independent ``IsaacSimTabletopEnv`` instances reset with the
    same seed produce byte-identical state-obs. The mock world's
    ``reset`` is a no-op so this exercises the seed-driven cube/target
    XY randomisation + EE teleport path cleanly.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    a = IsaacSimTabletopEnv()
    b = IsaacSimTabletopEnv()
    try:
        obs_a, _ = a.reset(seed=42)
        obs_b, _ = b.reset(seed=42)
        _assert_obs_equal(_obs_snapshot(obs_a), _obs_snapshot(obs_b))
    finally:
        a.close()
        b.close()


def test_extended_state_only_rollout_byte_identical() -> None:
    """Step 7 — fixed 20-step rollout produces byte-identical
    ``(obs, reward, terminated, truncated)`` tuples on two
    independent envs at the same seed.

    Under the mock ``_FakeWorld.step`` is a no-op, so this exercises
    the Python-side state machine: ``_apply_ee_command`` integrates
    the action into the EE pose, ``_update_grasp_state`` flips the
    grasp flag, ``_snap_cube_to_ee`` overrides cube pose when
    grasped. The contract is: same input + same seed -> identical
    trajectory across instances.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    def _rollout() -> list[tuple[dict[str, np.ndarray], float, bool, bool]]:
        env = IsaacSimTabletopEnv(max_steps=25)
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


def test_state_affecting_perturbation_byte_identical_across_independent_envs() -> None:
    """For each STATE-AFFECTING axis (cosmetic axes in
    ``VISUAL_ONLY_AXES`` are skipped per Task 17 spec), two
    independent envs queued with the same ``(name, value)`` and
    reset at the same seed produce byte-identical state obs.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    for axis, value in _STATE_AFFECTING_AXIS_CASES:
        a = IsaacSimTabletopEnv()
        b = IsaacSimTabletopEnv()
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


def test_restore_baseline_then_reset_matches_clean_reset_for_every_axis() -> None:
    """Episode N: queue an axis, reset (apply + drain queue).
    Episode N+1: nothing queued, reset at the same seed must
    match a clean env's first ``reset(seed=S)`` byte-for-byte.

    Iterates all 7 axes — cosmetic ones included, because the
    queue-drain + shadow-attribute restore happens regardless of
    whether the axis surfaces in state obs (RFC-009 §6.6 + the
    ``restore_baseline`` body that resets ``_light_intensity``,
    ``_cam_offset``, ``_texture_choice``).
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    for axis, value in _ALL_AXIS_CASES:
        ref = IsaacSimTabletopEnv()
        try:
            ref_obs, _ = ref.reset(seed=99)
            ref_snap = _obs_snapshot(ref_obs)
        finally:
            ref.close()

        env = IsaacSimTabletopEnv()
        try:
            env.set_perturbation(axis, value)
            env.reset(seed=99)
            obs2, _ = env.reset(seed=99)
            _assert_obs_equal(ref_snap, _obs_snapshot(obs2))
        finally:
            env.close()


def test_restore_baseline_resets_cosmetic_shadow_attrs() -> None:
    """Direct check: after ``set_perturbation`` queues a cosmetic
    value, applies it via ``reset``, then ``restore_baseline``,
    the shadow attrs must be back at their neutral defaults
    (``_light_intensity=1.0``, ``_cam_offset=zeros(2)``,
    ``_texture_choice=0``).

    Locks the shadow-attribute reset path that the cross-episode
    test above only exercises indirectly via the obs comparison
    (cosmetic axes don't surface in state obs, so the obs match
    alone could miss a leak).
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("lighting_intensity", 0.3)
        env.set_perturbation("camera_offset_x", 0.05)
        env.set_perturbation("camera_offset_y", -0.05)
        env.set_perturbation("object_texture", 1.0)
        env.reset(seed=11)
        # Pending queue is now empty; shadow attrs reflect the
        # perturbation values.
        assert env._light_intensity == pytest.approx(0.3)
        assert env._cam_offset[0] == pytest.approx(0.05)
        assert env._cam_offset[1] == pytest.approx(-0.05)
        assert env._texture_choice == 1

        env.restore_baseline()
        # All four shadows back at neutral.
        assert env._light_intensity == 1.0
        assert env._cam_offset[0] == 0.0
        assert env._cam_offset[1] == 0.0
        assert env._texture_choice == 0
    finally:
        env.close()


# ----------------------------------------------- cross-backend state-shape parity


def test_isaac_state_obs_space_matches_mujoco() -> None:
    """Isaac's 5-key state observation_space matches MuJoCo's
    byte-for-byte (Phase 2.5 Task 17 case 2).

    Lives in this file rather than ``tests/test_determinism_cross_backend.py``
    because the autouse mock fixture in ``tests/isaac/conftest.py``
    only applies to this directory — the top-level cross-backend
    file cannot import ``IsaacSimTabletopEnv`` cleanly.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv
    from gauntlet.env.tabletop import TabletopEnv

    isaac = IsaacSimTabletopEnv()
    mj = TabletopEnv()
    try:
        os_isaac = isaac.observation_space
        os_mj = mj.observation_space
        assert isinstance(os_isaac, Dict)
        assert isinstance(os_mj, Dict)
        expected_keys = {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
        assert set(os_isaac.spaces.keys()) == expected_keys
        assert set(os_mj.spaces.keys()) == expected_keys
        for key in expected_keys:
            sub_i = os_isaac.spaces[key]
            sub_m = os_mj.spaces[key]
            assert isinstance(sub_i, Box)
            assert isinstance(sub_m, Box)
            assert sub_i == sub_m, f"isaac {key!r}: {sub_i!r} != mujoco {sub_m!r}"
    finally:
        isaac.close()
        mj.close()


def test_isaac_action_space_matches_mujoco() -> None:
    """Isaac's 7-D action_space Box matches MuJoCo's byte-for-byte.

    Same rationale as the obs-space test above for living here.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv
    from gauntlet.env.tabletop import TabletopEnv

    isaac = IsaacSimTabletopEnv()
    mj = TabletopEnv()
    try:
        a_i = isaac.action_space
        a_m = mj.action_space
        assert isinstance(a_i, Box)
        assert isinstance(a_m, Box)
        assert a_i == a_m, f"isaac action: {a_i!r} != mujoco {a_m!r}"
    finally:
        isaac.close()
        mj.close()
