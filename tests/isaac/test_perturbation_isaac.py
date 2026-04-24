"""Per-axis perturbation contract tests — RFC-009 §6 / §8.

Covers the seven canonical axes against the mocked `omni.isaac.core`
prim surface. State-affecting axes assert on the right
``set_world_pose`` calls landing on the right prim with the right
value; cosmetic axes assert that the shadow attribute updates and
no prim was teleported (until the rendering follow-up RFC lands).

All tests run under the conftest's autouse fake-`isaacsim` namespace.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------- set_perturbation validation


def test_set_perturbation_rejects_unknown_axis() -> None:
    """Unknown axis -> ValueError listing the valid axis names."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        with pytest.raises(ValueError, match="unknown perturbation axis 'bogus'"):
            env.set_perturbation("bogus", 1.0)
    finally:
        env.close()


@pytest.mark.parametrize("bad_count", [-1, 11, 100])
def test_set_perturbation_rejects_out_of_range_distractor_count(
    bad_count: int,
) -> None:
    """`distractor_count` enforces integer `[0, 10]` (parity with PyBullet/Genesis)."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        with pytest.raises(ValueError, match=r"distractor_count must be in \[0, 10\]"):
            env.set_perturbation("distractor_count", float(bad_count))
    finally:
        env.close()


@pytest.mark.parametrize(
    "axis",
    [
        "lighting_intensity",
        "camera_offset_x",
        "camera_offset_y",
        "object_texture",
        "object_initial_pose_x",
        "object_initial_pose_y",
        "distractor_count",
    ],
)
def test_set_perturbation_accepts_every_known_axis(axis: str) -> None:
    """All seven canonical axes accepted (RFC-009 §6.7).

    Reset drains the queue; no assertion on per-axis behaviour here
    (separate tests cover that). Just verifies no raise from the
    validator.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        value = 3.0 if axis == "distractor_count" else 0.05
        env.set_perturbation(axis, value)
        env.reset(seed=42)
    finally:
        env.close()


# --------------------------------------------- state-affecting axes


def test_object_initial_pose_x_overrides_random_cube_x() -> None:
    """RFC-009 §6.4 — axis value overrides seed-driven random X."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("object_initial_pose_x", 0.123)
        obs, _ = env.reset(seed=42)
        assert abs(float(obs["cube_pos"][0]) - 0.123) < 1e-9
    finally:
        env.close()


def test_object_initial_pose_y_overrides_random_cube_y() -> None:
    """Counterpart axis, Y channel."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("object_initial_pose_y", -0.077)
        obs, _ = env.reset(seed=42)
        assert abs(float(obs["cube_pos"][1]) - (-0.077)) < 1e-9
    finally:
        env.close()


def test_object_initial_pose_x_preserves_random_y() -> None:
    """Setting X must NOT clobber the seed-driven random Y."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env_baseline = IsaacSimTabletopEnv()
    env_override = IsaacSimTabletopEnv()
    try:
        baseline, _ = env_baseline.reset(seed=42)
        env_override.set_perturbation("object_initial_pose_x", 0.1)
        overridden, _ = env_override.reset(seed=42)
        # Y should match (same seed, same RNG draw); X should be the override.
        assert abs(float(overridden["cube_pos"][1]) - float(baseline["cube_pos"][1])) < 1e-9
        assert abs(float(overridden["cube_pos"][0]) - 0.1) < 1e-9
    finally:
        env_baseline.close()
        env_override.close()


def test_distractor_count_teleport_semantics() -> None:
    """RFC-009 §6.5 — first `count` distractors at rest_z, rest at hidden_z.

    Reaches into ``env._distractors`` (the fake prims) — accepted
    test coupling, matches the equivalent Genesis test for the same
    invariant.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("distractor_count", 4)
        env.reset(seed=42)
        zs = [float(d.get_world_pose()[0][2]) for d in env._distractors]
        visible = sum(1 for z in zs if z > 0.0)
        hidden = sum(1 for z in zs if z < -1.0)
        assert visible == 4
        assert hidden == 6
    finally:
        env.close()


def test_distractor_count_zero_hides_all() -> None:
    """`distractor_count=0` == post-`restore_baseline` state."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("distractor_count", 0)
        env.reset(seed=0)
        for d in env._distractors:
            z = float(d.get_world_pose()[0][2])
            assert z < -1.0
    finally:
        env.close()


def test_distractor_count_full_reveals_all_ten() -> None:
    """`distractor_count=10` reveals every distractor."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("distractor_count", 10)
        env.reset(seed=0)
        for d in env._distractors:
            z = float(d.get_world_pose()[0][2])
            assert z > 0.0
    finally:
        env.close()


# ----------------------------------------------------- cosmetic axes


def test_lighting_intensity_stores_on_shadow() -> None:
    """`lighting_intensity` writes the shadow attribute and leaves
    the cube/EE/target prims untouched (state obs invariant)."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        baseline_obs, _ = env.reset(seed=42)
        env.set_perturbation("lighting_intensity", 0.4)
        after_obs, _ = env.reset(seed=42)
        assert env._light_intensity == 0.4
        # State obs unchanged (same seed, only cosmetic axis queued).
        for k in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
            assert np.allclose(baseline_obs[k], after_obs[k], atol=1e-9), (
                f"state obs['{k}'] diverged across cosmetic-only perturbation"
            )
    finally:
        env.close()


def test_camera_offset_x_stores_on_shadow() -> None:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("camera_offset_x", 0.02)
        env.reset(seed=42)
        assert float(env._cam_offset[0]) == 0.02
    finally:
        env.close()


def test_camera_offset_y_stores_on_shadow() -> None:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("camera_offset_y", -0.03)
        env.reset(seed=42)
        assert float(env._cam_offset[1]) == -0.03
    finally:
        env.close()


def test_object_texture_stores_on_shadow_and_is_zero_or_one() -> None:
    """`object_texture` rounds value to {0, 1}. Same shape Genesis used."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("object_texture", 1.0)
        env.reset(seed=0)
        assert env._texture_choice == 1

        env.set_perturbation("object_texture", 0.0)
        env.reset(seed=0)
        assert env._texture_choice == 0
    finally:
        env.close()


def test_cosmetic_axes_leave_state_obs_unchanged() -> None:
    """Compound check: queueing all four cosmetic axes does not
    change any state obs key (state-only first cut, RFC-009 §6.6).
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        baseline, _ = env.reset(seed=42)
        env.set_perturbation("lighting_intensity", 0.3)
        env.set_perturbation("camera_offset_x", 0.05)
        env.set_perturbation("camera_offset_y", -0.05)
        env.set_perturbation("object_texture", 1.0)
        after, _ = env.reset(seed=42)

        for k in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
            assert np.allclose(baseline[k], after[k], atol=1e-9), (
                f"state obs['{k}'] diverged across cosmetic-only perturbation"
            )
    finally:
        env.close()


# --------------------------------------------------- restore_baseline


def test_restore_baseline_hides_all_distractors() -> None:
    """Post-`restore_baseline` every distractor is at `_DISTRACTOR_HIDDEN_Z`."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.restore_baseline()
        zs = [float(d.get_world_pose()[0][2]) for d in env._distractors]
        assert all(z == -10.0 for z in zs)
    finally:
        env.close()


def test_restore_baseline_resets_cosmetic_shadows() -> None:
    """Cosmetic shadow attrs return to neutral defaults."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        # Pollute the shadows by hand (no perturbation queue).
        env._light_intensity = 0.7
        env._cam_offset = np.array([0.04, -0.02], dtype=np.float64)
        env._texture_choice = 1
        env.restore_baseline()
        assert env._light_intensity == 1.0
        assert float(env._cam_offset[0]) == 0.0
        assert float(env._cam_offset[1]) == 0.0
        assert env._texture_choice == 0
    finally:
        env.close()


def test_restore_baseline_is_idempotent() -> None:
    """Two consecutive `restore_baseline` calls are observationally identical."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.restore_baseline()
        zs1 = [float(d.get_world_pose()[0][2]) for d in env._distractors]
        env.restore_baseline()
        zs2 = [float(d.get_world_pose()[0][2]) for d in env._distractors]
        assert zs1 == zs2
    finally:
        env.close()


def test_set_perturbation_queue_drains_after_reset() -> None:
    """The pending queue is empty after `reset`; a second reset under
    the same seed produces baseline obs (proves we drained, not stored
    forever).
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        env.set_perturbation("object_initial_pose_x", 0.123)
        first, _ = env.reset(seed=42)
        # Queue must be empty now.
        assert env._pending_perturbations == {}
        # Second reset under the same seed should return to the
        # un-perturbed cube X (the seed-driven random draw).
        second, _ = env.reset(seed=42)
        # If the queue had survived, second cube X would still be 0.123.
        assert abs(float(first["cube_pos"][0]) - 0.123) < 1e-9
        assert abs(float(second["cube_pos"][0]) - 0.123) > 1e-3
    finally:
        env.close()
