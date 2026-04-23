"""PyBullet per-axis perturbation tests — RFC-005 §9.1 cases 7 and 8.

One test per canonical axis (§6 table). State-effecting axes assert
the observable world change on a real reset; cosmetic axes assert the
internal scratchpad is updated (state-only obs cannot prove more until
the rendering follow-up RFC lands).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pybullet as p

if TYPE_CHECKING:
    from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

pytestmark = pytest.mark.pybullet


def _distractor_alpha(env: PyBulletTabletopEnv, index: int) -> float:
    """Read the current rgba alpha of the ``index``-th distractor body.

    ``p.getVisualShapeData`` returns a list of per-link visual tuples;
    we only have link -1 on each distractor body.
    """
    body_id = env._distractor_ids[index]
    client = env._client
    data = p.getVisualShapeData(body_id, physicsClientId=client)
    # Tuple layout: (bodyId, linkIndex, visualGeometryType, dimensions,
    #                meshAssetFileName, localVisualFramePosition,
    #                localVisualFrameOrientation, rgbaColor, textureUniqueId)
    return float(data[0][7][3])


def _cube_texture_id(env: PyBulletTabletopEnv) -> int:
    """Return the texture UID currently bound to the cube.

    PyBullet's ``getVisualShapeData`` doesn't include the texture UID
    in its default return tuple, so we read the env's own scratchpad
    (``self._current_texture_id``) — the branch in
    ``_apply_one_perturbation`` writes here each time
    ``changeVisualShape(textureUniqueId=...)`` fires.
    """
    return int(env._current_texture_id)


def test_object_initial_pose_x_overrides_random_cube_xy() -> None:
    """RFC-005 §6 — state-effecting axis. ``obs["cube_pos"][0]`` must
    equal the queued value after :meth:`reset`, NOT the RNG draw.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("object_initial_pose_x", 0.1234)
        obs, _ = env.reset(seed=999)
        assert obs["cube_pos"][0] == pytest.approx(0.1234, abs=1e-6)
    finally:
        env.close()


def test_object_initial_pose_y_overrides_random_cube_xy() -> None:
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("object_initial_pose_y", -0.08)
        obs, _ = env.reset(seed=999)
        assert obs["cube_pos"][1] == pytest.approx(-0.08, abs=1e-6)
    finally:
        env.close()


def test_distractor_count_reveals_first_n_bodies() -> None:
    """RFC-005 §6 — state-effecting. N visible + solid distractors,
    (10 - N) hidden + ghosted. The visibility proxy is rgba.alpha on
    each distractor body.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("distractor_count", 4.0)
        env.reset(seed=1)
        active_alphas = [_distractor_alpha(env, i) for i in range(4)]
        inactive_alphas = [_distractor_alpha(env, i) for i in range(4, 10)]
        assert all(a > 0.0 for a in active_alphas), active_alphas
        assert all(a == 0.0 for a in inactive_alphas), inactive_alphas
    finally:
        env.close()


def test_distractor_count_zero_leaves_all_hidden() -> None:
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("distractor_count", 0.0)
        env.reset(seed=1)
        for i in range(10):
            assert _distractor_alpha(env, i) == 0.0
    finally:
        env.close()


def test_distractor_count_validator_rejects_out_of_range() -> None:
    """Parity with MuJoCo — set_perturbation must reject values
    outside [0, 10] at queue time (before reset applies the branch).
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        with pytest.raises(ValueError, match="distractor_count"):
            env.set_perturbation("distractor_count", 11.0)
        with pytest.raises(ValueError, match="distractor_count"):
            env.set_perturbation("distractor_count", -1.0)
    finally:
        env.close()


def test_object_texture_swaps_bound_texture_on_cube() -> None:
    """RFC-005 §6 — cosmetic axis. ``object_texture=1`` must bind the
    alt texture UID; value ``0`` binds the default. State-only obs are
    unaffected (no assertion on obs).
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("object_texture", 1.0)
        env.reset(seed=1)
        assert _cube_texture_id(env) == env._tex_alt_id

        env.set_perturbation("object_texture", 0.0)
        env.reset(seed=1)
        assert _cube_texture_id(env) == env._tex_default_id
    finally:
        env.close()


def test_lighting_intensity_stored_as_scratchpad() -> None:
    """RFC-005 §6.1 — headless DIRECT mode has no runtime light API;
    the axis writes to ``self._light_intensity`` and is consumed by the
    rendering RFC's getCameraImage path. State-only obs are unaffected.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("lighting_intensity", 0.42)
        env.reset(seed=1)
        assert env._light_intensity == pytest.approx(0.42)
    finally:
        env.close()


def test_camera_offset_x_stored_as_scratchpad() -> None:
    """Camera pose on PyBullet is a rendering-time concept (no camera
    body in the physics scene). The axis writes to ``self._cam_eye``
    and is consumed by the rendering follow-up RFC.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("camera_offset_x", 0.07)
        env.reset(seed=1)
        assert env._cam_eye_offset[0] == pytest.approx(0.07)
        assert env._cam_eye_offset[1] == pytest.approx(0.0)
    finally:
        env.close()


def test_camera_offset_y_stored_as_scratchpad() -> None:
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        env.set_perturbation("camera_offset_y", -0.03)
        env.reset(seed=1)
        assert env._cam_eye_offset[0] == pytest.approx(0.0)
        assert env._cam_eye_offset[1] == pytest.approx(-0.03)
    finally:
        env.close()


def test_restore_baseline_reverts_queued_perturbations_next_episode() -> None:
    """RFC-005 §9.1 case 8. Setting all seven axes, running one reset,
    then running a no-perturbation reset on the same seed must produce
    a baseline trajectory — the reset handler must have cleared the
    queue after applying.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        # Perturbed episode.
        for name, value in [
            ("lighting_intensity", 0.4),
            ("camera_offset_x", 0.01),
            ("camera_offset_y", -0.01),
            ("object_texture", 1.0),
            ("object_initial_pose_x", 0.12),
            ("object_initial_pose_y", -0.10),
            ("distractor_count", 3.0),
        ]:
            env.set_perturbation(name, value)
        env.reset(seed=5)

        # Second reset without queueing anything.
        assert env._pending_perturbations == {}
        obs_b, _ = env.reset(seed=5)

        # Baseline reference.
        env2 = PyBulletTabletopEnv()
        try:
            obs_ref, _ = env2.reset(seed=5)
            assert np.allclose(obs_b["cube_pos"], obs_ref["cube_pos"], atol=1e-9)
            assert np.allclose(
                obs_b["target_pos"], obs_ref["target_pos"], atol=1e-9
            )
        finally:
            env2.close()
    finally:
        env.close()
