"""Experimental image observations on the Isaac backend.

Runs against the conftest's fake ``isaacsim`` namespace. The fake camera
encodes light intensity, cube colour and camera x position into the R,
G and B channels, so these tests check that each cosmetic axis is
routed to the right scene object and reaches ``obs["image"]``. They do
not (and cannot) check what the real RTX renderer produces.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest


def _make(**kwargs: Any) -> Any:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsaacSimTabletopEnv(**kwargs)


def _reset_with(env: Any, **axes: float) -> dict[str, Any]:
    for name, value in axes.items():
        env.set_perturbation(name, value)
    obs, _ = env.reset(seed=0)
    return dict(obs)


def test_state_only_default_has_no_image_and_no_camera() -> None:
    from tests.isaac.conftest import _FAKE_STAGE

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no experimental warning by default
        env = _make()
    try:
        obs = _reset_with(env)
        assert "image" not in obs
        assert "image" not in env.observation_space.spaces
        assert "camera" not in _FAKE_STAGE
    finally:
        env.close()


def test_render_in_obs_warns_experimental() -> None:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    with pytest.warns(UserWarning, match="not hardware-verified"):
        env = IsaacSimTabletopEnv(render_in_obs=True)
    env.close()


def test_image_shape_and_space_on_reset_and_step() -> None:
    env = _make(render_in_obs=True, render_size=(48, 64))
    try:
        obs = _reset_with(env)
        assert obs["image"].shape == (48, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert env.observation_space["image"].shape == (48, 64, 3)
        obs, *_ = env.step(np.zeros(7))
        assert obs["image"].shape == (48, 64, 3)
    finally:
        env.close()


def test_camera_initialised_after_world_reset_and_warmed_up() -> None:
    from tests.isaac.conftest import _FAKE_STAGE

    env = _make(render_in_obs=True)
    try:
        camera = _FAKE_STAGE["camera"]
        assert camera.initialized is False
        obs = _reset_with(env)
        assert camera.initialized is True
        # The fake returns no frame until a render; reset must render.
        assert _FAKE_STAGE["renders"] >= 1
        assert obs["image"][..., 0].max() > 0
    finally:
        env.close()


def test_lighting_axis_drives_the_key_light() -> None:
    env = _make(render_in_obs=True)
    try:
        dim = _reset_with(env, lighting_intensity=0.5)["image"][0, 0, 0]
        bright = _reset_with(env, lighting_intensity=1.5)["image"][0, 0, 0]
        assert bright > dim
    finally:
        env.close()


def test_texture_axis_swaps_cube_material() -> None:
    env = _make(render_in_obs=True)
    try:
        default = _reset_with(env, object_texture=0.0)["image"][0, 0, 1]
        alt = _reset_with(env, object_texture=1.0)["image"][0, 0, 1]
        assert alt != default
    finally:
        env.close()


def test_camera_offset_moves_camera_in_world_axes() -> None:
    from tests.isaac.conftest import _FAKE_STAGE

    env = _make(render_in_obs=True)
    try:
        base = _reset_with(env)["image"][0, 0, 2]
        shifted = _reset_with(env, camera_offset_x=0.1)["image"][0, 0, 2]
        assert shifted != base
        last = _FAKE_STAGE["camera"].set_world_pose_calls[-1]
        assert last["camera_axes"] == "world"
    finally:
        env.close()


def test_cosmetic_state_does_not_leak_across_resets() -> None:
    env = _make(render_in_obs=True)
    try:
        baseline = _reset_with(env)["image"].copy()
        _reset_with(env, lighting_intensity=1.5, object_texture=1.0, camera_offset_x=0.1)
        again = _reset_with(env)["image"]
        np.testing.assert_array_equal(again, baseline)
    finally:
        env.close()


def test_look_at_quaternion_points_camera_forward_axis_at_target() -> None:
    from gauntlet.env.isaac.rendering import CAMERA_POS, CAMERA_TARGET, look_at_quat_world

    w, x, y, z = look_at_quat_world(CAMERA_POS, CAMERA_TARGET)
    # Rotate the camera's +X (forward in "world" camera axes) into the world frame.
    forward = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    expected = (CAMERA_TARGET - CAMERA_POS) / np.linalg.norm(CAMERA_TARGET - CAMERA_POS)
    np.testing.assert_allclose(forward, expected, atol=1e-9)


def test_render_size_validated() -> None:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    with pytest.raises(ValueError, match="render_size"):
        IsaacSimTabletopEnv(render_size=(0, 64))


def test_step_image_matches_post_step_scene() -> None:
    """The frame returned by step() shows the cube where obs['cube_pos'] says it is.

    Regression: rendering inside world.step and *then* snapping a grasped
    cube to the end-effector made carried-cube frames one step stale.
    """
    env = _make(render_in_obs=True)
    try:
        obs = _reset_with(env)
        # Put the EE on the cube, close the gripper, then carry it along +x.
        env._ee.set_world_pose(position=obs["cube_pos"].copy())
        close = np.zeros(7)
        close[6] = -1.0
        env.step(close)
        carry = close.copy()
        carry[0] = 1.0
        obs, *_ = env.step(carry)
        expected_alpha = round(1000 * obs["cube_pos"][0]) % 256
        assert obs["image"].shape[2] == 3
        assert env._renderer._camera.get_rgba()[0, 0, 3] == expected_alpha
    finally:
        env.close()
