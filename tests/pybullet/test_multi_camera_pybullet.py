"""Tests for the PyBullet multi-camera observation surface.

Mirrors the contracts in ``tests/test_multi_camera.py`` (which pin the
MuJoCo backend) — same CameraSpec validation, same per-cam dict
shape, same legacy ``obs['image']`` aliasing rule. PyBullet's
``getCameraImage`` is a per-call API with no scene-graph state, so
the multi-cam codepath here is straightforwardly per-spec view +
projection matrix.

``@pytest.mark.pybullet`` so the default (pybullet-free) pytest run
skips this whole module — same gating as
``tests/pybullet/test_render_pybullet.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from gauntlet.env import CameraSpec
from gauntlet.env.pybullet import PyBulletTabletopEnv

pytestmark = pytest.mark.pybullet


_PHASE_1_OBS_KEYS = frozenset({"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"})


def _two_specs() -> list[CameraSpec]:
    return [
        CameraSpec(name="wrist", pose=(0.3, -0.3, 0.6, 0.7, 0.0, 0.4), size=(64, 96)),
        CameraSpec(name="top", pose=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0), size=(48, 48)),
    ]


class TestSingleCameraDefaultUnchangedPyBullet:
    """The pybullet single-cam default contract is sacred — same regression
    pin as the MuJoCo equivalent in tests/test_multi_camera.py.
    """

    def test_default_obs_keys_unchanged(self) -> None:
        env = PyBulletTabletopEnv()
        try:
            obs, _ = env.reset(seed=0)
            assert set(obs.keys()) == _PHASE_1_OBS_KEYS
            assert "image" not in obs
            assert "images" not in obs
        finally:
            env.close()

    def test_default_observation_space_unchanged(self) -> None:
        env = PyBulletTabletopEnv()
        try:
            obs_space = env.observation_space
            assert isinstance(obs_space, spaces.Dict)
            assert set(obs_space.spaces.keys()) == _PHASE_1_OBS_KEYS
            assert "images" not in obs_space.spaces
        finally:
            env.close()

    def test_empty_cameras_list_treated_as_none(self) -> None:
        env = PyBulletTabletopEnv(cameras=[])
        try:
            obs, _ = env.reset(seed=0)
            assert "images" not in obs
            assert set(obs.keys()) == _PHASE_1_OBS_KEYS
        finally:
            env.close()


class TestCameraSpecValidationPyBullet:
    def test_rejects_duplicate_name(self) -> None:
        specs = [
            CameraSpec(name="wrist", pose=(0, 0, 1, 0, 0, 0), size=(32, 32)),
            CameraSpec(name="wrist", pose=(0, 0, 2, 0, 0, 0), size=(32, 32)),
        ]
        with pytest.raises(ValueError, match="duplicate camera name"):
            PyBulletTabletopEnv(cameras=specs)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="camera name must be a non-empty string"):
            PyBulletTabletopEnv(
                cameras=[CameraSpec(name="", pose=(0, 0, 1, 0, 0, 0), size=(32, 32))]
            )

    def test_rejects_nonpositive_size(self) -> None:
        with pytest.raises(ValueError, match="size must be"):
            PyBulletTabletopEnv(
                cameras=[CameraSpec(name="x", pose=(0, 0, 1, 0, 0, 0), size=(0, 32))]
            )


class TestMultiCameraObsPyBullet:
    def test_observation_space_advertises_per_camera_box(self) -> None:
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs_space = env.observation_space
            assert isinstance(obs_space, spaces.Dict)
            assert "images" in obs_space.spaces
            images_space = obs_space.spaces["images"]
            assert isinstance(images_space, spaces.Dict)
            assert set(images_space.spaces.keys()) == {"wrist", "top"}
            assert images_space.spaces["wrist"].shape == (64, 96, 3)
            assert images_space.spaces["top"].shape == (48, 48, 3)
        finally:
            env.close()

    def test_legacy_image_alias_advertised_at_top_level(self) -> None:
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs_space = env.observation_space
            assert isinstance(obs_space, spaces.Dict)
            assert "image" in obs_space.spaces
            img_space = obs_space.spaces["image"]
            assert isinstance(img_space, spaces.Box)
            # First spec wins — wrist is (64, 96, 3).
            assert img_space.shape == (64, 96, 3)
        finally:
            env.close()

    def test_reset_returns_per_camera_dict(self) -> None:
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs, _ = env.reset(seed=0)
            assert "images" in obs
            images = obs["images"]
            assert isinstance(images, dict)
            assert set(images.keys()) == {"wrist", "top"}
            assert images["wrist"].shape == (64, 96, 3)
            assert images["wrist"].dtype == np.uint8
            assert images["top"].shape == (48, 48, 3)
            assert images["top"].dtype == np.uint8
        finally:
            env.close()

    def test_image_alias_matches_first_camera_byte_for_byte(self) -> None:
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs, _ = env.reset(seed=0)
            np.testing.assert_array_equal(obs["image"], obs["images"]["wrist"])
        finally:
            env.close()

    def test_image_alias_is_independent_copy(self) -> None:
        # Same defensive-copy contract as the MuJoCo backend — an in-
        # place mutation of obs['image'] must not corrupt the per-cam
        # entry. Pinned so a future refactor cannot regress the
        # aliasing contract silently.
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs, _ = env.reset(seed=0)
            assert obs["image"] is not obs["images"]["wrist"]
            assert not np.shares_memory(obs["image"], obs["images"]["wrist"])
            original = obs["images"]["wrist"].copy()
            obs["image"].fill(123)
            np.testing.assert_array_equal(obs["images"]["wrist"], original)
        finally:
            env.close()

    def test_step_preserves_multi_camera_contract(self) -> None:
        env = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            env.reset(seed=0)
            obs_step, *_ = env.step(np.zeros(7, dtype=np.float64))
            assert "images" in obs_step
            assert obs_step["images"]["wrist"].shape == (64, 96, 3)
            assert obs_step["images"]["top"].shape == (48, 48, 3)
            np.testing.assert_array_equal(obs_step["image"], obs_step["images"]["wrist"])
        finally:
            env.close()

    def test_multi_cam_overrides_render_in_obs(self) -> None:
        # cameras takes precedence over render_in_obs (RFC §2). The
        # legacy 224x224 frame must NOT win over the multi-cam alias.
        env = PyBulletTabletopEnv(render_in_obs=True, render_size=(224, 224), cameras=_two_specs())
        try:
            obs, _ = env.reset(seed=0)
            assert obs["image"].shape == (64, 96, 3)
            assert "images" in obs
        finally:
            env.close()


class TestMultiCameraDeterminismPyBullet:
    def test_two_resets_same_seed_same_frames(self) -> None:
        env_a = PyBulletTabletopEnv(cameras=_two_specs())
        env_b = PyBulletTabletopEnv(cameras=_two_specs())
        try:
            obs_a, _ = env_a.reset(seed=42)
            obs_b, _ = env_b.reset(seed=42)
            np.testing.assert_array_equal(obs_a["images"]["wrist"], obs_b["images"]["wrist"])
            np.testing.assert_array_equal(obs_a["images"]["top"], obs_b["images"]["top"])
        finally:
            env_a.close()
            env_b.close()
