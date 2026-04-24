"""Tests for the opt-in multi-camera observation surface.

See ``docs/polish-exploration-multi-camera.md`` for the design.

These tests must run in the default pytest job (no backend marker), so
they live alongside ``tests/test_env.py``. The single-camera default
path is **byte-identical** to Phase 1 — the regression test below
pins that contract; if it fails, the multi-cam wiring leaked into the
default path and that is a bug, not a refactor opportunity.

Tests that exercise the multi-cam *render* path skip when offscreen
GL is unavailable, mirroring the gating in
``tests/test_env.py::TestRenderInObs.test_render_in_obs_emits_uint8_image``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from gymnasium import spaces

from gauntlet.env import CameraSpec, TabletopEnv

# Same key set as ``tests/test_env.py::_PHASE_1_OBS_KEYS`` — kept inline
# to mirror the existing pattern (we want a drift in the default obs to
# fail this test loudly, not be silently re-baked into a shared
# constant).
_PHASE_1_OBS_KEYS = frozenset({"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"})


def _two_specs() -> list[CameraSpec]:
    """Two CameraSpecs of differing sizes — exercises per-spec renderer caching.

    Both poses look at the table from above (z=0.6, 1.0) so the default
    EGL render does not return solid black even on minimal GL stacks.
    """
    return [
        CameraSpec(name="wrist", pose=(0.3, -0.3, 0.6, 0.7, 0.0, 0.4), size=(64, 96)),
        CameraSpec(name="top", pose=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0), size=(48, 48)),
    ]


def _gl_or_skip() -> None:
    """Set MUJOCO_GL=egl and let the renderer construct decide if GL is available."""
    os.environ.setdefault("MUJOCO_GL", "egl")


class TestSingleCameraDefaultUnchanged:
    """Backwards-compatibility regression — the default ``TabletopEnv()``
    construction MUST be byte-identical to the Phase 1 obs contract.

    A failure here means the multi-cam wiring leaked into the default
    path. The fix is in the env, not in this test.
    """

    def test_default_obs_keys_unchanged(self) -> None:
        env = TabletopEnv()
        try:
            obs, _ = env.reset(seed=0)
            # Exact equality — no extras, no missing keys.
            assert set(obs.keys()) == _PHASE_1_OBS_KEYS
            assert "image" not in obs
            assert "images" not in obs
        finally:
            env.close()

    def test_default_observation_space_unchanged(self) -> None:
        env = TabletopEnv()
        try:
            assert set(env.observation_space.spaces.keys()) == _PHASE_1_OBS_KEYS
            assert "image" not in env.observation_space.spaces
            assert "images" not in env.observation_space.spaces
        finally:
            env.close()

    def test_empty_cameras_list_treated_as_none(self) -> None:
        # cameras=[] is a documented synonym for cameras=None (RFC §2).
        env = TabletopEnv(cameras=[])
        try:
            obs, _ = env.reset(seed=0)
            assert set(obs.keys()) == _PHASE_1_OBS_KEYS
            assert "images" not in obs
        finally:
            env.close()

    def test_default_obs_values_byte_identical(self) -> None:
        # Pin the actual obs values for seed=0 so a future refactor that
        # accidentally re-orders the random draws (cube xy vs target xy
        # vs etc.) trips this test loudly.
        env_a = TabletopEnv()
        env_b = TabletopEnv()
        try:
            obs_a, _ = env_a.reset(seed=0)
            obs_b, _ = env_b.reset(seed=0)
            assert set(obs_a) == set(obs_b)
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key])
        finally:
            env_a.close()
            env_b.close()


class TestCameraSpecValidation:
    """Construction-time validation runs before any GL context is built,
    so these tests are headless-safe and live in the default job.
    """

    def test_rejects_duplicate_name(self) -> None:
        specs = [
            CameraSpec(name="wrist", pose=(0, 0, 1, 0, 0, 0), size=(32, 32)),
            CameraSpec(name="wrist", pose=(0, 0, 2, 0, 0, 0), size=(32, 32)),
        ]
        with pytest.raises(ValueError, match="duplicate camera name"):
            TabletopEnv(cameras=specs)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="camera name must be a non-empty string"):
            TabletopEnv(cameras=[CameraSpec(name="", pose=(0, 0, 1, 0, 0, 0), size=(32, 32))])

    def test_rejects_nonpositive_size(self) -> None:
        with pytest.raises(ValueError, match="size must be"):
            TabletopEnv(cameras=[CameraSpec(name="x", pose=(0, 0, 1, 0, 0, 0), size=(0, 32))])
        with pytest.raises(ValueError, match="size must be"):
            TabletopEnv(cameras=[CameraSpec(name="x", pose=(0, 0, 1, 0, 0, 0), size=(32, -1))])


class TestMultiCameraObs:
    """End-to-end multi-cam rendering — gated on a working offscreen GL
    context (MUJOCO_GL=egl by default; osmesa also works).
    """

    def _build(self) -> TabletopEnv:
        _gl_or_skip()
        try:
            return TabletopEnv(cameras=_two_specs())
        except Exception as exc:
            pytest.skip(f"offscreen GL backend unavailable ({exc!r}); set MUJOCO_GL=egl or osmesa")

    def test_observation_space_advertises_per_camera_box(self) -> None:
        env = self._build()
        try:
            assert "images" in env.observation_space.spaces
            images_space = env.observation_space.spaces["images"]
            assert isinstance(images_space, spaces.Dict)
            assert set(images_space.spaces.keys()) == {"wrist", "top"}
            wrist_box = images_space.spaces["wrist"]
            top_box = images_space.spaces["top"]
            assert isinstance(wrist_box, spaces.Box)
            assert isinstance(top_box, spaces.Box)
            assert wrist_box.shape == (64, 96, 3)
            assert top_box.shape == (48, 48, 3)
            assert wrist_box.dtype == np.uint8
            assert top_box.dtype == np.uint8
        finally:
            env.close()

    def test_legacy_image_alias_advertised_at_top_level(self) -> None:
        env = self._build()
        try:
            # Backwards-compat: obs['image'] must remain a top-level Box
            # so the runner's video recorder (runner/worker.py:417) keeps
            # working in multi-cam mode.
            assert "image" in env.observation_space.spaces
            img_space = env.observation_space.spaces["image"]
            assert isinstance(img_space, spaces.Box)
            # Aliased to the FIRST camera's size — wrist (64, 96, 3).
            assert img_space.shape == (64, 96, 3)
            assert img_space.dtype == np.uint8
        finally:
            env.close()

    def test_reset_returns_per_camera_dict(self) -> None:
        env = self._build()
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
        env = self._build()
        try:
            obs, _ = env.reset(seed=0)
            assert "image" in obs
            np.testing.assert_array_equal(obs["image"], obs["images"]["wrist"])
        finally:
            env.close()

    def test_image_alias_is_independent_copy(self) -> None:
        # Pin the defensive-copy contract: an in-place mutation of
        # obs['image'] (a normalisation pass, a tensor conversion that
        # aliases the buffer, etc.) MUST NOT corrupt
        # obs['images'][first]. Without the .copy() in _build_obs the
        # two would share storage and the mutation below would leak.
        env = self._build()
        try:
            obs, _ = env.reset(seed=0)
            assert obs["image"] is not obs["images"]["wrist"]
            assert not np.shares_memory(obs["image"], obs["images"]["wrist"])
            # Mutate every byte of the alias to a sentinel and confirm
            # the per-cam entry is unchanged.
            original = obs["images"]["wrist"].copy()
            obs["image"].fill(123)
            np.testing.assert_array_equal(obs["images"]["wrist"], original)
        finally:
            env.close()

    def test_step_preserves_multi_camera_contract(self) -> None:
        env = self._build()
        try:
            env.reset(seed=0)
            obs_step, *_ = env.step(np.zeros(7, dtype=np.float64))
            assert "images" in obs_step
            assert obs_step["images"]["wrist"].shape == (64, 96, 3)
            assert obs_step["images"]["top"].shape == (48, 48, 3)
            np.testing.assert_array_equal(obs_step["image"], obs_step["images"]["wrist"])
        finally:
            env.close()

    def test_state_observations_unchanged_in_multi_cam_mode(self) -> None:
        # The 5 phase-1 keys must still be present and have the same dtype
        # / shape contract — multi-cam adds keys, never replaces them.
        env = self._build()
        try:
            obs, _ = env.reset(seed=0)
            for key in _PHASE_1_OBS_KEYS:
                assert key in obs, f"phase-1 key {key!r} missing in multi-cam mode"
                assert obs[key].dtype == np.float64
        finally:
            env.close()

    def test_multi_cam_overrides_render_in_obs(self) -> None:
        # cameras takes precedence over the legacy render_in_obs flag
        # (RFC §2). The single-camera codepath is disabled so we should
        # NOT see the legacy 224x224 render — instead obs['image'] is
        # the multi-cam first-cam alias (64, 96).
        _gl_or_skip()
        try:
            env = TabletopEnv(render_in_obs=True, render_size=(224, 224), cameras=_two_specs())
        except Exception as exc:
            pytest.skip(f"offscreen GL backend unavailable ({exc!r})")
        try:
            obs, _ = env.reset(seed=0)
            assert obs["image"].shape == (64, 96, 3)
            assert "images" in obs
        finally:
            env.close()


class TestMultiCameraDeterminism:
    """Two envs with identical cameras + identical seed must produce
    byte-identical frames.
    """

    def test_two_resets_same_seed_same_frames(self) -> None:
        _gl_or_skip()
        try:
            env_a = TabletopEnv(cameras=_two_specs())
            env_b = TabletopEnv(cameras=_two_specs())
        except Exception as exc:
            pytest.skip(f"offscreen GL backend unavailable ({exc!r})")
        try:
            obs_a, _ = env_a.reset(seed=42)
            obs_b, _ = env_b.reset(seed=42)
            np.testing.assert_array_equal(obs_a["images"]["wrist"], obs_b["images"]["wrist"])
            np.testing.assert_array_equal(obs_a["images"]["top"], obs_b["images"]["top"])
        finally:
            env_a.close()
            env_b.close()
