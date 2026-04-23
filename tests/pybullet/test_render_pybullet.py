"""Tests for the PyBullet rendering path (RFC-006).

The ``render_in_obs`` kwarg opts into a headless ``obs["image"]`` stream
on ``PyBulletTabletopEnv``. Marker ``@pytest.mark.pybullet`` so the
default (torch-/pybullet-free) pytest run does not try to import
pybullet.

Each test is written to be self-contained: construct the env fresh,
exercise one behaviour, call ``close()``. The ``pybullet-tests`` CI job
installs the ``[pybullet]`` extra and the ``pybullet-dev`` group, so
real environments run — no MagicMock.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("pybullet")
pytest.importorskip("pybullet_data")

from gauntlet.env.pybullet import PyBulletTabletopEnv

pytestmark = pytest.mark.pybullet


class TestRenderInObsSpace:
    """The ``observation_space`` extension when ``render_in_obs=True``."""

    def test_image_absent_by_default(self) -> None:
        """``render_in_obs=False`` (the default) — observation_space has no 'image' key."""
        env = PyBulletTabletopEnv()
        try:
            assert isinstance(env.observation_space, spaces.Dict)
            assert "image" not in env.observation_space.spaces
            # Five canonical keys unchanged (RFC-005 §7.2 byte-parity).
            assert set(env.observation_space.spaces.keys()) == {
                "cube_pos",
                "cube_quat",
                "ee_pos",
                "gripper",
                "target_pos",
            }
        finally:
            env.close()

    def test_image_shape_matches_render_size(self) -> None:
        """``render_in_obs=True, render_size=(H, W)`` → ``image`` Box is (H, W, 3) uint8."""
        env = PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 96))
        try:
            assert isinstance(env.observation_space, spaces.Dict)
            img = env.observation_space.spaces["image"]
            assert isinstance(img, spaces.Box)
            assert img.shape == (64, 96, 3)
            assert img.dtype == np.uint8
            assert int(img.low.min()) == 0
            assert int(img.high.max()) == 255
        finally:
            env.close()

    def test_bad_render_size_rejected(self) -> None:
        """Non-positive H or W raises the same message TabletopEnv raises."""
        with pytest.raises(ValueError, match=r"render_size must be a \(height, width\)"):
            PyBulletTabletopEnv(render_in_obs=True, render_size=(0, 32))
        with pytest.raises(ValueError, match=r"render_size must be a \(height, width\)"):
            PyBulletTabletopEnv(render_in_obs=True, render_size=(32, -1))


class TestRenderShapeDtype:
    """``obs["image"]`` shape / dtype / value range after a real render."""

    def test_reset_emits_uint8_image(self) -> None:
        """``reset`` with ``render_in_obs=True`` fills a (H, W, 3) uint8 array."""
        env = PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 96))
        try:
            obs, _ = env.reset(seed=0)
            img = obs["image"]
            assert img.shape == (64, 96, 3)
            assert img.dtype == np.uint8
            assert int(img.min()) >= 0
            assert int(img.max()) <= 255
            # Non-trivial render — there is a scene visible, not a flat colour.
            # The scene has a plane + table + cube + EE body; ≥ 4 unique RGB
            # triples is a conservative lower bound that the renderer actually ran.
            uniq = len(np.unique(img.reshape(-1, 3), axis=0))
            assert uniq > 3, f"expected a non-trivial render; got only {uniq} unique colours"
        finally:
            env.close()

    def test_step_emits_uint8_image(self) -> None:
        """``step`` returns an ``obs["image"]`` with the same contract as reset."""
        env = PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 64))
        try:
            env.reset(seed=0)
            # Zero action — no-op step, but render still fires.
            obs, _, _, _, _ = env.step(np.zeros(7, dtype=np.float64))
            img = obs["image"]
            assert img.shape == (64, 64, 3)
            assert img.dtype == np.uint8
        finally:
            env.close()


class TestRenderDeterminism:
    """Bit-determinism contract — RFC-006 §4."""

    def test_reset_same_seed_matches_images(self) -> None:
        """Two independent envs, both ``reset(seed=42)`` → byte-equal images."""
        env_a = PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 64))
        env_b = PyBulletTabletopEnv(render_in_obs=True, render_size=(64, 64))
        try:
            obs_a, _ = env_a.reset(seed=42)
            obs_b, _ = env_b.reset(seed=42)
            assert np.array_equal(obs_a["image"], obs_b["image"])
        finally:
            env_a.close()
            env_b.close()

    def test_post_step_determinism_fixed_actions(self) -> None:
        """Same seed + same action sequence on two instances → equal per-step images."""
        actions = [
            np.array([0.1, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.1, -0.05, 0.0, 0.0, 0.1, 1.0], dtype=np.float64),
            np.array([-0.05, -0.05, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64),
        ] * 5  # 15 steps — keeps the test fast while covering the render path.

        env_a = PyBulletTabletopEnv(render_in_obs=True, render_size=(48, 48))
        env_b = PyBulletTabletopEnv(render_in_obs=True, render_size=(48, 48))
        try:
            env_a.reset(seed=7)
            env_b.reset(seed=7)
            for i, a in enumerate(actions):
                obs_a, *_ = env_a.step(a)
                obs_b, *_ = env_b.step(a)
                assert np.array_equal(
                    obs_a["image"], obs_b["image"]
                ), f"images diverged at step {i}"
        finally:
            env_a.close()
            env_b.close()
