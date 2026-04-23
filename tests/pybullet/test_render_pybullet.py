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
            img = env.observation_space["image"]
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
