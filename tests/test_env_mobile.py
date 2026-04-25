"""Tests for the kinematic mobile-base wrapper (B-13).

Style mirrors :mod:`tests.test_env` — small focused classes, no
fixtures more clever than necessary, headless-only (never
``env.render()``). The suite intentionally exercises the phase-1
contract: SE(2) base pose plumbing, 10-D action space, AND-of-(nav,
pick) success — *not* a real wheeled-base physics rebake.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from gauntlet.env import GauntletEnv, MobileTabletopEnv
from gauntlet.env.registry import get_env_factory


def _zero_action() -> np.ndarray:
    return np.zeros(10, dtype=np.float64)


class TestConstruction:
    def test_instantiates_and_satisfies_protocol(self) -> None:
        env = MobileTabletopEnv()
        try:
            assert env is not None
            assert isinstance(env, GauntletEnv)
        finally:
            env.close()

    def test_rejects_nonpositive_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive"):
            MobileTabletopEnv(max_steps=0)


class TestSpaces:
    def test_action_space_dim_10_and_bounded(self) -> None:
        env = MobileTabletopEnv()
        try:
            assert isinstance(env.action_space, spaces.Box)
            assert env.action_space.shape == (10,)
            assert env.action_space.dtype == np.float64
            assert float(env.action_space.low.min()) == -1.0
            assert float(env.action_space.high.max()) == 1.0
        finally:
            env.close()

    def test_observation_includes_pose_and_inner_keys(self) -> None:
        env = MobileTabletopEnv()
        try:
            assert isinstance(env.observation_space, spaces.Dict)
            keys = set(env.observation_space.spaces.keys())
            # Inner tabletop keys preserved, plus our SE(2) pose key.
            assert {
                "cube_pos",
                "cube_quat",
                "ee_pos",
                "gripper",
                "target_pos",
                "pose",
            } <= keys
            pose_box = env.observation_space.spaces["pose"]
            assert isinstance(pose_box, spaces.Box)
            assert pose_box.shape == (3,)
            assert pose_box.dtype == np.float64
        finally:
            env.close()

    def test_reset_obs_pose_starts_at_origin(self) -> None:
        env = MobileTabletopEnv()
        try:
            obs, _info = env.reset(seed=0)
            assert "pose" in obs
            assert obs["pose"].shape == (3,)
            assert np.allclose(obs["pose"], 0.0)
        finally:
            env.close()


class TestRollouts:
    def test_random_50_steps_does_not_crash(self) -> None:
        env = MobileTabletopEnv()
        try:
            env.reset(seed=42)
            rng = np.random.default_rng(0)
            for _ in range(50):
                action = rng.uniform(-1.0, 1.0, size=10).astype(np.float64)
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                assert isinstance(reward, float)
                assert isinstance(info, dict)
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_scripted_navigate_then_pick_succeeds(self) -> None:
        """Drive the base to the target table, then snap-grasp the cube.

        Phase 1 is kinematic: the inner cube does not move with the
        base. We exploit that by overwriting the inner mocap to land
        on top of the cube once nav is done, then close the gripper —
        same protocol the inner env's grasp model exercises.
        """
        env = MobileTabletopEnv(max_steps=200)
        try:
            env.reset(seed=0)
            # Phase A — navigate vx=+1 until base XY is within
            # NAV_RADIUS of the target table. With BASE_DT=0.1 and
            # MAX_BASE_LINEAR=1.0 we cover 0.1m per step; 13 steps
            # land at x=1.3 (>= 1.5 - NAV_RADIUS).
            drive = np.zeros(10, dtype=np.float64)
            drive[7] = 1.0  # base_vx = +1
            nav_done = False
            for _ in range(40):
                _obs, _r, _t, _tr, info = env.step(drive)
                if info.get("nav_done"):
                    nav_done = True
                    break
            assert nav_done, "base should reach the target table within 40 steps"

            # Phase B — snap the EE onto the cube and close the gripper.
            inner = env._inner
            cube_pos = np.array(inner._data.xpos[inner._cube_body_id])
            inner._data.mocap_pos[inner._ee_mocap_id] = cube_pos
            close_grip = np.zeros(10, dtype=np.float64)
            close_grip[6] = -1.0  # gripper close
            _obs, _r, terminated, _tr, info = env.step(close_grip)
            assert info["success"], f"expected nav+grasp success; info={info}"
            assert terminated

        finally:
            env.close()

    def test_drive_wrong_direction_never_succeeds(self) -> None:
        env = MobileTabletopEnv(max_steps=50)
        try:
            env.reset(seed=0)
            backwards = np.zeros(10, dtype=np.float64)
            backwards[7] = -1.0  # base_vx = -1, away from target
            for _ in range(50):
                _obs, _r, terminated, truncated, info = env.step(backwards)
                assert not info["success"]
                assert not terminated
                if truncated:
                    break
            # Base should be solidly negative-x.
            assert env._base_pose[0] < -0.3
        finally:
            env.close()


class TestAxesAndRegistry:
    def test_axis_names_is_empty_on_phase1(self) -> None:
        assert frozenset() == MobileTabletopEnv.AXIS_NAMES

    def test_set_perturbation_rejects_every_key(self) -> None:
        env = MobileTabletopEnv()
        try:
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                env.set_perturbation("lighting_intensity", 1.0)
        finally:
            env.close()

    def test_registry_resolves_tabletop_mobile(self) -> None:
        factory = get_env_factory("tabletop-mobile")
        env = factory()
        try:
            assert isinstance(env, MobileTabletopEnv)
        finally:
            env.close()
