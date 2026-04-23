"""Tests for the deterministic MuJoCo tabletop environment.

Style mirrors ``tests/test_policy.py`` — small focused test classes with
explicit names, no fixtures more clever than what we need.

These tests must run headless (no GL): they never call ``env.render()``
and never instantiate ``mujoco.Renderer`` unless they are explicitly
exercising the opt-in ``render_in_obs`` path.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from gymnasium import spaces

from gauntlet.env import TabletopEnv


def _zero_action() -> np.ndarray:
    return np.zeros(7, dtype=np.float64)


class TestConstruction:
    def test_instantiates(self) -> None:
        env = TabletopEnv()
        try:
            assert env is not None
        finally:
            env.close()

    def test_rejects_nonpositive_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive"):
            TabletopEnv(max_steps=0)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            TabletopEnv(max_steps=-1)

    def test_rejects_nonpositive_n_substeps(self) -> None:
        with pytest.raises(ValueError, match="n_substeps must be positive"):
            TabletopEnv(n_substeps=0)


class TestSpaces:
    def test_action_space_shape_and_dtype(self) -> None:
        env = TabletopEnv()
        try:
            assert isinstance(env.action_space, spaces.Box)
            assert env.action_space.shape == (7,)
            assert env.action_space.dtype == np.float64
            assert float(env.action_space.low.min()) == -1.0
            assert float(env.action_space.high.max()) == 1.0
        finally:
            env.close()

    def test_observation_space_keys_and_shapes(self) -> None:
        env = TabletopEnv()
        try:
            assert isinstance(env.observation_space, spaces.Dict)
            keys = set(env.observation_space.spaces.keys())
            assert keys == {
                "cube_pos",
                "cube_quat",
                "ee_pos",
                "gripper",
                "target_pos",
            }
            shapes = {
                "cube_pos": (3,),
                "cube_quat": (4,),
                "ee_pos": (3,),
                "gripper": (1,),
                "target_pos": (3,),
            }
            for key, shape in shapes.items():
                box = env.observation_space.spaces[key]
                assert isinstance(box, spaces.Box)
                assert box.shape == shape
                assert box.dtype == np.float64
        finally:
            env.close()

    def test_action_space_sample_is_accepted_by_step(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            action = env.action_space.sample()
            assert action.shape == (7,)
            assert action.dtype == np.float64
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert env.observation_space.contains(obs)
        finally:
            env.close()


class TestReset:
    def test_reset_returns_obs_and_info(self) -> None:
        env = TabletopEnv()
        try:
            obs, info = env.reset(seed=0)
            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            for key, value in obs.items():
                assert isinstance(value, np.ndarray), key
                assert value.dtype == np.float64, key
            assert env.observation_space.contains(obs)
            assert info["success"] is False
            assert info["grasped"] is False
            assert info["step"] == 0
        finally:
            env.close()

    def test_cube_starts_on_table(self) -> None:
        env = TabletopEnv()
        try:
            obs, _ = env.reset(seed=7)
            cube_z = float(obs["cube_pos"][2])
            # Cube rests on top of the table (table_top.z = 0.42, cube half = 0.025).
            assert 0.4 <= cube_z <= 0.5
            # XY within the randomisation window we configured.
            assert abs(float(obs["cube_pos"][0])) <= 0.2
            assert abs(float(obs["cube_pos"][1])) <= 0.2
        finally:
            env.close()

    def test_target_z_on_table_top(self) -> None:
        env = TabletopEnv()
        try:
            obs, _ = env.reset(seed=3)
            assert float(obs["target_pos"][2]) == pytest.approx(0.42)
        finally:
            env.close()


class TestStep:
    def test_step_returns_five_tuple_with_correct_types(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            obs, reward, terminated, truncated, info = env.step(_zero_action())
            assert isinstance(obs, dict)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert {"success", "grasped", "step"} <= set(info.keys())
        finally:
            env.close()

    def test_rejects_wrong_shape_action(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            with pytest.raises(ValueError, match=r"shape \(7,\)"):
                env.step(np.zeros(6, dtype=np.float64))
        finally:
            env.close()

    def test_truncates_at_max_steps(self) -> None:
        env = TabletopEnv(max_steps=4)
        try:
            env.reset(seed=0)
            for i in range(3):
                _, _, terminated, truncated, _ = env.step(_zero_action())
                assert not terminated, f"step {i} terminated unexpectedly"
                assert not truncated, f"step {i} truncated early"
            _, _, terminated, truncated, info = env.step(_zero_action())
            # With a zero action and randomised target, success at step 4 is
            # extremely unlikely; assert truncation is the trigger.
            assert truncated
            assert not terminated
            assert info["step"] == 4
        finally:
            env.close()


class TestDeterminism:
    """Spec §6 hard rule: fixed seed + fixed actions => bit-identical obs."""

    @staticmethod
    def _rollout(env: TabletopEnv, seed: int, actions: np.ndarray) -> list[dict[str, np.ndarray]]:
        obs, _ = env.reset(seed=seed)
        trace = [obs]
        for action in actions:
            obs, _, _, _, _ = env.step(action)
            trace.append(obs)
        return trace

    def test_same_seed_same_actions_yields_bit_identical_obs(self) -> None:
        action_rng = np.random.default_rng(2026)
        actions = action_rng.uniform(-1.0, 1.0, size=(20, 7)).astype(np.float64)

        env_a = TabletopEnv()
        env_b = TabletopEnv()
        try:
            trace_a = self._rollout(env_a, seed=123, actions=actions)
            trace_b = self._rollout(env_b, seed=123, actions=actions)
        finally:
            env_a.close()
            env_b.close()

        assert len(trace_a) == len(trace_b) == 21
        for step_idx, (oa, ob) in enumerate(zip(trace_a, trace_b, strict=True)):
            assert oa.keys() == ob.keys(), f"key set diverged at step {step_idx}"
            for key in oa:
                np.testing.assert_array_equal(
                    oa[key],
                    ob[key],
                    err_msg=f"obs diverged at step {step_idx}, key {key}",
                )

    def test_different_seeds_yield_different_initial_state(self) -> None:
        env = TabletopEnv()
        try:
            obs_a, _ = env.reset(seed=1)
            obs_b, _ = env.reset(seed=2)
            assert not np.array_equal(obs_a["cube_pos"], obs_b["cube_pos"])
            assert not np.array_equal(obs_a["target_pos"], obs_b["target_pos"])
        finally:
            env.close()

    def test_reset_with_same_seed_is_idempotent(self) -> None:
        env = TabletopEnv()
        try:
            obs1, _ = env.reset(seed=42)
            # Take a few steps to perturb internal state.
            for _ in range(5):
                env.step(_zero_action())
            obs2, _ = env.reset(seed=42)
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
        finally:
            env.close()


class TestSuccess:
    """The success signal is what the entire Gauntlet harness measures.

    A bug here silently returns 0% for every policy across every perturbation
    sweep, so we test the success pathway end-to-end before anything else
    builds on it.
    """

    def test_success_fires_when_cube_in_target_zone(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            # Move target to wherever the cube happens to be, so a zero
            # action immediately triggers success.
            cube_pos = np.array(env._data.xpos[env._cube_body_id])
            env._target_pos = np.array(
                [cube_pos[0], cube_pos[1], TabletopEnv._TABLE_TOP_Z],
                dtype=np.float64,
            )
            _, _, terminated, truncated, info = env.step(_zero_action())
            assert info["success"] is True
            assert terminated is True
            assert truncated is False
        finally:
            env.close()

    def test_success_reflected_in_observation_target_pos(self) -> None:
        env = TabletopEnv()
        try:
            obs, _ = env.reset(seed=0)
            cube_pos = np.array(env._data.xpos[env._cube_body_id])
            env._target_pos = np.array(
                [cube_pos[0], cube_pos[1], TabletopEnv._TABLE_TOP_Z],
                dtype=np.float64,
            )
            obs, _, _, _, _ = env.step(_zero_action())
            # The observed target_pos reflects the override.
            np.testing.assert_array_equal(obs["target_pos"], env._target_pos)
        finally:
            env.close()

    def test_success_latches_after_first_trigger(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            cube_pos = np.array(env._data.xpos[env._cube_body_id])
            env._target_pos = np.array(
                [cube_pos[0], cube_pos[1], TabletopEnv._TABLE_TOP_Z],
                dtype=np.float64,
            )
            _, _, terminated_1, _, info_1 = env.step(_zero_action())
            assert info_1["success"] is True
            assert terminated_1 is True
            # Move target far away. Success must NOT flicker off.
            env._target_pos = np.array([5.0, 5.0, TabletopEnv._TABLE_TOP_Z], dtype=np.float64)
            _, _, terminated_2, _, info_2 = env.step(_zero_action())
            assert info_2["success"] is True
            assert terminated_2 is True
        finally:
            env.close()

    def test_success_does_not_fire_when_cube_outside_target(self) -> None:
        env = TabletopEnv()
        try:
            env.reset(seed=0)
            cube_pos = np.array(env._data.xpos[env._cube_body_id])
            # Target far enough that even random reset XY can't satisfy it
            # (cube XY is randomised within ±0.15, target offset by 1.0 is
            # well outside TARGET_RADIUS=0.05).
            env._target_pos = np.array(
                [cube_pos[0] + 1.0, cube_pos[1] + 1.0, TabletopEnv._TABLE_TOP_Z],
                dtype=np.float64,
            )
            _, _, terminated, _, info = env.step(_zero_action())
            assert info["success"] is False
            assert terminated is False
        finally:
            env.close()


class TestCloseAndGrasp:
    def test_close_does_not_raise(self) -> None:
        env = TabletopEnv()
        env.reset(seed=0)
        env.close()  # must not raise

    def test_grasp_engages_when_ee_close_and_gripper_closed(self) -> None:
        """End-to-end check of the grasp heuristic.

        Drive the EE down to the cube with the gripper open, then close
        the gripper. The cube should latch onto the EE and lift with it.
        """
        env = TabletopEnv()
        try:
            obs, _ = env.reset(seed=0)
            # EE starts 0.15 m above the cube. With MAX_LINEAR_STEP=0.05,
            # three full-down steps land the EE on top of the cube.
            descend = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
            for _ in range(3):
                obs, _, _, _, _ = env.step(descend)
            assert obs["gripper"][0] == TabletopEnv.GRIPPER_OPEN
            # Sanity check: EE is now within GRASP_RADIUS of the cube.
            ee_to_cube = float(np.linalg.norm(obs["ee_pos"] - obs["cube_pos"]))
            assert ee_to_cube <= TabletopEnv.GRASP_RADIUS
            # Close the gripper for a step to latch.
            close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, _, _, _, info = env.step(close)
            assert obs["gripper"][0] == TabletopEnv.GRIPPER_CLOSED
            assert info["grasped"] is True
            # Lift: cube z should follow EE z.
            lift = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            for _ in range(5):
                obs, _, _, _, _ = env.step(lift)
            cube_z = float(obs["cube_pos"][2])
            ee_z = float(obs["ee_pos"][2])
            # Snapped: cube tracks EE exactly (within float noise).
            assert cube_z == pytest.approx(ee_z, abs=1e-9)
        finally:
            env.close()


# Phase 1 key set — used by the render_in_obs=False regression test. Kept
# inline here rather than imported so a drift in the env's default obs is
# caught by this test, not silently re-baked into a shared constant.
_PHASE_1_OBS_KEYS = frozenset({"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"})


class TestRenderInObs:
    """RFC Phase 2 Task 1 §5 — the opt-in ``render_in_obs`` kwarg is the one
    core-side change; default behaviour must stay byte-identical.
    """

    def test_default_obs_keys_unchanged(self) -> None:
        env = TabletopEnv()  # render_in_obs=False by default
        try:
            obs, _ = env.reset(seed=0)
            assert set(obs.keys()) == _PHASE_1_OBS_KEYS
            assert "image" not in env.observation_space.spaces
        finally:
            env.close()

    def test_render_in_obs_emits_uint8_image(self) -> None:
        # Offscreen rendering needs a GL backend; EGL is the one we target
        # in CI (osmesa would work too). If neither is available the test
        # surfaces a clear skip rather than a cryptic GL crash.
        os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            env = TabletopEnv(render_in_obs=True, render_size=(64, 96))
        except Exception as exc:
            # GL init failures wrap many underlying exception types; catching
            # ``Exception`` surfaces them as a skip rather than a cryptic crash.
            pytest.skip(f"offscreen GL backend unavailable ({exc!r}); set MUJOCO_GL=egl or osmesa")
        try:
            obs, _ = env.reset(seed=0)
            assert "image" in obs
            img = obs["image"]
            assert img.dtype == np.uint8
            assert img.shape == (64, 96, 3)

            # observation_space advertises the matching Box.
            assert "image" in env.observation_space.spaces
            img_space = env.observation_space.spaces["image"]
            assert isinstance(img_space, spaces.Box)
            assert img_space.shape == (64, 96, 3)
            assert img_space.dtype == np.uint8

            # Step preserves the image contract.
            obs_step, *_ = env.step(np.zeros(7, dtype=np.float64))
            assert obs_step["image"].shape == (64, 96, 3)
            assert obs_step["image"].dtype == np.uint8
        finally:
            env.close()
