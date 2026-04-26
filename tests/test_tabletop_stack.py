"""Tests for the long-horizon 3-cube stacking env (B-09 stub).

State-only — these tests never touch ``env.render()`` or
``mujoco.Renderer`` so they run on headless CI workers without a GL
context. The scripted-policy test is the load-bearing assertion that
the 3-subtask chain is genuinely reachable from the action surface;
everything else (spaces, determinism, monotonicity) pins the
contract surface around it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from gauntlet.env import TabletopStackEnv


def _zero_action() -> NDArray[np.float64]:
    return np.zeros(7, dtype=np.float64)


# Shared scripted policy used by the milestone test. Lives at module
# scope so tests can import it without recreating the closure each
# time. Driving the EE via a small proportional controller against
# the cached cube poses is the cheapest end-to-end validation that
# the action surface, the grasp tracker, and the latch predicates
# all line up. Anything fancier (planning, RL) would be hostile to
# debugging when this fails.


def _move_to(
    env: TabletopStackEnv,
    target: Iterable[float],
    gripper: float,
    *,
    tol: float = 0.003,
    max_steps: int = 80,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Drive the EE toward ``target`` (3-tuple) at the env's per-step velocity.

    Issues no-op step at the end to land a fresh ``obs`` / ``info``
    pair even when the EE is already at the target on entry. ``gripper``
    is held constant for the duration of the move (the caller controls
    grasp / release timing by varying it across calls).
    """
    obs_out: dict[str, NDArray[np.float64]] | None = None
    info_out: dict[str, Any] | None = None
    target_arr = np.asarray(list(target), dtype=np.float64)
    for _ in range(max_steps):
        cur = np.asarray(env._data.mocap_pos[env._ee_mocap_id], dtype=np.float64)
        delta = target_arr - cur
        dist = float(np.linalg.norm(delta))
        if dist < tol:
            break
        unit = delta / max(dist, 1e-9)
        step_xyz = unit * min(env.MAX_LINEAR_STEP, dist)
        action = np.array(
            [
                step_xyz[0] / env.MAX_LINEAR_STEP,
                step_xyz[1] / env.MAX_LINEAR_STEP,
                step_xyz[2] / env.MAX_LINEAR_STEP,
                0.0,
                0.0,
                0.0,
                gripper,
            ],
            dtype=np.float64,
        )
        obs_out, _, terminated, _, info_out = env.step(action)
        if terminated:
            return obs_out, info_out
    # Trail with a single no-op (velocity zero) at the requested gripper
    # state so the caller always gets back a fresh obs/info even when the
    # main loop never iterated.
    final_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float64)
    obs_out, _, _, _, info_out = env.step(final_action)
    return obs_out, info_out


def _hold(
    env: TabletopStackEnv, gripper: float, n: int
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Run ``n`` no-op steps at the given gripper state — used for settling."""
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float64)
    obs: dict[str, NDArray[np.float64]] = {}
    info: dict[str, Any] = {}
    for _ in range(n):
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break
    return obs, info


class TestConstruction:
    def test_instantiates(self) -> None:
        env = TabletopStackEnv()
        try:
            assert env is not None
        finally:
            env.close()

    def test_rejects_nonpositive_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive"):
            TabletopStackEnv(max_steps=0)

    def test_rejects_nonpositive_n_substeps(self) -> None:
        with pytest.raises(ValueError, match="n_substeps must be positive"):
            TabletopStackEnv(n_substeps=0)

    def test_n_subtasks_classvar_is_three(self) -> None:
        # Both the class-level attribute and an instance read should
        # report ``3`` — the SubtaskMilestone contract pins
        # ``n_subtasks`` for the env lifetime.
        assert TabletopStackEnv.n_subtasks == 3
        env = TabletopStackEnv()
        try:
            assert env.n_subtasks == 3
        finally:
            env.close()

    def test_axis_names_is_empty(self) -> None:
        # Stub env publishes no perturbation axes — see module docstring.
        # ``frozenset()`` on the LHS dodges ruff SIM300's Yoda-condition
        # warning (the empty literal is the constant we're comparing
        # *to*, not the value under test).
        assert frozenset() == TabletopStackEnv.AXIS_NAMES
        assert frozenset() == TabletopStackEnv.VISUAL_ONLY_AXES


class TestSpaces:
    def test_action_space_matches_tabletop(self) -> None:
        env = TabletopStackEnv()
        try:
            assert isinstance(env.action_space, spaces.Box)
            assert env.action_space.shape == (7,)
            assert env.action_space.dtype == np.float64
            assert float(env.action_space.low.min()) == -1.0
            assert float(env.action_space.high.max()) == 1.0
        finally:
            env.close()

    def test_observation_space_keys_and_shapes(self) -> None:
        env = TabletopStackEnv()
        try:
            assert isinstance(env.observation_space, spaces.Dict)
            keys = set(env.observation_space.spaces.keys())
            assert keys == {
                "cube_a_pos",
                "cube_a_quat",
                "cube_b_pos",
                "cube_b_quat",
                "cube_c_pos",
                "cube_c_quat",
                "ee_pos",
                "gripper",
            }
            shapes = {
                "cube_a_pos": (3,),
                "cube_a_quat": (4,),
                "cube_b_pos": (3,),
                "cube_b_quat": (4,),
                "cube_c_pos": (3,),
                "cube_c_quat": (4,),
                "ee_pos": (3,),
                "gripper": (1,),
            }
            for key, shape in shapes.items():
                box = env.observation_space.spaces[key]
                assert isinstance(box, spaces.Box)
                assert box.shape == shape
                assert box.dtype == np.float64
        finally:
            env.close()


class TestReset:
    def test_reset_with_seed_is_deterministic(self) -> None:
        env_a = TabletopStackEnv()
        env_b = TabletopStackEnv()
        try:
            obs_a, _ = env_a.reset(seed=12345)
            obs_b, _ = env_b.reset(seed=12345)
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key])
        finally:
            env_a.close()
            env_b.close()

    def test_reset_returns_zero_subtask_completion(self) -> None:
        env = TabletopStackEnv()
        try:
            _, info = env.reset(seed=42)
            assert info["subtask_completion"] == [False, False, False]
            assert info["success"] is False
            assert info["grasped_cube_idx"] is None
        finally:
            env.close()

    def test_set_perturbation_always_raises(self) -> None:
        # Stub env publishes no axes; every queued name is rejected.
        env = TabletopStackEnv()
        try:
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                env.set_perturbation("lighting_intensity", 1.0)
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                env.set_perturbation("anything", 0.0)
        finally:
            env.close()


class TestStep:
    def test_step_returns_canonical_five_tuple(self) -> None:
        env = TabletopStackEnv()
        try:
            env.reset(seed=0)
            result = env.step(_zero_action())
            assert len(result) == 5
            obs, reward, terminated, truncated, info = result
            assert isinstance(obs, dict)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        finally:
            env.close()

    def test_step_rejects_wrong_action_shape(self) -> None:
        env = TabletopStackEnv()
        try:
            env.reset(seed=0)
            with pytest.raises(ValueError, match="action must have shape"):
                env.step(np.zeros(5, dtype=np.float64))
        finally:
            env.close()

    def test_subtask_completion_is_length_three(self) -> None:
        env = TabletopStackEnv()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(_zero_action())
            sub = info["subtask_completion"]
            assert isinstance(sub, list)
            assert len(sub) == 3
            assert all(isinstance(b, bool) for b in sub)
        finally:
            env.close()

    def test_subtask_completion_is_monotonic_under_noop(self) -> None:
        # Driving with zero actions cannot complete any milestone, so
        # the latch must remain ``[False, False, False]`` for the
        # entire (truncated) rollout. Run for 50 steps — well past
        # any plausible spurious-flip window.
        env = TabletopStackEnv()
        try:
            env.reset(seed=7)
            prev: list[bool] = [False, False, False]
            for _ in range(50):
                _, _, _, _, info = env.step(_zero_action())
                cur: list[bool] = info["subtask_completion"]
                # Monotonic non-decreasing over the rollout.
                for a, b in zip(prev, cur, strict=True):
                    assert (not a) or b, f"subtask_completion regressed: prev={prev} cur={cur}"
                prev = cur
            # Specifically: zero-action policies never latch any subtask.
            assert prev == [False, False, False]
        finally:
            env.close()


class TestScriptedMilestoneChain:
    """Canned scripted policy reaches all 3 milestones from scratch.

    The policy lives entirely in this test (no scripted-policy plumbing
    in the production env) so the env's milestone definitions and the
    test's geometric reasoning are both load-bearing. If this test
    ever drifts from the env, it points at a real bug.
    """

    def test_full_chain_reaches_success(self) -> None:
        env = TabletopStackEnv(max_steps=600)
        try:
            obs, _ = env.reset(seed=42)
            cube_a = obs["cube_a_pos"].copy()
            cube_b = obs["cube_b_pos"].copy()
            cube_c = obs["cube_c_pos"].copy()

            # ----- Subtask 0: place A on B.
            # 1. Hover above A.
            obs, info = _move_to(
                env,
                [cube_a[0], cube_a[1], 0.6],
                gripper=env.GRIPPER_OPEN,
            )
            # 2. Descend to A.
            obs, info = _move_to(
                env,
                [cube_a[0], cube_a[1], cube_a[2] + 0.005],
                gripper=env.GRIPPER_OPEN,
            )
            # 3. Close gripper (in place — no further travel).
            obs, info = _move_to(
                env,
                [cube_a[0], cube_a[1], cube_a[2] + 0.005],
                gripper=env.GRIPPER_CLOSED,
                max_steps=4,
            )
            assert info["grasped_cube_idx"] == 0, (
                f"expected to grasp cube A (idx 0); got {info['grasped_cube_idx']}"
            )
            # 4. Lift A.
            obs, info = _move_to(
                env,
                [cube_a[0], cube_a[1], 0.6],
                gripper=env.GRIPPER_CLOSED,
            )
            # 5. Carry to above B at stack height.
            obs, info = _move_to(
                env,
                [cube_b[0], cube_b[1], cube_b[2] + 0.05],
                gripper=env.GRIPPER_CLOSED,
            )
            # 6. Release and let A settle on B.
            obs, info = _hold(env, gripper=env.GRIPPER_OPEN, n=40)
            sub_after_first = info["subtask_completion"]
            assert sub_after_first[0] is True, f"expected subtask 0 (A on B); got {sub_after_first}"

            # ----- Subtask 1: pick C.
            obs, info = _move_to(
                env,
                [cube_c[0], cube_c[1], 0.6],
                gripper=env.GRIPPER_OPEN,
            )
            obs, info = _move_to(
                env,
                [cube_c[0], cube_c[1], cube_c[2] + 0.005],
                gripper=env.GRIPPER_OPEN,
            )
            obs, info = _move_to(
                env,
                [cube_c[0], cube_c[1], cube_c[2] + 0.005],
                gripper=env.GRIPPER_CLOSED,
                max_steps=4,
            )
            assert info["grasped_cube_idx"] == 2, (
                f"expected to grasp cube C (idx 2); got {info['grasped_cube_idx']}"
            )
            sub_after_grasp_c = info["subtask_completion"]
            assert sub_after_grasp_c[1] is True, (
                f"expected subtask 1 (C grasped); got {sub_after_grasp_c}"
            )

            # ----- Subtask 2: place C on top of (now-stacked) A.
            cube_a_now = obs["cube_a_pos"].copy()
            obs, info = _move_to(
                env,
                [cube_a_now[0], cube_a_now[1], 0.6],
                gripper=env.GRIPPER_CLOSED,
            )
            obs, info = _move_to(
                env,
                [cube_a_now[0], cube_a_now[1], cube_a_now[2] + 0.05],
                gripper=env.GRIPPER_CLOSED,
            )
            # Release and settle.
            obs, info = _hold(env, gripper=env.GRIPPER_OPEN, n=60)

            sub_final = info["subtask_completion"]
            assert sub_final == [True, True, True], f"expected all milestones; got {sub_final}"
            assert info["success"] is True
        finally:
            env.close()


class TestClose:
    def test_close_is_idempotent(self) -> None:
        env = TabletopStackEnv()
        env.reset(seed=0)
        env.close()
        # Second close must not raise.
        env.close()
