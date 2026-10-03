"""TabletopPushEnv (``env: tabletop-push``) and the scripted push controller.

Push shares the tabletop scene, action space and perturbation axes with
pick-and-place, but the cube can only be moved by contact: the gripper
never grasps, the end-effector sphere collides, and success needs the
cube to come to rest inside the target (sliding through it does not
count).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from gauntlet.env.registry import get_env_factory
from gauntlet.env.tabletop import TabletopEnv
from gauntlet.env.tabletop_push import TabletopPushEnv
from gauntlet.policy.registry import resolve_policy_factory
from gauntlet.policy.scripted import ScriptedPushPolicy


def _close_gripper_on_cube(env: TabletopEnv) -> None:
    """Teleport the EE onto the cube and command the gripper closed."""
    cube = np.array(env._data.xpos[env._cube_body_id])
    env._data.mocap_pos[env._ee_mocap_id] = cube
    action = np.zeros(7)
    action[6] = -1.0
    env.step(action)


def test_registered_under_tabletop_push() -> None:
    assert get_env_factory("tabletop-push") is cast(object, TabletopPushEnv)


def test_shares_tabletop_axes() -> None:
    assert TabletopPushEnv.AXIS_NAMES == TabletopEnv.AXIS_NAMES


def test_never_grasps() -> None:
    env = TabletopPushEnv()
    try:
        env.reset(seed=0)
        _close_gripper_on_cube(env)
        assert env._grasped is False
    finally:
        env.close()


def test_pick_and_place_env_still_grasps() -> None:
    env = TabletopEnv()
    try:
        env.reset(seed=0)
        _close_gripper_on_cube(env)
        assert env._grasped is True
    finally:
        env.close()


def test_end_effector_pushes_the_cube() -> None:
    env = TabletopPushEnv()
    try:
        obs, _ = env.reset(seed=0)
        start = obs["cube_pos"].copy()
        # Park the EE beside the cube at cube height, then sweep +x.
        env._data.mocap_pos[env._ee_mocap_id] = start + np.array([-0.07, 0.0, 0.0])
        for _ in range(8):
            obs, *_ = env.step(np.array([0.5, 0, 0, 0, 0, 0, 1.0]))
        assert obs["cube_pos"][0] > start[0] + 0.03
        # A hard shove can tip the cube onto an edge mid-push; once the
        # EE stops it settles flat on the table again.
        for _ in range(10):
            obs, *_ = env.step(np.zeros(7))
        assert obs["cube_pos"][2] == pytest.approx(TabletopEnv._CUBE_REST_Z, abs=0.005)
    finally:
        env.close()


def test_success_requires_cube_at_rest_in_target() -> None:
    env = TabletopPushEnv()
    try:
        env.reset(seed=0)
        target = env._target_pos
        adr, vadr = env._cube_qpos_adr, env._cube_qvel_adr
        # Inside the target but sliding fast: not a success.
        env._data.qpos[adr : adr + 2] = target[:2]
        env._data.qvel[vadr : vadr + 3] = [0.5, 0.0, 0.0]
        env._data.mocap_pos[env._ee_mocap_id] = target + np.array([0.0, 0.0, 0.3])
        _, _, terminated, _, info = env.step(np.zeros(7))
        assert not info["success"]
        assert not terminated
        # Inside the target and at rest: success.
        env._data.qpos[adr : adr + 3] = [target[0], target[1], TabletopEnv._CUBE_REST_Z]
        env._data.qvel[vadr : vadr + 6] = 0.0
        _, _, terminated, _, info = env.step(np.zeros(7))
        assert info["success"]
        assert terminated
    finally:
        env.close()


def test_pushing_contact_is_not_a_collision() -> None:
    env = TabletopPushEnv()
    try:
        obs, _ = env.reset(seed=0)
        env._data.mocap_pos[env._ee_mocap_id] = obs["cube_pos"] + np.array([-0.07, 0.0, 0.0])
        collisions = 0
        near = 0
        for _ in range(8):
            _, _, _, _, info = env.step(np.array([0.5, 0, 0, 0, 0, 0, 1.0]))
            collisions += info["safety_n_collisions_delta"]
            near += info["behavior_near_collision_delta"]
        assert collisions == 0
        assert near == 0
    finally:
        env.close()


@pytest.mark.parametrize("seed", range(10))
def test_scripted_push_policy_solves_baseline(seed: int) -> None:
    env = TabletopPushEnv(max_steps=150)
    policy = ScriptedPushPolicy()
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset(np.random.default_rng(seed))
        info: dict[str, object] = {}
        for _ in range(150):
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if terminated or truncated:
                break
        assert info["success"] is True
    finally:
        env.close()


def test_scripted_push_is_a_builtin_policy_spec() -> None:
    policy = resolve_policy_factory("scripted-push")()
    assert isinstance(policy, ScriptedPushPolicy)


def test_gymnasium_id() -> None:
    import gymnasium as gym

    import gauntlet  # noqa: F401  (registers the gym ids)

    env = gym.make("gauntlet/TabletopPush-v0")
    try:
        assert isinstance(env.unwrapped, TabletopPushEnv)
    finally:
        env.close()
