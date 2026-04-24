"""Isaac Sim adapter contract tests — RFC-009 §8.

Mock-driven (the conftest fixture in this directory injects a fake
``isaacsim`` / ``omni.isaac.core`` namespace into ``sys.modules``
before each test). These tests run in the default torch-free CI
job; the ``isaac`` pytest marker is reserved for the future
real-runtime tests gated to a GPU runner.

Test matrix lives in RFC-009 §8 — this file groups by:

* Protocol conformance + spaces / axis-name parity with Genesis.
* Reset-seed determinism.
* Step contract (return-tuple shape + types).
* Per-axis branches land in step 6's test file
  (``test_perturbation_isaac.py``); this file only covers the
  baseline state-only contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv

if TYPE_CHECKING:
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv


# --------------------------------------------------------- protocol / registry


def test_protocol_conformance() -> None:
    """RFC-009 §8 — `isinstance` against the runtime-checkable Protocol.

    The mocked env exposes the right attribute / method shape, so
    `isinstance(env, GauntletEnv)` passes.  Important guard against
    accidentally dropping a Protocol member from the adapter.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        assert isinstance(env, GauntletEnv)
    finally:
        env.close()


def test_subpackage_import_registers_tabletop_isaac() -> None:
    """Importing `gauntlet.env.isaac` registers the `tabletop-isaac`
    key in the registry and routes lookups back to
    :class:`IsaacSimTabletopEnv`.
    """
    import gauntlet.env.isaac  # noqa: F401 — trigger registration
    from gauntlet.env.isaac import IsaacSimTabletopEnv
    from gauntlet.env.registry import get_env_factory, registered_envs

    assert "tabletop-isaac" in registered_envs()
    factory = get_env_factory("tabletop-isaac")
    assert factory is IsaacSimTabletopEnv


# -------------------------------------------------------- spaces / axis names


def test_action_space_matches_genesis_state_only() -> None:
    """Same 7-D action Box as every other backend (RFC-009 §3)."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        action = env.action_space
        assert isinstance(action, Box)
        assert action.shape == (7,)
        assert action.dtype == np.float64
        assert float(action.low[0]) == -1.0
        assert float(action.high[0]) == 1.0
    finally:
        env.close()


def test_observation_space_keys_match_genesis_state_only() -> None:
    """Same five state-obs keys as `GenesisTabletopEnv(render_in_obs=False)`.

    The exact dict is `{cube_pos, cube_quat, ee_pos, gripper, target_pos}`
    — RFC-009 §3 / RFC-007 §3. `step` lives on `info`, not obs.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        obs_space = env.observation_space
        assert isinstance(obs_space, Dict)
        expected = {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
        assert set(obs_space.spaces.keys()) == expected
        assert obs_space.spaces["cube_pos"].shape == (3,)
        assert obs_space.spaces["cube_quat"].shape == (4,)
        assert obs_space.spaces["ee_pos"].shape == (3,)
        assert obs_space.spaces["gripper"].shape == (1,)
        assert obs_space.spaces["target_pos"].shape == (3,)
    finally:
        env.close()


def test_axis_names_are_canonical_seven() -> None:
    """`AXIS_NAMES` matches the canonical 7 (RFC-009 §6.7)."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    expected = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
        }
    )
    assert expected == IsaacSimTabletopEnv.AXIS_NAMES


def test_visual_only_axes_are_four_cosmetic() -> None:
    """State-only first cut declares the four cosmetic axes
    `VISUAL_ONLY_AXES` (RFC-009 §6.6). The Suite loader's
    `_reject_purely_visual_suites` guard fires off this set; tests
    pinning the rejection live in `test_suite_loader_isaac.py`.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    expected = frozenset(
        {"lighting_intensity", "camera_offset_x", "camera_offset_y", "object_texture"}
    )
    assert expected == IsaacSimTabletopEnv.VISUAL_ONLY_AXES


# ----------------------------------------------------- reset / step / determinism


def _stack_obs(
    env: IsaacSimTabletopEnv,
    seed: int,
    n_steps: int,
    action: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Roll out n steps from a fresh reset and return keyed stacks."""
    obs, _ = env.reset(seed=seed)
    frames: dict[str, list[NDArray[np.float64]]] = {k: [v.copy()] for k, v in obs.items()}
    for _ in range(n_steps):
        obs, *_ = env.step(action)
        for k, v in obs.items():
            frames[k].append(v.copy())
    return {k: np.stack(v) for k, v in frames.items()}


def test_reset_seed_determinism_under_mock() -> None:
    """RFC-009 §7.1 — two `env.reset(seed=42)` + 5 steps under the
    same action sequence produce byte-identical obs (the fake never
    introduces non-determinism).

    Constructs two independent envs so the test isolates the
    seed-driven RNG path from any per-instance state pollution.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    a = IsaacSimTabletopEnv()
    b = IsaacSimTabletopEnv()
    try:
        action = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        sa = _stack_obs(a, seed=7, n_steps=5, action=action)
        sb = _stack_obs(b, seed=7, n_steps=5, action=action)
        for k in sa:
            assert np.array_equal(sa[k], sb[k]), f"obs['{k}'] diverged across runs under the mock"
    finally:
        a.close()
        b.close()


def test_reset_returns_expected_shapes_and_dtypes() -> None:
    """Reset's obs dict matches the declared observation_space."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        obs, info = env.reset(seed=42)
        assert obs["cube_pos"].shape == (3,)
        assert obs["cube_pos"].dtype == np.float64
        assert obs["cube_quat"].shape == (4,)
        assert obs["cube_quat"].dtype == np.float64
        assert obs["ee_pos"].shape == (3,)
        assert obs["gripper"].shape == (1,)
        assert obs["target_pos"].shape == (3,)
        # info carries the step counter + grasp + success flags
        # (parity with Genesis).
        assert {"success", "grasped", "step"} <= set(info)
        assert info["step"] == 0
        assert info["success"] is False
        assert info["grasped"] is False
    finally:
        env.close()


def test_reset_seeds_cube_xy_within_init_halfrange() -> None:
    """Seeded reset places the cube within `[-CUBE_INIT_HALFRANGE,
    CUBE_INIT_HALFRANGE]` on each XY axis. Same property other backends
    test.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        for seed in (0, 1, 42, 12345):
            obs, _ = env.reset(seed=seed)
            assert -0.15 <= float(obs["cube_pos"][0]) <= 0.15
            assert -0.15 <= float(obs["cube_pos"][1]) <= 0.15
    finally:
        env.close()


def test_step_returns_5_tuple_with_correct_python_types() -> None:
    """`step` matches the `Env` Protocol's
    `(obs, reward: float, terminated: bool, truncated: bool, info: dict)`.
    """
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv(max_steps=3)
    try:
        env.reset(seed=0)
        action = np.zeros(7, dtype=np.float64)
        action[6] = 1.0  # gripper open
        obs, reward, terminated, truncated, info = env.step(action)

        # obs is the full state dict.
        assert set(obs.keys()) == {
            "cube_pos",
            "cube_quat",
            "ee_pos",
            "gripper",
            "target_pos",
        }
        # reward / flags / info typed exactly as the Protocol declares.
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert info["step"] == 1
    finally:
        env.close()


def test_step_advances_ee_in_x_by_max_linear_step() -> None:
    """Action `[+1, 0, 0, 0, 0, 0, 1]` moves EE by +MAX_LINEAR_STEP in x
    (kinematic teleport, no contact dynamics)."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    try:
        obs, _ = env.reset(seed=0)
        ee0 = obs["ee_pos"].copy()
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        obs2, *_ = env.step(action)
        delta_x = float(obs2["ee_pos"][0] - ee0[0])
        # Pure-kinematic EE — exact up to fp noise.
        assert abs(delta_x - 0.05) < 1e-9, f"expected ~+0.05, got {delta_x}"
    finally:
        env.close()


def test_step_truncates_at_max_steps() -> None:
    """`max_steps=3` -> step 3 returns `truncated=True` if not already
    `terminated`. Confirms the rollout-length cap matches MuJoCo /
    PyBullet / Genesis."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv(max_steps=3)
    try:
        env.reset(seed=0)
        # Move EE far from cube so success doesn't trigger first.
        action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        for i in range(3):
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if i < 2:
                assert truncated is False
            else:
                # On step 3 (i == 2), step count hits max_steps.
                # success is fp-distance-dependent so we only assert
                # that EITHER terminated or truncated is set; in
                # practice terminated stays False because the cube is
                # nowhere near the target after pure-vertical EE
                # motion.
                assert terminated or truncated
    finally:
        env.close()


# --------------------------------------------------- close idempotence


def test_close_is_idempotent() -> None:
    """`close()` twice doesn't raise."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    env = IsaacSimTabletopEnv()
    env.close()
    env.close()  # must be a no-op the second time


def test_invalid_max_steps_rejected() -> None:
    """`max_steps <= 0` -> ValueError. Same as Genesis."""
    from gauntlet.env.isaac.tabletop_isaac import IsaacSimTabletopEnv

    with pytest.raises(ValueError, match="max_steps must be positive"):
        IsaacSimTabletopEnv(max_steps=0)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        IsaacSimTabletopEnv(max_steps=-1)
