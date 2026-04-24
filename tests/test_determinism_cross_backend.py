"""Cross-backend space-parity audit — Phase 2.5 Task 17 cases 2 + 3.

Each backend RFC promises that its env exposes the SAME 5-key state
observation_space and the SAME 7-D action_space as the MuJoCo
``TabletopEnv`` reference, modulo numerics. This file is the
single, default-job place that pins that promise.

* **Case 2 — state-shape parity.** ``tabletop``, ``tabletop-pybullet``,
  ``tabletop-genesis`` (and ``tabletop-isaac``, exercised under its
  own mocked conftest in ``tests/isaac/test_determinism_isaac.py``)
  all expose the SAME 5-key state ``observation_space`` (same keys,
  same dtype, same shape, same low/high) when ``render_in_obs=False``.
  Pixel parity is explicitly NOT asserted (RFC-007 §7.3, RFC-008 §8 Q2).
* **Case 3 — action-space parity.** Same 7-D EE-twist+gripper Box
  across all four backends. Same dtype, same bounds.

Backends whose extras are not installed in the active env are
``importorskip``'d cleanly so the default-job invocation stays green
on a torch-free / pybullet-free / genesis-free worktree. CI ships
each extra in its own job; the default job picks up MuJoCo only.
Isaac is covered in ``tests/isaac/test_determinism_isaac.py`` which
has the autouse ``sys.modules`` fake injection.
"""

from __future__ import annotations

from typing import Any

import pytest
from gymnasium.spaces import Box, Dict

from gauntlet.env.tabletop import TabletopEnv

# Canonical 5-key state-obs schema every backend must expose at
# ``render_in_obs=False``. Pinned here so a future backend adding a
# 6th key has to either justify it in an RFC or skip this test.
_EXPECTED_STATE_OBS_KEYS: frozenset[str] = frozenset(
    {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
)


def _state_only_env(env_cls: type[Any]) -> Any:
    """Construct ``env_cls`` in state-only mode.

    All backends accept ``render_in_obs=False`` as a kwarg except
    Isaac (state-only first cut, no rendering kwarg). Isaac is not
    exercised here — see module docstring.
    """
    return env_cls(render_in_obs=False)


def _assert_action_space_matches_mujoco(other: Any, label: str) -> None:
    """Action space byte-for-byte identical to MuJoCo's reference."""
    mj = TabletopEnv()
    try:
        a_other = other.action_space
        a_mj = mj.action_space
        assert isinstance(a_other, Box), f"{label}: action_space is not a Box"
        assert isinstance(a_mj, Box)
        # Box.__eq__ compares low + high + shape + dtype. Use it as the
        # primary check; the per-field asserts below give a clearer
        # failure message when the equality fails.
        assert a_other == a_mj, (
            f"{label} action_space != MuJoCo: {label}={a_other!r} vs mujoco={a_mj!r}"
        )
        assert a_other.shape == (7,)
        assert a_other.dtype == a_mj.dtype
        assert float(a_other.low.min()) == -1.0
        assert float(a_other.high.max()) == 1.0
    finally:
        mj.close()


def _assert_state_obs_space_matches_mujoco(other: Any, label: str) -> None:
    """Observation space (5-key state Dict) byte-for-byte identical to MuJoCo."""
    mj = TabletopEnv()
    try:
        os_other = other.observation_space
        os_mj = mj.observation_space
        assert isinstance(os_other, Dict), f"{label}: observation_space is not a Dict"
        assert isinstance(os_mj, Dict)
        # Same key set first (more useful failure than a Box mismatch
        # if a backend silently adds a sixth key).
        keys_other = set(os_other.spaces.keys())
        keys_mj = set(os_mj.spaces.keys())
        assert keys_other == _EXPECTED_STATE_OBS_KEYS, (
            f"{label}: state-obs keys {keys_other} != expected {_EXPECTED_STATE_OBS_KEYS}"
        )
        assert keys_other == keys_mj
        for key in _EXPECTED_STATE_OBS_KEYS:
            sub_other = os_other.spaces[key]
            sub_mj = os_mj.spaces[key]
            assert isinstance(sub_other, Box), f"{label}: {key!r} sub-space is not a Box"
            assert isinstance(sub_mj, Box)
            assert sub_other == sub_mj, (
                f"{label}: {key!r} sub-space differs from MuJoCo: "
                f"{label}={sub_other!r} vs mujoco={sub_mj!r}"
            )
    finally:
        mj.close()


# -------------------------------------------------------------- MuJoCo (self-check)


def test_mujoco_state_obs_keys_pinned() -> None:
    """Self-check: the canonical key set IS the MuJoCo reference's
    key set. Catches a future MJCF addition that would silently
    drift the 5-key contract.
    """
    mj = TabletopEnv()
    try:
        os_mj = mj.observation_space
        assert isinstance(os_mj, Dict)
        assert set(os_mj.spaces.keys()) == _EXPECTED_STATE_OBS_KEYS
    finally:
        mj.close()


# -------------------------------------------------------------- PyBullet


def test_pybullet_state_obs_space_matches_mujoco() -> None:
    """RFC-005 §7.2 — state-only ``observation_space`` identical to MuJoCo."""
    pytest.importorskip(
        "pybullet",
        reason="PyBullet extra not installed (uv sync --extra pybullet)",
    )
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    pb = _state_only_env(PyBulletTabletopEnv)
    try:
        _assert_state_obs_space_matches_mujoco(pb, label="tabletop-pybullet")
    finally:
        pb.close()


def test_pybullet_action_space_matches_mujoco() -> None:
    """RFC-005 §7.2 — 7-D action Box identical to MuJoCo."""
    pytest.importorskip(
        "pybullet",
        reason="PyBullet extra not installed (uv sync --extra pybullet)",
    )
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    pb = _state_only_env(PyBulletTabletopEnv)
    try:
        _assert_action_space_matches_mujoco(pb, label="tabletop-pybullet")
    finally:
        pb.close()


# -------------------------------------------------------------- Genesis


def test_genesis_state_obs_space_matches_mujoco() -> None:
    """RFC-007 §3 — state-only ``observation_space`` identical to MuJoCo.

    The Genesis adapter spends ~5 s on first-time ``gs.init`` + scene
    build. Acceptable for a single-test boot in the genesis CI job.
    """
    pytest.importorskip(
        "genesis",
        reason="Genesis extra not installed (uv sync --extra genesis)",
    )
    from gauntlet.env.genesis import GenesisTabletopEnv

    g = _state_only_env(GenesisTabletopEnv)
    try:
        _assert_state_obs_space_matches_mujoco(g, label="tabletop-genesis")
    finally:
        g.close()


def test_genesis_action_space_matches_mujoco() -> None:
    """RFC-007 §3 — 7-D action Box identical to MuJoCo."""
    pytest.importorskip(
        "genesis",
        reason="Genesis extra not installed (uv sync --extra genesis)",
    )
    from gauntlet.env.genesis import GenesisTabletopEnv

    g = _state_only_env(GenesisTabletopEnv)
    try:
        _assert_action_space_matches_mujoco(g, label="tabletop-genesis")
    finally:
        g.close()


# -------------------------------------------------------------- canonical-key pin


def test_expected_state_obs_keys_are_exactly_five() -> None:
    """Lock the 5-key contract so a future addition is loud.

    A 6th key would force every backend adapter to add a matching
    ``observation_space`` entry — the test's job is to be loud, not
    to wave through silent additions.
    """
    expected = frozenset({"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"})
    assert len(_EXPECTED_STATE_OBS_KEYS) == 5
    # Two named locals; SIM300 fires on either ordering against an
    # ALL_CAPS constant. Suppress — both sides are real symbols.
    assert _EXPECTED_STATE_OBS_KEYS == expected  # noqa: SIM300
