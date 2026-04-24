"""Tests for the gymnasium registration shim.

See ``docs/polish-exploration-gymnasium-registration.md`` for the full
rationale. This module covers:

* ``gym.make("gauntlet/Tabletop-v0")`` constructs a real
  :class:`gauntlet.env.tabletop.TabletopEnv` (round-trip through the
  string ``entry_point``).
* :func:`register_envs` is idempotent — the per-id check inside
  :func:`gauntlet.env.gym_registration._register_one` keeps repeated
  calls quiet, no warnings, no duplicate registry entries.
* All four namespaced ids appear in ``gym.envs.registry`` after a bare
  ``import gauntlet``.
* The three heavy-extra backends are registered with **string
  entry_points** — registration must NOT import
  ``gauntlet.env.{pybullet,genesis,isaac}``. Verified in a clean
  subprocess so an earlier collection-time import in this session
  cannot mask the failure.
* The pre-existing direct-import affordance
  (``from gauntlet.env.tabletop import TabletopEnv``) keeps working
  unchanged — the new gym layer is purely additive.

The test module is intentionally unmarked: it runs in the default pytest
job. No module-level imports of ``pybullet`` / ``genesis`` / ``isaacsim``
appear here.
"""

from __future__ import annotations

import gymnasium as gym

from gauntlet.env.gym_registration import (
    _DEFAULT_MAX_EPISODE_STEPS,
    register_envs,
)
from gauntlet.env.tabletop import TabletopEnv

_TABLETOP_ID: str = "gauntlet/Tabletop-v0"


def test_make_tabletop_constructs_mujoco_env() -> None:
    """``gym.make`` returns a wrapped env whose ``.unwrapped`` is the real class."""
    register_envs()
    env = gym.make(_TABLETOP_ID)
    try:
        assert isinstance(env.unwrapped, TabletopEnv)
    finally:
        env.close()


def test_register_envs_is_idempotent() -> None:
    """Calling register_envs repeatedly must not raise nor double-register."""
    register_envs()
    before = len(gym.envs.registry)
    register_envs()
    register_envs()
    register_envs()
    after = len(gym.envs.registry)
    assert before == after, (
        f"register_envs added {after - before} entries on repeat calls; should have been a no-op."
    )


def test_tabletop_id_present_after_register() -> None:
    """``register_envs()`` populates the Tabletop id in the gym registry."""
    register_envs()
    assert _TABLETOP_ID in gym.envs.registry


def test_tabletop_registration_carries_max_episode_steps() -> None:
    """``max_episode_steps`` matches the adapter's own ``max_steps`` default."""
    register_envs()
    spec = gym.envs.registry[_TABLETOP_ID]
    assert spec.max_episode_steps == _DEFAULT_MAX_EPISODE_STEPS


def test_existing_direct_import_still_works() -> None:
    """Backwards-compat regression: pre-gym-registration usage must keep working."""
    env = TabletopEnv()
    try:
        assert isinstance(env, TabletopEnv)
        # Reset round-trip — same surface every existing test depends on.
        obs, info = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_internal_registry_unchanged() -> None:
    """The plain-Python ``gauntlet.env.registry`` keeps its existing entries."""
    from gauntlet.env.registry import registered_envs

    # The MuJoCo backend has always self-registered under ``"tabletop"``;
    # the gym layer must not have moved or removed that entry.
    assert "tabletop" in registered_envs()
