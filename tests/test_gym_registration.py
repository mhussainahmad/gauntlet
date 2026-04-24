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

import subprocess
import sys

import gymnasium as gym

from gauntlet.env.gym_registration import (
    _DEFAULT_MAX_EPISODE_STEPS,
    register_envs,
)
from gauntlet.env.tabletop import TabletopEnv

_TABLETOP_ID: str = "gauntlet/Tabletop-v0"
_PYBULLET_ID: str = "gauntlet/TabletopPyBullet-v0"
_GENESIS_ID: str = "gauntlet/TabletopGenesis-v0"
_ISAAC_ID: str = "gauntlet/TabletopIsaac-v0"

_ALL_IDS: tuple[str, ...] = (_TABLETOP_ID, _PYBULLET_ID, _GENESIS_ID, _ISAAC_ID)
_HEAVY_BACKEND_MODULES: tuple[str, ...] = (
    "gauntlet.env.pybullet",
    "gauntlet.env.genesis",
    "gauntlet.env.isaac",
)


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


def test_all_four_ids_registered() -> None:
    """Every shipped backend has a ``gauntlet/Tabletop<...>-v0`` id."""
    register_envs()
    for env_id in _ALL_IDS:
        assert env_id in gym.envs.registry, f"{env_id} missing from gym registry"


def test_heavy_backend_specs_use_string_entry_points() -> None:
    """String entry_points are required for lazy resolution of the heavy extras.

    A direct class reference would force the optional-extra import at
    registration time and re-introduce the ImportError that the lazy
    pattern exists to avoid.
    """
    register_envs()
    for env_id in (_PYBULLET_ID, _GENESIS_ID, _ISAAC_ID):
        spec = gym.envs.registry[env_id]
        assert isinstance(spec.entry_point, str), (
            f"{env_id} entry_point is {type(spec.entry_point)!r}; must be a string for lazy import."
        )
        # The string must point at the real module:class — sanity check
        # that the colon-separator convention is honoured.
        assert ":" in spec.entry_point


def test_heavy_backend_modules_not_imported_by_register_envs() -> None:
    """``register_envs()`` must NOT pull pybullet / genesis / isaac into sys.modules.

    Run in a clean subprocess so an earlier collection-time import (or a
    prior test in the same session) cannot mask a regression. The check
    is the whole point of the string-entry_point design — if any of the
    three heavy subpackages appears in ``sys.modules`` after
    ``import gauntlet``, a user without the corresponding extra will hit
    an ImportError just from importing the package.
    """
    script = (
        "import sys\n"
        "import gauntlet  # noqa: F401\n"
        "from gauntlet.env.gym_registration import register_envs\n"
        "register_envs()\n"
        "leaked = [m for m in "
        f"{list(_HEAVY_BACKEND_MODULES)!r}"
        " if m in sys.modules]\n"
        "if leaked:\n"
        "    print('LEAKED:' + ','.join(leaked))\n"
        "    sys.exit(1)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_bare_import_gauntlet_registers_all_ids() -> None:
    """A bare ``import gauntlet`` registers every shipped backend.

    Run in a clean subprocess: the in-session module cache already has
    ``gauntlet`` loaded from the test-collection step, so an in-process
    assertion would tautologically pass. The subprocess proves the
    registration side effect actually fires from a cold ``import``.
    """
    script = (
        "import gymnasium as gym\n"
        "import gauntlet  # noqa: F401\n"
        f"ids = {list(_ALL_IDS)!r}\n"
        "missing = [env_id for env_id in ids if env_id not in gym.envs.registry]\n"
        "if missing:\n"
        "    print('MISSING:' + ','.join(missing))\n"
        "    raise SystemExit(1)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_register_envs_reexported_from_top_level() -> None:
    """``from gauntlet import register_envs`` is the public surface."""
    import gauntlet

    assert hasattr(gauntlet, "register_envs")
    assert "register_envs" in gauntlet.__all__
