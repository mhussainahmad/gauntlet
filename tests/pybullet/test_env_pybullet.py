"""PyBullet backend tests — step-7 subset (RFC-005 §9.1).

Only the import / registration round-trip tests live here today. The
full per-axis + determinism + rollout battery (RFC-005 §9.1 cases 1 to 11)
lands with steps 10 and 11 of the RFC-005 §13 checklist, once the real
backend body replaces the step-7 stub.

All tests are marked ``@pytest.mark.pybullet`` and de-selected from the
default ``pytest`` run by ``pyproject.toml``. Install the extra + run:

    uv sync --extra pybullet --group pybullet-dev
    uv run pytest -m pybullet -q
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.pybullet


def test_subpackage_import_registers_tabletop_pybullet() -> None:
    """RFC-005 §9.1 case 2 (registry round-trip). Importing the
    subpackage must register ``"tabletop-pybullet"`` and route a direct
    factory lookup back to :class:`PyBulletTabletopEnv`.
    """
    import gauntlet.env.pybullet  # noqa: F401 — trigger registration
    from gauntlet.env.pybullet import PyBulletTabletopEnv
    from gauntlet.env.registry import get_env_factory, registered_envs

    assert "tabletop-pybullet" in registered_envs()
    # The factory is wrapped in typing.cast at the registration site;
    # identity survives the cast.
    assert get_env_factory("tabletop-pybullet") is PyBulletTabletopEnv


def test_pybullet_backend_satisfies_gauntlet_env_protocol() -> None:
    """RFC-005 §9.1 case 1 — runtime Protocol conformance.

    Structural check — ``runtime_checkable`` + matching attrs. The stub
    already declares ``AXIS_NAMES``, ``VISUAL_ONLY_AXES``, the spaces,
    and the five Protocol methods, so this passes even before step 9
    fills in the bodies. Regression guard for future refactors.
    """
    from gauntlet.env.base import GauntletEnv
    from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

    env = PyBulletTabletopEnv()
    try:
        assert isinstance(env, GauntletEnv)
    finally:
        env.close()


def test_pybullet_backend_declares_visual_only_axes() -> None:
    """Pins the step-5 loader-rejection contract (RFC-005 §6.2).

    The PyBullet backend's ``VISUAL_ONLY_AXES`` must be exactly
    ``{"lighting_intensity", "object_texture"}`` — these two axes
    mutate the scene but cannot change a state-only observation, so
    the Suite loader rejects suites that vary only these axes at
    load-time (step 12 wires the check in).
    """
    from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

    expected_visual = frozenset({"lighting_intensity", "object_texture"})
    expected_all = frozenset(
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
    assert expected_visual == PyBulletTabletopEnv.VISUAL_ONLY_AXES
    assert expected_all == PyBulletTabletopEnv.AXIS_NAMES
