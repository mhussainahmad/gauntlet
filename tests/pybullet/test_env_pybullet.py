"""PyBullet backend tests — import + registration + rollout smoke.

Covers RFC-005 §9.1 cases 1, 2, 9, 10 — plus a shared asset-resolution
test. The determinism battery (cases 3-6) lives in
tests/pybullet/test_determinism_pybullet.py; the per-axis table
(cases 7, 8) lives in tests/pybullet/test_perturbation_pybullet.py.
Case 12 (missing-extra ImportError) lives in tests/test_suite.py to
avoid pulling the ``@pytest.mark.pybullet`` gate on a test whose
premise is "pybullet is not installed."

All tests are marked ``@pytest.mark.pybullet`` and de-selected from the
default ``pytest`` run by ``pyproject.toml``. Install the extra + run:

    uv sync --extra pybullet --group pybullet-dev
    uv run pytest -m pybullet -q
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from gauntlet.env.base import GauntletEnv

if TYPE_CHECKING:
    from gauntlet.policy.base import Policy

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
    """Pins the RFC-006-updated contract plus the B-06 / B-42 axes.

    RFC-006 made every cosmetic axis observable on ``obs["image"]`` so
    the four cosmetic axes are no longer in ``VISUAL_ONLY_AXES``.
    ``object_swap`` (B-06) is the lone non-empty entry — PyBullet
    ships no alternate asset library. ``camera_extrinsics`` (B-42) is
    a real impl on the single-cam render path so it is NOT in
    ``VISUAL_ONLY_AXES``.
    """
    from gauntlet.env.pybullet.tabletop_pybullet import PyBulletTabletopEnv

    expected_all = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
            "object_swap",
            "camera_extrinsics",
        }
    )
    assert frozenset({"object_swap"}) == PyBulletTabletopEnv.VISUAL_ONLY_AXES
    assert expected_all == PyBulletTabletopEnv.AXIS_NAMES


def make_fast_pybullet_env() -> GauntletEnv:
    """Module-level factory for Runner.run (pickle-friendly under spawn).

    Matches the ``make_fast_env`` pattern in tests/test_runner.py — short
    max_steps keeps case-10 runtime under a second on CI.
    """
    from gauntlet.env.pybullet import PyBulletTabletopEnv

    return cast(GauntletEnv, PyBulletTabletopEnv(max_steps=10))


def make_random_policy() -> Policy:
    """Module-level policy factory."""
    from gauntlet.policy.random import RandomPolicy

    return RandomPolicy(action_dim=7, seed=None)


def test_ten_rollouts_with_random_policy_no_crash_no_nan() -> None:
    """RFC-005 §9.1 case 9 — 10 rollouts with RandomPolicy on seeds 0..9.

    No crashes, no NaN/Inf in observations, cube_pos[2] stays roughly
    on the table (catches the "cube fell through the floor via bad
    collision mask" failure mode).
    """
    import numpy as np

    from gauntlet.env.pybullet import PyBulletTabletopEnv
    from gauntlet.policy.random import RandomPolicy

    # Semi-loose floor check — the table top is at z = 0.42; below
    # 0.40 means the cube escaped through the table.
    _FLOOR_CHECK = 0.40

    for seed in range(10):
        env = PyBulletTabletopEnv(max_steps=20)
        try:
            policy = RandomPolicy(action_dim=7, seed=seed)
            obs, _ = env.reset(seed=seed)
            # Up to 20 steps — break early on terminate/truncate.
            for _ in range(20):
                action = policy.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                for k, v in obs.items():
                    arr = np.asarray(v, dtype=np.float64)
                    assert np.all(np.isfinite(arr)), f"seed {seed}: {k!r} not finite"
                cube_z = float(np.asarray(obs["cube_pos"], dtype=np.float64)[2])
                assert cube_z >= _FLOOR_CHECK, (
                    f"seed {seed}: cube fell through table (z={cube_z:.3f})"
                )
                if terminated or truncated:
                    break
        finally:
            env.close()


def test_runner_integration_tabletop_pybullet_baseline_sweep() -> None:
    """RFC-005 §9.1 case 10 — a small suite (1 axis, 2 steps, 2 eps)
    drives end-to-end through :class:`gauntlet.runner.Runner` using the
    PyBullet backend via the Protocol seam; Episodes pass the Episode
    schema validator.
    """
    from gauntlet.runner import Runner
    from gauntlet.runner.episode import Episode
    from gauntlet.suite.schema import AxisSpec, Suite

    suite = Suite.model_validate(
        {
            "name": "pybullet-runner-smoke",
            "env": "tabletop-pybullet",
            "episodes_per_cell": 2,
            "seed": 99,
            "axes": {
                "object_initial_pose_x": {
                    "low": -0.05,
                    "high": 0.05,
                    "steps": 2,
                },
            },
        }
    )
    # 2 steps x 2 episodes_per_cell = 4 rollouts.
    runner = Runner(n_workers=1, env_factory=make_fast_pybullet_env)
    episodes = runner.run(
        policy_factory=make_random_policy,
        suite=suite,
    )

    assert len(episodes) == 4
    # Every return value validates as a real Episode (not a subclass).
    for ep in episodes:
        assert isinstance(ep, Episode)
        Episode.model_validate(ep.model_dump())
        # perturbation_config carries the cell's axis value.
        assert "object_initial_pose_x" in ep.perturbation_config
        # perturbation_config values are within the sweep bounds.
        val = ep.perturbation_config["object_initial_pose_x"]
        assert AxisSpec(low=-0.05, high=0.05, steps=2).low is not None
        assert -0.05 - 1e-9 <= val <= 0.05 + 1e-9

    # Episodes from two distinct cells: two unique axis values.
    unique_vals = {ep.perturbation_config["object_initial_pose_x"] for ep in episodes}
    assert len(unique_vals) == 2


def test_texture_assets_resolve_via_importlib_resources() -> None:
    """RFC-005 §5.1 / §6.1 / §12 Q6 — the two texture PNGs ship inside
    the wheel under ``gauntlet/env/pybullet/assets/``.

    Using :func:`importlib.resources.files` (not ``Path(__file__)``)
    keeps the resolution wheel-portable: the assets load whether the
    package is running from the repo, a venv install, or a zipimport.
    """
    from importlib.resources import files

    assets = files("gauntlet.env.pybullet") / "assets"
    default_tex = assets / "cube_default.png"
    alt_tex = assets / "cube_alt.png"

    assert default_tex.is_file()
    assert alt_tex.is_file()

    # Cheap sanity — first 8 bytes are the PNG signature.
    assert default_tex.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert alt_tex.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # The two textures must be distinct — step-9 object_texture swap
    # relies on them producing visibly different cube colours.
    assert default_tex.read_bytes() != alt_tex.read_bytes()
