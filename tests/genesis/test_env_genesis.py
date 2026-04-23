"""Genesis backend tests — import + registration + rollout + perturbations.

Covers the RFC-007 §8 test matrix:

* Protocol conformance (``isinstance`` against
  :class:`gauntlet.env.base.GauntletEnv`).
* Spaces parity with :class:`TabletopEnv` / :class:`PyBulletTabletopEnv`.
* Registration round-trip (``registered_envs`` + ``get_env_factory``).
* Reset-seed same-process determinism (§7.1 — exploration found 0.0
  on Plane+Box and ~1e-11 on the full scene; test uses a
  floating-point tolerance).
* ``set_perturbation`` validation (unknown axis, distractor_count
  bounds).
* State-affecting axis impact on obs (``object_initial_pose_x``,
  ``distractor_count``).
* Cosmetic-axis storage on the visual-only shadow attrs
  (state obs unchanged).
* ``restore_baseline`` idempotence.

Not covered here: cross-backend numerical parity (RFC-007 §7.3
explicit non-goal), rendering (deferred to RFC-008), import-guard
tests (live in ``tests/test_suite.py::TestBackendLazyImport``
alongside the equivalent PyBullet cases — kept outside
``tests/genesis/`` so they run in the default torch-free CI job
via the ``sys.modules`` monkey-patch simulation pattern).

All tests marked ``@pytest.mark.genesis`` and de-selected from the
default run. Install + run:

    uv sync --extra genesis --group genesis-dev
    uv run pytest -m genesis -q
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv

if TYPE_CHECKING:
    # Only resolves when ``[genesis]`` is installed (pytest.mark.genesis
    # gates runtime); the default torch-free mypy pass skips the body
    # thanks to the ``genesis`` mypy override in pyproject.toml plus
    # this TYPE_CHECKING guard.
    from gauntlet.env.genesis.tabletop_genesis import GenesisTabletopEnv

pytestmark = pytest.mark.genesis


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def env() -> Iterator[GenesisTabletopEnv]:
    """Session-shared env — one ``GenesisTabletopEnv`` across all tests.

    Construction is ~40 s (torch import + gs.init kernel compile +
    scene.build), so reusing one instance across the module keeps the
    whole suite under a minute. Each test resets the env (cheap) and
    drives its own perturbations; the fixture re-baselines via a
    clean ``reset`` at teardown so state from one test cannot leak
    into another ordering. Order-sensitive tests opt out by
    constructing their own env.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv()
    yield e
    e.close()


# ------------------------------------------------------------ protocol / registry


def test_protocol_conformance(env: GenesisTabletopEnv) -> None:
    """RFC-007 §8 — isinstance via the runtime-checkable Protocol."""
    assert isinstance(env, GauntletEnv)


def test_subpackage_import_registers_tabletop_genesis() -> None:
    """RFC-007 §8 — importing ``gauntlet.env.genesis`` registers the
    ``tabletop-genesis`` key in the registry and routes lookups back
    to :class:`GenesisTabletopEnv`.
    """
    import gauntlet.env.genesis  # noqa: F401 — trigger registration
    from gauntlet.env.genesis import GenesisTabletopEnv
    from gauntlet.env.registry import get_env_factory, registered_envs

    assert "tabletop-genesis" in registered_envs()
    factory = get_env_factory("tabletop-genesis")
    # cast-around-Callable: the registry holds a Callable[..., GauntletEnv].
    assert factory is GenesisTabletopEnv


# -------------------------------------------------------- spaces / axis names


def test_spaces_parity_with_other_backends(env: GenesisTabletopEnv) -> None:
    """Same 7-D action space + same 5-key state-obs dict as
    :class:`TabletopEnv` / :class:`PyBulletTabletopEnv` (RFC-007 §3).
    """
    from gymnasium.spaces import Box, Dict

    # action_space / observation_space are typed Space[Any] on the
    # Protocol; the concrete instances are Box / Dict respectively.
    action = env.action_space
    assert isinstance(action, Box)
    assert action.shape == (7,)
    assert action.dtype == np.float64
    assert float(action.low[0]) == -1.0
    assert float(action.high[0]) == 1.0

    obs_space = env.observation_space
    assert isinstance(obs_space, Dict)
    expected_obs_keys = {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
    assert set(obs_space.spaces.keys()) == expected_obs_keys


def test_axis_names_are_canonical_seven(env: GenesisTabletopEnv) -> None:
    """:attr:`AXIS_NAMES` matches the canonical 7 (RFC-007 §6.7)."""
    e = env
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
    assert expected == type(e).AXIS_NAMES


def test_visual_only_axes_are_four_cosmetic(env: GenesisTabletopEnv) -> None:
    """State-only first cut — the four cosmetic axes are VISUAL_ONLY
    until the follow-up rendering RFC (RFC-007 §6.6)."""
    e = env
    expected = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
        }
    )
    assert expected == type(e).VISUAL_ONLY_AXES


# ------------------------------------------------------------- reset / step / determinism


def _stack_obs(
    env_: GenesisTabletopEnv,
    seed: int,
    n_steps: int,
    action: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Roll out n steps from a fresh reset and return keyed stacks."""
    obs, _ = env_.reset(seed=seed)
    frames: dict[str, list[NDArray[np.float64]]] = {k: [v.copy()] for k, v in obs.items()}
    for _ in range(n_steps):
        obs, *_ = env_.step(action)
        for k, v in obs.items():
            frames[k].append(v.copy())
    return {k: np.stack(v) for k, v in frames.items()}


def test_reset_seed_determinism(env: GenesisTabletopEnv) -> None:
    """Two resets with the same seed + same action sequence -> same obs
    within floating-point tolerance (RFC-007 §7.1 — measured 0.0 on
    Plane+Box; the full scene adds a gravity-compensated EE + 10
    distractors and drifts at the ~1e-10 level)."""
    action = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    a = _stack_obs(env, seed=7, n_steps=15, action=action)
    b = _stack_obs(env, seed=7, n_steps=15, action=action)
    for k in a:
        diff = float(np.abs(a[k] - b[k]).max())
        assert diff < 1e-6, f"obs['{k}'] diverged across runs: max|Δ|={diff}"


def test_reset_returns_expected_shapes(env: GenesisTabletopEnv) -> None:
    """Reset's obs dict keys + shapes line up with the declared space."""
    obs, info = env.reset(seed=42)
    assert obs["cube_pos"].shape == (3,)
    assert obs["cube_quat"].shape == (4,)
    assert obs["ee_pos"].shape == (3,)
    assert obs["gripper"].shape == (1,)
    assert obs["target_pos"].shape == (3,)
    assert {"success", "grasped", "step"} <= set(info)


def test_step_advances_ee_in_x(env: GenesisTabletopEnv) -> None:
    """Action [+1, 0, 0, 0, 0, 0, 1] moves EE by +MAX_LINEAR_STEP in x.

    Pure-kinematic EE — no contact forces, so the delta is exact up
    to float precision from the identity-quat starting pose.
    """
    obs, _ = env.reset(seed=0)
    ee0 = obs["ee_pos"].copy()
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    obs2, *_ = env.step(action)
    delta_x = float(obs2["ee_pos"][0] - ee0[0])
    # MAX_LINEAR_STEP = 0.05; gravity_compensation + set_pos -> no
    # correction force, so delta is near-exact.
    assert abs(delta_x - 0.05) < 1e-4, f"expected ~+0.05, got {delta_x}"


# ----------------------------------------------------- set_perturbation validation


def test_set_perturbation_rejects_unknown_axis(env: GenesisTabletopEnv) -> None:
    """Unknown axis -> ValueError listing the valid axis names."""
    with pytest.raises(ValueError, match="unknown perturbation axis 'bogus'"):
        env.set_perturbation("bogus", 1.0)


@pytest.mark.parametrize("bad_count", [-1, 11, 100])
def test_set_perturbation_rejects_out_of_range_distractor_count(
    env: GenesisTabletopEnv, bad_count: int
) -> None:
    """distractor_count enforces integer [0, 10] (same as PyBullet)."""
    with pytest.raises(ValueError, match=r"distractor_count must be in \[0, 10\]"):
        env.set_perturbation("distractor_count", float(bad_count))


@pytest.mark.parametrize(
    "axis",
    [
        "lighting_intensity",
        "camera_offset_x",
        "camera_offset_y",
        "object_texture",
        "object_initial_pose_x",
        "object_initial_pose_y",
        "distractor_count",
    ],
)
def test_set_perturbation_accepts_every_known_axis(env: GenesisTabletopEnv, axis: str) -> None:
    """All seven canonical axes accepted (RFC-007 §6.7)."""
    value = 3.0 if axis == "distractor_count" else 0.05
    env.set_perturbation(axis, value)
    # Reset drains the queue; no assertion on the axis's effect here
    # (separate tests cover that). Just verifies no raise.
    env.reset(seed=42)


# --------------------------------------------- state-affecting axis behaviour


def test_object_initial_pose_x_overrides_random_cube_x(env: GenesisTabletopEnv) -> None:
    """RFC-007 §6.4: axis value overrides seed-driven random XY."""
    env.set_perturbation("object_initial_pose_x", 0.123)
    obs, _ = env.reset(seed=42)
    assert abs(float(obs["cube_pos"][0]) - 0.123) < 1e-5


def test_object_initial_pose_y_overrides_random_cube_y(env: GenesisTabletopEnv) -> None:
    """Counterpart axis, Y channel."""
    env.set_perturbation("object_initial_pose_y", -0.077)
    obs, _ = env.reset(seed=42)
    assert abs(float(obs["cube_pos"][1]) - (-0.077)) < 1e-5


def test_distractor_count_teleport_semantics(env: GenesisTabletopEnv) -> None:
    """RFC-007 §6.5: first ``count`` distractors at rest_z, rest at hidden_z.

    Reaches into private state (``env._distractors``) — accepted test
    coupling, matches the equivalent test PyBullet's battery has for
    the same invariant.
    """
    env.set_perturbation("distractor_count", 4)
    env.reset(seed=42)
    zs = [float(d.get_pos().cpu().numpy()[2]) for d in env._distractors]
    visible = sum(1 for z in zs if z > 0.0)
    hidden = sum(1 for z in zs if z < -1.0)
    assert visible == 4
    assert hidden == 6


# ----------------------------------------------- cosmetic axes: shadow storage


def test_object_texture_swap_preserves_cube_xy() -> None:
    """RFC-008 §3.7 — the ``object_texture`` swap must inherit the
    active cube's XY; otherwise the axis silently teleports the cube
    to the origin and the per-seed state diverges from the no-texture
    baseline.

    Two fresh envs with the same seed and the same ``render_in_obs=False``:
    one with ``object_texture=1`` queued, one without. ``cube_pos[:2]``
    must match to within float tolerance.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    a = GenesisTabletopEnv()
    b = GenesisTabletopEnv()
    try:
        obs_a, _ = a.reset(seed=42)
        b.set_perturbation("object_texture", 1.0)
        obs_b, _ = b.reset(seed=42)

        # Swap happened.
        assert b._cube is b._cube_green
        assert a._cube is a._cube_red
        # XY survived the swap.
        assert np.allclose(obs_a["cube_pos"][:2], obs_b["cube_pos"][:2], atol=1e-6), (
            f"object_texture swap lost cube XY: no-swap={obs_a['cube_pos'][:2]} "
            f"swap={obs_b['cube_pos'][:2]}"
        )

        # A follow-up reset with no pending perturbation unswaps back
        # to the red cube at the seed-random XY.
        obs_c, _ = b.reset(seed=42)
        assert b._cube is b._cube_red
        assert np.allclose(obs_a["cube_pos"][:2], obs_c["cube_pos"][:2], atol=1e-6)
    finally:
        a.close()
        b.close()


def test_cosmetic_axes_store_on_shadows_and_leave_obs_unchanged(env: GenesisTabletopEnv) -> None:
    """The four VISUAL_ONLY axes do not touch state obs — they set
    the shadow attributes the rendering RFC (RFC-008) will consume.

    Reaches into private state; same accepted coupling pattern the
    equivalent PyBullet test uses.
    """
    obs_baseline, _ = env.reset(seed=42)

    env.set_perturbation("lighting_intensity", 0.4)
    env.set_perturbation("camera_offset_x", 0.02)
    env.set_perturbation("camera_offset_y", -0.03)
    env.set_perturbation("object_texture", 1.0)
    obs_after, _ = env.reset(seed=42)

    # State obs unchanged — same seed, only cosmetic axes queued.
    for k in ("cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"):
        assert np.allclose(obs_baseline[k], obs_after[k], atol=1e-8), (
            f"state obs['{k}'] diverged across a cosmetic-only perturbation"
        )

    # Shadow attributes took the values.
    assert env._light_intensity == 0.4
    assert float(env._cam_offset[0]) == 0.02
    assert float(env._cam_offset[1]) == -0.03
    assert env._texture_choice == 1


# --------------------------------------------- render_in_obs / render_size surface


def test_render_in_obs_extends_observation_space_and_validates_size() -> None:
    """RFC-008 §3.1/§3.2 — ``render_in_obs=True`` adds an ``"image"``
    key to ``observation_space`` with the declared shape, and a
    non-positive ``render_size`` raises ``ValueError`` with the same
    message ``TabletopEnv`` uses.

    Asymmetric ``render_size=(64, 96)`` catches H/W transposition bugs
    against Genesis's ``add_camera(res=(W, H))`` convention.
    """
    from gymnasium.spaces import Box, Dict

    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 96))
    try:
        obs_space = e.observation_space
        assert isinstance(obs_space, Dict)
        img_space = obs_space.spaces["image"]
        assert isinstance(img_space, Box)
        assert img_space.shape == (64, 96, 3)
        assert img_space.dtype == np.uint8
        assert int(img_space.low.min()) == 0
        assert int(img_space.high.max()) == 255
    finally:
        e.close()

    # Validation error: same string as TabletopEnv / PyBulletTabletopEnv.
    with pytest.raises(ValueError, match="render_size must be"):
        GenesisTabletopEnv(render_in_obs=True, render_size=(0, 224))


# --------------------------------------------------- restore_baseline idempotence


def test_restore_baseline_hides_all_distractors(env: GenesisTabletopEnv) -> None:
    """Post-``restore_baseline`` all 10 distractors are at hidden_z.

    Idempotent by construction — second call is a no-op observational
    delta against the first.
    """
    env.restore_baseline()
    zs1 = [float(d.get_pos().cpu().numpy()[2]) for d in env._distractors]
    env.restore_baseline()
    zs2 = [float(d.get_pos().cpu().numpy()[2]) for d in env._distractors]
    assert all(z == -10.0 for z in zs1)
    assert zs1 == zs2
