"""Genesis backend rendering tests — image-observation contract.

Covers the RFC-008 §7 test matrix:

* Shape / dtype / range of ``obs["image"]`` under ``render_in_obs=True``.
* Absence of ``"image"`` under the default ``render_in_obs=False``.
* Observation-space parity with :class:`TabletopEnv` (MuJoCo).
* Within-run determinism — two independent envs reset + rollout
  under the same seed produce byte-identical images.
* Per-axis pixel-sensitivity on the four cosmetic axes
  (``lighting_intensity``, ``object_texture``, ``camera_offset_x``,
  ``camera_offset_y``).
* RandomPolicy end-to-end smoke with rendering on.
* Suite-loader acceptance of cosmetic-only sweeps on
  ``tabletop-genesis`` (was: rejected; is: accepted).

Not covered: cross-backend numerical pixel parity (RFC-007 §7.3
explicit non-goal). The cross-backend shape-parity test compares the
``observation_space["image"]`` Box (low, high, shape, dtype) only.

All tests marked ``@pytest.mark.genesis`` and de-selected from the
default run. Install + run:

    uv sync --extra genesis --group genesis-dev
    uv run pytest -m genesis tests/genesis/test_render_genesis.py -q
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from gauntlet.env.genesis.tabletop_genesis import GenesisTabletopEnv

pytestmark = pytest.mark.genesis


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def render_env() -> Iterator[GenesisTabletopEnv]:
    """Shared render-enabled env at 64x64 — pays the one-time
    Rasterizer shader compile once for the whole module.

    64x64 keeps the first-render tax (~100 ms on the laptop measured
    in the exploration pass) low vs. the 224x224 production default
    that'd cost seconds. Shape / dtype / bounds are scale-invariant so
    tests that assert on those only don't care about resolution.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64))
    yield e
    e.close()


# ------------------------------------------------------------ shape / dtype


def test_image_shape_dtype_and_bounds(render_env: GenesisTabletopEnv) -> None:
    """RFC-008 §7 case 1 — ``obs["image"]`` is a uint8 ``(H, W, 3)``
    array with values in ``[0, 255]``. Asymmetric sizes are handled in
    a separate test; this one uses the fixture's square ``(64, 64)``.
    """
    obs, _ = render_env.reset(seed=0)
    assert "image" in obs
    img = obs["image"]
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert int(img.min()) >= 0
    assert int(img.max()) <= 255


def test_asymmetric_render_size_respects_height_width_order() -> None:
    """Asymmetric ``render_size=(H=64, W=96)`` must produce ``(64, 96, 3)``
    — catches H/W transposition against Genesis's
    ``add_camera(res=(W, H))`` convention (RFC-008 §3.6).
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 96))
    try:
        obs, _ = e.reset(seed=0)
        assert obs["image"].shape == (64, 96, 3)
    finally:
        e.close()


def test_image_absent_when_render_in_obs_false() -> None:
    """RFC-008 §7 case 6 — default path produces the state-only obs
    dict. Locks the RFC-007 state-only contract.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv()
    try:
        obs, _ = e.reset(seed=0)
        assert "image" not in obs
        assert set(obs.keys()) == {"cube_pos", "cube_quat", "ee_pos", "gripper", "target_pos"}
    finally:
        e.close()


# ---------------------------------------------------- within-run determinism


def test_reset_seed_produces_byte_identical_image() -> None:
    """RFC-008 §7 case 3 — two independent envs, same seed, same
    ``render_in_obs=True`` path produce ``np.array_equal`` images.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    a = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64))
    b = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64))
    try:
        obs_a, _ = a.reset(seed=42)
        obs_b, _ = b.reset(seed=42)
        assert np.array_equal(obs_a["image"], obs_b["image"])
    finally:
        a.close()
        b.close()


def test_post_step_determinism_across_instances() -> None:
    """RFC-008 §7 case 4 — same seed + same fixed action sequence on
    two independent envs -> ``obs["image"]`` matches byte-for-byte at
    every stepped frame. Composes Rasterizer determinism (back-to-back
    renders are bit-equal) with the existing physics-side within-run
    determinism.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    a = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64), max_steps=10)
    b = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64), max_steps=10)
    try:
        a.reset(seed=42)
        b.reset(seed=42)
        rng = np.random.default_rng(0)
        action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float64)
        for step_index in range(10):
            obs_a, *_ = a.step(action)
            obs_b, *_ = b.step(action)
            assert np.array_equal(obs_a["image"], obs_b["image"]), (
                f"image diverged across instances at step {step_index}"
            )
    finally:
        a.close()
        b.close()


def test_every_step_image_is_valid_uint8() -> None:
    """RFC-008 §7 case 8 — every per-step ``obs["image"]`` is a
    contiguous uint8 ``(64, 64, 3)`` array under rollout. Direct env
    step loop (not Runner-mediated) so the assertion hits every
    frame, not just episode boundaries.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64), max_steps=5)
    try:
        e.reset(seed=7)
        rng = np.random.default_rng(0)
        for _ in range(5):
            action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float64)
            obs, *_ = e.step(action)
            img = obs["image"]
            assert img.shape == (64, 64, 3)
            assert img.dtype == np.uint8
            assert img.flags["C_CONTIGUOUS"]
            assert int(img.min()) >= 0
            assert int(img.max()) <= 255
    finally:
        e.close()


# ------------------------------------------------- cross-backend shape parity


# --------------------------------------------- runner-level smoke with rendering


def test_random_policy_smoke_with_rendering() -> None:
    """RFC-008 §7 case 8 — end-to-end smoke: a Runner drives
    :class:`RandomPolicy` across a tiny cosmetic-only grid with
    ``render_in_obs=True`` and the rollout completes cleanly.

    Library-level integration — not via the CLI. Asserts (a) the
    Runner can pickle the env factory with rendering enabled, (b)
    every emitted :class:`Episode` has a concrete ``success``
    verdict, (c) the sweep actually produces rollouts rather than
    being rejected at load time (the ``VISUAL_ONLY_AXES`` relaxation
    from step 8).

    ``max_steps=3`` keeps each rollout under a second after the
    first scene build; the whole test budget lands well under the
    ``genesis-tests`` CI job's time allowance.
    """
    from functools import partial

    from gauntlet.env.genesis import GenesisTabletopEnv
    from gauntlet.policy import RandomPolicy
    from gauntlet.runner import Runner
    from gauntlet.suite import AxisSpec, Suite

    env_factory = partial(
        GenesisTabletopEnv,
        max_steps=3,
        render_in_obs=True,
        render_size=(64, 64),
    )
    suite = Suite(
        name="genesis-render-smoke",
        env="tabletop-genesis",
        seed=0,
        episodes_per_cell=1,
        axes={
            "lighting_intensity": AxisSpec(values=[0.5, 1.5]),
            "object_texture": AxisSpec(values=[0.0, 1.0]),
        },
    )
    runner = Runner(n_workers=1, env_factory=env_factory)
    episodes = runner.run(
        policy_factory=partial(RandomPolicy, action_dim=7),
        suite=suite,
    )

    # 2 lighting x 2 texture = 4 cells, 1 episode per cell = 4 rollouts.
    assert len(episodes) == 4
    for ep in episodes:
        assert isinstance(ep.success, bool)


# ------------------------------------------------- axis sensitivity (four cases)


def _image_under_axis(axis: str, value: float, seed: int = 42) -> np.ndarray:
    """Fresh env, queue the axis at the given value, reset at ``seed``,
    return the rendered image. Small-resolution to keep the test cheap.
    """
    from gauntlet.env.genesis import GenesisTabletopEnv

    e = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64))
    try:
        e.set_perturbation(axis, value)
        obs, _ = e.reset(seed=seed)
        return np.asarray(obs["image"], dtype=np.uint8)
    finally:
        e.close()


def test_lighting_intensity_changes_pixels() -> None:
    """RFC-008 §7 case 5a — ``lighting_intensity=0.3`` vs ``=1.5``
    produce distinct images under identical seed + scene state.
    """
    low = _image_under_axis("lighting_intensity", 0.3)
    high = _image_under_axis("lighting_intensity", 1.5)
    assert not np.array_equal(low, high), (
        "lighting_intensity did not affect the render — pyrender light "
        "intensity mutation (RFC-008 §4) may not be hooked up"
    )


def test_object_texture_changes_pixels() -> None:
    """RFC-008 §7 case 5b — texture choice 0 (red) vs 1 (green) swap
    the active cube and therefore the rendered colour.
    """
    red = _image_under_axis("object_texture", 0.0)
    green = _image_under_axis("object_texture", 1.0)
    assert not np.array_equal(red, green), (
        "object_texture swap did not affect the render — check the "
        "dual-cube teleport in _apply_one_perturbation"
    )


def test_camera_offset_x_changes_pixels() -> None:
    """RFC-008 §7 case 5c — camera pans along X."""
    left = _image_under_axis("camera_offset_x", -0.05)
    right = _image_under_axis("camera_offset_x", 0.05)
    assert not np.array_equal(left, right)


def test_camera_offset_y_changes_pixels() -> None:
    """RFC-008 §7 case 5d — camera pans along Y."""
    back = _image_under_axis("camera_offset_y", -0.05)
    front = _image_under_axis("camera_offset_y", 0.05)
    assert not np.array_equal(back, front)


def test_loader_accepts_cosmetic_only_sweep_on_tabletop_genesis() -> None:
    """RFC-008 §7 case 7 — post-RFC-008 the Suite loader's
    cosmetic-only rejection is a no-op on ``tabletop-genesis``
    because ``VISUAL_ONLY_AXES == frozenset()``. A sweep whose every
    axis is cosmetic now loads.

    The ``_reject_purely_visual_suites`` branch short-circuits on an
    empty ``VISUAL_ONLY_AXES``, so no loader code change was needed
    — only the classvar flip in step 8. This test locks that
    behaviour in.
    """
    # Register the genesis backend.
    import gauntlet.env.genesis  # noqa: F401
    from gauntlet.suite.loader import load_suite_from_string

    yaml_text = """
name: genesis-cosmetic-only
env: tabletop-genesis
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.5
    high: 1.5
    steps: 2
  object_texture:
    values: [0.0, 1.0]
"""
    suite = load_suite_from_string(yaml_text)
    assert suite.env == "tabletop-genesis"
    assert set(suite.axes.keys()) == {"lighting_intensity", "object_texture"}


def test_image_space_matches_mujoco() -> None:
    """RFC-008 §7 case 2 — ``observation_space["image"]`` Box is equal
    under ``gym.spaces.Box.__eq__`` to ``TabletopEnv``'s. Pixel values
    explicitly not compared (RFC-007 §7.3).
    """
    from gymnasium.spaces import Box, Dict

    from gauntlet.env.genesis import GenesisTabletopEnv
    from gauntlet.env.tabletop import TabletopEnv

    g = GenesisTabletopEnv(render_in_obs=True, render_size=(64, 64))
    m = TabletopEnv(render_in_obs=True, render_size=(64, 64))
    try:
        # Cast through the runtime-checked Dict so mypy --strict knows
        # ``.spaces`` is available; the env Protocol declares
        # ``observation_space: gym.spaces.Space[Any]`` (no .spaces attr
        # on the base Space) so the narrowing is needed once per
        # backend in the test layer.
        assert isinstance(g.observation_space, Dict)
        assert isinstance(m.observation_space, Dict)
        g_img = g.observation_space.spaces["image"]
        m_img = m.observation_space.spaces["image"]
        assert isinstance(g_img, Box)
        assert isinstance(m_img, Box)
        assert g_img == m_img  # Box.__eq__ compares low, high, shape, dtype
    finally:
        g.close()
        m.close()
