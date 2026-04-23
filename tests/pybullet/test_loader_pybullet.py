"""PyBullet-aware loader tests — RFC-005 §6.2 / §9.1 case 11 / §12 Q1
(updated by RFC-006 §3.5).

RFC-005 shipped the backend state-only, which meant cosmetic axes
(``lighting_intensity`` / ``object_texture``) produced byte-identical
state obs. The Suite loader rejected any YAML whose every axis was in
``VISUAL_ONLY_AXES`` so the run could not silently look like a broken
harness.

RFC-006 adds the image-observation path. With the renderer online the
cosmetic axes are observable on ``obs["image"]`` — so
``PyBulletTabletopEnv.VISUAL_ONLY_AXES`` drops to ``frozenset()`` and
the loader's rejection becomes a no-op on the PyBullet backend (same
shape MuJoCo has always had). These tests pin the relaxed contract.

Tests marked ``@pytest.mark.pybullet`` because they still exercise the
PyBullet-registration import path; they no longer depend on a
non-empty ``VISUAL_ONLY_AXES``.
"""

from __future__ import annotations

import pytest

from gauntlet.suite.loader import load_suite_from_string

pytestmark = pytest.mark.pybullet


_VISUAL_ONLY_YAML = """
name: cosmetic-only
env: tabletop-pybullet
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.2
    steps: 2
  object_texture:
    values: [0.0, 1.0]
"""


_MIXED_YAML = """
name: mixed
env: tabletop-pybullet
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.2
    steps: 2
  object_initial_pose_x:
    low: -0.05
    high: 0.05
    steps: 2
"""


_STATE_ONLY_YAML = """
name: state-only
env: tabletop-pybullet
episodes_per_cell: 1
axes:
  distractor_count:
    low: 0
    high: 4
    steps: 3
"""


_MUJOCO_COSMETIC_YAML = """
name: mujoco-cosmetic
env: tabletop
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.2
    steps: 2
  object_texture:
    values: [0.0, 1.0]
"""


def test_accepts_cosmetic_only_suite_once_rendering_exists() -> None:
    """RFC-006 §3.5: cosmetic-only sweeps load on PyBullet once rendering is on.

    Was ``test_rejects_suite_whose_every_axis_is_visual_only`` under the
    state-only contract. Rendering (RFC-006) makes every cosmetic axis
    observable on ``obs["image"]``, so ``VISUAL_ONLY_AXES`` is empty and
    the rejection becomes a no-op. A user running a cosmetic-only sweep
    with ``render_in_obs=False`` still gets pairwise-identical state-only
    cells — documented on the env class, same property MuJoCo has always
    had, not a harness bug.
    """
    suite = load_suite_from_string(_VISUAL_ONLY_YAML)
    assert suite.env == "tabletop-pybullet"
    assert set(suite.axes.keys()) == {"lighting_intensity", "object_texture"}


def test_accepts_mix_of_visual_and_state_effecting_axes() -> None:
    """One visual axis + one state-effecting axis is fine — the run
    will still show observable variation along the state axis.
    """
    suite = load_suite_from_string(_MIXED_YAML)
    assert suite.env == "tabletop-pybullet"
    assert set(suite.axes.keys()) == {"lighting_intensity", "object_initial_pose_x"}


def test_accepts_state_only_axes() -> None:
    """distractor_count is state-effecting → load succeeds."""
    suite = load_suite_from_string(_STATE_ONLY_YAML)
    assert suite.env == "tabletop-pybullet"


def test_mujoco_backend_accepts_cosmetic_only_suites() -> None:
    """``TabletopEnv.VISUAL_ONLY_AXES`` is the empty frozenset (the
    MuJoCo renderer consumes cosmetic axes via render_in_obs), so
    suites varying only visual axes on the MuJoCo backend must NOT
    be rejected.
    """
    suite = load_suite_from_string(_MUJOCO_COSMETIC_YAML)
    assert suite.env == "tabletop"
