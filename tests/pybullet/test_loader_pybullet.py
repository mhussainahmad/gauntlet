"""PyBullet-aware loader tests — RFC-005 §6.2 / §9.1 case 11 / §12 Q1.

The Suite loader rejects a YAML whose every declared axis is in the
backend's :attr:`~gauntlet.env.base.GauntletEnv.VISUAL_ONLY_AXES` —
those axes mutate the PyBullet scene but cannot change a state-only
observation dict, so the run would report pairwise-identical cell
success rates and look like a broken harness.

Tests marked ``@pytest.mark.pybullet`` because the rejection only
fires once ``gauntlet.env.pybullet`` has registered and declared a
non-empty ``VISUAL_ONLY_AXES`` set.
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


def test_rejects_suite_whose_every_axis_is_visual_only() -> None:
    """Both declared axes are in PyBullet VISUAL_ONLY_AXES → reject."""
    with pytest.raises(ValueError) as excinfo:
        load_suite_from_string(_VISUAL_ONLY_YAML)
    msg = str(excinfo.value)
    assert "cosmetic on a state-only backend" in msg
    # The error names the declared axes and the cosmetic set.
    assert "lighting_intensity" in msg
    assert "object_texture" in msg


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
