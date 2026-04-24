"""Suite loader integration for the Isaac Sim backend — RFC-009 §8.

The loader's `_reject_purely_visual_suites` guard reads
`VISUAL_ONLY_AXES` off the registered factory. Because
`IsaacSimTabletopEnv.VISUAL_ONLY_AXES` is non-empty on this state-only
first cut (RFC-009 §6.6), the rejection happens for free for any YAML
naming `env: tabletop-isaac` and only cosmetic axes.

These tests pin the full loader contract on the Isaac Sim backend
under the conftest's autouse fake-`isaacsim` namespace:

* State-only sweep loads cleanly.
* Mixed (state + cosmetic) sweep loads cleanly.
* Cosmetic-only sweep is rejected with a useful message.
"""

from __future__ import annotations

import pytest

from gauntlet.suite.loader import load_suite_from_string

_STATE_ONLY_YAML = """
name: isaac-state-smoke
env: tabletop-isaac
episodes_per_cell: 1
axes:
  object_initial_pose_x:
    low: -0.05
    high: 0.05
    steps: 3
  distractor_count:
    values: [0, 3]
"""


_COSMETIC_ONLY_YAML = """
name: isaac-cosmetic-only
env: tabletop-isaac
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
name: isaac-mixed
env: tabletop-isaac
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


def test_loader_accepts_state_only_isaac_sweep() -> None:
    """A state-only sweep on `tabletop-isaac` loads via
    `load_suite_from_string` — no loader guard fires."""
    suite = load_suite_from_string(_STATE_ONLY_YAML)
    assert suite.env == "tabletop-isaac"
    assert set(suite.axes.keys()) == {"object_initial_pose_x", "distractor_count"}
    # Sanity: cell enumeration works (Cartesian product of 3 * 2 = 6 cells).
    assert suite.num_cells() == 6


def test_loader_accepts_mixed_axes_isaac_sweep() -> None:
    """One cosmetic axis + one state-affecting axis is accepted —
    the run will still show observable variation along the state
    axis, so the rejection guard does not fire."""
    suite = load_suite_from_string(_MIXED_YAML)
    assert suite.env == "tabletop-isaac"
    assert set(suite.axes.keys()) == {"lighting_intensity", "object_initial_pose_x"}


def test_loader_rejects_cosmetic_only_isaac_sweep() -> None:
    """A YAML naming only cosmetic axes on `tabletop-isaac` is
    rejected at load time. RFC-009 §6.6 — the loader's
    `_reject_purely_visual_suites` guard fires because
    `VISUAL_ONLY_AXES = {lighting_intensity, camera_offset_x,
    camera_offset_y, object_texture}` is non-empty.

    Was the same shape Genesis had pre-RFC-008 and PyBullet had
    pre-RFC-006; they relaxed once their rendering follow-up landed.
    """
    with pytest.raises(ValueError, match=r"every declared axis"):
        load_suite_from_string(_COSMETIC_ONLY_YAML)


def test_loader_rejection_message_lists_the_cosmetic_axes() -> None:
    """The rejection message names which axes are cosmetic on this
    backend so the user knows exactly what to add to the sweep."""
    with pytest.raises(ValueError) as excinfo:
        load_suite_from_string(_COSMETIC_ONLY_YAML)
    msg = str(excinfo.value)
    # One axis name from each of the four cosmetic axes will appear in
    # the sorted-axis list rendered by the loader.
    assert "lighting_intensity" in msg
    assert "object_texture" in msg
    # Helper hint about the resolution (state-effecting axes or rendering RFC).
    assert "object_initial_pose" in msg or "distractor_count" in msg
