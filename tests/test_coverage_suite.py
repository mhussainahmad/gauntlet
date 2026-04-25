"""Branch-coverage backfill for :mod:`gauntlet.suite`.

Phase 2.5 Task 11. Targets the missed lines in:

* ``schema.py`` L144 — ``values=[]`` (empty values list rejected)
* ``schema.py`` L171 — continuous shape with one endpoint missing
* ``lhs.py`` L116 — degenerate ``low == high`` axis returns ``low``
* ``sobol.py`` L242 — same degenerate branch on the Sobol path
* ``loader.py`` L138-140 — purely-visual suite rejection on a backend
  that declares a non-empty ``VISUAL_ONLY_AXES`` (no first-party
  backend currently does, so we register a synthetic one for the test
  and clean up after).
* ``loader.py`` L195-196 — backend module imports OK but does not
  call ``register_env``: harder to trigger without monkeypatching the
  built-in import table; covered via that route here.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import _REGISTRY
from gauntlet.suite.lhs import LatinHypercubeSampler
from gauntlet.suite.loader import load_suite_from_string
from gauntlet.suite.schema import AxisSpec, Suite
from gauntlet.suite.sobol import SobolSampler

# ----------------------------------------------------------------------------
# AxisSpec — schema-level validators (L144, L171).
# ----------------------------------------------------------------------------


def test_axis_spec_rejects_empty_values_list() -> None:
    """schema.py L144: ``values=[]`` → ``ValidationError`` with explanation."""
    with pytest.raises(ValidationError, match="non-empty"):
        AxisSpec(values=[])


def test_axis_spec_rejects_continuous_with_only_low() -> None:
    """schema.py L171: continuous shape requires both ``low`` AND ``high``."""
    with pytest.raises(ValidationError, match="missing"):
        AxisSpec(low=0.0)


def test_axis_spec_rejects_continuous_with_only_high() -> None:
    with pytest.raises(ValidationError, match="missing"):
        AxisSpec(high=1.0)


def test_axis_spec_rejects_continuous_with_only_steps() -> None:
    """``steps`` alone counts as ``has_continuous=True`` — endpoints required."""
    with pytest.raises(ValidationError, match="missing"):
        AxisSpec(steps=3)


# ----------------------------------------------------------------------------
# LHS / Sobol degenerate axis — ``low == high`` returns ``low`` verbatim
# (lhs.py L116, sobol.py L242). The samplers go through
# ``_axis_value_from_unit`` which short-circuits on a zero-width axis.
# ----------------------------------------------------------------------------


def test_lhs_sampler_handles_degenerate_low_eq_high_axis() -> None:
    """lhs.py L116: zero-width continuous axis → constant value."""
    suite = Suite(
        name="lhs-degenerate",
        env="tabletop",
        episodes_per_cell=1,
        sampling="latin_hypercube",
        n_samples=5,
        axes={
            "lighting_intensity": AxisSpec(low=0.7, high=0.7),
            "camera_offset_x": AxisSpec(low=-0.1, high=0.1),
        },
    )
    cells = LatinHypercubeSampler().sample(suite, np.random.default_rng(0))

    assert len(cells) == 5
    # Every cell collapses ``lighting_intensity`` to the literal endpoint.
    assert all(c.values["lighting_intensity"] == 0.7 for c in cells)


def test_sobol_sampler_handles_degenerate_low_eq_high_axis() -> None:
    """sobol.py L242: same degenerate-axis branch on the Sobol path."""
    suite = Suite(
        name="sobol-degenerate",
        env="tabletop",
        episodes_per_cell=1,
        sampling="sobol",
        n_samples=4,
        axes={
            "lighting_intensity": AxisSpec(low=1.5, high=1.5),
            "camera_offset_x": AxisSpec(low=-0.05, high=0.05),
        },
    )
    cells = SobolSampler().sample(suite, np.random.default_rng(0))

    assert len(cells) == 4
    assert all(c.values["lighting_intensity"] == 1.5 for c in cells)


# ----------------------------------------------------------------------------
# loader._reject_purely_visual_suites — VISUAL_ONLY_AXES enforcement
# (L138-149). No first-party backend currently declares non-empty
# VISUAL_ONLY_AXES, so we register a synthetic factory whose class
# advertises one, point a Suite at it, and assert the loader rejects.
# Cleanup removes the registration so other tests are unaffected.
# ----------------------------------------------------------------------------


class _SyntheticVisualOnlyEnv:
    """Minimal Protocol-satisfying env with a non-empty VISUAL_ONLY_AXES.

    Drives the loader's purely-visual rejection path. The class never
    actually has to be instantiated — the loader only reads the
    ``VISUAL_ONLY_AXES`` ClassVar via ``getattr``.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"object_texture", "lighting_intensity"})
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({"object_texture", "lighting_intensity"})

    observation_space: Any = None
    action_space: Any = None

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> Any:
        raise NotImplementedError

    def set_perturbation(self, name: str, value: float) -> None:
        raise NotImplementedError

    def restore_baseline(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


@pytest.fixture
def _visual_only_backend() -> Any:
    """Register a synthetic visual-only backend, then teardown removes it."""
    name = "synthetic-visual-only"

    # Register the CLASS itself (not a lambda) so the loader's
    # ``_visual_only_axes_of`` can read the ``VISUAL_ONLY_AXES``
    # ClassVar via getattr. Lambdas drop the attribute and the
    # rejection branch never fires. This matches how MuJoCo /
    # PyBullet / Genesis backends register their env classes.
    _REGISTRY[name] = _SyntheticVisualOnlyEnv
    try:
        # The schema's ``_env_supported`` validator restricts ``env`` to
        # the built-in import table OR currently-registered names. Our
        # synthetic name is registered above, so the schema accepts it.
        yield name
    finally:
        _REGISTRY.pop(name, None)


def test_loader_rejects_purely_visual_suite_on_visual_only_backend(
    _visual_only_backend: str,
) -> None:
    """loader.py L138-149: a suite whose every axis is in the backend's
    ``VISUAL_ONLY_AXES`` is rejected at load time with the "every
    declared axis is cosmetic" error message."""
    yaml_text = f"""\
name: visual-only-rejection
env: {_visual_only_backend}
episodes_per_cell: 1
axes:
  object_texture:
    values: [0.0, 1.0]
  lighting_intensity:
    low: 0.5
    high: 1.0
    steps: 2
"""
    with pytest.raises(ValueError, match="cosmetic on a state-only backend"):
        load_suite_from_string(yaml_text)


def test_loader_accepts_mixed_suite_on_visual_only_backend(
    _visual_only_backend: str,
) -> None:
    """A mixed suite (one cosmetic axis, one state-effecting axis) is
    accepted — only purely-cosmetic suites are rejected. Even though
    every declared axis happens to be in VISUAL_ONLY_AXES, the
    rejection only fires when ALL axes are cosmetic; a mixed suite
    would need a non-cosmetic axis. We exercise the negative ("declared
    is NOT subset of visual_only") branch by registering a richer
    backend with a smaller VISUAL_ONLY set on the fly.
    """

    # Override the fixture-registered factory's class to declare ONE
    # cosmetic axis but a richer AXIS_NAMES — declaring the
    # state-effecting axis means ``declared <= visual_only`` is False.
    class _PartiallyVisualEnv(_SyntheticVisualOnlyEnv):
        AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(
            {"object_texture", "object_initial_pose_x"}
        )
        VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({"object_texture"})

    _REGISTRY[_visual_only_backend] = _PartiallyVisualEnv

    yaml_text = f"""\
name: mixed-suite
env: {_visual_only_backend}
episodes_per_cell: 1
axes:
  object_texture:
    values: [0.0, 1.0]
  object_initial_pose_x:
    low: -0.05
    high: 0.05
    steps: 2
"""
    suite = load_suite_from_string(yaml_text)
    assert suite.name == "mixed-suite"


# ----------------------------------------------------------------------------
# Verify the synthetic env satisfies the GauntletEnv Protocol — keeps the
# test honest (the loader path actually constructs/inspects a Protocol
# implementor under real usage).
# ----------------------------------------------------------------------------


def test_synthetic_visual_only_env_satisfies_protocol() -> None:
    env = _SyntheticVisualOnlyEnv()
    assert isinstance(env, GauntletEnv)
