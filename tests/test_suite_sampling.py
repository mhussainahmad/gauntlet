"""Sampler-protocol tests — see ``src/gauntlet/suite/sampling.py``.

Two responsibilities:

* Pin :class:`CartesianSampler` to a byte-identical reproduction of
  the historical :meth:`Suite.cells` enumeration, so the sampler
  refactor is invisible to existing callers.
* Cover the :func:`build_sampler` dispatch table — including the
  :class:`SobolSampler` placeholder that raises a clear
  :class:`NotImplementedError` until the follow-up PR lands.

The LHS sampler has its own dedicated suite in ``test_lhs_sampler.py``;
this file only covers the protocol layer + the cartesian regression
pin + the Sobol placeholder.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from gauntlet.suite import Suite, load_suite, load_suite_from_string
from gauntlet.suite.sampling import (
    CartesianSampler,
    Sampler,
    SobolSampler,
    build_sampler,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SMOKE = REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
EXAMPLE_BASIC = REPO_ROOT / "examples" / "suites" / "tabletop-basic-v1.yaml"


_REGRESSION_YAML = """
name: cartesian-regression
env: tabletop
seed: 7
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
    steps: 4
  camera_offset_x:
    low: -0.05
    high: 0.05
    steps: 3
  object_texture:
    values: [0, 1]
  distractor_count:
    low: 0
    high: 5
    steps: 6
"""


def _historical_cells(suite: Suite) -> list[tuple[int, dict[str, float]]]:
    """Reproduce the pre-refactor ``Suite.cells`` body inline.

    Used as the regression oracle: every refactor (and every later
    sampler-aware dispatch) must produce a list that compares equal
    to this for cartesian sampling.
    """
    axis_names = tuple(suite.axes.keys())
    per_axis = tuple(spec.enumerate() for spec in suite.axes.values())
    out: list[tuple[int, dict[str, float]]] = []
    for index, combo in enumerate(itertools.product(*per_axis)):
        out.append((index, dict(zip(axis_names, combo, strict=True))))
    return out


class TestCartesianSampler:
    def test_satisfies_sampler_protocol(self) -> None:
        # Runtime-checkable Protocol: structural conformance.
        sampler = CartesianSampler()
        assert isinstance(sampler, Sampler)

    def test_byte_identical_to_historical_cells_inline(self) -> None:
        suite = load_suite_from_string(_REGRESSION_YAML)
        rng = np.random.default_rng(0)
        cells = CartesianSampler().sample(suite, rng)
        actual = [(c.index, dict(c.values)) for c in cells]
        assert actual == _historical_cells(suite)

    def test_byte_identical_to_suite_dot_cells(self) -> None:
        # The currently-shipping ``Suite.cells`` is the source of truth
        # we must not drift from. After commit 5 wires the sampler
        # into ``Suite.cells``, this becomes a tautology — until then,
        # it pins the contract directly.
        suite = load_suite_from_string(_REGRESSION_YAML)
        from_method = [(c.index, dict(c.values)) for c in suite.cells()]
        from_sampler = [
            (c.index, dict(c.values))
            for c in CartesianSampler().sample(suite, np.random.default_rng(0))
        ]
        assert from_method == from_sampler

    def test_byte_identical_on_shipped_smoke_yaml(self) -> None:
        suite = load_suite(EXAMPLE_SMOKE)
        from_method = [(c.index, dict(c.values)) for c in suite.cells()]
        from_sampler = [
            (c.index, dict(c.values))
            for c in CartesianSampler().sample(suite, np.random.default_rng(0))
        ]
        assert from_method == from_sampler

    def test_byte_identical_on_shipped_basic_v1_yaml(self) -> None:
        suite = load_suite(EXAMPLE_BASIC)
        from_method = [(c.index, dict(c.values)) for c in suite.cells()]
        from_sampler = [
            (c.index, dict(c.values))
            for c in CartesianSampler().sample(suite, np.random.default_rng(0))
        ]
        assert from_method == from_sampler
        # Cell count is the documented 4 * 3 * 2 * 6 product.
        assert len(from_sampler) == 4 * 3 * 2 * 6

    def test_rng_is_unused_for_cartesian(self) -> None:
        suite = load_suite_from_string(_REGRESSION_YAML)
        a = CartesianSampler().sample(suite, np.random.default_rng(0))
        b = CartesianSampler().sample(suite, np.random.default_rng(99))
        assert [(c.index, dict(c.values)) for c in a] == [(c.index, dict(c.values)) for c in b]


class TestSobolSampler:
    def test_satisfies_sampler_protocol(self) -> None:
        assert isinstance(SobolSampler(), Sampler)

    def test_raises_clear_not_implemented_with_followup_hint(self) -> None:
        suite = load_suite_from_string(_REGRESSION_YAML)
        with pytest.raises(NotImplementedError, match="follow-up"):
            SobolSampler().sample(suite, np.random.default_rng(0))


class TestBuildSampler:
    def test_cartesian_dispatch(self) -> None:
        assert isinstance(build_sampler("cartesian"), CartesianSampler)

    def test_sobol_dispatch(self) -> None:
        assert isinstance(build_sampler("sobol"), SobolSampler)

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown sampling mode"):
            build_sampler("not-a-mode")


class TestSuiteCellsDispatch:
    """Pin :meth:`Suite.cells` to the right sampler keyed off ``sampling``."""

    _LHS_YAML = """
name: lhs-dispatch
env: tabletop
seed: 42
sampling: latin_hypercube
n_samples: 24
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
  camera_offset_x:
    low: -0.05
    high: 0.05
"""

    _SOBOL_YAML = """
name: sobol-dispatch
env: tabletop
seed: 42
sampling: sobol
n_samples: 8
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.5
"""

    def test_default_cells_unchanged_for_cartesian(self) -> None:
        # The flagship backwards-compat assertion: every shipped suite
        # YAML produces the exact same cell list it produced before
        # this PR. ``CartesianSampler`` is byte-identical to the
        # historical inline body (pinned in TestCartesianSampler);
        # this test pins that ``Suite.cells`` keeps using it.
        suite = load_suite(EXAMPLE_BASIC)
        # Default sampling is cartesian; no n_samples; no LHS budget.
        assert suite.sampling == "cartesian"
        first_pass = [(c.index, dict(c.values)) for c in suite.cells()]
        second_pass = [(c.index, dict(c.values)) for c in suite.cells()]
        assert first_pass == second_pass
        # And matches the sampler-level oracle:
        sampler_pass = [
            (c.index, dict(c.values))
            for c in CartesianSampler().sample(suite, np.random.default_rng(0))
        ]
        assert first_pass == sampler_pass
        assert len(first_pass) == 4 * 3 * 2 * 6

    def test_lhs_cells_emits_n_samples_rows(self) -> None:
        suite = load_suite_from_string(self._LHS_YAML)
        cells = list(suite.cells())
        assert len(cells) == 24
        assert suite.num_cells() == 24

    def test_lhs_cells_deterministic_across_calls(self) -> None:
        # Two ``Suite.cells()`` calls on the same Suite must produce
        # the same list — the dispatch layer seeds a fresh RNG from
        # ``suite.seed`` each time, so calls are idempotent.
        suite = load_suite_from_string(self._LHS_YAML)
        a = [(c.index, dict(c.values)) for c in suite.cells()]
        b = [(c.index, dict(c.values)) for c in suite.cells()]
        assert a == b

    def test_lhs_cells_differ_under_different_seeds(self) -> None:
        suite_a = load_suite_from_string(self._LHS_YAML.replace("seed: 42", "seed: 1"))
        suite_b = load_suite_from_string(self._LHS_YAML.replace("seed: 42", "seed: 2"))
        rows_a = [dict(c.values) for c in suite_a.cells()]
        rows_b = [dict(c.values) for c in suite_b.cells()]
        assert rows_a != rows_b

    def test_sobol_cells_propagates_not_implemented(self) -> None:
        suite = load_suite_from_string(self._SOBOL_YAML)
        with pytest.raises(NotImplementedError, match="follow-up"):
            list(suite.cells())

    def test_lhs_num_cells_matches_n_samples_without_iterating(self) -> None:
        # ``num_cells`` for non-cartesian must read off ``n_samples``
        # directly; iterating the sampler would be wasteful.
        suite = load_suite_from_string(self._LHS_YAML)
        assert suite.num_cells() == suite.n_samples
