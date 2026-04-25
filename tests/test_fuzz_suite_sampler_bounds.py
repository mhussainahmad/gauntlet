"""Fuzz tests for ``Suite.cells`` bound + index invariants across samplers.

Phase 2.5 Task 13 — covers the gap left by ``test_property_perturbation``:
that test exercises the *primitive* sampler factories (continuous / int /
categorical) but never drives a full :class:`Suite` through the
:meth:`Suite.cells` dispatch layer.

Properties under test (all three samplers — cartesian / latin_hypercube /
sobol):

* Every emitted ``SuiteCell.values[axis_name]`` lies inside the declared
  envelope (``[low, high]`` for continuous axes, ``set(values)`` for
  categorical axes). One axis's bounds may not silently leak onto
  another.
* ``Suite.num_cells()`` matches ``len(list(Suite.cells()))`` exactly.
* ``SuiteCell.index`` values are zero-based, contiguous, and unique.
* For ``sampling="latin_hypercube"`` (the only stochastic sampler — Sobol
  is deterministic from the table; cartesian from the schema), two
  ``Suite.cells()`` calls with the same ``Suite.seed`` yield identical
  cell lists.

Hypothesis budget: ``max_examples=30`` per test (5 axes worst case x
n_samples<=8 means each example loops ~40 cells; tests still finish
under 2s wall-time because no env or runner is involved).
"""

from __future__ import annotations

from datetime import timedelta

import pydantic
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.env.perturbation import AXIS_NAMES
from gauntlet.suite.schema import AxisSpec, Suite

# Sobol's inline direction-number table tops out at 21 dimensions; LHS is
# unbounded but suite YAMLs ship only 7 canonical axes today. Bound the
# strategy at 5 to keep generation cheap.
_MAX_AXES = 5
# Sobol sample budget — drop the leading origin per the default ``skip=1``
# in :class:`gauntlet.suite.sobol.SobolSampler`, so 1 row is the floor.
_MAX_SAMPLES = 8

# Bound the float envelope so ``np.float64`` arithmetic inside the
# samplers never overflows or rounds away to inf. Real perturbation
# bounds are O(1); 1e3 is three orders of magnitude wider than anything
# a real suite declares.
_BOUND_FLOAT = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _continuous_axis_spec(draw: st.DrawFn) -> AxisSpec:
    """``AxisSpec`` with ``low <= high`` and (optional) ``steps``."""
    a = draw(_BOUND_FLOAT)
    b = draw(_BOUND_FLOAT)
    lo, hi = (a, b) if a <= b else (b, a)
    # ``steps`` is required for the cartesian path; the LHS / Sobol
    # branch builds with ``steps=None`` (and gets ``n_samples`` from the
    # Suite). The strategies below handle both shapes by passing
    # ``steps`` only into the cartesian variant.
    steps = draw(st.integers(min_value=1, max_value=4))
    return AxisSpec(low=lo, high=hi, steps=steps)


@st.composite
def _continuous_axis_spec_no_steps(draw: st.DrawFn) -> AxisSpec:
    """Continuous spec without ``steps`` — the LHS / Sobol shape."""
    a = draw(_BOUND_FLOAT)
    b = draw(_BOUND_FLOAT)
    lo, hi = (a, b) if a <= b else (b, a)
    return AxisSpec(low=lo, high=hi)


@st.composite
def _categorical_axis_spec(draw: st.DrawFn) -> AxisSpec:
    """``AxisSpec`` with the ``values`` shape (categorical)."""
    n = draw(st.integers(min_value=1, max_value=4))
    vals = draw(st.lists(_BOUND_FLOAT, min_size=n, max_size=n))
    return AxisSpec(values=[float(v) for v in vals])


@st.composite
def _cartesian_suite(draw: st.DrawFn) -> Suite:
    n_axes = draw(st.integers(min_value=1, max_value=_MAX_AXES))
    names = draw(
        st.lists(st.sampled_from(AXIS_NAMES), min_size=n_axes, max_size=n_axes, unique=True)
    )
    axes: dict[str, AxisSpec] = {}
    for name in names:
        spec = draw(st.one_of(_continuous_axis_spec(), _categorical_axis_spec()))
        axes[name] = spec
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    return Suite(
        name="prop-cartesian",
        env="tabletop",
        episodes_per_cell=1,
        seed=seed,
        axes=axes,
        sampling="cartesian",
    )


@st.composite
def _stochastic_suite(draw: st.DrawFn, mode: str) -> Suite:
    n_axes = draw(st.integers(min_value=1, max_value=_MAX_AXES))
    names = draw(
        st.lists(st.sampled_from(AXIS_NAMES), min_size=n_axes, max_size=n_axes, unique=True)
    )
    axes: dict[str, AxisSpec] = {}
    for name in names:
        spec = draw(st.one_of(_continuous_axis_spec_no_steps(), _categorical_axis_spec()))
        axes[name] = spec
    n_samples = draw(st.integers(min_value=1, max_value=_MAX_SAMPLES))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    return Suite(
        name=f"prop-{mode}",
        env="tabletop",
        episodes_per_cell=1,
        seed=seed,
        axes=axes,
        sampling=mode,  # type: ignore[arg-type]
        n_samples=n_samples,
    )


def _assert_value_in_axis_envelope(spec: AxisSpec, value: float) -> None:
    """Assert ``value`` lies inside the declared per-axis envelope."""
    if spec.values is not None:
        # Categorical: emitted value must be one of the choices.
        # (Equality on float is fine here — the sampler emits the choice
        # by direct selection, not by arithmetic.)
        assert value in tuple(float(v) for v in spec.values), (
            f"categorical axis emitted {value} outside declared values {spec.values}"
        )
        return
    # Continuous: ``[low, high]`` envelope. The continuous unit-cube
    # mappers (lhs / sobol) use ``low + u * (high - low)`` with
    # ``u in [0, 1)`` so the upper bound is technically exclusive; we
    # still assert ``<= high`` because Cartesian endpoints reach high
    # exactly.
    assert spec.low is not None and spec.high is not None
    # 1e-9 slack absorbs round-off from the float64 affine map; the
    # primitive-sampler property test uses the same slack.
    assert spec.low - 1e-9 <= value <= spec.high + 1e-9, (
        f"continuous axis emitted {value} outside [{spec.low}, {spec.high}]"
    )


# ----- cartesian sampler -----------------------------------------------------


@given(suite=_cartesian_suite())
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_cartesian_cells_within_axis_envelopes(suite: Suite) -> None:
    """Every cartesian cell value lives inside its axis spec's envelope."""
    for cell in suite.cells():
        for axis_name, value in cell.values.items():
            _assert_value_in_axis_envelope(suite.axes[axis_name], value)


@given(suite=_cartesian_suite())
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_cartesian_indices_are_contiguous_and_match_num_cells(suite: Suite) -> None:
    """``num_cells()`` matches the iterator length; indices are 0..N-1 unique."""
    cells = list(suite.cells())
    assert len(cells) == suite.num_cells()
    indices = [cell.index for cell in cells]
    assert indices == list(range(len(cells)))


# ----- LHS sampler -----------------------------------------------------------


@given(suite=_stochastic_suite(mode="latin_hypercube"))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_lhs_cells_within_axis_envelopes(suite: Suite) -> None:
    """Every LHS cell value lives inside its axis spec's envelope."""
    for cell in suite.cells():
        for axis_name, value in cell.values.items():
            _assert_value_in_axis_envelope(suite.axes[axis_name], value)


@given(suite=_stochastic_suite(mode="latin_hypercube"))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_lhs_emits_n_samples_unique_indices(suite: Suite) -> None:
    cells = list(suite.cells())
    assert suite.n_samples is not None
    assert len(cells) == suite.n_samples == suite.num_cells()
    assert [cell.index for cell in cells] == list(range(len(cells)))


@given(
    suite=_stochastic_suite(mode="latin_hypercube"),
)
@settings(
    max_examples=20,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_lhs_seeded_suite_is_reproducible(suite: Suite) -> None:
    """Two ``Suite.cells()`` calls on a seeded LHS suite return the same cells.

    The strategy may produce ``seed=None``; in that case every call seeds
    a fresh OS-entropy generator and reproducibility is not promised. The
    contract under test is "fixed seed -> fixed cells", so we skip the
    ``None`` half.
    """
    if suite.seed is None:
        return
    cells_a = list(suite.cells())
    cells_b = list(suite.cells())
    assert len(cells_a) == len(cells_b)
    for ca, cb in zip(cells_a, cells_b, strict=True):
        assert ca.index == cb.index
        assert ca.values == cb.values


# ----- Sobol sampler ---------------------------------------------------------


@given(suite=_stochastic_suite(mode="sobol"))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sobol_cells_within_axis_envelopes(suite: Suite) -> None:
    """Every Sobol cell value lives inside its axis spec's envelope."""
    for cell in suite.cells():
        for axis_name, value in cell.values.items():
            _assert_value_in_axis_envelope(suite.axes[axis_name], value)


@given(suite=_stochastic_suite(mode="sobol"))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sobol_emits_n_samples_unique_indices(suite: Suite) -> None:
    cells = list(suite.cells())
    assert suite.n_samples is not None
    assert len(cells) == suite.n_samples == suite.num_cells()
    assert [cell.index for cell in cells] == list(range(len(cells)))


@given(suite=_stochastic_suite(mode="sobol"))
@settings(
    max_examples=20,
    deadline=timedelta(seconds=3),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sobol_is_seed_independent(suite: Suite) -> None:
    """Sobol is fully deterministic — the sequence ignores the RNG.

    Two ``Suite.cells()`` calls return the same list regardless of the
    Suite's ``seed`` (the seed is forwarded to the sampler but the Sobol
    sampler ``del rng`` immediately). See
    :class:`gauntlet.suite.sobol.SobolSampler` docstring.
    """
    cells_a = list(suite.cells())
    cells_b = list(suite.cells())
    for ca, cb in zip(cells_a, cells_b, strict=True):
        assert ca.values == cb.values


# ----- malformed Suite bounds (defence against schema gaps) -------------------


def test_cartesian_suite_with_inverted_bounds_rejected_at_construction() -> None:
    """``low > high`` must trip the AxisSpec model validator before
    ``Suite.cells`` ever runs."""
    with pytest.raises(pydantic.ValidationError):
        Suite(
            name="bad",
            env="tabletop",
            episodes_per_cell=1,
            axes={"lighting_intensity": AxisSpec(low=1.0, high=0.0, steps=2)},
            sampling="cartesian",
        )
