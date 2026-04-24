"""Property-based tests for ``gauntlet.env.perturbation`` samplers.

Phase 2.5 Task 13 — turns the hand-rolled bound checks in
``tests/test_perturbation.py::TestSamplerReproducibility`` into
property-based assertions over the full bound envelope. Each sampler
factory has a single contract: every emitted scalar lies inside the
declared ``[low, high]`` range; the int-and-categorical wrappers narrow
that contract further.

Hypothesis budget: ``max_examples=50`` per test (well under the task's
30-second budget across the full property suite). ``deadline=2s`` keeps
a slow draw from ballooning the run.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gauntlet.env.perturbation import (
    DEFAULT_BOUNDS,
    axis_for,
    distractor_count,
)
from gauntlet.env.perturbation.base import (
    make_categorical_sampler,
    make_continuous_sampler,
    make_int_sampler,
)

# ----- continuous sampler ----------------------------------------------------


# Bound the floats so a 64-bit RNG draw inside numpy.uniform never
# overflows to inf. The real perturbation envelopes are O(1); 1e6 is
# six orders of magnitude wider than anything a real suite declares.
_BOUND_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _ordered_float_pair(draw: st.DrawFn) -> tuple[float, float]:
    """Draw a ``(low, high)`` pair with ``low <= high``."""
    a = draw(_BOUND_FLOAT)
    b = draw(_BOUND_FLOAT)
    return (min(a, b), max(a, b))


@given(
    bounds=_ordered_float_pair(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_make_continuous_sampler_value_in_bounds(bounds: tuple[float, float], seed: int) -> None:
    lo, hi = bounds
    sampler = make_continuous_sampler(lo, hi)
    rng = np.random.default_rng(seed)
    value = sampler(rng)
    assert lo <= value <= hi


@given(low=_BOUND_FLOAT, high=_BOUND_FLOAT)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_make_continuous_sampler_inverted_bounds_raise(low: float, high: float) -> None:
    if low <= high:
        # Skip the well-formed half — covered by the in-bounds property.
        return
    with pytest.raises(ValueError, match="low must be <= high"):
        make_continuous_sampler(low, high)


# ----- int sampler -----------------------------------------------------------


_BOUND_INT = st.integers(min_value=-1_000_000, max_value=1_000_000)


@st.composite
def _ordered_int_pair(draw: st.DrawFn) -> tuple[int, int]:
    a = draw(_BOUND_INT)
    b = draw(_BOUND_INT)
    return (min(a, b), max(a, b))


@given(
    bounds=_ordered_int_pair(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_make_int_sampler_value_in_bounds_and_integer(bounds: tuple[int, int], seed: int) -> None:
    lo, hi = bounds
    sampler = make_int_sampler(lo, hi)
    rng = np.random.default_rng(seed)
    value = sampler(rng)
    # Returned as a float per the protocol, but the integer-sampler
    # contract guarantees it round-trips through ``int(...)`` losslessly.
    assert value == float(int(value))
    assert lo <= int(value) <= hi


@given(low=_BOUND_INT, high=_BOUND_INT)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_make_int_sampler_inverted_bounds_raise(low: int, high: int) -> None:
    if low <= high:
        return
    with pytest.raises(ValueError, match="low must be <= high"):
        make_int_sampler(low, high)


# ----- categorical sampler ---------------------------------------------------


_CATEGORY_LIST = st.lists(_BOUND_FLOAT, min_size=1, max_size=20)


@given(
    values=_CATEGORY_LIST,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_make_categorical_sampler_value_in_choice_set(values: list[float], seed: int) -> None:
    choices = tuple(float(v) for v in values)
    sampler = make_categorical_sampler(choices)
    rng = np.random.default_rng(seed)
    value = sampler(rng)
    assert value in choices


def test_make_categorical_sampler_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_categorical_sampler(())


# ----- registered axes -------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_all_registered_axes_emit_values_within_default_bounds(seed: int) -> None:
    """For every axis name in the canonical registry, the default-built
    axis emits scalar values inside ``axis.low <= value <= axis.high``
    (with a 1e-9 slack for round-off on the float-cast comparison)."""
    rng = np.random.default_rng(seed)
    for name in DEFAULT_BOUNDS:
        axis = axis_for(name)
        for _ in range(5):  # five draws per axis per example keeps total fast.
            value = axis.sample(rng)
            assert axis.low - 1e-9 <= value <= axis.high + 1e-9, (
                f"axis {name!r} emitted {value} outside [{axis.low}, {axis.high}]"
            )


@given(
    low=st.integers(min_value=0, max_value=10),
    high=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_distractor_count_factory_in_or_out_of_envelope(low: int, high: int) -> None:
    """Distractor bound check raises iff bounds escape ``[0, 10]``;
    well-formed bounds produce a sampler whose draws stay within them."""
    if low > high:
        with pytest.raises(ValueError):
            distractor_count(low=low, high=high)
        return
    if low < 0 or high > 10:
        with pytest.raises(ValueError, match="within"):
            distractor_count(low=low, high=high)
        return
    axis = distractor_count(low=low, high=high)
    rng = np.random.default_rng(2026)
    for _ in range(10):
        value = axis.sample(rng)
        assert low <= int(value) <= high


# ----- determinism property --------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_axis_sample_is_deterministic_under_seeded_rng(seed: int) -> None:
    """Spec §6 hard rule: same seed -> same sample sequence."""
    for name in DEFAULT_BOUNDS:
        axis = axis_for(name)
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        seq_a = [axis.sample(rng_a) for _ in range(5)]
        seq_b = [axis.sample(rng_b) for _ in range(5)]
        assert seq_a == seq_b
