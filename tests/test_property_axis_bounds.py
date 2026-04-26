"""Property tests for axis bounds invariants and env-side validation.

Phase 2.5 Task 13 — covers two gaps not exercised by
``tests/test_property_perturbation.py``:

* **Per-axis stability across re-seeded generators.** The existing
  ``test_axis_sample_is_deterministic_under_seeded_rng`` walks
  :data:`DEFAULT_BOUNDS` in a single test function but does not assert
  the cross-axis bounds invariant ``axis.low <= axis.high``. We pin it
  here as a one-shot for every name in :data:`AXIS_NAMES`, plus a
  hypothesis-driven property over arbitrary rng seeds that re-asserts
  the in-bounds + same-seed-same-value contract on a per-axis basis.
* **Env-side hard-bound validation.** Four axes have hard validation in
  :meth:`gauntlet.env.tabletop.TabletopEnv.set_perturbation`:
  ``distractor_count`` ([0, 10]), ``initial_state_ood`` (>= 0),
  ``object_swap`` (in [0, len(classes))), ``camera_extrinsics`` (in
  [0, len(registry))). The contract under fuzz: any out-of-range value
  raises :class:`ValueError`; any in-range value is accepted into
  ``_pending_perturbations``. The categorical sampler's emission set is
  the in-range surface, so ``axis.sample(rng)`` outputs MUST always be
  accepted by ``set_perturbation`` without the hypothesis fuzz needing
  to know the axis-specific shape.

All tests are tagged ``hypothesis_property`` so the targeted run
``pytest -m hypothesis_property`` picks them up; the per-test
``@settings(...)`` blocks override the conftest profile only for the
slow env-construction path (the in-process MuJoCo tests cap at
``max_examples=20`` to keep wall-time bounded).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.env.perturbation import (
    AXIS_NAMES,
    DEFAULT_BOUNDS,
    OBJECT_SWAP_CLASSES,
    axis_for,
)
from gauntlet.env.tabletop import TabletopEnv

pytestmark = pytest.mark.hypothesis_property


# ----- structural invariants over the full AXIS_NAMES registry --------------


def test_every_registered_axis_has_low_le_high() -> None:
    """``axis.low <= axis.high`` for every default-built axis.

    Pinned as a one-shot rather than fuzzed: the registry is fixed at
    import time, so 200 hypothesis examples would all hit the same
    14 axes. The invariant is the contract every sampler factory keys
    off (``make_continuous_sampler`` / ``make_int_sampler`` would raise
    on inverted bounds; this test asserts the registry never trips that
    raise).
    """
    for name in AXIS_NAMES:
        axis = axis_for(name)
        assert axis.name == name, f"axis_for({name!r}) returned axis named {axis.name!r}"
        assert axis.low <= axis.high, f"axis {name!r}: low={axis.low} must be <= high={axis.high}"


def test_axis_names_matches_default_bounds_keys() -> None:
    """Every name in :data:`AXIS_NAMES` has a :data:`DEFAULT_BOUNDS` row.

    Ratchet against silent registry drift: a new axis added to
    ``AXIS_NAMES`` without a partner row in ``DEFAULT_BOUNDS`` would
    trip ``_resolve_bounds``'s ``KeyError`` only at constructor call
    time. This test surfaces it at collection time instead.
    """
    assert set(AXIS_NAMES) == set(DEFAULT_BOUNDS.keys()), (
        f"AXIS_NAMES {set(AXIS_NAMES)} drifted from DEFAULT_BOUNDS {set(DEFAULT_BOUNDS.keys())}"
    )


# ----- per-axis stability across re-seeded generators -----------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_every_axis_value_is_stable_across_rng_reseeds(seed: int) -> None:
    """For every registered axis, two freshly-seeded generators yield
    identical sample sequences. This is the spec §6 reproducibility
    contract restated per-axis (the existing
    ``test_axis_sample_is_deterministic_under_seeded_rng`` covers the
    same property; we re-pin it here so a regression on one axis does
    not silently coast through under another axis's name).

    Iterates :data:`AXIS_NAMES` rather than :data:`DEFAULT_BOUNDS` so a
    new axis added to the registry cannot bypass the property.
    """
    for name in AXIS_NAMES:
        axis = axis_for(name)
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        seq_a = [axis.sample(rng_a) for _ in range(5)]
        seq_b = [axis.sample(rng_b) for _ in range(5)]
        assert seq_a == seq_b, f"axis {name!r}: same seed produced divergent sequence"


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_every_axis_value_lies_within_declared_bounds(seed: int) -> None:
    """For every registered axis name and any rng seed, every drawn
    sample satisfies ``axis.low <= value <= axis.high`` (with a 1e-9
    slack for the float-cast comparison; matches the slack used by
    the existing primitive-sampler property test).

    1000 samples per (seed, axis) pair across 200 hypothesis examples
    works out to ~2.8M draws over the full property suite — far above
    the 1000-sample target the plan asks for, while still wall-time-
    cheap because no env is involved.
    """
    rng = np.random.default_rng(seed)
    for name in AXIS_NAMES:
        axis = axis_for(name)
        for _ in range(5):
            value = axis.sample(rng)
            assert axis.low - 1e-9 <= value <= axis.high + 1e-9, (
                f"axis {name!r}: sample {value} outside [{axis.low}, {axis.high}]"
            )


# ----- env-side hard-bound validation ---------------------------------------


@pytest.fixture(scope="module")
def env() -> Iterator[TabletopEnv]:
    """One env per module — MuJoCo model load is ~50 ms cold; reuse
    cuts the wall-time of the env-side property tests by ~Nx."""
    e = TabletopEnv(max_steps=4)
    try:
        yield e
    finally:
        e.close()


@given(
    bad_count=st.one_of(
        st.integers(min_value=-1_000_000, max_value=-1),
        st.integers(min_value=11, max_value=1_000_000),
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_set_perturbation_distractor_count_rejects_out_of_range(
    env: TabletopEnv, bad_count: int
) -> None:
    """``distractor_count`` is hard-bounded to ``[0, 10]`` by the env's
    setter (the underlying MJCF ships exactly 10 distractor slots).
    Any out-of-range value must raise :class:`ValueError` with a
    message naming the legal envelope."""
    with pytest.raises(ValueError, match=r"distractor_count must be in \[0, 10\]"):
        env.set_perturbation("distractor_count", float(bad_count))


@given(
    good_count=st.integers(min_value=0, max_value=10),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_set_perturbation_distractor_count_accepts_in_range(
    env: TabletopEnv, good_count: int
) -> None:
    """The companion of the rejection property: every value in the
    legal envelope is accepted (no exception raised). Values from the
    sampler always land in this set, so the categorical contract that
    ``axis.sample(rng)`` -> ``set_perturbation`` round-trips is pinned
    here at the env layer."""
    env.set_perturbation("distractor_count", float(good_count))


@given(
    bad_sigma=st.floats(
        min_value=-1e6,
        max_value=-1e-9,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_set_perturbation_initial_state_ood_rejects_negative_sigma(
    env: TabletopEnv, bad_sigma: float
) -> None:
    """``initial_state_ood`` is a unitless sigma magnitude (LIBERO-PRO /
    LIBERO-Plus framing). Negative values are nonsense — the sign is
    drawn per-dim from the env seed at apply time — and must be
    rejected at queue time so a buggy YAML surface fails loud rather
    than silently swallowing the negative."""
    with pytest.raises(ValueError, match=r"initial_state_ood: sigma multiplier must be >= 0"):
        env.set_perturbation("initial_state_ood", bad_sigma)


@given(
    bad_idx=st.one_of(
        st.integers(min_value=-1_000_000, max_value=-1),
        st.integers(min_value=len(OBJECT_SWAP_CLASSES), max_value=1_000_000),
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_set_perturbation_object_swap_rejects_out_of_range_index(
    env: TabletopEnv, bad_idx: int
) -> None:
    """``object_swap`` is a categorical index into the active class
    registry (defaults to :data:`OBJECT_SWAP_CLASSES`). Out-of-range
    indices must raise :class:`ValueError` rather than silently snap to
    the baseline cube — silent fallback would hide a YAML bug where a
    user trims the registry but forgets to retighten the suite values."""
    with pytest.raises(ValueError, match=r"object_swap: index"):
        env.set_perturbation("object_swap", float(bad_idx))


@given(
    bad_idx=st.one_of(
        st.integers(min_value=-1_000_000, max_value=-1),
        # The env defaults to a 1-element extrinsics registry (index 0
        # = baseline). Any index >= 1 is out of range until the suite
        # rebinds via ``set_camera_extrinsics_list``.
        st.integers(min_value=1, max_value=1_000_000),
    ),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_set_perturbation_camera_extrinsics_rejects_out_of_range_index(
    env: TabletopEnv, bad_idx: int
) -> None:
    """``camera_extrinsics`` is a categorical index into the active
    extrinsics registry (default 1-element registry; suite YAML rebinds
    via :meth:`TabletopEnv.set_camera_extrinsics_list`). Out-of-range
    indices must raise :class:`ValueError` rather than silently snap to
    the baseline pose."""
    with pytest.raises(ValueError, match=r"camera_extrinsics: index"):
        env.set_perturbation("camera_extrinsics", float(bad_idx))


def test_set_perturbation_unknown_axis_rejected(env: TabletopEnv) -> None:
    """Defence against silent typos in the suite YAML / runner:
    ``set_perturbation`` rejects any name not in the env's
    ``AXIS_NAMES`` ClassVar with a typed :class:`ValueError`."""
    with pytest.raises(ValueError, match=r"unknown perturbation axis"):
        env.set_perturbation("nonexistent_axis_name", 0.0)
