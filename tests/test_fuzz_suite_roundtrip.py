"""Fuzz tests for ``Suite`` ↔ YAML round-trip identity.

Phase 2.5 Task 13 — covers the gap left by ``test_property_suite_loader``:
that test fuzzes the loader's *no-crash* envelope but never asserts the
YAML round-trip property. Concretely:

* Hypothesis builds a valid :class:`Suite`.
* We ``yaml.safe_dump(suite.model_dump(...))`` it back to a string.
* We feed that string to :func:`load_suite_from_string`.
* The resulting :class:`Suite` must equal the original.

The interesting bit here is the cross-product of:

* Cartesian / LHS / Sobol sampling modes (each with its own
  ``n_samples`` / per-axis ``steps`` rule).
* Continuous (``low/high/steps``) and categorical (``values``) axis
  shapes.
* Insertion-order preservation — Pydantic v2 + PyYAML 6 both promise
  it; this is the round-trip surface that catches a regression.
* ``ConfigDict(extra="forbid")`` interaction with ``model_dump`` —
  every dumped field must be a known field, otherwise re-validation
  raises.

Hypothesis budget: ``max_examples=50`` per test, sub-millisecond per
case (no env, no I/O — just YAML serialisation).
"""

from __future__ import annotations

from datetime import timedelta

import pydantic
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.env.perturbation import AXIS_NAMES
from gauntlet.suite.loader import load_suite_from_string
from gauntlet.suite.schema import AxisSpec, Suite

# Bound the float envelope so (a) a YAML dump never produces a value
# Python can't parse back exactly, and (b) the affine-map samplers
# ``low + u * (high - low)`` don't overflow inside the schema validator.
_BOUND_FLOAT = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
# Distractor bounds must lie within [0, 10] (see ``axes.distractor_count``)
# — strategies that emit values outside that range trip a ValueError at
# Suite construction time, which would short-circuit the round-trip
# property. Keep the hypothesis at "valid Suites" by avoiding the
# distractor axis name in the round-trip strategies.
_AXIS_NAME_NO_DISTRACTOR = st.sampled_from(
    tuple(name for name in AXIS_NAMES if name != "distractor_count")
)


@st.composite
def _continuous_axis_with_steps(draw: st.DrawFn) -> AxisSpec:
    a = draw(_BOUND_FLOAT)
    b = draw(_BOUND_FLOAT)
    lo, hi = (a, b) if a <= b else (b, a)
    steps = draw(st.integers(min_value=1, max_value=4))
    return AxisSpec(low=lo, high=hi, steps=steps)


@st.composite
def _continuous_axis_without_steps(draw: st.DrawFn) -> AxisSpec:
    a = draw(_BOUND_FLOAT)
    b = draw(_BOUND_FLOAT)
    lo, hi = (a, b) if a <= b else (b, a)
    return AxisSpec(low=lo, high=hi)


@st.composite
def _categorical_axis(draw: st.DrawFn) -> AxisSpec:
    n = draw(st.integers(min_value=1, max_value=4))
    vals = draw(st.lists(_BOUND_FLOAT, min_size=n, max_size=n))
    return AxisSpec(values=[float(v) for v in vals])


@st.composite
def _cartesian_suite(draw: st.DrawFn) -> Suite:
    n_axes = draw(st.integers(min_value=1, max_value=3))
    names = draw(st.lists(_AXIS_NAME_NO_DISTRACTOR, min_size=n_axes, max_size=n_axes, unique=True))
    axes: dict[str, AxisSpec] = {
        name: draw(st.one_of(_continuous_axis_with_steps(), _categorical_axis())) for name in names
    }
    eps = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    return Suite(
        name="prop-roundtrip-cart",
        env="tabletop",
        episodes_per_cell=eps,
        seed=seed,
        axes=axes,
        sampling="cartesian",
    )


@st.composite
def _stochastic_suite(draw: st.DrawFn, mode: str) -> Suite:
    n_axes = draw(st.integers(min_value=1, max_value=3))
    names = draw(st.lists(_AXIS_NAME_NO_DISTRACTOR, min_size=n_axes, max_size=n_axes, unique=True))
    axes: dict[str, AxisSpec] = {
        name: draw(st.one_of(_continuous_axis_without_steps(), _categorical_axis()))
        for name in names
    }
    eps = draw(st.integers(min_value=1, max_value=4))
    n_samples = draw(st.integers(min_value=1, max_value=8))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    return Suite(
        name=f"prop-roundtrip-{mode}",
        env="tabletop",
        episodes_per_cell=eps,
        seed=seed,
        axes=axes,
        sampling=mode,  # type: ignore[arg-type]
        n_samples=n_samples,
    )


def _dump_suite_yaml(suite: Suite) -> str:
    """Dump a :class:`Suite` to a YAML string the loader can re-parse.

    ``model_dump(exclude_none=True)`` strips fields whose value is
    ``None`` (e.g. ``seed`` when unset, ``n_samples`` for cartesian
    suites, ``steps`` on a non-cartesian axis). Without this the dumped
    YAML carries ``seed: null`` / ``n_samples: null`` keys; the loader's
    cross-field validator forbids ``n_samples`` on a cartesian Suite, so
    a literal ``null`` would trip the validator.

    ``sort_keys=False`` preserves Pydantic's declared field order, which
    matters for the round-trip equality on ``axes`` (insertion order is
    the contract per the schema docstring).
    """
    payload = suite.model_dump(exclude_none=True)
    return yaml.safe_dump(payload, sort_keys=False)


def _suites_equal(a: Suite, b: Suite) -> bool:
    """Structural equality on the dumped representation.

    ``Suite`` does not implement ``__eq__`` (Pydantic v2 generates one
    for BaseModel subclasses by default — but only on field-by-field
    comparison). We compare the ``model_dump(exclude_none=True)``
    payload directly so the test is explicit about what "equal" means.
    """
    return a.model_dump(exclude_none=True) == b.model_dump(exclude_none=True)


# ----- cartesian suites ------------------------------------------------------


@given(suite=_cartesian_suite())
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_cartesian_suite_yaml_round_trip_identity(suite: Suite) -> None:
    """``Suite -> YAML -> Suite`` is the identity for cartesian suites."""
    yaml_text = _dump_suite_yaml(suite)
    rt = load_suite_from_string(yaml_text)
    assert _suites_equal(suite, rt), (
        f"round-trip drift:\n  before={suite.model_dump(exclude_none=True)}\n"
        f"  after ={rt.model_dump(exclude_none=True)}\n  yaml=\n{yaml_text}"
    )


# ----- stochastic suites -----------------------------------------------------


@given(suite=_stochastic_suite(mode="latin_hypercube"))
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_lhs_suite_yaml_round_trip_identity(suite: Suite) -> None:
    yaml_text = _dump_suite_yaml(suite)
    rt = load_suite_from_string(yaml_text)
    assert _suites_equal(suite, rt)


@given(suite=_stochastic_suite(mode="sobol"))
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sobol_suite_yaml_round_trip_identity(suite: Suite) -> None:
    yaml_text = _dump_suite_yaml(suite)
    rt = load_suite_from_string(yaml_text)
    assert _suites_equal(suite, rt)


# ----- axis insertion order is preserved -------------------------------------


@given(
    perm=st.permutations([n for n in AXIS_NAMES if n != "distractor_count"]).map(lambda p: p[:3]),
)
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_axis_insertion_order_survives_round_trip(perm: list[str]) -> None:
    """``Suite.axes`` is an insertion-ordered dict; the round-trip must
    not permute the keys (Runner / Report key off this order)."""
    axes = {name: AxisSpec(low=0.0, high=1.0, steps=2) for name in perm}
    suite = Suite(
        name="prop-roundtrip-order",
        env="tabletop",
        episodes_per_cell=1,
        axes=axes,
        sampling="cartesian",
    )
    rt = load_suite_from_string(_dump_suite_yaml(suite))
    assert list(rt.axes.keys()) == perm


# ----- cells() output is identical pre- and post-round-trip ------------------


@given(suite=_cartesian_suite())
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_round_trip_preserves_cells_enumeration(suite: Suite) -> None:
    """The downstream Runner consumes ``Suite.cells()``; the round-trip
    must not change the enumerated cell list (which would silently
    change run-output identity)."""
    rt = load_suite_from_string(_dump_suite_yaml(suite))
    cells_before = list(suite.cells())
    cells_after = list(rt.cells())
    assert len(cells_before) == len(cells_after)
    for ca, cb in zip(cells_before, cells_after, strict=True):
        assert ca.index == cb.index
        assert ca.values == cb.values


# ----- non-round-trip pin: dumped payload re-validates -----------------------


@given(suite=_cartesian_suite())
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_dumped_payload_revalidates_through_model_validate(suite: Suite) -> None:
    """``Suite.model_validate(suite.model_dump(exclude_none=True))`` must
    succeed. If the dump emits a field the validator forbids (or vice
    versa), every YAML round-trip is broken and this test catches it
    earlier in the pipeline than the YAML one."""
    payload = suite.model_dump(exclude_none=True)
    rt = Suite.model_validate(payload)
    assert _suites_equal(suite, rt)


# ----- a plain regression: well-known YAML survives the round-trip ----------


def test_known_good_yaml_survives_round_trip() -> None:
    """Anchors the property tests: a hand-written YAML the project ships
    with must re-emerge byte-equivalent (per ``model_dump``) after a
    full round-trip."""
    yaml_text = (
        "name: example\n"
        "env: tabletop\n"
        "episodes_per_cell: 2\n"
        "seed: 42\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    low: 0.3\n"
        "    high: 1.5\n"
        "    steps: 3\n"
        "  object_initial_pose_x:\n"
        "    values: [-0.1, 0.0, 0.1]\n"
    )
    suite = load_suite_from_string(yaml_text)
    rt = load_suite_from_string(_dump_suite_yaml(suite))
    assert _suites_equal(suite, rt)


def test_dumped_yaml_omits_none_fields_so_loader_accepts_it() -> None:
    """Defence against a regression where ``model_dump`` would emit
    ``n_samples: null`` for a cartesian Suite; the loader's cross-field
    validator would then reject it."""
    suite = Suite(
        name="cart",
        env="tabletop",
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(low=0.3, high=1.5, steps=2)},
        sampling="cartesian",
    )
    payload = suite.model_dump(exclude_none=True)
    assert "n_samples" not in payload, "cartesian Suite must not emit n_samples"
    # And re-loading the dumped string still works.
    rt = load_suite_from_string(yaml.safe_dump(payload, sort_keys=False))
    assert _suites_equal(suite, rt)


def test_unknown_field_in_dumped_payload_is_rejected_by_extra_forbid() -> None:
    """Pin ``ConfigDict(extra='forbid')`` on Suite — silent additions
    would defeat the round-trip identity property."""
    payload = {
        "name": "bad",
        "env": "tabletop",
        "episodes_per_cell": 1,
        "axes": {"lighting_intensity": {"low": 0.3, "high": 1.5, "steps": 2}},
        "future_field_we_did_not_declare": True,
    }
    with pytest.raises(pydantic.ValidationError):
        Suite.model_validate(payload)
