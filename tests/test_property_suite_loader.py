"""Property-based tests for ``gauntlet.suite.loader.load_suite_from_string``.

Phase 2.5 Task 13 — fuzzes the YAML loader with both well-formed and
malformed inputs. The contract under test is the *no-crash* envelope:
every input either parses to a valid :class:`Suite` or raises one of
the documented error types (``ValueError``, ``pydantic.ValidationError``,
``yaml.YAMLError``). Anything else propagates and is treated as a bug.

Hypothesis budget: ``max_examples=50`` per test; small per-iteration
cost because we never instantiate an env.
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

# Documented error types: any input that is not a valid Suite must raise
# one of these; anything else is an uncaught crash and a real bug.
_DOCUMENTED_ERRORS = (ValueError, pydantic.ValidationError, yaml.YAMLError)


# ----- well-formed YAML strategy ---------------------------------------------


# All Phase 1 / Phase 2 envs known to the schema before any optional
# extra is loaded. ``tabletop`` is always registered; the rest live behind
# ``BUILTIN_BACKEND_IMPORTS`` and trigger a lazy import. To keep the
# fuzzer fast and offline-safe we stick to ``tabletop``.
_ENV_NAME = st.just("tabletop")

_AXIS_NAME = st.sampled_from(AXIS_NAMES)


@st.composite
def _continuous_axis_yaml(draw: st.DrawFn) -> str:
    """One axis block with the ``low/high/steps`` shape."""
    lo = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo, max_value=lo + 5.0, allow_nan=False, allow_infinity=False))
    steps = draw(st.integers(min_value=1, max_value=4))
    return f"    low: {lo}\n    high: {hi}\n    steps: {steps}\n"


@st.composite
def _categorical_axis_yaml(draw: st.DrawFn) -> str:
    """One axis block with the ``values`` shape."""
    n = draw(st.integers(min_value=1, max_value=4))
    vals = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    rendered = ", ".join(str(v) for v in vals)
    return f"    values: [{rendered}]\n"


@st.composite
def _well_formed_suite_yaml(draw: st.DrawFn) -> str:
    """Generate a YAML string that the loader should parse successfully."""
    # Wrap the name in double quotes so YAML always parses it as a
    # string — bareword names like ``null`` / ``true`` would otherwise
    # be coerced to None / bool by ``yaml.safe_load``.
    raw_name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_-", min_size=1, max_size=20))
    name = f'"{raw_name}"'
    env = draw(_ENV_NAME)
    eps = draw(st.integers(min_value=1, max_value=3))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1)))
    n_axes = draw(st.integers(min_value=1, max_value=3))
    axis_names = draw(
        st.lists(_AXIS_NAME, min_size=n_axes, max_size=n_axes, unique=True),
    )
    axis_blocks: list[str] = []
    for axis_name in axis_names:
        body = draw(st.one_of(_continuous_axis_yaml(), _categorical_axis_yaml()))
        axis_blocks.append(f"  {axis_name}:\n{body}")

    seed_line = f"seed: {seed}\n" if seed is not None else ""
    return f"name: {name}\nenv: {env}\nepisodes_per_cell: {eps}\n{seed_line}axes:\n" + "".join(
        axis_blocks
    )


@given(yaml_text=_well_formed_suite_yaml())
@settings(
    max_examples=50,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_well_formed_yaml_parses_to_valid_suite(yaml_text: str) -> None:
    """Sanity property: the well-formed strategy round-trips through the
    loader without raising. If this regresses, either the strategy or
    the schema is genuinely wrong."""
    suite = load_suite_from_string(yaml_text)
    assert suite.env == "tabletop"
    assert suite.episodes_per_cell >= 1
    assert len(suite.axes) >= 1
    # ``num_cells`` must be the product of each axis's enumerated grid.
    expected = 1
    for spec in suite.axes.values():
        expected *= len(spec.enumerate())
    assert suite.num_cells() == expected


# ----- malformed YAML -- never crashes uncaught ------------------------------


@given(yaml_text=st.text(max_size=200))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_arbitrary_text_either_parses_or_raises_documented_error(yaml_text: str) -> None:
    """Any string at all may go in. The loader either:
    a) returns a valid :class:`Suite` (vanishingly rare for random text), or
    b) raises one of ``(ValueError, pydantic.ValidationError, yaml.YAMLError)``.

    Anything else propagates and is treated as a real defect by hypothesis."""
    try:
        load_suite_from_string(yaml_text)
    except _DOCUMENTED_ERRORS:
        return  # acceptable failure mode


@given(
    raw_top=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.text(max_size=20),
        st.lists(st.integers(), max_size=5),
    ),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_non_mapping_top_level_rejected_with_value_error(raw_top: object) -> None:
    """A YAML doc whose top-level node is not a mapping (scalar / null /
    list) must produce a clear ``ValueError`` before pydantic ever
    sees the payload — see ``loader._validate``'s explicit guard."""
    yaml_text = yaml.safe_dump(raw_top)
    with pytest.raises(ValueError):
        load_suite_from_string(yaml_text)


# ----- structural mutations to a known-good YAML -----------------------------


_GOOD_YAML = (
    "name: prop-fuzz\n"
    "env: tabletop\n"
    "episodes_per_cell: 1\n"
    "seed: 7\n"
    "axes:\n"
    "  lighting_intensity:\n"
    "    low: 0.3\n"
    "    high: 0.9\n"
    "    steps: 2\n"
)


@given(missing_key=st.sampled_from(["name", "env", "episodes_per_cell", "axes"]))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_missing_required_field_raises_validation_error(missing_key: str) -> None:
    """Removing any required top-level key turns the YAML into a
    schema-violating mapping — pydantic must reject it cleanly."""
    lines = [ln for ln in _GOOD_YAML.splitlines() if not ln.startswith(f"{missing_key}:")]
    if missing_key == "axes":
        # The 'axes:' header line is followed by indented children; drop them too.
        lines = [
            ln
            for ln in lines
            if not (ln.startswith("  ") and not ln.startswith("    ")) and not ln.startswith("    ")
        ]
    mutated = "\n".join(lines) + "\n"
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(mutated)


@given(
    bad_eps=st.integers(min_value=-100, max_value=0),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_episodes_per_cell_must_be_positive(bad_eps: int) -> None:
    """``episodes_per_cell <= 0`` must raise the documented validation
    error from the field validator in ``Suite._episodes_positive``."""
    yaml_text = _GOOD_YAML.replace("episodes_per_cell: 1", f"episodes_per_cell: {bad_eps}")
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(yaml_text)


@given(unknown_axis=st.text(alphabet="abcdefghij_", min_size=1, max_size=15))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_unknown_axis_name_rejected(unknown_axis: str) -> None:
    """Any axis key outside :data:`AXIS_NAMES` must be rejected by the
    field validator on ``Suite.axes``."""
    if unknown_axis in AXIS_NAMES:
        return  # the well-formed half is covered elsewhere.
    yaml_text = (
        "name: bad-axis\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        f"  {unknown_axis}:\n"
        "    values: [0.0]\n"
    )
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(yaml_text)


@given(
    bad_steps=st.integers(min_value=-100, max_value=0),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_axis_steps_must_be_positive(bad_steps: int) -> None:
    """``steps < 1`` must trip ``AxisSpec._steps_positive``."""
    yaml_text = (
        "name: bad-steps\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    low: 0.3\n"
        "    high: 0.9\n"
        f"    steps: {bad_steps}\n"
    )
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(yaml_text)


@given(
    lo=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    hi=st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_axis_low_above_high_rejected(lo: float, hi: float) -> None:
    """``low > high`` on a continuous axis must trip the model-level
    bounds check in ``AxisSpec._check_shape_exclusive``."""
    yaml_text = (
        "name: inverted\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        "  lighting_intensity:\n"
        f"    low: {lo}\n"
        f"    high: {hi}\n"
        "    steps: 2\n"
    )
    with pytest.raises(_DOCUMENTED_ERRORS):
        load_suite_from_string(yaml_text)
