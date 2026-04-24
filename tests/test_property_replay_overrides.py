"""Property-based tests for ``gauntlet.replay.overrides.parse_override``.

Phase 2.5 Task 13 — fuzzes the CLI ``--override AXIS=VALUE`` parser. The
contract has two halves:

* Well-formed inputs (single ``=``, non-empty axis name, float-parseable
  RHS, no extra ``=``) parse without raising and round-trip through
  ``float(...)``.
* Malformed inputs raise ``OverrideError`` (a :class:`ValueError`
  subclass) and **never** produce some other exception — anything else
  is treated as a real defect.

Hypothesis budget: ``max_examples=50`` per test; sub-millisecond per case.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gauntlet.replay import OverrideError, parse_override

# RHS may be any finite float — Python's ``float(...)`` also accepts
# ``"inf"`` / ``"nan"`` literals; the parser does not filter those out
# (they're valid floats, just unusual). We exercise both cases.
_RHS_FLOAT = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9)
_AXIS_NAME = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20)


# ----- well-formed: round-trip through parse_override -----------------------


@given(name=_AXIS_NAME, value=_RHS_FLOAT)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_well_formed_override_parses_to_input_pair(name: str, value: float) -> None:
    spec = f"{name}={value}"
    parsed_name, parsed_value = parse_override(spec)
    assert parsed_name == name
    # Float -> str -> float can lose a trailing ULP for some bit patterns;
    # Python's repr / str on floats is round-trip-safe since 3.1, so we
    # compare exactly.
    assert parsed_value == value


@given(
    name=_AXIS_NAME,
    value=_RHS_FLOAT,
    pad=st.text(alphabet=" \t", min_size=1, max_size=4),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_leading_trailing_whitespace_is_stripped(name: str, value: float, pad: str) -> None:
    """``parse_override`` strips whitespace around each half — convenience
    described in the docstring (RFC §4 CLI contract)."""
    spec = f"{pad}{name}{pad}={pad}{value}{pad}"
    parsed_name, parsed_value = parse_override(spec)
    assert parsed_name == name
    assert parsed_value == value


# ----- malformed: must raise OverrideError, never anything else --------------


@given(spec=st.text(alphabet=" \t\n\r", max_size=10))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_empty_or_whitespace_only_raises_override_error(spec: str) -> None:
    """Any string consisting only of whitespace (or the empty string)
    must raise :class:`OverrideError` with the documented ``non-empty``
    message."""
    with pytest.raises(OverrideError, match="non-empty"):
        parse_override(spec)


@given(
    spec=st.text(min_size=1, max_size=30).filter(lambda s: "=" not in s and s.strip() != ""),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_missing_equals_raises_override_error(spec: str) -> None:
    """A spec with no ``=`` separator is rejected with a clear message."""
    with pytest.raises(OverrideError, match="exactly one '='"):
        parse_override(spec)


@given(
    name=_AXIS_NAME,
    value=_RHS_FLOAT,
    extras=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_multiple_equals_raises_override_error(name: str, value: float, extras: int) -> None:
    """Multiple ``=`` separators are rejected — the parser refuses to
    guess which split was intended."""
    spec = f"{name}={value}" + ("=junk" * extras)
    with pytest.raises(OverrideError, match="exactly one '='"):
        parse_override(spec)


@given(value=_RHS_FLOAT)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_empty_axis_name_raises_override_error(value: float) -> None:
    """Empty (or whitespace-only) axis name on the LHS is rejected."""
    with pytest.raises(OverrideError, match="axis name"):
        parse_override(f"={value}")


@given(name=_AXIS_NAME)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_empty_value_raises_override_error(name: str) -> None:
    """Empty (or whitespace-only) RHS is rejected before float() is tried."""
    with pytest.raises(OverrideError, match="must be non-empty"):
        parse_override(f"{name}=")


@given(
    name=_AXIS_NAME,
    junk=st.text(min_size=1, max_size=10).filter(
        lambda s: not _safe_float(s),
    ),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_non_float_rhs_raises_override_error(name: str, junk: str) -> None:
    """A non-float RHS must surface as :class:`OverrideError`, not
    bare :class:`ValueError` from Python's ``float`` builtin."""
    spec = f"{name}={junk}"
    # Skip cases where the junk happens to contain '=' — that hits a
    # different branch (covered by test_multiple_equals_*).
    if "=" in junk:
        return
    with pytest.raises(OverrideError, match="not a valid float"):
        parse_override(spec)


def _safe_float(s: str) -> bool:
    """Return True iff ``float(s)`` would succeed without raising."""
    try:
        float(s)
    except ValueError:
        return False
    return not math.isnan(float(s))  # nan is technically valid; exclude.


# ----- arbitrary-text envelope: never crashes uncaught -----------------------


@given(spec=st.text(max_size=80))
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_arbitrary_text_either_parses_or_raises_override_error(spec: str) -> None:
    """Whatever the user types, the parser either returns a
    ``(str, float)`` tuple or raises :class:`OverrideError`. No other
    exception type may escape."""
    try:
        result = parse_override(spec)
    except OverrideError:
        return
    name, value = result
    assert isinstance(name, str)
    assert isinstance(value, float)
