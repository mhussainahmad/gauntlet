"""Parse and validate ``--override AXIS=VALUE`` flags for ``gauntlet replay``.

See ``docs/phase2-rfc-004-trajectory-replay.md`` §5. Two-layer validation:

* **Name**: the axis must appear in ``suite.axes``. Axes outside the
  suite's declared set are rejected — replaying against an axis the
  original run never varied would mean extending the perturbation
  surface beyond what the user ran, which is a different experiment
  (and is what ``gauntlet run`` is for).
* **Value**: for continuous axes (``{low, high, steps}`` shape) the
  parsed float must lie within ``[low, high]`` inclusive. For
  categorical axes (``{values}`` shape) the value must match one of the
  declared values within a 1e-9 tolerance. Off-grid values are
  permitted — the whole point of replay is exploring *off* the original
  grid; only the axis's declared envelope is enforced.

For axes with hard env-side bounds beyond the YAML envelope (only
``distractor_count`` today: ``TabletopEnv`` clamps to
``[0, N_DISTRACTOR_SLOTS]``), we re-assert the env's constraint at the
CLI boundary so the user sees the error before the env is constructed.

All errors raise :class:`OverrideError` — a :class:`ValueError` subclass
so existing ``except ValueError`` handlers keep working, but the CLI
glue can catch this specific type to format its output cleanly.
"""

from __future__ import annotations

from gauntlet.env.tabletop import N_DISTRACTOR_SLOTS
from gauntlet.suite.schema import AxisSpec, Suite

__all__ = [
    "OverrideError",
    "parse_override",
    "validate_overrides",
]


# Tolerance for matching a floated user value against a categorical
# axis's declared values. 1e-9 is below any realistic float literal in
# a suite YAML and well above numpy/yaml round-trip noise.
_CATEGORICAL_TOL: float = 1e-9


class OverrideError(ValueError):
    """Raised when a ``--override`` flag is malformed or rejected.

    Subclasses :class:`ValueError` so callers that already catch
    ``ValueError`` keep working without a breaking change.
    """


def parse_override(spec: str) -> tuple[str, float]:
    """Parse ``"axis=value"`` into a ``(name, float(value))`` tuple.

    The format is deliberately strict to match the RFC §4 CLI contract:
    exactly one ``=``, non-empty axis name on the left, a float on the
    right. Leading / trailing whitespace around each half is stripped
    (a convenience, not a looseness).

    Raises:
        OverrideError: if *spec* is empty, missing ``=``, has more than
            one ``=``, has an empty axis name, or the right-hand side
            does not parse as a float.
    """
    if not spec or not spec.strip():
        raise OverrideError("override spec must be a non-empty string")
    raw = spec.strip()
    # Reject zero or multiple '=' up-front so we can produce a single
    # clear error rather than piecemeal messages from str.split.
    if raw.count("=") != 1:
        raise OverrideError(
            f"override spec {spec!r}: expected 'AXIS=VALUE' with exactly one '=' "
            f"(got {raw.count('=')})"
        )
    axis_name, raw_value = raw.split("=", 1)
    axis_name = axis_name.strip()
    raw_value = raw_value.strip()
    if not axis_name:
        raise OverrideError(
            f"override spec {spec!r}: axis name on the left of '=' must be non-empty"
        )
    if not raw_value:
        raise OverrideError(f"override spec {spec!r}: value on the right of '=' must be non-empty")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise OverrideError(
            f"override spec {spec!r}: value {raw_value!r} is not a valid float"
        ) from exc
    return axis_name, value


def _value_allowed(axis: AxisSpec, value: float) -> bool:
    """Check whether *value* lies within *axis*'s declared envelope.

    Continuous / int shape (``{low, high, steps}``): value is accepted
    if it lies in ``[low, high]`` inclusive. Off-grid values are
    permitted — replay's whole point is exploring *off* the enumerated
    grid points.

    Categorical shape (``{values}``): value must match one of the
    declared values within :data:`_CATEGORICAL_TOL`. This handles YAML
    round-trip noise (e.g. ``0.1`` -> ``0.10000000000000001`` after a
    float cast) without permitting arbitrary off-set values.
    """
    if axis.values is not None:
        # ``axis.values`` widened to ``list[float] | list[str]`` after
        # B-05 added the categorical-string ``instruction_paraphrase``
        # axis. ``parse_override`` constrains the user's value to
        # ``float``, so str-valued entries can never match by
        # construction; filter to numeric entries before subtracting so
        # mypy --strict accepts the operand and so a future float / str
        # mixed-list axis (none today) silently degrades to "no match"
        # rather than ``TypeError`` at runtime.
        return any(
            isinstance(v, (int, float)) and abs(value - v) <= _CATEGORICAL_TOL for v in axis.values
        )
    # Continuous: _check_shape_exclusive guarantees low and high are set.
    assert axis.low is not None
    assert axis.high is not None
    return axis.low <= value <= axis.high


def _format_envelope(axis: AxisSpec) -> str:
    """Render an axis's declared envelope for error messages."""
    if axis.values is not None:
        return "values=" + ", ".join(repr(v) for v in axis.values)
    return f"[{axis.low}, {axis.high}]"


def validate_overrides(overrides: dict[str, float], suite: Suite) -> None:
    """Validate *overrides* against *suite*'s declared axes.

    Applies the two-layer check from RFC §5: axis name must be declared
    in ``suite.axes``, value must lie within the declared axis
    envelope. The env-level bound on ``distractor_count``
    (``[0, N_DISTRACTOR_SLOTS]``) is re-asserted here so users see the
    error at the CLI boundary instead of inside a worker traceback.

    Raises:
        OverrideError: on the first offending axis or value. The error
            message names the legal axes (for unknown names) or the
            declared envelope (for out-of-range values) so the user
            can retype without digging into the suite YAML.
    """
    for axis_name, value in overrides.items():
        if axis_name not in suite.axes:
            legal = ", ".join(sorted(suite.axes.keys()))
            raise OverrideError(
                f"override axis {axis_name!r} is not declared in suite {suite.name!r}; "
                f"legal axes: {legal}"
            )
        axis = suite.axes[axis_name]
        if not _value_allowed(axis, value):
            raise OverrideError(
                f"override {axis_name}={value}: value is outside the suite's declared "
                f"envelope {_format_envelope(axis)}"
            )
        # Re-assert the env-level hard bound for distractor_count so
        # the user sees the error before env construction. Other axes
        # trust the YAML envelope (env/tabletop.py has no other hard
        # bounds today).
        if axis_name == "distractor_count":
            count = round(float(value))
            if count < 0 or count > N_DISTRACTOR_SLOTS:
                raise OverrideError(
                    f"override distractor_count={value}: env bound is "
                    f"[0, {N_DISTRACTOR_SLOTS}]; got {count} after rounding"
                )
