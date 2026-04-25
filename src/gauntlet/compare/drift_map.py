"""Cross-backend embodiment-transfer drift map — backlog B-29.

``gauntlet compare --allow-cross-backend`` already gates and warns when
two reports come from different sim backends, but the comparison itself
returns the same regression / improvement payload as a same-backend
compare — i.e. a yes/no answer. B-29 extends that with a structured
*drift map*: per-axis-value, how far does the same policy-and-seed
differ across backends?

The drift map is intentionally one analysis step removed from a
sim-vs-real correlation (B-28) — it scores sim-vs-sim disparity, which
is the SIMPLER (arxiv 2405.05941) "control and visual disparities"
framing. Useful as a triangulation signal: a high-drift axis-value is a
spot the perturbation does not transfer across embodiments, which is
exactly the kind of structural fragility that hides under a same-
backend pass.

Public API:

* :func:`compute_drift_map` — given two cross-backend
  :class:`gauntlet.report.Report` objects (plus the policy_label and
  suite_hash captured by the caller from the underlying episodes),
  emit a :class:`DriftMap` with per-axis-value
  :class:`AxisDrift` rows sortable by ``abs_delta``.
* :func:`top_axis_drifts` — flatten and sort the per-axis tables for
  the CLI's terminal-table render (top-N most-divergent rows).
* :func:`render_drift_map_table` — render the top-N table with rich
  for the terminal.

Validation invariants enforced inside :func:`compute_drift_map`:

* ``report_a.suite_env != report_b.suite_env`` — the inverse of the
  CLI's same-backend default; a same-backend "drift map" is a category
  error, not a degenerate-but-valid input.
* ``report_a.suite_name == report_b.suite_name`` — comparing reports
  from different suites is meaningless even with matched envs.
* ``report_a.suite_env`` and ``report_b.suite_env`` must both be set.
  Pre-RFC-005 reports lacking the env slug cannot establish
  embodiment identity; reject loudly rather than silently emit a
  half-attributed drift map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rich.console import Console

    from gauntlet.report import Report

__all__ = [
    "AxisDrift",
    "DriftMap",
    "DriftMapError",
    "compute_drift_map",
    "render_drift_map_table",
    "top_axis_drifts",
]


class DriftMapError(ValueError):
    """Raised when the inputs to :func:`compute_drift_map` are incompatible."""


class AxisDrift(BaseModel):  # type: ignore[explicit-any]
    """Per-(axis, value) drift row.

    All four numeric fields are computed as
    ``b - a`` — i.e. ``rate_b`` minus ``rate_a`` is the signed
    direction of the drift; ``abs_delta`` is the magnitude;
    ``relative_delta`` is the fractional change against the
    ``a``-baseline. ``relative_delta`` is ``None`` when ``rate_a`` is
    exactly ``0.0`` (avoiding ``+inf`` / ``-inf`` JSON poisoning).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    axis: str
    value: float
    rate_a: float
    rate_b: float
    n_a: int
    n_b: int
    delta: float
    abs_delta: float
    relative_delta: float | None


class DriftMap(BaseModel):  # type: ignore[explicit-any]
    """Structured cross-backend drift map (B-29).

    ``axes`` keys are axis names (matching ``Report.per_axis[*].name``);
    each value is the list of :class:`AxisDrift` rows for that axis,
    sorted by ``abs_delta`` descending then ``value`` ascending so the
    most-divergent values come first.

    ``total_drift`` is the mean ``abs_delta`` over every row (across
    all axes). Defined as ``0.0`` for an empty drift map (no shared
    axis-values) so the JSON round-trips without ``NaN``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    backend_a: str
    backend_b: str
    policy_label: str
    suite_hash: str
    suite_name: str
    axes: dict[str, list[AxisDrift]] = Field(default_factory=dict)
    total_drift: float


def _intersect_axis(
    name: str,
    a_rates: dict[float, float],
    b_rates: dict[float, float],
    a_counts: dict[float, int],
    b_counts: dict[float, int],
) -> list[AxisDrift]:
    """Build the per-value drift rows for one axis."""
    shared_values = sorted(set(a_rates.keys()) & set(b_rates.keys()))
    rows: list[AxisDrift] = []
    for value in shared_values:
        rate_a = a_rates[value]
        rate_b = b_rates[value]
        delta = rate_b - rate_a
        relative: float | None = None if rate_a == 0.0 else delta / rate_a
        rows.append(
            AxisDrift(
                axis=name,
                value=value,
                rate_a=rate_a,
                rate_b=rate_b,
                n_a=a_counts.get(value, 0),
                n_b=b_counts.get(value, 0),
                delta=delta,
                abs_delta=abs(delta),
                relative_delta=relative,
            )
        )
    rows.sort(key=lambda r: (-r.abs_delta, r.value))
    return rows


def compute_drift_map(
    report_a: Report,
    report_b: Report,
    *,
    policy_label: str,
    suite_hash: str,
) -> DriftMap:
    """Compute the per-axis-value drift between two cross-backend reports.

    Both reports MUST come from the same suite-and-policy-and-seed run
    against *different* sim backends; the caller is responsible for
    establishing that identity (typically by reading
    ``Episode.suite_hash`` off the underlying episodes.json files and
    threading the policy_label through ``gauntlet compare``).
    """
    if report_a.suite_env is None or report_b.suite_env is None:
        raise DriftMapError(
            "drift map requires both reports to carry suite_env; "
            "pre-RFC-005 reports without an env slug cannot establish "
            "cross-backend identity"
        )
    if report_a.suite_env == report_b.suite_env:
        raise DriftMapError(
            f"drift map requires different backends; got "
            f"suite_env={report_a.suite_env!r} on both sides"
        )
    if report_a.suite_name != report_b.suite_name:
        raise DriftMapError(
            f"drift map requires identical suite_name; "
            f"got a={report_a.suite_name!r}, b={report_b.suite_name!r}"
        )

    a_axes = {ax.name: ax for ax in report_a.per_axis}
    b_axes = {ax.name: ax for ax in report_b.per_axis}
    shared_axis_names = sorted(a_axes.keys() & b_axes.keys())

    axes: dict[str, list[AxisDrift]] = {}
    all_rows: list[AxisDrift] = []
    for name in shared_axis_names:
        rows = _intersect_axis(
            name,
            a_axes[name].rates,
            b_axes[name].rates,
            a_axes[name].counts,
            b_axes[name].counts,
        )
        if rows:
            axes[name] = rows
            all_rows.extend(rows)

    total_drift = sum(r.abs_delta for r in all_rows) / len(all_rows) if all_rows else 0.0

    return DriftMap(
        backend_a=report_a.suite_env,
        backend_b=report_b.suite_env,
        policy_label=policy_label,
        suite_hash=suite_hash,
        suite_name=report_a.suite_name,
        axes=axes,
        total_drift=total_drift,
    )


def top_axis_drifts(drift_map: DriftMap, *, limit: int = 5) -> list[AxisDrift]:
    """Return the ``limit`` most-divergent rows across all axes.

    Sort key matches :func:`_intersect_axis` (``-abs_delta`` then
    ``value`` ascending) with axis name as the final tie-breaker so two
    rows with identical magnitude render in a stable order across runs.
    """
    flat: list[AxisDrift] = []
    for rows in drift_map.axes.values():
        flat.extend(rows)
    flat.sort(key=lambda r: (-r.abs_delta, r.axis, r.value))
    return flat[:limit]


def render_drift_map_table(
    drift_map: DriftMap,
    console: Console,
    *,
    limit: int = 5,
) -> None:
    """Render the top-``limit`` drift rows as a rich Table on ``console``.

    Coloring follows the rest of the CLI: negative deltas are bad
    (regression vs. ``a``), positive are good. ``relative_delta`` of
    ``None`` (zero ``a``-baseline) renders as a dash so the column
    width is stable.
    """
    # Lazy import — keeps `gauntlet --help` snappy when no drift map
    # is requested (mirrors compare's github_summary lazy import).
    from rich.table import Table

    rows = top_axis_drifts(drift_map, limit=limit)
    title = (
        f"Top-{len(rows)} drift: {drift_map.backend_a} vs "
        f"{drift_map.backend_b} (policy={drift_map.policy_label}, "
        f"total_drift={drift_map.total_drift:.3f})"
    )
    table = Table(title=title, show_lines=False)
    table.add_column("Axis", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Rate A", justify="right")
    table.add_column("Rate B", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("|Delta|", justify="right")
    table.add_column("Rel.", justify="right")
    for row in rows:
        delta_style = (
            "delta.down" if row.delta < 0 else ("delta.up" if row.delta > 0 else "delta.zero")
        )
        rel_str = "-" if row.relative_delta is None else f"{row.relative_delta:+.2f}"
        table.add_row(
            row.axis,
            f"{row.value:g}",
            f"{row.rate_a * 100:.1f}%",
            f"{row.rate_b * 100:.1f}%",
            f"[{delta_style}]{row.delta * 100:+.1f}%[/]",
            f"{row.abs_delta * 100:.1f}%",
            rel_str,
        )
    console.print(table)
