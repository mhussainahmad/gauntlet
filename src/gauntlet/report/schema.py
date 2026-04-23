"""Pydantic models for failure-analysis reports.

See ``GAUNTLET_SPEC.md`` §3 (Report "takes a list of Episode results +
produces per-axis breakdowns, failure clusters, and an HTML artifact")
and §6 ("never aggregate away failures — every report must show the
breakdown before the mean").

Design notes:

* Every model uses ``ConfigDict(extra="forbid")`` — silent additions are
  a contract violation, not a feature.
* The ``Report`` field order is intentional: breakdowns first
  (``per_axis``, ``per_cell``, ``failure_clusters``, ``heatmap_2d``),
  scalar means last (``overall_success_rate``, ``overall_failure_rate``).
  Per spec §6 the breakdowns are the headline.
* All float axis values that appear as ``dict`` keys are normalized to 9
  decimal places at construction time (see
  :func:`gauntlet.report.analyze._norm`); this avoids splitting one
  intended grid value across two buckets when floating-point arithmetic
  drifts by 1e-15 or so.
* :class:`Heatmap2D` keys in :attr:`Report.heatmap_2d` follow
  ``f"{axis_x}__{axis_y}"`` where ``axis_x`` and ``axis_y`` come from
  ``itertools.combinations`` over the union axis order; the convention
  is documented here so downstream HTML / JSON consumers don't guess.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "AxisBreakdown",
    "CellBreakdown",
    "FailureCluster",
    "Heatmap2D",
    "Report",
]


class AxisBreakdown(BaseModel):
    """Marginal success rate for one axis, broken down by axis value.

    ``rates``, ``counts`` and ``successes`` share the same set of keys
    (the unique axis values that appear in the episode list, sorted
    ascending). Keys are float-normalized — see module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rates: dict[float, float]
    counts: dict[float, int]
    successes: dict[float, int]


class CellBreakdown(BaseModel):
    """Aggregate stats for a single ``(cell_index, perturbation_config)`` cell.

    The Runner guarantees that all episodes sharing a ``cell_index``
    also share their ``perturbation_config``; we still group on both so
    the model is robust to hand-constructed Episode lists.
    """

    model_config = ConfigDict(extra="forbid")

    cell_index: int
    perturbation_config: dict[str, float]
    n_episodes: int
    n_success: int
    success_rate: float


class FailureCluster(BaseModel):
    """An axis-PAIR value combination with elevated failure rate.

    Per spec §6 ("never aggregate away failures"), this is the surface
    that lets a human see *which* axis-value combinations break the
    policy. ``axes`` always has exactly two entries (one per axis in the
    pair). Inclusion criterion (computed in
    :mod:`gauntlet.report.analyze`):

    * ``n_episodes >= min_cluster_size`` (default 3), AND
    * ``failure_rate >= cluster_multiple * baseline_failure_rate``.

    ``lift`` is ``failure_rate / baseline_failure_rate``; the
    ``failure_clusters`` list is sorted by ``lift`` descending then
    ``failure_rate`` descending for stable presentation.
    """

    model_config = ConfigDict(extra="forbid")

    axes: dict[str, float]
    n_episodes: int
    n_success: int
    failure_rate: float
    lift: float


class Heatmap2D(BaseModel):
    """2D success-rate matrix for a pair of axes.

    Layout:

    * ``axis_x``, ``axis_y`` — axis names. ``axis_x`` varies along
      columns, ``axis_y`` along rows.
    * ``x_values`` / ``y_values`` — sorted-ascending unique values for
      the corresponding axis (float-normalized).
    * ``success_rate[y_index][x_index]`` — success rate of the episodes
      that hit *both* ``axis_x = x_values[x_index]`` AND ``axis_y =
      y_values[y_index]``. Cells with no episodes are ``float("nan")``.
    """

    model_config = ConfigDict(extra="forbid")

    axis_x: str
    axis_y: str
    x_values: list[float]
    y_values: list[float]
    success_rate: list[list[float]]


class Report(BaseModel):
    """Top-level failure-analysis report.

    Built by :func:`gauntlet.report.analyze.build_report` from a list
    of :class:`gauntlet.runner.Episode` results. The field order is
    deliberately breakdown-first (§6).
    """

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    # Optional for Phase-1 compat: old report JSONs (pre RFC-005) do
    # not carry the env slug and we accept them unchanged. New reports
    # written by :func:`gauntlet.report.analyze.build_report` always
    # set it when the caller passes the Suite. :command:`gauntlet compare`
    # uses this to detect cross-backend comparisons (§12 Q2).
    suite_env: str | None = None
    n_episodes: int
    n_success: int
    per_axis: list[AxisBreakdown]
    per_cell: list[CellBreakdown]
    failure_clusters: list[FailureCluster]
    heatmap_2d: dict[str, Heatmap2D]
    overall_success_rate: float
    overall_failure_rate: float
    cluster_multiple: float
