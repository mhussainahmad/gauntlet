"""Pure ``Report`` -> ``ReportDiff`` differ.

Design notes:

* The differ operates on the public :class:`gauntlet.report.Report`
  surface only — no private helpers from :mod:`gauntlet.report.analyze`
  are imported. Float keys are normalized defensively to 9 decimal
  places (mirroring ``analyze._norm``) so that values stored as JSON
  string keys and parsed back round-trip through ``frozenset`` lookups
  without spurious mismatches.
* Cell identity is the ``frozenset`` of the perturbation_config items —
  same convention as :func:`gauntlet.cli._cell_key`.
* Cluster identity is the ``frozenset`` of the cluster's ``axes``
  dictionary — every cluster :class:`gauntlet.report.FailureCluster`
  carries exactly two ``(axis_name, axis_value)`` pairs.
* ``cell_flip_threshold`` and ``cluster_intensify_threshold`` mirror
  the existing ``compare`` ergonomics. ``cell_flip_threshold`` is a
  two-sided gate (``|delta| >= threshold``); ``cluster_intensify_threshold``
  is one-sided on lift growth (``b_lift - a_lift >= threshold``) — we
  surface "things that got worse", not symmetric churn.
* Axis-value coverage: an :class:`AxisDelta` is emitted for the union of
  the axis's ``rates`` keys across both reports; values present on only
  one side are skipped (the other side has no episodes at that grid
  point — there is no defensible "delta" to report). Axes that appear
  in only one report are also skipped — a different axis schema is a
  different experiment, not a delta.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauntlet.report import FailureCluster, Report

__all__ = [
    "AxisDelta",
    "CellFlip",
    "ClusterDelta",
    "ReportDiff",
    "diff_reports",
]


# ──────────────────────────────────────────────────────────────────────
# Schema.
# ──────────────────────────────────────────────────────────────────────


class AxisDelta(BaseModel):
    """Per-axis-value rate deltas (``b - a``) for one axis name.

    Keys of ``rate_deltas`` are the float-normalized axis values present
    in *both* reports' :class:`gauntlet.report.AxisBreakdown` rates.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    name: str
    rate_deltas: dict[float, float]


class CellFlip(BaseModel):
    """A single cell whose success rate changed by ``>= threshold``.

    ``direction`` is ``"regressed"`` when ``b_success_rate < a_success_rate``,
    ``"improved"`` otherwise.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    cell_index: int
    perturbation_config: dict[str, float]
    a_success_rate: float
    b_success_rate: float
    direction: Literal["regressed", "improved"]


class ClusterDelta(BaseModel):
    """A failure cluster present in both reports whose lift rose."""

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    axes: dict[str, float]
    a_lift: float
    b_lift: float
    delta: float


class ReportDiff(BaseModel):
    """Top-level structured diff between two reports.

    Field order is "headline first": the scalar overall delta and per-
    axis breakdowns precede the cell- and cluster-level surfacing.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    a_label: str
    b_label: str
    a_suite_name: str
    b_suite_name: str
    n_episodes_delta: int
    overall_success_rate_delta: float
    axis_deltas: dict[str, AxisDelta]
    cell_flips: list[CellFlip]
    cluster_added: list[FailureCluster]
    cluster_removed: list[FailureCluster]
    cluster_intensified: list[ClusterDelta]


# ──────────────────────────────────────────────────────────────────────
# Helpers.
# ──────────────────────────────────────────────────────────────────────


def _norm(value: float) -> float:
    """Normalize a float key to 9 decimal places.

    Mirrors :func:`gauntlet.report.analyze._norm` defensively so the
    differ does not depend on a private helper from the report package
    (which a sibling refactor branch may re-type/rename).
    """
    return round(float(value), 9)


def _cell_key(perturbation_config: dict[str, float]) -> frozenset[tuple[str, float]]:
    """Stable hashable identity for a per-cell perturbation config."""
    return frozenset((k, _norm(v)) for k, v in perturbation_config.items())


def _cluster_key(cluster: FailureCluster) -> frozenset[tuple[str, float]]:
    """Stable hashable identity for a failure cluster's axis pair."""
    return frozenset((k, _norm(v)) for k, v in cluster.axes.items())


# ──────────────────────────────────────────────────────────────────────
# Public differ.
# ──────────────────────────────────────────────────────────────────────


def diff_reports(
    a: Report,
    b: Report,
    *,
    a_label: str = "a",
    b_label: str = "b",
    cell_flip_threshold: float = 0.10,
    cluster_intensify_threshold: float = 0.5,
) -> ReportDiff:
    """Compute the structured per-axis / per-cell / per-cluster diff.

    Parameters
    ----------
    a, b
        The two reports to diff. ``b`` is the candidate; deltas are
        ``b - a``.
    a_label, b_label
        Display labels (typically the file paths). Surfaced verbatim by
        :func:`gauntlet.diff.render.render_text`.
    cell_flip_threshold
        Inclusive minimum ``|b.success_rate - a.success_rate|`` for a
        per-cell flip to be reported. Two-sided.
    cluster_intensify_threshold
        Inclusive minimum ``b.lift - a.lift`` for a *shared* failure
        cluster to be reported as intensified. One-sided (lift growth).

    Returns
    -------
    :class:`ReportDiff`
        Structured delta. ``cluster_added`` / ``cluster_removed`` are the
        cluster-set differences (same identity as the union axes pair).
        ``cluster_intensified`` only contains clusters present in both
        reports whose lift growth crossed the threshold.

    Raises
    ------
    ValueError
        If either threshold is negative.
    """
    if cell_flip_threshold < 0:
        raise ValueError(f"cell_flip_threshold must be >= 0; got {cell_flip_threshold}")
    if cluster_intensify_threshold < 0:
        raise ValueError(
            f"cluster_intensify_threshold must be >= 0; got {cluster_intensify_threshold}"
        )

    # Per-axis rate deltas — only over axes present in both reports.
    a_axes = {ax.name: ax for ax in a.per_axis}
    b_axes = {ax.name: ax for ax in b.per_axis}
    axis_deltas: dict[str, AxisDelta] = {}
    for axis_name in sorted(a_axes.keys() & b_axes.keys()):
        a_ax = a_axes[axis_name]
        b_ax = b_axes[axis_name]
        a_rates = {_norm(k): v for k, v in a_ax.rates.items()}
        b_rates = {_norm(k): v for k, v in b_ax.rates.items()}
        shared_values = sorted(a_rates.keys() & b_rates.keys())
        rate_deltas: dict[float, float] = {
            value: b_rates[value] - a_rates[value] for value in shared_values
        }
        axis_deltas[axis_name] = AxisDelta(name=axis_name, rate_deltas=rate_deltas)

    # Per-cell flips — shared cells whose success_rate moved by the threshold.
    a_cells = {_cell_key(c.perturbation_config): c for c in a.per_cell}
    b_cells = {_cell_key(c.perturbation_config): c for c in b.per_cell}
    flips: list[CellFlip] = []
    for key in a_cells.keys() & b_cells.keys():
        ca = a_cells[key]
        cb = b_cells[key]
        delta = cb.success_rate - ca.success_rate
        if abs(delta) >= cell_flip_threshold:
            direction: Literal["regressed", "improved"] = "regressed" if delta < 0 else "improved"
            flips.append(
                CellFlip(
                    cell_index=cb.cell_index,
                    perturbation_config={k: _norm(v) for k, v in cb.perturbation_config.items()},
                    a_success_rate=ca.success_rate,
                    b_success_rate=cb.success_rate,
                    direction=direction,
                )
            )
    # Stable order: regressions first (most negative), then improvements
    # (most positive). Tie-break on cell_index then repr for determinism.
    flips.sort(key=lambda f: (f.b_success_rate - f.a_success_rate, f.cell_index))

    # Cluster set difference + intensification.
    a_clusters = {_cluster_key(c): c for c in a.failure_clusters}
    b_clusters = {_cluster_key(c): c for c in b.failure_clusters}
    cluster_added: list[FailureCluster] = [
        b_clusters[k] for k in sorted(b_clusters.keys() - a_clusters.keys(), key=repr)
    ]
    cluster_removed: list[FailureCluster] = [
        a_clusters[k] for k in sorted(a_clusters.keys() - b_clusters.keys(), key=repr)
    ]
    cluster_intensified: list[ClusterDelta] = []
    for k in a_clusters.keys() & b_clusters.keys():
        ca_cluster = a_clusters[k]
        cb_cluster = b_clusters[k]
        lift_delta = cb_cluster.lift - ca_cluster.lift
        if lift_delta >= cluster_intensify_threshold:
            cluster_intensified.append(
                ClusterDelta(
                    axes={kk: _norm(vv) for kk, vv in cb_cluster.axes.items()},
                    a_lift=ca_cluster.lift,
                    b_lift=cb_cluster.lift,
                    delta=lift_delta,
                )
            )
    # Worst intensification first.
    cluster_intensified.sort(key=lambda c: (-c.delta, repr(c.axes)))

    return ReportDiff(
        a_label=a_label,
        b_label=b_label,
        a_suite_name=a.suite_name,
        b_suite_name=b.suite_name,
        n_episodes_delta=b.n_episodes - a.n_episodes,
        overall_success_rate_delta=b.overall_success_rate - a.overall_success_rate,
        axis_deltas=axis_deltas,
        cell_flips=flips,
        cluster_added=cluster_added,
        cluster_removed=cluster_removed,
        cluster_intensified=cluster_intensified,
    )
