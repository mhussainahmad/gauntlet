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

from gauntlet.diff.paired import PairedCellDelta, PairedComparison
from gauntlet.report import FailureCluster, Report

__all__ = [
    "VERDICT_IMPROVED",
    "VERDICT_REGRESSED",
    "VERDICT_WITHIN_NOISE",
    "AxisDelta",
    "CellFlip",
    "ClusterDelta",
    "ReportDiff",
    "Verdict",
    "diff_reports",
]


# B-20: regression-vs-noise attribution. ``Verdict`` reflects whether a
# cell-flip's delta crossed the threshold *and* the per-cell CI bracket
# (Wilson on each side from B-03, or the CRN-paired Newcombe/Tango on
# the difference from B-08) provides any evidence the flip is real
# rather than sample noise.
Verdict = Literal["regressed", "improved", "within_noise"]
VERDICT_REGRESSED: Verdict = "regressed"
VERDICT_IMPROVED: Verdict = "improved"
VERDICT_WITHIN_NOISE: Verdict = "within_noise"

# McNemar override threshold: when the paired CI clears zero but the
# McNemar p-value on the discordant-pair contingency table exceeds
# this, downgrade the verdict to ``within_noise``. The 0.05 cut-off
# matches the conventional alpha used elsewhere in the project (see
# ``gauntlet.diff.paired`` docstring on Agresti / SciPy defaults).
VERDICT_MCNEMAR_ALPHA = 0.05


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

    ``paired`` is ``True`` when the flip was computed from a CRN-paired
    run (``--paired``) and the ``delta_ci_low`` / ``delta_ci_high`` /
    ``mcnemar_p_value`` fields carry the paired-CI bracket plus the
    McNemar p-value on the per-episode pass/fail contingency table
    (B-08). Under the unpaired path all four fields are ``None`` /
    ``False`` so ``ReportDiff`` JSONs from the legacy code path remain
    structurally identical except for the additional defaulted keys.

    B-20 (regression-vs-noise attribution) consumes the paired CI to
    decide whether a flip is ``regressed`` / ``improved`` in earnest or
    ``within_noise``; without B-08's CRN reduction the bracket on the
    delta is too loose to support that decision.

    ``verdict`` is the B-20 tag — ``regressed`` / ``improved`` /
    ``within_noise`` — and is computed as follows (see
    :func:`_compute_verdict`):

    * **Paired path** (``paired=True`` and the delta CI populated): the
      bracket on the paired delta directly answers whether zero is in
      scope. ``within_noise`` if it brackets zero or if the McNemar
      p-value exceeds 0.05; otherwise the sign of the point delta picks
      ``regressed`` vs ``improved``.
    * **Unpaired-but-Wilson** path (CIs on both ``CellBreakdown`` sides,
      no paired delta CI): the conservative independent-Wilson bracket
      on the delta is ``[b.ci_low - a.ci_high, b.ci_high - a.ci_low]``.
      ``within_noise`` if it straddles zero; else point-delta sign picks.
    * **Legacy** path (any CI missing): falls back to the unsafe binary
      direction tag — ``regressed`` if delta < 0 else ``improved``.

    Defaults to ``None`` so legacy diff JSONs (pre-B-20) round-trip; new
    diffs always populate it through :func:`diff_reports`.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    cell_index: int
    perturbation_config: dict[str, float]
    a_success_rate: float
    b_success_rate: float
    direction: Literal["regressed", "improved"]
    paired: bool = False
    delta_ci_low: float | None = None
    delta_ci_high: float | None = None
    mcnemar_p_value: float | None = None
    verdict: Verdict | None = None


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

    ``paired`` and ``paired_comparison`` are populated when
    :func:`diff_reports` is given a paired CRN payload (B-08). They are
    ``False`` / ``None`` for the legacy unpaired call path, which keeps
    the old JSON shape structurally compatible -- every existing
    consumer (the dashboard's diff renderer, ``gauntlet diff --json |
    jq`` recipes, B-20's regression-vs-noise attribution) sees the new
    fields default cleanly.
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
    paired: bool = False
    paired_comparison: PairedComparison | None = None


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


def _compute_verdict(
    delta: float,
    *,
    paired_cell: PairedCellDelta | None,
    a_ci_low: float | None,
    a_ci_high: float | None,
    b_ci_low: float | None,
    b_ci_high: float | None,
) -> Verdict:
    """Tag a cell-flip as ``regressed`` / ``improved`` / ``within_noise`` (B-20).

    Three-tier decision tree (preferred-path-first):

    1. **Paired CRN** (``paired_cell is not None``): the Newcombe / Tango
       Wald CI on the paired success-rate difference (B-08) brackets zero
       → ``within_noise``. McNemar's two-sided p-value above
       :data:`VERDICT_MCNEMAR_ALPHA` also forces ``within_noise`` —
       discordant-pair imbalance is the only paired evidence we have, and
       a high p means we cannot reject H0 even when the Wald bracket is
       narrow. Otherwise the sign of ``delta`` (``b_success_rate -
       a_success_rate``) picks ``regressed`` (negative) or ``improved``
       (positive).
    2. **Independent Wilson** (both sides expose ``ci_low`` / ``ci_high``
       from B-03): the worst-case CI on the delta is ``[b_low - a_high,
       b_high - a_low]``. If it straddles zero the flip is within noise;
       else the point delta sign picks. This bracket is wider than the
       paired one (Var[X-Y] = Var[X]+Var[Y] under independence) so it
       under-flags more aggressively — that's the price of skipping CRN.
    3. **Legacy** (any CI is ``None`` — pre-B-03 report.json): no
       statistical evidence available. Fall back to the binary point-delta
       direction so the verdict field is always populated; users can spot
       the legacy path by the absence of CIs in the JSON output.

    Caller is expected to feed ``delta = b_success_rate - a_success_rate``;
    this matches :class:`CellFlip` and the paired delta convention.
    """
    if (
        paired_cell is not None
        and paired_cell.delta_ci_low is not None
        and paired_cell.delta_ci_high is not None
    ):
        if paired_cell.delta_ci_low <= 0.0 <= paired_cell.delta_ci_high:
            return VERDICT_WITHIN_NOISE
        if paired_cell.mcnemar.p_value > VERDICT_MCNEMAR_ALPHA:
            return VERDICT_WITHIN_NOISE
        return VERDICT_REGRESSED if delta < 0 else VERDICT_IMPROVED

    if (
        a_ci_low is not None
        and a_ci_high is not None
        and b_ci_low is not None
        and b_ci_high is not None
    ):
        delta_low = b_ci_low - a_ci_high
        delta_high = b_ci_high - a_ci_low
        if delta_low <= 0.0 <= delta_high:
            return VERDICT_WITHIN_NOISE
        return VERDICT_REGRESSED if delta < 0 else VERDICT_IMPROVED

    # Legacy fallback: no CI evidence — keep the unsafe binary tag so the
    # verdict field is never null in fresh diffs, but do not pretend it
    # is statistically defensible.
    return VERDICT_REGRESSED if delta < 0 else VERDICT_IMPROVED


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
    paired_comparison: PairedComparison | None = None,
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
    paired_comparison
        Optional CRN paired-comparison artefact (B-08). When set, every
        :class:`CellFlip` is enriched with the paired CI bracket on the
        delta and the McNemar p-value, and the top-level
        :attr:`ReportDiff.paired` flag flips to ``True``. The artefact
        comes from :func:`gauntlet.diff.compute_paired_cells`. Per-cell
        rows whose ``perturbation_config`` does not appear in the paired
        artefact are left with the unpaired defaults (the diff is
        permissive across mismatched grids).

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

    # Index the paired comparison by ``cell_index`` for O(1) lookup
    # while building CellFlip rows. ``cell_index`` is the most stable
    # join key (the runner derives it from grid enumeration order); the
    # perturbation_config pass-through stays as a defensive fallback for
    # tests that hand-construct paired payloads with non-matching cell
    # indices.
    paired_by_cell_index: dict[int, PairedCellDelta] = {}
    paired_by_config: dict[frozenset[tuple[str, float]], PairedCellDelta] = {}
    if paired_comparison is not None:
        for entry in paired_comparison.cells:
            paired_by_cell_index[entry.cell_index] = entry
            paired_by_config[_cell_key(entry.perturbation_config)] = entry

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
            paired_cell: PairedCellDelta | None = paired_by_cell_index.get(cb.cell_index)
            if paired_cell is None:
                paired_cell = paired_by_config.get(key)
            verdict = _compute_verdict(
                delta,
                paired_cell=paired_cell,
                a_ci_low=ca.ci_low,
                a_ci_high=ca.ci_high,
                b_ci_low=cb.ci_low,
                b_ci_high=cb.ci_high,
            )
            if paired_cell is not None:
                flips.append(
                    CellFlip(
                        cell_index=cb.cell_index,
                        perturbation_config={
                            k: _norm(v) for k, v in cb.perturbation_config.items()
                        },
                        a_success_rate=ca.success_rate,
                        b_success_rate=cb.success_rate,
                        direction=direction,
                        paired=True,
                        delta_ci_low=paired_cell.delta_ci_low,
                        delta_ci_high=paired_cell.delta_ci_high,
                        mcnemar_p_value=paired_cell.mcnemar.p_value,
                        verdict=verdict,
                    )
                )
            else:
                flips.append(
                    CellFlip(
                        cell_index=cb.cell_index,
                        perturbation_config={
                            k: _norm(v) for k, v in cb.perturbation_config.items()
                        },
                        a_success_rate=ca.success_rate,
                        b_success_rate=cb.success_rate,
                        direction=direction,
                        verdict=verdict,
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
        paired=paired_comparison is not None,
        paired_comparison=paired_comparison,
    )
