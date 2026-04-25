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

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AxisBreakdown",
    "CellBreakdown",
    "FailureCluster",
    "Heatmap2D",
    "Report",
    "SensitivityIndex",
]


class AxisBreakdown(BaseModel):
    """Marginal success rate for one axis, broken down by axis value.

    ``rates``, ``counts``, ``successes``, ``ci_low``, and ``ci_high``
    share the same set of keys (the unique axis values that appear in
    the episode list, sorted ascending). Keys are float-normalized —
    see module docstring.

    ``ci_low`` / ``ci_high`` carry the per-bucket Wilson 95%
    confidence interval on the bucket's success rate (B-03). Each
    value is ``None`` only if the bucket is empty (``n == 0``), which
    the analyse layer never produces — included for schema-level
    completeness so downstream consumers can rely on the field shape.
    Both fields default to empty dicts so old report.json files
    written before B-03 still validate via ``Report.model_validate``.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    name: str
    rates: dict[float, float]
    counts: dict[float, int]
    successes: dict[float, int]
    ci_low: dict[float, float | None] = Field(default_factory=dict)
    ci_high: dict[float, float | None] = Field(default_factory=dict)


class CellBreakdown(BaseModel):
    """Aggregate stats for a single ``(cell_index, perturbation_config)`` cell.

    The Runner guarantees that all episodes sharing a ``cell_index``
    also share their ``perturbation_config``; we still group on both so
    the model is robust to hand-constructed Episode lists.

    ``video_paths`` is the list of per-episode MP4 paths that the
    Polish "rollout video recording" feature populates when
    ``Runner(record_video=True)`` is used. Default is an empty list so
    pre-PR Episode dicts (which have no ``video_path`` field) round-
    trip through ``build_report`` unchanged.

    ``ci_low`` / ``ci_high`` carry the Wilson 95% confidence interval
    on ``success_rate`` (B-03). ``None`` only when ``n_episodes == 0``
    (which :func:`gauntlet.report.analyze.build_report` never produces);
    both default to ``None`` so old report.json files written before
    B-03 still validate.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    cell_index: int
    perturbation_config: dict[str, float]
    n_episodes: int
    n_success: int
    success_rate: float
    ci_low: float | None = None
    ci_high: float | None = None
    video_paths: list[str] = Field(default_factory=list)


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

    ``video_paths`` is the list of MP4 paths for the failed episodes
    inside this cluster, populated when ``Runner(record_video=True)``.
    Default empty list — pre-PR reports round-trip unchanged.

    ``ci_low`` / ``ci_high`` carry the Wilson 95% confidence interval
    on ``failure_rate`` (B-03). The CI is computed on failures (not
    successes), so a (5/5) all-failed cluster reports a high
    ``ci_low`` — the right framing for "we are confident this combo
    breaks the policy". ``None`` only when ``n_episodes == 0`` (which
    the analyse layer never produces). Both default to ``None`` so old
    report.json files written before B-03 still validate. Consumed by
    B-20 (regression-vs-noise attribution in ``gauntlet diff``).
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    axes: dict[str, float]
    n_episodes: int
    n_success: int
    failure_rate: float
    lift: float
    ci_low: float | None = None
    ci_high: float | None = None
    video_paths: list[str] = Field(default_factory=list)

    # B-21: per-cluster aggregates of the rollout-level actuator
    # telemetry that lives on :class:`gauntlet.runner.Episode`. Each
    # field is the mean across the cluster's episodes that *did*
    # report a value; episodes whose backend left telemetry unset
    # (``Episode.actuator_energy is None``) are dropped from both the
    # numerator and the denominator so a partial-coverage cluster is
    # not biased to zero. ``None`` means "no episode in this cluster
    # carried telemetry" — typically a non-MuJoCo backend or a fake
    # env in a unit test. The HTML report renders ``None`` as a dash
    # ("-") instead of "0.00".
    mean_actuator_energy: float | None = None
    mean_peak_torque_norm: float | None = None
    # B-18: per-cluster mean of :attr:`gauntlet.runner.Episode.action_variance`
    # across the cluster's episodes that reported one. Same partial-
    # coverage handling as the actuator trio above: episodes whose
    # ``action_variance is None`` (greedy policies, or runs that did
    # not opt into ``--measure-action-consistency``) are dropped from
    # both numerator and denominator. ``None`` here means "no episode
    # in this cluster carried action-variance telemetry" — the HTML
    # report renders this as a dash, distinct from a measured
    # ``0.0`` (which would mean true mode collapse).
    mean_action_variance: float | None = None

    # B-30: per-cluster aggregates of the safety-violation telemetry on
    # :class:`gauntlet.runner.Episode`. Each field is the cluster mean
    # across the episodes that *did* report a value — episodes whose
    # backend left the field ``None`` are dropped from both numerator
    # and denominator (same partial-coverage handling as the actuator
    # trio above). ``None`` here means "no episode in this cluster
    # carried safety telemetry"; the HTML report renders ``None`` as a
    # dash, never as ``0``. Failure clusters surface these so a human
    # reading the report can see "this combo fails AND collides 4x per
    # rollout" — the safety-vs-success asymmetry the B-30 spec calls
    # out.
    mean_collisions: float | None = None
    mean_joint_excursions: float | None = None


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

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    axis_x: str
    axis_y: str
    x_values: list[float]
    y_values: list[float]
    success_rate: list[list[float]]


class SensitivityIndex(BaseModel):
    """Per-axis Sobol sensitivity index (B-19).

    ``first_order`` (``S_i``) is the share of outcome variance
    explained by axis ``i`` alone; ``total_order`` (``S_T_i``) is the
    share of variance lost when ``i`` is held fixed (``>= first_order``;
    the gap is the interaction share). Both are in ``[0, 1]`` when
    populated.

    Both fields are ``None`` only when the run is all-success or
    all-fail (``Var(Y) == 0`` and the indices are undefined). A
    single-value axis is reported as ``0.0`` for both — by construction
    it can carry no variance and the closed-form total-order would
    otherwise leak the within-cell noise into a non-zero number.

    ``approximate`` is set when the source suite is not Sobol-sampled
    (e.g. cartesian or LHS): the closed-form decomposition is still
    well-defined, but the sample structure does not satisfy the
    quasi-MC assumptions Saltelli's bias bounds rely on, so the
    indices read as a useful indicator rather than a calibrated
    estimate. The HTML report carries this flag through to a "this is
    approximate" callout above the chart.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    first_order: float | None
    total_order: float | None
    approximate: bool


class Report(BaseModel):
    """Top-level failure-analysis report.

    Built by :func:`gauntlet.report.analyze.build_report` from a list
    of :class:`gauntlet.runner.Episode` results. The field order is
    deliberately breakdown-first (§6).
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

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
    # B-19: per-axis Sobol sensitivity indices, keyed by axis name.
    # ``None`` for backwards-compat: report.json files written before
    # B-19 do not carry the field; ``Report.model_validate`` accepts
    # them unchanged.
    sensitivity_indices: dict[str, SensitivityIndex] | None = None
    # B-26: cell index at which Optuna-style early-stop pruning kicked
    # in, or ``None`` when the run completed every cell. Populated by
    # the CLI from :attr:`gauntlet.runner.Runner.last_pruned_at_cell`
    # after the run finishes; ``build_report`` itself never sets it.
    # ``None`` is the byte-identical default for runs that did not opt
    # into pruning (the common case) — old report.json files written
    # before B-26 still validate via ``Report.model_validate``.
    #
    # Note: pruning destroys the "fair sample of the perturbation
    # space" reading by definition (we stop sampling early). When this
    # field is non-None the report's ``per_cell`` list has fewer than
    # the suite's nominal cell count and any axis-marginal or
    # heatmap-derived statistic is biased toward whichever cells the
    # Runner happened to schedule first. Force-disabled when
    # ``suite.sampling == "sobol"`` (Sobol indices need full sample
    # structure per B-19).
    pruned_at_cell: int | None = None
    # B-30: fraction of *successful* episodes that were also safe (zero
    # safety violations recorded). ``None`` when no episode carried
    # safety telemetry at all (the partial-coverage dataset case) or
    # when there were no successes to begin with — distinct from
    # ``0.0`` (which would mean "every success was unsafe"). The
    # ``success_unsafe`` framing the B-30 spec calls out is the
    # complement: ``success_unsafe_rate = success_rate - success_safe_rate``
    # whenever both are defined. Old report.json files written before
    # B-30 round-trip via ``Report.model_validate`` because the field
    # defaults to ``None``.
    success_safe_rate: float | None = None
