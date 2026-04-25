"""Pure-function analysis over a list of :class:`Episode` results.

Computes the schemas defined in :mod:`gauntlet.report.schema`. The
single public entry point is :func:`build_report`; everything else in
this module is an internal helper.

Computation rules — see ``GAUNTLET_SPEC.md`` §5 task 7 for the contract
and §6 for the "never aggregate away failures" guarantee:

* **Overall success rate** = ``n_success / n_episodes``. An empty list
  raises :class:`ValueError`.
* **Per-axis marginals** group episodes by the (normalized) value of
  one axis at a time.
* **Per-cell** groups by ``(cell_index, frozenset(perturbation_config))``.
* **Failure clusters** iterate over every unordered pair of distinct
  axes and every (value_a, value_b) combination that actually appears
  in the episode list. A cluster is reported when the pair's
  ``failure_rate >= cluster_multiple * baseline_failure_rate`` AND
  ``n_episodes >= min_cluster_size``.
* **2D heatmaps** are built for every unordered axis pair; cells with
  no episodes are ``float("nan")``.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable

from gauntlet.report.abstention import compute_abstention_metrics
from gauntlet.report.schema import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Heatmap2D,
    Report,
    SensitivityIndex,
)
from gauntlet.report.sobol_indices import compute_sobol_indices
from gauntlet.report.wilson import DEFAULT_CONFIDENCE, wilson_interval
from gauntlet.runner.episode import Episode

__all__ = ["DEFAULT_CONFIDENCE", "build_report", "episode_has_safety_violation"]


def episode_has_safety_violation(ep: Episode) -> bool:
    """True iff the episode recorded any safety violation (B-30).

    ``None`` on any field is treated as "not measured", NOT as a
    violation — a backend that leaves all four fields ``None`` cannot
    be tagged unsafe (Safety-Gymnasium NeurIPS 2023 / safe-control-gym
    arXiv 2109.06325 framing: absence of evidence is not evidence of
    safety, but it is not evidence of unsafety either). The dual-use
    convention keeps the cross-backend anti-feature honest.
    """
    return (
        (ep.n_collisions is not None and ep.n_collisions > 0)
        or (ep.n_joint_limit_excursions is not None and ep.n_joint_limit_excursions > 0)
        or (ep.energy_over_budget is True)
        or (ep.n_workspace_excursions is not None and ep.n_workspace_excursions > 0)
    )


def _episode_has_any_safety_field(ep: Episode) -> bool:
    """True iff any of the four B-30 safety fields is non-None.

    Used to decide whether ``Report.success_safe_rate`` is defined for
    a given dataset: if no episode carries safety telemetry at all the
    rate is ``None`` (not ``0.0`` or ``success_rate``), distinct from
    "every success was safe" (no violations across the dataset → rate
    == ``success_rate``).
    """
    return (
        ep.n_collisions is not None
        or ep.n_joint_limit_excursions is not None
        or ep.energy_over_budget is not None
        or ep.n_workspace_excursions is not None
    )


# Number of decimal places used to normalize float axis values before
# they are used as dict keys / set members. Picked to absorb arithmetic
# drift on the order of 1e-15 (typical IEEE-754 round-off) while still
# distinguishing legitimate grid points spaced 1e-6 apart.
_FLOAT_DECIMALS = 9


def _norm(value: float) -> float:
    """Normalize a float so 1e-15 jitter doesn't split it across buckets.

    Applied to every axis value that ends up as a ``dict`` key, set
    member, or sortable comparable. Centralising the rounding in one
    helper keeps the per-axis, per-cell, heatmap, and cluster paths
    consistent.
    """
    return round(float(value), _FLOAT_DECIMALS)


def _ordered_axis_names(episodes: Iterable[Episode]) -> tuple[str, ...]:
    """Union of ``perturbation_config`` keys across episodes, in first-seen order.

    Python dict preserves insertion order, so ``dict.fromkeys`` over the
    iterator gives a stable, deterministic axis order without a manual
    "seen" set.
    """
    seen: dict[str, None] = {}
    for ep in episodes:
        for axis_name in ep.perturbation_config:
            if axis_name not in seen:
                seen[axis_name] = None
    return tuple(seen.keys())


def _per_axis_breakdowns(
    episodes: list[Episode],
    axis_names: tuple[str, ...],
    *,
    confidence: float,
) -> list[AxisBreakdown]:
    """Compute one :class:`AxisBreakdown` per axis name.

    Only axes that actually carry a value in *every* episode contribute
    to the marginal; missing keys are skipped (they should not occur in
    Runner-produced episodes, but tolerating them keeps the function
    safe under hand-constructed inputs).

    ``ci_low`` / ``ci_high`` carry the per-bucket Wilson interval at
    the requested ``confidence`` level (B-03).
    """
    breakdowns: list[AxisBreakdown] = []
    for axis in axis_names:
        successes: dict[float, int] = defaultdict(int)
        counts: dict[float, int] = defaultdict(int)
        for ep in episodes:
            if axis not in ep.perturbation_config:
                continue
            value = _norm(ep.perturbation_config[axis])
            counts[value] += 1
            if ep.success:
                successes[value] += 1
        # Sort keys ascending so JSON / HTML rendering is stable.
        sorted_keys = sorted(counts.keys())
        rates = {k: successes[k] / counts[k] for k in sorted_keys}
        ci_low: dict[float, float | None] = {}
        ci_high: dict[float, float | None] = {}
        for k in sorted_keys:
            lo, hi = wilson_interval(successes[k], counts[k], confidence=confidence)
            ci_low[k] = lo
            ci_high[k] = hi
        breakdowns.append(
            AxisBreakdown(
                name=axis,
                rates=rates,
                counts={k: counts[k] for k in sorted_keys},
                successes={k: successes[k] for k in sorted_keys},
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return breakdowns


def _per_cell_breakdowns(
    episodes: list[Episode],
    *,
    confidence: float,
) -> list[CellBreakdown]:
    """Group episodes by ``(cell_index, frozenset(normalized_config))``.

    Returns one :class:`CellBreakdown` per group, sorted by
    ``cell_index`` ascending. Cell-id collisions across distinct configs
    (which the Runner never produces) become separate entries — the
    sort key is just the index, so order is still stable.

    ``video_paths`` is populated from ``Episode.video_path`` for any
    episodes that carry one (Polish "rollout video recording"
    feature). Order matches the input ``episodes`` enumeration order
    so re-running the analysis on the same Episode list is byte-
    identical.

    ``ci_low`` / ``ci_high`` carry the Wilson interval on
    ``success_rate`` at the requested ``confidence`` level (B-03).
    """
    groups: dict[tuple[int, frozenset[tuple[str, float]]], list[Episode]] = defaultdict(list)
    configs: dict[tuple[int, frozenset[tuple[str, float]]], dict[str, float]] = {}
    for ep in episodes:
        config_norm = {k: _norm(v) for k, v in ep.perturbation_config.items()}
        key = (ep.cell_index, frozenset(config_norm.items()))
        groups[key].append(ep)
        # Keep one representative dict per group for the output model.
        configs.setdefault(key, config_norm)

    rows: list[CellBreakdown] = []
    for key in sorted(groups.keys(), key=lambda k: k[0]):
        eps = groups[key]
        n = len(eps)
        n_success = sum(1 for e in eps if e.success)
        videos = [e.video_path for e in eps if e.video_path is not None]
        ci_low, ci_high = wilson_interval(n_success, n, confidence=confidence)
        rows.append(
            CellBreakdown(
                cell_index=key[0],
                perturbation_config=configs[key],
                n_episodes=n,
                n_success=n_success,
                success_rate=n_success / n,
                ci_low=ci_low,
                ci_high=ci_high,
                video_paths=videos,
            )
        )
    return rows


def _failure_clusters(
    episodes: list[Episode],
    axis_names: tuple[str, ...],
    *,
    baseline_failure_rate: float,
    cluster_multiple: float,
    min_cluster_size: int,
    confidence: float,
) -> list[FailureCluster]:
    """Find axis-PAIR value combinations with elevated failure rates.

    Iterates over every unordered pair of distinct axes in
    ``axis_names`` and every (value_a, value_b) tuple that actually
    appears in the episode list. With fewer than 2 axes the result is
    necessarily empty.

    Short-circuits to an empty list when ``baseline_failure_rate == 0``
    (no failures to cluster on — the test case for the all-success
    suite).
    """
    if baseline_failure_rate <= 0.0:
        return []
    if len(axis_names) < 2:
        return []

    clusters: list[FailureCluster] = []
    for axis_a, axis_b in itertools.combinations(axis_names, 2):
        # (value_a, value_b) -> (n_total, n_success)
        pair_counts: dict[tuple[float, float], list[int]] = defaultdict(lambda: [0, 0])
        # (value_a, value_b) -> [video_path, ...] for failed episodes
        # only. Tracked separately so the cluster surfaces *actionable*
        # videos: the "failed cluster" table is the first place a human
        # looks when reading a report. Successful videos belong on the
        # per-cell row, not here.
        pair_videos: dict[tuple[float, float], list[str]] = defaultdict(list)
        # B-21 actuator-cost aggregates. We sum + count separately
        # (instead of carrying a running mean) so a backend that emits
        # telemetry on some episodes but not others doesn't bias the
        # cluster mean toward zero — episodes whose
        # ``actuator_energy is None`` are dropped from BOTH the
        # numerator and the denominator.
        pair_energy_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_energy_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_peak_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_peak_count: dict[tuple[float, float], int] = defaultdict(int)
        # B-18 mode-collapse aggregate. Same partial-coverage handling
        # as the actuator trio above — episodes whose action_variance
        # is None (greedy policies, or runs that did not opt into the
        # measurement) are dropped from BOTH numerator and denominator
        # so the cluster mean is not biased toward zero.
        pair_var_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_var_count: dict[tuple[float, float], int] = defaultdict(int)
        # B-30 safety-violation aggregates. Same partial-coverage
        # handling as the actuator / variance trios above — episodes
        # whose backend left the field ``None`` are dropped from BOTH
        # numerator and denominator so the cluster mean is not biased
        # toward zero on a dataset that mixes MuJoCo runs (telemetry)
        # with PyBullet/Genesis runs (no telemetry).
        pair_collisions_sum: dict[tuple[float, float], int] = defaultdict(int)
        pair_collisions_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_joint_sum: dict[tuple[float, float], int] = defaultdict(int)
        pair_joint_count: dict[tuple[float, float], int] = defaultdict(int)
        # B-02 behavioural-metric aggregates. Same partial-coverage
        # handling as the actuator / safety trios above — episodes
        # whose backend left the field ``None`` are dropped from BOTH
        # numerator and denominator so a partial-coverage cluster mean
        # is not biased toward zero. The five fields are independent
        # (a successful episode has ``time_to_success`` populated but
        # ``jerk_rms`` may be ``None`` if the rollout was T < 4
        # samples), so each carries its own pair of dicts.
        pair_tts_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_tts_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_plr_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_plr_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_jerk_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_jerk_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_nearcol_sum: dict[tuple[float, float], int] = defaultdict(int)
        pair_nearcol_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_peakforce_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_peakforce_count: dict[tuple[float, float], int] = defaultdict(int)
        # B-37 inference-latency aggregates. Same partial-coverage
        # handling as the actuator / safety / behavioural trios above
        # — episodes whose Episode.inference_latency_ms_p99 is None
        # (legacy pre-B-37 episodes.json, or T=0 rollouts) are dropped
        # from BOTH numerator and denominator. The budget-violation
        # rate has its own pair of dicts because a violation only
        # surfaces when the user opted into ``--max-inference-ms`` —
        # treating it like a separate column for the partial-coverage
        # / "absent means not measured" gate.
        pair_lat_p50_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_lat_p50_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_lat_p99_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_lat_p99_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_lat_max_sum: dict[tuple[float, float], float] = defaultdict(float)
        pair_lat_max_count: dict[tuple[float, float], int] = defaultdict(int)
        pair_budget_violations: dict[tuple[float, float], int] = defaultdict(int)
        pair_budget_observed: dict[tuple[float, float], int] = defaultdict(int)
        for ep in episodes:
            if axis_a not in ep.perturbation_config or axis_b not in ep.perturbation_config:
                continue
            va = _norm(ep.perturbation_config[axis_a])
            vb = _norm(ep.perturbation_config[axis_b])
            bucket = pair_counts[(va, vb)]
            bucket[0] += 1
            if ep.success:
                bucket[1] += 1
            elif ep.video_path is not None:
                pair_videos[(va, vb)].append(ep.video_path)
            if ep.actuator_energy is not None:
                pair_energy_sum[(va, vb)] += ep.actuator_energy
                pair_energy_count[(va, vb)] += 1
            if ep.peak_torque_norm is not None:
                pair_peak_sum[(va, vb)] += ep.peak_torque_norm
                pair_peak_count[(va, vb)] += 1
            if ep.action_variance is not None:
                pair_var_sum[(va, vb)] += ep.action_variance
                pair_var_count[(va, vb)] += 1
            if ep.n_collisions is not None:
                pair_collisions_sum[(va, vb)] += ep.n_collisions
                pair_collisions_count[(va, vb)] += 1
            if ep.n_joint_limit_excursions is not None:
                pair_joint_sum[(va, vb)] += ep.n_joint_limit_excursions
                pair_joint_count[(va, vb)] += 1
            if ep.time_to_success is not None:
                pair_tts_sum[(va, vb)] += ep.time_to_success
                pair_tts_count[(va, vb)] += 1
            if ep.path_length_ratio is not None:
                pair_plr_sum[(va, vb)] += ep.path_length_ratio
                pair_plr_count[(va, vb)] += 1
            if ep.jerk_rms is not None:
                pair_jerk_sum[(va, vb)] += ep.jerk_rms
                pair_jerk_count[(va, vb)] += 1
            if ep.near_collision_count is not None:
                pair_nearcol_sum[(va, vb)] += ep.near_collision_count
                pair_nearcol_count[(va, vb)] += 1
            if ep.peak_force is not None:
                pair_peakforce_sum[(va, vb)] += ep.peak_force
                pair_peakforce_count[(va, vb)] += 1
            # B-37 inference-latency aggregation. The three percentile
            # fields are populated together by the worker (always-on
            # measurement); we still gate each independently so a
            # mixed dataset of pre-B-37 and post-B-37 episodes.json
            # files renders honestly. The budget-violation rate is
            # derived from ``metadata['inference_budget_violated']`` —
            # we only count the episode toward the rate when it had
            # at least one latency sample (i.e. the budget compare was
            # actually exercised), so a cluster that mixes "budget on"
            # with "budget off" runs reports the rate over the subset
            # that ran with a budget.
            if ep.inference_latency_ms_p50 is not None:
                pair_lat_p50_sum[(va, vb)] += ep.inference_latency_ms_p50
                pair_lat_p50_count[(va, vb)] += 1
            if ep.inference_latency_ms_p99 is not None:
                pair_lat_p99_sum[(va, vb)] += ep.inference_latency_ms_p99
                pair_lat_p99_count[(va, vb)] += 1
                # Observed denominator for the budget-violation rate:
                # any episode whose p99 is reported has a meaningful
                # latency measurement, so it can be the denominator
                # for the violation-rate fraction. The numerator is
                # the episodes whose metadata flag is True.
                pair_budget_observed[(va, vb)] += 1
                if bool(ep.metadata.get("inference_budget_violated", False)):
                    pair_budget_violations[(va, vb)] += 1
            if ep.inference_latency_ms_max is not None:
                pair_lat_max_sum[(va, vb)] += ep.inference_latency_ms_max
                pair_lat_max_count[(va, vb)] += 1

        for (va, vb), (n_total, n_success) in pair_counts.items():
            if n_total < min_cluster_size:
                continue
            failure_rate = (n_total - n_success) / n_total
            if failure_rate < cluster_multiple * baseline_failure_rate:
                continue
            # CI is on the failure rate (not the success rate) — the
            # "we are confident this combo breaks the policy" framing
            # the report leads with. Wilson is symmetric on the
            # binomial, so this is just ``wilson_interval(n_total -
            # n_success, n_total)``.
            ci_low, ci_high = wilson_interval(
                n_total - n_success,
                n_total,
                confidence=confidence,
            )
            energy_n = pair_energy_count.get((va, vb), 0)
            mean_energy: float | None = (
                pair_energy_sum[(va, vb)] / energy_n if energy_n > 0 else None
            )
            peak_n = pair_peak_count.get((va, vb), 0)
            mean_peak: float | None = pair_peak_sum[(va, vb)] / peak_n if peak_n > 0 else None
            var_n = pair_var_count.get((va, vb), 0)
            mean_var: float | None = pair_var_sum[(va, vb)] / var_n if var_n > 0 else None
            col_n = pair_collisions_count.get((va, vb), 0)
            mean_collisions: float | None = (
                pair_collisions_sum[(va, vb)] / col_n if col_n > 0 else None
            )
            joint_n = pair_joint_count.get((va, vb), 0)
            mean_joint: float | None = pair_joint_sum[(va, vb)] / joint_n if joint_n > 0 else None
            tts_n = pair_tts_count.get((va, vb), 0)
            mean_tts: float | None = pair_tts_sum[(va, vb)] / tts_n if tts_n > 0 else None
            plr_n = pair_plr_count.get((va, vb), 0)
            mean_plr: float | None = pair_plr_sum[(va, vb)] / plr_n if plr_n > 0 else None
            jerk_n = pair_jerk_count.get((va, vb), 0)
            mean_jerk: float | None = pair_jerk_sum[(va, vb)] / jerk_n if jerk_n > 0 else None
            nearcol_n = pair_nearcol_count.get((va, vb), 0)
            mean_nearcol: float | None = (
                pair_nearcol_sum[(va, vb)] / nearcol_n if nearcol_n > 0 else None
            )
            peakforce_n = pair_peakforce_count.get((va, vb), 0)
            mean_peakforce: float | None = (
                pair_peakforce_sum[(va, vb)] / peakforce_n if peakforce_n > 0 else None
            )
            # B-37 inference-latency aggregates. The three percentile
            # means follow the standard partial-coverage pattern. The
            # budget-violation rate is reported only when at least one
            # episode in the cluster was flagged — absent (i.e.
            # ``None``) means "no budget configured for this run, OR
            # every episode came in under the budget". The HTML report
            # hides the column entirely when no cluster reports a
            # rate, so a run without --max-inference-ms produces a
            # visually identical layout to the pre-B-37 report.
            lat_p50_n = pair_lat_p50_count.get((va, vb), 0)
            mean_lat_p50: float | None = (
                pair_lat_p50_sum[(va, vb)] / lat_p50_n if lat_p50_n > 0 else None
            )
            lat_p99_n = pair_lat_p99_count.get((va, vb), 0)
            mean_lat_p99: float | None = (
                pair_lat_p99_sum[(va, vb)] / lat_p99_n if lat_p99_n > 0 else None
            )
            lat_max_n = pair_lat_max_count.get((va, vb), 0)
            mean_lat_max: float | None = (
                pair_lat_max_sum[(va, vb)] / lat_max_n if lat_max_n > 0 else None
            )
            budget_n = pair_budget_violations.get((va, vb), 0)
            budget_obs = pair_budget_observed.get((va, vb), 0)
            budget_rate: float | None = (
                budget_n / budget_obs if budget_n > 0 and budget_obs > 0 else None
            )
            clusters.append(
                FailureCluster(
                    axes={axis_a: va, axis_b: vb},
                    n_episodes=n_total,
                    n_success=n_success,
                    failure_rate=failure_rate,
                    lift=failure_rate / baseline_failure_rate,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    video_paths=list(pair_videos.get((va, vb), [])),
                    mean_actuator_energy=mean_energy,
                    mean_peak_torque_norm=mean_peak,
                    mean_action_variance=mean_var,
                    mean_collisions=mean_collisions,
                    mean_joint_excursions=mean_joint,
                    mean_time_to_success=mean_tts,
                    mean_path_length_ratio=mean_plr,
                    mean_jerk_rms=mean_jerk,
                    mean_near_collision_count=mean_nearcol,
                    mean_peak_force=mean_peakforce,
                    mean_inference_latency_ms_p50=mean_lat_p50,
                    mean_inference_latency_ms_p99=mean_lat_p99,
                    mean_inference_latency_ms_max=mean_lat_max,
                    inference_budget_violation_rate=budget_rate,
                )
            )

    # Sort by lift desc, then failure_rate desc — stable presentation.
    clusters.sort(key=lambda c: (-c.lift, -c.failure_rate))
    return clusters


def _heatmaps_2d(
    episodes: list[Episode],
    axis_names: tuple[str, ...],
) -> dict[str, Heatmap2D]:
    """Build one :class:`Heatmap2D` per unordered axis pair.

    Returns an empty dict when fewer than two axes are present
    (single-axis suites have no 2D structure). Cells with no episodes
    populate as ``float("nan")``.

    Key format: ``f"{axis_x}__{axis_y}"`` where ``axis_x`` is the first
    axis in :func:`itertools.combinations` order over ``axis_names``;
    documented in :mod:`gauntlet.report.schema`.
    """
    if len(axis_names) < 2:
        return {}

    heatmaps: dict[str, Heatmap2D] = {}
    for axis_x, axis_y in itertools.combinations(axis_names, 2):
        # (x_value, y_value) -> [n_total, n_success]
        cell_counts: dict[tuple[float, float], list[int]] = defaultdict(lambda: [0, 0])
        x_seen: set[float] = set()
        y_seen: set[float] = set()
        for ep in episodes:
            if axis_x not in ep.perturbation_config or axis_y not in ep.perturbation_config:
                continue
            xv = _norm(ep.perturbation_config[axis_x])
            yv = _norm(ep.perturbation_config[axis_y])
            x_seen.add(xv)
            y_seen.add(yv)
            bucket = cell_counts[(xv, yv)]
            bucket[0] += 1
            if ep.success:
                bucket[1] += 1

        x_values = sorted(x_seen)
        y_values = sorted(y_seen)
        matrix: list[list[float]] = []
        for yv in y_values:
            row: list[float] = []
            for xv in x_values:
                bucket_or_none = cell_counts.get((xv, yv))
                if bucket_or_none is None or bucket_or_none[0] == 0:
                    row.append(float("nan"))
                else:
                    row.append(bucket_or_none[1] / bucket_or_none[0])
            matrix.append(row)

        key = f"{axis_x}__{axis_y}"
        heatmaps[key] = Heatmap2D(
            axis_x=axis_x,
            axis_y=axis_y,
            x_values=x_values,
            y_values=y_values,
            success_rate=matrix,
        )
    return heatmaps


def build_report(
    episodes: list[Episode],
    *,
    cluster_multiple: float = 2.0,
    min_cluster_size: int = 3,
    suite_env: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    sampling: str | None = None,
) -> Report:
    """Aggregate a list of :class:`Episode` into a :class:`Report`.

    Pure function — no I/O, no globals, deterministic from its input
    (Python guarantees dict insertion order, which we lean on).

    Args:
        episodes: All rollouts from a single suite. Must be non-empty
            and share one ``suite_name`` value.
        cluster_multiple: Failure-rate multiplier above baseline that
            qualifies an axis-pair as a failure cluster. Defaults to
            2.0 per ``GAUNTLET_SPEC.md`` §5 task 7. Must be ``> 0``.
        min_cluster_size: Minimum number of episodes required at an
            axis-pair value combination before it can be reported as a
            cluster. Defaults to 3.
        suite_env: Optional env slug carried through onto the report
            (RFC-005 §12 Q2). Old report.json files written before
            RFC-005 are accepted unchanged when ``suite_env`` is
            ``None``.
        confidence: Two-sided coverage level for the Wilson score
            interval attached to every per-cell, per-axis-value, and
            failure-cluster rate (B-03). Defaults to
            :data:`DEFAULT_CONFIDENCE` (0.95). Must be in ``(0, 1)``.
        sampling: Suite sampling mode (``"sobol"``, ``"lhs"``,
            ``"cartesian"`` ...) carried through to the per-axis Sobol
            sensitivity indices (B-19). Anything other than
            ``"sobol"`` (including the default ``None``) tags the
            indices ``approximate=True`` so the HTML report can warn
            that the sample structure is not the quasi-MC grid Sobol
            indices were derived for.

    Returns:
        A fully populated :class:`Report`.

    Raises:
        ValueError: if ``episodes`` is empty, contains rows from more
            than one suite, or ``cluster_multiple <= 0``, or
            ``confidence`` is not in ``(0, 1)``.
    """
    if cluster_multiple <= 0:
        raise ValueError(
            f"cluster_multiple must be > 0; got {cluster_multiple}",
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence must be in (0, 1); got {confidence}",
        )
    if len(episodes) == 0:
        raise ValueError("cannot build report from zero episodes")

    suite_names = {ep.suite_name for ep in episodes}
    if len(suite_names) > 1:
        offending = sorted(suite_names)
        raise ValueError(
            f"all episodes must share a single suite_name; got {offending}",
        )
    suite_name = next(iter(suite_names))

    n_episodes = len(episodes)
    n_success = sum(1 for ep in episodes if ep.success)
    overall_success_rate = n_success / n_episodes
    overall_failure_rate = 1.0 - overall_success_rate

    # B-30: ``success_safe_rate`` is the fraction of successful episodes
    # with ZERO recorded safety violations. Defined only when at least
    # one episode in the dataset carried safety telemetry AND there is
    # at least one success — otherwise ``None`` (distinct from ``0.0``,
    # which would mean "every success was unsafe"). Successes whose
    # backend left safety fields ``None`` are treated as "not measured"
    # — neither safe nor unsafe — and dropped from BOTH numerator and
    # denominator so a partial-coverage dataset does not bias the rate.
    success_safe_rate: float | None = None
    if n_success > 0 and any(_episode_has_any_safety_field(ep) for ep in episodes):
        measured_successes = [
            ep for ep in episodes if ep.success and _episode_has_any_safety_field(ep)
        ]
        if measured_successes:
            n_safe = sum(1 for ep in measured_successes if not episode_has_safety_violation(ep))
            success_safe_rate = n_safe / len(measured_successes)

    axis_names = _ordered_axis_names(episodes)

    per_axis = _per_axis_breakdowns(episodes, axis_names, confidence=confidence)
    per_cell = _per_cell_breakdowns(episodes, confidence=confidence)
    failure_clusters = _failure_clusters(
        episodes,
        axis_names,
        baseline_failure_rate=overall_failure_rate,
        cluster_multiple=cluster_multiple,
        min_cluster_size=min_cluster_size,
        confidence=confidence,
    )
    heatmap_2d = _heatmaps_2d(episodes, axis_names)

    sensitivity_indices: dict[str, SensitivityIndex] | None
    if axis_names:
        approximate = sampling != "sobol"
        raw = compute_sobol_indices(episodes, axis_names)
        sensitivity_indices = {
            name: SensitivityIndex(
                first_order=raw[name][0],
                total_order=raw[name][1],
                approximate=approximate,
            )
            for name in axis_names
            if name in raw
        }
    else:
        sensitivity_indices = None

    # B-04: calibration-aware abstention scoring. ``None`` when no
    # episode in the dataset carries a populated ``failure_score`` —
    # see :func:`gauntlet.report.abstention.compute_abstention_metrics`
    # for the gate.
    abstention_metrics = compute_abstention_metrics(episodes)

    return Report(
        suite_name=suite_name,
        suite_env=suite_env,
        n_episodes=n_episodes,
        n_success=n_success,
        per_axis=per_axis,
        per_cell=per_cell,
        failure_clusters=failure_clusters,
        heatmap_2d=heatmap_2d,
        overall_success_rate=overall_success_rate,
        overall_failure_rate=overall_failure_rate,
        cluster_multiple=cluster_multiple,
        sensitivity_indices=sensitivity_indices,
        success_safe_rate=success_safe_rate,
        abstention_metrics=abstention_metrics,
    )
