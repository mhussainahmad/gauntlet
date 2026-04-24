"""Pure-function aggregation across a list of :class:`Report` objects.

The single public entry points are:

* :func:`discover_run_files` — recursive ``report.json`` glob.
* :func:`aggregate_reports` — pure transform from per-run reports to
  the fleet-level :class:`FleetReport`.
* :func:`aggregate_directory` — convenience: discover + load + aggregate.

Everything else is an internal helper. See
``docs/phase3-rfc-019-fleet-aggregate.md`` for the algorithm.

This module performs *no* I/O beyond the directory glob and per-file
``open`` in :func:`aggregate_directory`. The pure
:func:`aggregate_reports` step is testable without touching the
filesystem and is what the CLI subcommand ultimately drives.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from gauntlet.aggregate.schema import FleetReport, FleetRun
from gauntlet.report.analyze import _norm
from gauntlet.report.schema import AxisBreakdown, FailureCluster, Report

__all__ = [
    "aggregate_directory",
    "aggregate_reports",
    "discover_run_files",
]


# ---------------------------------------------------------------------------
# Discovery.
# ---------------------------------------------------------------------------


def discover_run_files(directory: Path) -> list[Path]:
    """Recursively find files literally named ``report.json``.

    Returns an absolute, sorted list of matches. Sort order is ``Path``-
    natural (the Python default lexical sort), which makes downstream
    aggregation deterministic.

    Raises:
        FileNotFoundError: if *directory* does not exist or is not a
            directory.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    matches = list(directory.rglob("report.json"))
    return sorted(matches)


# ---------------------------------------------------------------------------
# Aggregation primitives.
# ---------------------------------------------------------------------------


def _ordered_axis_names(reports: Iterable[Report]) -> tuple[str, ...]:
    """Union of per-report ``per_axis`` names in first-appearance order.

    Mirrors :func:`gauntlet.report.analyze._ordered_axis_names` — Python
    dict preserves insertion order so ``dict.fromkeys`` yields a stable,
    deterministic ordering without an explicit ``seen`` set.
    """
    seen: dict[str, None] = {}
    for rep in reports:
        for ab in rep.per_axis:
            if ab.name not in seen:
                seen[ab.name] = None
    return tuple(seen.keys())


def _aggregate_per_axis(
    reports: list[Report],
    axis_names: tuple[str, ...],
) -> dict[str, AxisBreakdown]:
    """Sum per-axis ``counts`` / ``successes`` across runs and recompute rates.

    Float keys are normalised through the shared ``_norm`` helper so
    1e-15 jitter doesn't split a bucket across two reports.
    """
    out: dict[str, AxisBreakdown] = {}
    for axis in axis_names:
        counts: dict[float, int] = defaultdict(int)
        successes: dict[float, int] = defaultdict(int)
        for rep in reports:
            ab = next((a for a in rep.per_axis if a.name == axis), None)
            if ab is None:
                continue
            for v, c in ab.counts.items():
                counts[_norm(v)] += c
            for v, s in ab.successes.items():
                successes[_norm(v)] += s
        sorted_keys = sorted(counts.keys())
        rates: dict[float, float] = {}
        for k in sorted_keys:
            n = counts[k]
            rates[k] = successes[k] / n if n > 0 else float("nan")
        out[axis] = AxisBreakdown(
            name=axis,
            rates=rates,
            counts={k: counts[k] for k in sorted_keys},
            successes={k: successes[k] for k in sorted_keys},
        )
    return out


def _cluster_fingerprint(
    cluster: FailureCluster,
) -> tuple[tuple[str, float], ...]:
    """Stable fingerprint for a :class:`FailureCluster`.

    Sort by axis-name so two clusters with the same (name, value) pairs
    in different insertion orders match. Float values are normalised
    through ``_norm``. Returns a ``tuple`` (hashable) so it can key a
    ``dict``.
    """
    return tuple(sorted((name, _norm(value)) for name, value in cluster.axes.items()))


@dataclass
class _ClusterAccumulator:
    """Internal scratch type — pools per-run cluster stats by fingerprint.

    Held as the value type of the ``pooled`` dict in
    :func:`_persistent_failure_clusters`. Defined as a dataclass (not
    a TypedDict) so mypy --strict can narrow the int / dict fields
    without per-line ``type: ignore`` casts.
    """

    axes: dict[str, float]
    n_episodes: int = 0
    n_success: int = 0
    appearances: int = 0


def _persistent_failure_clusters(
    reports: list[Report],
    *,
    persistence_threshold: float,
    fleet_baseline_failure_rate: float,
) -> list[FailureCluster]:
    """Return the fleet's persistent failure clusters.

    A cluster fingerprint is "persistent" if it appears in at least
    ``ceil(persistence_threshold * n_runs)`` runs (the test suite pins
    that the comparison is ``>=``). For each persistent fingerprint,
    ``n_episodes`` and ``n_success`` are SUMMED across the runs that
    carried the cluster; ``failure_rate`` is recomputed from those
    sums; ``lift`` is taken against the *fleet* baseline failure rate.

    The output is sorted by ``lift`` desc then ``failure_rate`` desc,
    matching :func:`gauntlet.report.analyze._failure_clusters` for
    consistent presentation.

    Empty result paths:

    * ``n_runs == 0`` → empty list (also caught upstream).
    * No cluster appears often enough → empty list.
    * ``fleet_baseline_failure_rate <= 0.0`` → empty list (no failures
      to lift against; mirrors the per-run "all-success" short-circuit).
    """
    if fleet_baseline_failure_rate <= 0.0:
        return []
    n_runs = len(reports)
    if n_runs == 0:
        return []

    pooled: dict[tuple[tuple[str, float], ...], _ClusterAccumulator] = {}
    for rep in reports:
        # Each run's cluster set may carry the same fingerprint multiple
        # times only when build_report mis-emits — in practice
        # build_report dedupes; the per-run set is a defensive guard.
        seen_in_run: set[tuple[tuple[str, float], ...]] = set()
        for cluster in rep.failure_clusters:
            fp = _cluster_fingerprint(cluster)
            entry = pooled.setdefault(fp, _ClusterAccumulator(axes=dict(fp)))
            entry.n_episodes += cluster.n_episodes
            entry.n_success += cluster.n_success
            if fp not in seen_in_run:
                entry.appearances += 1
                seen_in_run.add(fp)

    threshold_count = persistence_threshold * n_runs
    persistent: list[FailureCluster] = []
    for entry in pooled.values():
        if entry.appearances < threshold_count:
            continue
        if entry.n_episodes == 0:
            continue
        failure_rate = (entry.n_episodes - entry.n_success) / entry.n_episodes
        lift = failure_rate / fleet_baseline_failure_rate
        persistent.append(
            FailureCluster(
                axes=dict(entry.axes),
                n_episodes=entry.n_episodes,
                n_success=entry.n_success,
                failure_rate=failure_rate,
                lift=lift,
            )
        )

    persistent.sort(key=lambda c: (-c.lift, -c.failure_rate))
    return persistent


def _cross_run_success_distribution(
    reports: list[Report],
) -> dict[str, list[float]]:
    """Group each run's overall success rate by ``suite_name``.

    Order within each bucket matches the *input* order of ``reports``
    (which is sorted by source path under :func:`aggregate_directory`),
    so two aggregations of the same directory produce byte-identical
    output.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for rep in reports:
        out[rep.suite_name].append(rep.overall_success_rate)
    # Convert defaultdict → dict so the schema's strict dict typing
    # holds and so re-entering this dict elsewhere doesn't auto-create
    # missing keys.
    return dict(out)


# ---------------------------------------------------------------------------
# Public aggregation entry points.
# ---------------------------------------------------------------------------


def aggregate_reports(
    reports: list[Report],
    *,
    persistence_threshold: float = 0.5,
    runs: list[FleetRun] | None = None,
) -> FleetReport:
    """Aggregate a list of per-run :class:`Report` into a :class:`FleetReport`.

    Pure function — no I/O. ``runs`` is optional and threads through
    the directory loader's :class:`FleetRun` rows so the
    ``fleet_report.json`` carries the source-file metadata; passing
    ``None`` synthesises minimal :class:`FleetRun` rows from each
    report's ``suite_name`` / counts (no ``source_file`` to point at).

    Args:
        reports: per-run reports to aggregate. Must be non-empty.
        persistence_threshold: cluster fingerprint must appear in at
            least ``persistence_threshold * n_runs`` runs to be
            included. Comparison is ``>=`` (the threshold value itself
            is included). Must be in ``[0.0, 1.0]``.
        runs: optional :class:`FleetRun` rows already populated by the
            directory loader. When ``None``, minimal rows are derived
            from each report.

    Returns:
        A fully populated :class:`FleetReport`.

    Raises:
        ValueError: if ``reports`` is empty or
            ``persistence_threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= persistence_threshold <= 1.0:
        raise ValueError(
            f"persistence_threshold must be in [0.0, 1.0]; got {persistence_threshold}",
        )
    if len(reports) == 0:
        raise ValueError("cannot aggregate zero reports")

    n_runs = len(reports)
    n_total_episodes = sum(r.n_episodes for r in reports)
    n_total_success = sum(r.n_success for r in reports)
    fleet_failure_rate = (
        (n_total_episodes - n_total_success) / n_total_episodes if n_total_episodes > 0 else 0.0
    )

    axis_names = _ordered_axis_names(reports)
    per_axis_aggregate = _aggregate_per_axis(reports, axis_names)
    persistent_clusters = _persistent_failure_clusters(
        reports,
        persistence_threshold=persistence_threshold,
        fleet_baseline_failure_rate=fleet_failure_rate,
    )
    distribution = _cross_run_success_distribution(reports)

    rates = [r.overall_success_rate for r in reports]
    mean_rate = statistics.fmean(rates)
    std_rate = statistics.pstdev(rates) if n_runs > 1 else 0.0

    if runs is None:
        runs = [
            FleetRun(
                run_id=f"run-{i:04d}",
                policy_label=f"run-{i:04d}",
                suite_name=rep.suite_name,
                suite_env=rep.suite_env,
                n_episodes=rep.n_episodes,
                n_success=rep.n_success,
                success_rate=rep.overall_success_rate,
                source_file="",
            )
            for i, rep in enumerate(reports)
        ]

    suite_names = sorted({r.suite_name for r in reports})

    return FleetReport(
        runs=runs,
        n_runs=n_runs,
        n_total_episodes=n_total_episodes,
        per_axis_aggregate=per_axis_aggregate,
        persistent_failure_clusters=persistent_clusters,
        cross_run_success_distribution=distribution,
        persistence_threshold=persistence_threshold,
        fleet_baseline_failure_rate=fleet_failure_rate,
        mean_success_rate=mean_rate,
        std_success_rate=std_rate,
        suite_names=suite_names,
    )


def aggregate_directory(
    directory: Path,
    *,
    persistence_threshold: float = 0.5,
) -> FleetReport:
    """Discover ``report.json`` files under *directory* and aggregate them.

    Builds :class:`FleetRun` rows whose ``source_file`` is the path
    *relative to* ``directory`` (RFC §3) so the resulting
    ``fleet_report.json`` is movable.

    Args:
        directory: scan root.
        persistence_threshold: see :func:`aggregate_reports`.

    Returns:
        A fully populated :class:`FleetReport`.

    Raises:
        FileNotFoundError: if *directory* does not exist.
        ValueError: if no ``report.json`` files are found, if any
            file is malformed JSON, or if any file fails Pydantic
            validation as a :class:`Report`.
    """
    files = discover_run_files(directory)
    if not files:
        raise ValueError(f"no report.json files found under {directory}")

    reports: list[Report] = []
    runs: list[FleetRun] = []
    for path in files:
        rel = path.relative_to(directory).as_posix()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
        try:
            report = Report.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"{rel}: not a valid report.json: {exc}") from exc
        reports.append(report)
        # ``policy_label`` and ``run_id`` derive from the run dir name
        # (the parent of ``report.json``). For files placed directly in
        # ``directory`` the parent is ``"."`` — use the file stem as
        # a fallback so the label is never empty.
        parent = path.parent.name or path.stem
        runs.append(
            FleetRun(
                run_id=parent,
                policy_label=parent,
                suite_name=report.suite_name,
                suite_env=report.suite_env,
                n_episodes=report.n_episodes,
                n_success=report.n_success,
                success_rate=report.overall_success_rate,
                source_file=rel,
            )
        )

    return aggregate_reports(
        reports,
        persistence_threshold=persistence_threshold,
        runs=runs,
    )
