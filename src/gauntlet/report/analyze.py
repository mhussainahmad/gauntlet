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

from gauntlet.report.schema import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Heatmap2D,
    Report,
)
from gauntlet.runner.episode import Episode

__all__ = ["build_report"]


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
) -> list[AxisBreakdown]:
    """Compute one :class:`AxisBreakdown` per axis name.

    Only axes that actually carry a value in *every* episode contribute
    to the marginal; missing keys are skipped (they should not occur in
    Runner-produced episodes, but tolerating them keeps the function
    safe under hand-constructed inputs).
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
        breakdowns.append(
            AxisBreakdown(
                name=axis,
                rates=rates,
                counts={k: counts[k] for k in sorted_keys},
                successes={k: successes[k] for k in sorted_keys},
            )
        )
    return breakdowns


def _per_cell_breakdowns(episodes: list[Episode]) -> list[CellBreakdown]:
    """Group episodes by ``(cell_index, frozenset(normalized_config))``.

    Returns one :class:`CellBreakdown` per group, sorted by
    ``cell_index`` ascending. Cell-id collisions across distinct configs
    (which the Runner never produces) become separate entries — the
    sort key is just the index, so order is still stable.
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
        rows.append(
            CellBreakdown(
                cell_index=key[0],
                perturbation_config=configs[key],
                n_episodes=n,
                n_success=n_success,
                success_rate=n_success / n,
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
        for ep in episodes:
            if axis_a not in ep.perturbation_config or axis_b not in ep.perturbation_config:
                continue
            va = _norm(ep.perturbation_config[axis_a])
            vb = _norm(ep.perturbation_config[axis_b])
            bucket = pair_counts[(va, vb)]
            bucket[0] += 1
            if ep.success:
                bucket[1] += 1

        for (va, vb), (n_total, n_success) in pair_counts.items():
            if n_total < min_cluster_size:
                continue
            failure_rate = (n_total - n_success) / n_total
            if failure_rate < cluster_multiple * baseline_failure_rate:
                continue
            clusters.append(
                FailureCluster(
                    axes={axis_a: va, axis_b: vb},
                    n_episodes=n_total,
                    n_success=n_success,
                    failure_rate=failure_rate,
                    lift=failure_rate / baseline_failure_rate,
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

    Returns:
        A fully populated :class:`Report`.

    Raises:
        ValueError: if ``episodes`` is empty, contains rows from more
            than one suite, or ``cluster_multiple <= 0``.
    """
    if cluster_multiple <= 0:
        raise ValueError(
            f"cluster_multiple must be > 0; got {cluster_multiple}",
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

    axis_names = _ordered_axis_names(episodes)

    per_axis = _per_axis_breakdowns(episodes, axis_names)
    per_cell = _per_cell_breakdowns(episodes)
    failure_clusters = _failure_clusters(
        episodes,
        axis_names,
        baseline_failure_rate=overall_failure_rate,
        cluster_multiple=cluster_multiple,
        min_cluster_size=min_cluster_size,
    )
    heatmap_2d = _heatmaps_2d(episodes, axis_names)

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
    )
