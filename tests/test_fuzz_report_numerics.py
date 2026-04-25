"""Fuzz tests for ``Report`` numerical invariants.

Phase 2.5 Task 13 — covers the gap left by ``test_property_report_clusters``:
that test asserts the cluster invariants and the per-cell sum-to-total
property, but it does NOT do a general "every number in the report is
finite or NaN-only-where-documented" sweep. This file does.

Properties under test (over hypothesis-generated Episode lists):

* Every probability in the Report is in ``[0, 1]`` exactly:
  ``overall_success_rate``, ``overall_failure_rate``, every
  ``per_axis.rates`` value, every ``per_cell.success_rate``, every
  finite cell of every ``heatmap_2d`` matrix.
* Every count is a non-negative integer: ``n_episodes``, ``n_success``,
  every ``per_axis.counts`` / ``successes`` value, every
  ``per_cell.n_episodes`` / ``n_success``.
* Every Heatmap2D cell is either a probability in ``[0, 1]`` OR exactly
  ``float('nan')`` (the documented sentinel for an empty grid cell).
  No infs anywhere — clusters cannot reach inf because the schema
  forbids zero-baseline failure_rate from reaching the cluster filter
  (analyzed branch is short-circuited).
* Every ``FailureCluster.lift`` is finite and ``>= 1`` (the
  ``cluster_multiple`` floor of 1.0 means the cluster filter only
  admits clusters at or above baseline; with the default 2.0 multiple
  every reported lift is ``>= 2.0``).
* The Report itself round-trips through JSON. This pins
  ``ser_json_inf_nan="strings"`` for the heat-map NaN cells: a regression
  to the pydantic 2 default would emit ``null`` and refuse revalidation.

Hypothesis budget: ``max_examples=50`` per test, sub-millisecond per case
(no env, no I/O).
"""

from __future__ import annotations

import math
from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from gauntlet.report.analyze import build_report
from gauntlet.report.schema import Report
from gauntlet.runner.episode import Episode

# Two axes is the minimum for the cluster + heatmap branches to fire;
# three is the maximum we generate to keep the per-axis combinatorics
# bounded.
_MAX_AXES = 3
_AXIS_NAMES = ("axis_a", "axis_b", "axis_c")
_BOUND_VALUE = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)


@st.composite
def _episode_list(draw: st.DrawFn) -> list[Episode]:
    """Build an Episode list across an N-axis grid with random success bits.

    The grid is intentionally small so the Cartesian product never
    explodes (worst case: ``3 axes x 3 vals/axis x 3 eps = 81``
    episodes). Hypothesis shrinks down to single-cell suites for failing
    cases.
    """
    n_axes = draw(st.integers(min_value=2, max_value=_MAX_AXES))
    axis_names = list(_AXIS_NAMES[:n_axes])
    per_axis_vals = []
    for _ in axis_names:
        n_vals = draw(st.integers(min_value=1, max_value=3))
        vals = draw(st.lists(_BOUND_VALUE, min_size=n_vals, max_size=n_vals, unique=True))
        per_axis_vals.append(vals)
    eps_per_cell = draw(st.integers(min_value=1, max_value=3))

    # Build the cartesian product of axis values. Stable nested-loop
    # keeps the Episode list ordering deterministic for a given draw.
    cells: list[dict[str, float]] = [{}]
    for name, vals in zip(axis_names, per_axis_vals, strict=True):
        cells = [{**existing, name: v} for existing in cells for v in vals]

    n_eps_total = len(cells) * eps_per_cell
    successes = draw(st.lists(st.booleans(), min_size=n_eps_total, max_size=n_eps_total))

    episodes: list[Episode] = []
    success_iter = iter(successes)
    for cell_idx, config in enumerate(cells):
        for ep_idx in range(eps_per_cell):
            episodes.append(
                Episode(
                    suite_name="prop-numerics",
                    cell_index=cell_idx,
                    episode_index=ep_idx,
                    seed=cell_idx * 100 + ep_idx,
                    perturbation_config=dict(config),
                    success=next(success_iter),
                    terminated=True,
                    truncated=False,
                    step_count=1,
                    total_reward=0.0,
                )
            )
    return episodes


# ----- probability invariants ------------------------------------------------


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_overall_rates_are_probabilities(episodes: list[Episode]) -> None:
    report = build_report(episodes)
    assert 0.0 <= report.overall_success_rate <= 1.0
    assert 0.0 <= report.overall_failure_rate <= 1.0
    assert math.isfinite(report.overall_success_rate)
    assert math.isfinite(report.overall_failure_rate)


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_per_axis_rates_and_counts_are_well_formed(episodes: list[Episode]) -> None:
    """Every per-axis breakdown rate is a finite probability; every count
    is a non-negative int with ``successes <= counts``."""
    report = build_report(episodes)
    for breakdown in report.per_axis:
        for value, rate in breakdown.rates.items():
            assert math.isfinite(rate), (
                f"axis {breakdown.name} value {value} rate not finite: {rate}"
            )
            assert 0.0 <= rate <= 1.0, f"axis {breakdown.name} rate out of [0,1]: {rate}"
            assert breakdown.counts[value] >= 0
            assert breakdown.successes[value] >= 0
            assert breakdown.successes[value] <= breakdown.counts[value]


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_per_cell_rates_and_counts_are_well_formed(episodes: list[Episode]) -> None:
    report = build_report(episodes)
    for cell in report.per_cell:
        assert math.isfinite(cell.success_rate)
        assert 0.0 <= cell.success_rate <= 1.0
        assert cell.n_episodes >= 1
        assert 0 <= cell.n_success <= cell.n_episodes


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_heatmap_cells_are_probability_or_nan(episodes: list[Episode]) -> None:
    """Every heat-map matrix cell is either a probability in ``[0, 1]``
    OR exactly ``float('nan')`` (the documented empty-cell sentinel).
    Inf is never a legal cell value."""
    report = build_report(episodes)
    for heatmap in report.heatmap_2d.values():
        n_y = len(heatmap.y_values)
        n_x = len(heatmap.x_values)
        assert len(heatmap.success_rate) == n_y
        for row in heatmap.success_rate:
            assert len(row) == n_x
            for cell in row:
                if math.isnan(cell):
                    continue
                assert math.isfinite(cell), f"heat-map cell is +/- inf: {cell}"
                assert 0.0 <= cell <= 1.0, f"heat-map cell out of [0,1]: {cell}"


# ----- cluster invariants beyond the lift / size threshold tests --------------


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_cluster_lift_is_finite_and_at_least_multiple(episodes: list[Episode]) -> None:
    """Every reported cluster has ``lift >= cluster_multiple`` (the
    inclusion criterion) AND a finite, positive ``failure_rate`` and
    ``lift``. ``cluster_multiple`` defaults to 2.0."""
    multiple = 2.0
    report = build_report(episodes, cluster_multiple=multiple)
    for cluster in report.failure_clusters:
        assert math.isfinite(cluster.failure_rate)
        assert math.isfinite(cluster.lift)
        # Every reported failure_rate is at least ``multiple * baseline``,
        # so ``lift >= multiple``.
        assert cluster.lift >= multiple - 1e-12, (
            f"cluster lift {cluster.lift} below multiple {multiple}"
        )
        assert 0.0 < cluster.failure_rate <= 1.0
        assert cluster.n_episodes >= 1
        assert 0 <= cluster.n_success < cluster.n_episodes  # cluster requires at least one failure


# ----- JSON round-trip pins the NaN-as-string behaviour ----------------------


@given(episodes=_episode_list())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_report_json_round_trip_preserves_numerics(episodes: list[Episode]) -> None:
    """``Report.model_validate_json(report.model_dump_json())`` is an
    identity over hypothesis-generated reports. Heat-map NaN cells must
    round-trip via ``ser_json_inf_nan='strings'`` — the existing
    ``test_nonfinite_float_roundtrip`` only pins fixed examples; this
    exercises the same surface against hypothesis-generated empty
    cells."""
    report = build_report(episodes)
    js = report.model_dump_json()
    rt = Report.model_validate_json(js)
    assert rt.n_episodes == report.n_episodes
    assert rt.n_success == report.n_success
    assert rt.overall_success_rate == report.overall_success_rate
    assert len(rt.heatmap_2d) == len(report.heatmap_2d)
    for key, hm in report.heatmap_2d.items():
        rt_hm = rt.heatmap_2d[key]
        for y in range(len(hm.success_rate)):
            for x in range(len(hm.success_rate[y])):
                src = hm.success_rate[y][x]
                got = rt_hm.success_rate[y][x]
                if math.isnan(src):
                    assert math.isnan(got)
                else:
                    assert src == got
