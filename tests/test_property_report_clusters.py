"""Property-based tests for ``gauntlet.report.analyze.build_report``.

Phase 2.5 Task 13 — covers the cluster-extraction invariants stated in
``GAUNTLET_SPEC.md`` §6 and the ``FailureCluster`` docstring:

* ``n_episodes >= min_cluster_size`` for every reported cluster.
* ``failure_rate >= cluster_multiple * baseline_failure_rate``.
* ``lift == failure_rate / baseline_failure_rate`` and the cluster list
  is sorted by ``(-lift, -failure_rate)``.
* ``per_cell`` row counts sum to the input episode count (no
  double-counting, no drops).
* The overall success / failure rates sum to 1.0.

We synthesise Episode lists with hypothesis-generated grid shapes and
randomised success bits. Cost is sub-millisecond per case (no env
construction, no I/O).
"""

from __future__ import annotations

from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from gauntlet.report.analyze import build_report
from gauntlet.runner.episode import Episode

# Two axes always present on the synthetic episodes — lets us hit the
# axis-pair clustering branch (the report short-circuits when fewer
# than two axes are present).
_AXIS_A = "axis_a"
_AXIS_B = "axis_b"


@st.composite
def _episode_list_strategy(draw: st.DrawFn) -> list[Episode]:
    """Generate a list of Episodes spread across a 2-axis grid."""
    n_a_vals = draw(st.integers(min_value=1, max_value=3))
    n_b_vals = draw(st.integers(min_value=1, max_value=3))
    eps_per_cell = draw(st.integers(min_value=1, max_value=4))
    a_values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_a_vals,
            max_size=n_a_vals,
            unique=True,
        )
    )
    b_values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_b_vals,
            max_size=n_b_vals,
            unique=True,
        )
    )
    success_seq = draw(
        st.lists(
            st.booleans(),
            min_size=n_a_vals * n_b_vals * eps_per_cell,
            max_size=n_a_vals * n_b_vals * eps_per_cell,
        )
    )

    episodes: list[Episode] = []
    cell_index = 0
    success_iter = iter(success_seq)
    for a in a_values:
        for b in b_values:
            for ep_idx in range(eps_per_cell):
                episodes.append(
                    Episode(
                        suite_name="prop-clusters",
                        cell_index=cell_index,
                        episode_index=ep_idx,
                        seed=cell_index * 100 + ep_idx,
                        perturbation_config={_AXIS_A: a, _AXIS_B: b},
                        success=next(success_iter),
                        terminated=True,
                        truncated=False,
                        step_count=10,
                        total_reward=1.0,
                    )
                )
            cell_index += 1
    return episodes


# ----- per-cell + overall invariants -----------------------------------------


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_per_cell_episode_counts_sum_to_total(episodes: list[Episode]) -> None:
    """Every Episode appears in exactly one ``per_cell`` row; counts sum
    to the total. A double-count or drop would break the §6 "never
    aggregate away failures" guarantee."""
    report = build_report(episodes)
    assert sum(row.n_episodes for row in report.per_cell) == len(episodes)
    assert sum(row.n_success for row in report.per_cell) == sum(1 for ep in episodes if ep.success)


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_overall_success_failure_rates_sum_to_one(episodes: list[Episode]) -> None:
    """``overall_success_rate + overall_failure_rate == 1.0`` exactly
    (single division, not a difference of two divisions)."""
    report = build_report(episodes)
    assert report.overall_success_rate + report.overall_failure_rate == 1.0
    assert 0.0 <= report.overall_success_rate <= 1.0


# ----- cluster invariants ----------------------------------------------------


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_clusters_meet_min_size_threshold(episodes: list[Episode]) -> None:
    """Every reported cluster carries at least ``min_cluster_size``
    episodes. Tested at the default threshold."""
    report = build_report(episodes, min_cluster_size=3)
    for cluster in report.failure_clusters:
        assert cluster.n_episodes >= 3


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_clusters_meet_lift_threshold(episodes: list[Episode]) -> None:
    """Every reported cluster has ``failure_rate >= cluster_multiple
    * baseline``. With ``cluster_multiple=2.0`` and a positive baseline,
    this is the inclusion criterion documented on
    :class:`FailureCluster`."""
    multiple = 2.0
    report = build_report(episodes, cluster_multiple=multiple)
    if report.overall_failure_rate <= 0.0:
        # No failures -> no clusters reportable.
        assert report.failure_clusters == []
        return
    for cluster in report.failure_clusters:
        assert cluster.failure_rate >= multiple * report.overall_failure_rate - 1e-12
        # ``lift`` is the canonical ratio.
        assert cluster.lift == cluster.failure_rate / report.overall_failure_rate


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_clusters_sorted_by_lift_descending(episodes: list[Episode]) -> None:
    """Cluster output is sorted ``(-lift, -failure_rate)`` for stable
    presentation — see :func:`gauntlet.report.analyze._failure_clusters`."""
    report = build_report(episodes)
    if len(report.failure_clusters) < 2:
        return
    pairs = [(c.lift, c.failure_rate) for c in report.failure_clusters]
    for i in range(len(pairs) - 1):
        assert pairs[i] >= pairs[i + 1]


@given(episodes=_episode_list_strategy())
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_axis_breakdown_counts_match_total(episodes: list[Episode]) -> None:
    """Every per-axis breakdown's ``counts`` sums to the input episode
    count — every episode must contribute to every axis breakdown
    exactly once."""
    report = build_report(episodes)
    for axis_breakdown in report.per_axis:
        assert sum(axis_breakdown.counts.values()) == len(episodes)
        assert sum(axis_breakdown.successes.values()) == sum(1 for ep in episodes if ep.success)
        for value, rate in axis_breakdown.rates.items():
            count = axis_breakdown.counts[value]
            success = axis_breakdown.successes[value]
            assert rate == success / count


# ----- empty / single-axis edge cases ----------------------------------------


def test_empty_episode_list_raises() -> None:
    """The documented behaviour is to refuse the empty case loudly."""
    import pytest

    with pytest.raises(ValueError, match="zero episodes"):
        build_report([])


@given(
    successes=st.lists(st.booleans(), min_size=1, max_size=12),
)
@settings(max_examples=50, deadline=timedelta(seconds=2))
def test_single_axis_suite_yields_no_clusters(successes: list[bool]) -> None:
    """The cluster routine iterates over axis pairs; a single-axis
    suite must short-circuit to an empty cluster list."""
    episodes = [
        Episode(
            suite_name="single-axis",
            cell_index=i,
            episode_index=0,
            seed=i,
            perturbation_config={_AXIS_A: float(i % 3)},
            success=ok,
            terminated=True,
            truncated=False,
            step_count=1,
            total_reward=0.0,
        )
        for i, ok in enumerate(successes)
    ]
    report = build_report(episodes)
    assert report.failure_clusters == []
    # Heatmaps require >=2 axes too — must be empty.
    assert report.heatmap_2d == {}
