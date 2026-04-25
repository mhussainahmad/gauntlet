"""Branch-coverage backfill for :mod:`gauntlet.report`.

Phase 2.5 Task 11. Targets the specific missed-branch lines in
``gauntlet/report/analyze.py`` (92, 189, 197, 246) and
``gauntlet/report/html.py`` (86) that the existing report-test suite
does not exercise.

The four ``analyze.py`` gaps all live on "skip if axis missing on this
episode" branches in ``_per_axis_breakdowns``, ``_failure_clusters``,
and ``_heatmaps_2d``. Real Runner output never produces an episode
that omits an axis its sibling carries, but ``build_report`` is
documented as tolerant of hand-constructed mixed-shape inputs and the
branches are reachable via a heterogeneous episode list.

The ``html.py`` gap is the ``tuple`` branch of ``_nan_to_none`` —
sanitiser that the Jinja template never feeds tuples but is reachable
via a direct call. Covered by a unit-level test rather than rebuilding
a whole Report.
"""

from __future__ import annotations

import math

from gauntlet.report import build_report
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode


def _make_episode(
    *,
    cell_index: int,
    perturbation_config: dict[str, float],
    success: bool,
    video_path: str | None = None,
) -> Episode:
    return Episode(
        suite_name="cov-suite",
        cell_index=cell_index,
        episode_index=0,
        seed=cell_index + 1,
        perturbation_config=perturbation_config,
        success=success,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
        video_path=video_path,
    )


def test_per_axis_breakdown_skips_episode_missing_axis() -> None:
    """analyze.py L92: ``axis not in ep.perturbation_config`` → continue.

    Build episodes where some carry only ``axis_a`` and others only
    ``axis_b``. ``_ordered_axis_names`` returns both axes, so the inner
    loop hits the skip-branch on every episode that lacks the current
    axis.
    """
    episodes = [
        _make_episode(cell_index=0, perturbation_config={"axis_a": 0.1}, success=True),
        _make_episode(cell_index=1, perturbation_config={"axis_a": 0.2}, success=False),
        _make_episode(cell_index=2, perturbation_config={"axis_b": 1.0}, success=True),
        _make_episode(cell_index=3, perturbation_config={"axis_b": 2.0}, success=False),
    ]

    report = build_report(episodes)

    by_name = {b.name: b for b in report.per_axis}
    assert set(by_name) == {"axis_a", "axis_b"}
    # axis_a only saw the two episodes that carried it; same for axis_b.
    assert sum(by_name["axis_a"].counts.values()) == 2
    assert sum(by_name["axis_b"].counts.values()) == 2


def test_failure_cluster_skips_episode_missing_either_axis() -> None:
    """analyze.py L189: pair-loop continues when an axis is missing.

    Triggers the cluster path (>= 2 axes, baseline failure > 0). One
    episode carries only one of the pair-axes — the inner loop must
    skip it without contributing to ``pair_counts``.
    """
    episodes: list[Episode] = []
    # Failures concentrated on (axis_a=1.0, axis_b=2.0) so a cluster
    # actually fires; many baseline episodes spread elsewhere so the
    # baseline failure rate is non-zero but well below the cluster rate.
    for i in range(8):
        episodes.append(
            _make_episode(
                cell_index=i,
                perturbation_config={"axis_a": 1.0, "axis_b": 2.0},
                success=False,
            )
        )
    for i in range(20):
        episodes.append(
            _make_episode(
                cell_index=100 + i,
                perturbation_config={"axis_a": 0.0, "axis_b": 0.0},
                success=True,
            )
        )
    # The single skip-branch trigger: axis_b missing.
    episodes.append(
        _make_episode(
            cell_index=999,
            perturbation_config={"axis_a": 0.0},
            success=False,
        )
    )

    report = build_report(episodes, min_cluster_size=4, cluster_multiple=1.5)

    # Cluster on the (1.0, 2.0) corner is reported.
    assert any(
        c.axes.get("axis_a") == 1.0 and c.axes.get("axis_b") == 2.0 for c in report.failure_clusters
    ), "expected the failed corner to surface as a cluster"


def test_failure_cluster_collects_video_paths_from_failures() -> None:
    """analyze.py L197: ``elif ep.video_path is not None`` adds to pair_videos.

    A failed episode in the clustered cell that carries a
    ``video_path`` must surface that path on the resulting
    :class:`FailureCluster.video_paths` list.
    """
    episodes: list[Episode] = []
    for i in range(6):
        episodes.append(
            _make_episode(
                cell_index=i,
                perturbation_config={"x": 1.0, "y": 2.0},
                success=False,
                video_path=f"videos/fail_{i}.mp4",
            )
        )
    for i in range(20):
        episodes.append(
            _make_episode(
                cell_index=100 + i,
                perturbation_config={"x": 0.0, "y": 0.0},
                success=True,
            )
        )

    report = build_report(episodes, min_cluster_size=4, cluster_multiple=1.5)

    cluster = next(
        c for c in report.failure_clusters if c.axes.get("x") == 1.0 and c.axes.get("y") == 2.0
    )
    assert cluster.video_paths, "expected failed-episode videos to be attached"
    assert all(p.startswith("videos/fail_") for p in cluster.video_paths)


def test_heatmap_skips_episode_missing_either_axis() -> None:
    """analyze.py L246: heatmap inner-loop skip on missing axis.

    A 2D heatmap is only emitted for ``len(axis_names) >= 2``. Mix one
    full ``(x, y)`` episode with a singleton ``x``-only episode — the
    second must be skipped without polluting the matrix.
    """
    episodes = [
        _make_episode(cell_index=0, perturbation_config={"x": 0.0, "y": 0.0}, success=True),
        _make_episode(cell_index=1, perturbation_config={"x": 0.0, "y": 1.0}, success=False),
        # Trigger for L246 — only the x axis present.
        _make_episode(cell_index=2, perturbation_config={"x": 0.0}, success=False),
    ]

    report = build_report(episodes)

    assert "x__y" in report.heatmap_2d
    heatmap = report.heatmap_2d["x__y"]
    # Only the two (x=0.0, y={0.0,1.0}) cells contributed; nothing
    # extra rolled into the matrix from the orphan episode.
    total = sum(1 for row in heatmap.success_rate for v in row if not math.isnan(v))
    assert total == 2


def test_html_nan_to_none_handles_tuple_input() -> None:
    """html.py L86: ``isinstance(value, tuple)`` branch.

    The Jinja template path always feeds dicts/lists/floats but the
    sanitiser is reachable via a direct call from third-party
    consumers. A tuple containing both finite and non-finite floats
    must round-trip with NaNs collapsed to ``None`` and the outer
    container preserved as a tuple.
    """
    out = _nan_to_none((1.0, float("nan"), float("inf"), 2.5))

    assert isinstance(out, tuple)
    assert out == (1.0, None, None, 2.5)
