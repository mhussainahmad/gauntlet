"""Report tests — see ``GAUNTLET_SPEC.md`` §5 task 7 and §6.

The tests construct synthetic :class:`Episode` lists directly; the
Runner is exercised in :mod:`tests.test_runner`. The "never aggregate
away failures" rule (§6) is the load-bearing one — the marginal,
per-cell, and failure-cluster tests are what protect it.

Determinism / round-trip notes:

* Pydantic model equality reduces to field equality, and ``float("nan")
  != float("nan")``. The determinism + round-trip tests therefore use
  fixtures whose 2D heatmaps have no empty cells (full grids), or rely
  on the 1-axis case where ``heatmap_2d`` is empty.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from gauntlet.report import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Heatmap2D,
    Report,
    build_report,
)
from gauntlet.runner.episode import Episode

# Episode factory helpers — keep tests terse without hiding fields.

_SUITE = "test-suite-v1"


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    suite_name: str = _SUITE,
    seed: int = 0,
) -> Episode:
    """Build an Episode with sensible outcome defaults."""
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def _episodes_from_grid(
    *,
    axis_a: str,
    a_values: list[float],
    axis_b: str,
    b_values: list[float],
    successes_per_cell: int,
    failures_per_cell: int,
) -> list[Episode]:
    """Build a fully populated 2D grid with a fixed success/failure ratio per cell.

    Used by the determinism + round-trip tests where every heatmap cell
    must be populated (NaN poisons equality).
    """
    episodes: list[Episode] = []
    cell_index = 0
    for a in a_values:
        for b in b_values:
            ep_idx = 0
            for _ in range(successes_per_cell):
                episodes.append(
                    _ep(
                        cell_index=cell_index,
                        episode_index=ep_idx,
                        success=True,
                        config={axis_a: a, axis_b: b},
                    )
                )
                ep_idx += 1
            for _ in range(failures_per_cell):
                episodes.append(
                    _ep(
                        cell_index=cell_index,
                        episode_index=ep_idx,
                        success=False,
                        config={axis_a: a, axis_b: b},
                    )
                )
                ep_idx += 1
            cell_index += 1
    return episodes


# 1. Single-episode reports.
def test_single_episode_success_rate_one() -> None:
    report = build_report(
        [_ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.5})]
    )
    assert report.overall_success_rate == 1.0
    assert report.overall_failure_rate == 0.0
    assert report.n_episodes == 1
    assert report.n_success == 1


def test_single_episode_failure_rate_one() -> None:
    report = build_report(
        [_ep(cell_index=0, episode_index=0, success=False, config={"lighting_intensity": 0.5})]
    )
    assert report.overall_success_rate == 0.0
    assert report.overall_failure_rate == 1.0


# 2. Empty list.
def test_empty_episode_list_raises() -> None:
    with pytest.raises(ValueError, match="zero episodes"):
        build_report([])


# 3. Mixed suite_name.
def test_mixed_suite_names_raises() -> None:
    eps = [
        _ep(
            cell_index=0,
            episode_index=0,
            success=True,
            config={"lighting_intensity": 0.5},
            suite_name="alpha",
        ),
        _ep(
            cell_index=0,
            episode_index=1,
            success=True,
            config={"lighting_intensity": 0.5},
            suite_name="beta",
        ),
    ]
    with pytest.raises(ValueError, match="suite_name"):
        build_report(eps)


# 4. Marginal computation.
def test_per_axis_marginal_rates_exact() -> None:
    eps: list[Episode] = []
    # lighting=0.3 → 3 success, 1 failure.
    for i in range(3):
        eps.append(
            _ep(cell_index=0, episode_index=i, success=True, config={"lighting_intensity": 0.3})
        )
    eps.append(
        _ep(cell_index=0, episode_index=3, success=False, config={"lighting_intensity": 0.3})
    )
    # lighting=1.5 → 1 success, 3 failures.
    eps.append(_ep(cell_index=1, episode_index=0, success=True, config={"lighting_intensity": 1.5}))
    for i in range(3):
        eps.append(
            _ep(
                cell_index=1, episode_index=i + 1, success=False, config={"lighting_intensity": 1.5}
            )
        )

    report = build_report(eps)

    assert len(report.per_axis) == 1
    bd = report.per_axis[0]
    assert bd.name == "lighting_intensity"
    assert bd.rates == {0.3: 0.75, 1.5: 0.25}
    assert bd.counts == {0.3: 4, 1.5: 4}
    assert bd.successes == {0.3: 3, 1.5: 1}


# 5. Per-cell breakdown.
def test_per_cell_breakdown_two_cells_sorted_by_index() -> None:
    eps = [
        _ep(cell_index=1, episode_index=0, success=True, config={"lighting_intensity": 1.5}),
        _ep(cell_index=1, episode_index=1, success=False, config={"lighting_intensity": 1.5}),
        _ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.3}),
        _ep(cell_index=0, episode_index=1, success=True, config={"lighting_intensity": 0.3}),
    ]
    report = build_report(eps)

    assert [c.cell_index for c in report.per_cell] == [0, 1]
    assert report.per_cell[0].n_episodes == 2
    assert report.per_cell[0].n_success == 2
    assert report.per_cell[0].success_rate == 1.0
    assert report.per_cell[1].n_episodes == 2
    assert report.per_cell[1].n_success == 1
    assert report.per_cell[1].success_rate == 0.5
    assert report.per_cell[1].perturbation_config == {"lighting_intensity": 1.5}


# 6. Failure cluster fires when failure rate is >= 2x baseline.
def test_failure_cluster_fires_when_rate_exceeds_threshold() -> None:
    # Build a 2x2 axis grid:
    #   (0.3, 0.0) → all success (3/3)
    #   (0.3, 1.0) → all success (3/3)
    #   (1.5, 0.0) → all success (3/3)
    #   (1.5, 1.0) → all failure (0/3)
    # baseline_failure_rate = 3/12 = 0.25
    # The (1.5, 1.0) cell has failure_rate = 1.0 >= 2 * 0.25 = 0.5.
    eps: list[Episode] = []
    cell = 0
    for lighting in (0.3, 1.5):
        for camera in (0.0, 1.0):
            for i in range(3):
                success = not (lighting == 1.5 and camera == 1.0)
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=i,
                        success=success,
                        config={"lighting_intensity": lighting, "camera_offset_x": camera},
                    )
                )
            cell += 1

    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    assert len(report.failure_clusters) >= 1
    top = report.failure_clusters[0]
    assert top.axes == {"lighting_intensity": 1.5, "camera_offset_x": 1.0}
    assert top.n_episodes == 3
    assert top.n_success == 0
    assert top.failure_rate == 1.0
    assert top.lift == pytest.approx(1.0 / 0.25)


# 7. Failure cluster does NOT fire when count < min_cluster_size.
def test_failure_cluster_skipped_when_under_min_cluster_size() -> None:
    # Same shape as above, but only 1 episode in the bad cell — which
    # is below the default min_cluster_size of 3, so no cluster.
    # Also need enough failures elsewhere for a non-zero baseline.
    eps: list[Episode] = []
    # Three successful (0.3, 0.0) cells for baseline.
    for i in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=i,
                success=True,
                config={"lighting_intensity": 0.3, "camera_offset_x": 0.0},
            )
        )
    # One success and one failure at (1.5, 0.0) for baseline failures.
    eps.append(
        _ep(
            cell_index=1,
            episode_index=0,
            success=True,
            config={"lighting_intensity": 1.5, "camera_offset_x": 0.0},
        )
    )
    eps.append(
        _ep(
            cell_index=1,
            episode_index=1,
            success=False,
            config={"lighting_intensity": 1.5, "camera_offset_x": 0.0},
        )
    )
    # ONE failure at (1.5, 1.0) — below min_cluster_size.
    eps.append(
        _ep(
            cell_index=2,
            episode_index=0,
            success=False,
            config={"lighting_intensity": 1.5, "camera_offset_x": 1.0},
        )
    )

    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    # No cluster contains the (1.5, 1.0) combo, since it has only 1 episode.
    for cluster in report.failure_clusters:
        assert cluster.axes != {"lighting_intensity": 1.5, "camera_offset_x": 1.0}


# 8. Failure cluster empty when baseline failure rate is zero.
def test_failure_cluster_empty_when_all_success() -> None:
    eps = [
        _ep(
            cell_index=i // 3,
            episode_index=i % 3,
            success=True,
            config={"lighting_intensity": float(i // 3), "camera_offset_x": float(i % 3)},
        )
        for i in range(9)
    ]
    report = build_report(eps)
    assert report.overall_failure_rate == 0.0
    assert report.failure_clusters == []


# 9. failure_clusters sorted by lift desc.
def test_failure_clusters_sorted_by_lift_descending() -> None:
    # Construct a mostly-success baseline (low failure rate) with two
    # distinct failure-prone axis pair combinations of different
    # severities. Both must satisfy failure_rate >= 2 * baseline AND
    # min_cluster_size, and they must come out sorted by lift desc.
    eps: list[Episode] = []
    cell = 0

    # Mostly-success bulk: 80 successes covering a 4x4 axis grid (5
    # successes per cell, no failures). Drives baseline_failure_rate
    # towards 0 so even a moderate cluster fires.
    for lighting in (0.0, 0.5, 1.0, 1.5):
        for camera in (-0.05, 0.0, 0.05, 0.1):
            for i in range(5):
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=i,
                        success=True,
                        config={"lighting_intensity": lighting, "camera_offset_x": camera},
                    )
                )
            cell += 1

    # Severity A: (lighting=2.0, camera=2.0) — 5/5 failures (rate 1.0).
    for i in range(5):
        eps.append(
            _ep(
                cell_index=cell,
                episode_index=i,
                success=False,
                config={"lighting_intensity": 2.0, "camera_offset_x": 2.0},
            )
        )
    cell += 1

    # Severity B: (lighting=3.0, camera=3.0) — 3/4 failures (rate 0.75).
    for i in range(3):
        eps.append(
            _ep(
                cell_index=cell,
                episode_index=i,
                success=False,
                config={"lighting_intensity": 3.0, "camera_offset_x": 3.0},
            )
        )
    eps.append(
        _ep(
            cell_index=cell,
            episode_index=3,
            success=True,
            config={"lighting_intensity": 3.0, "camera_offset_x": 3.0},
        )
    )

    # baseline_failure_rate = 8 / 89 ≈ 0.0899
    # Severity A lift = 1.0 / 0.0899 ≈ 11.1
    # Severity B lift = 0.75 / 0.0899 ≈ 8.3
    # Both clear the 2.0 threshold.
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    lifts = [c.lift for c in report.failure_clusters]
    assert lifts == sorted(lifts, reverse=True)
    # And our two severities both made the cut.
    assert len(report.failure_clusters) >= 2
    # Severity A (lift ≈ 11) sorts above Severity B (lift ≈ 8).
    top_two = report.failure_clusters[:2]
    assert top_two[0].axes == {"lighting_intensity": 2.0, "camera_offset_x": 2.0}
    assert top_two[1].axes == {"lighting_intensity": 3.0, "camera_offset_x": 3.0}


# 10. 2D heatmap shape + NaN handling.
def test_heatmap_2d_shape_and_nan_for_empty_cells() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"a": 0.0, "b": 0.0}),
        _ep(cell_index=0, episode_index=1, success=False, config={"a": 0.0, "b": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"a": 1.0, "b": 1.0}),
        # Note: (a=0.0, b=1.0) and (a=1.0, b=0.0) are intentionally absent.
    ]
    # Build via raw axis names to avoid the AXIS_NAMES validation;
    # synthetic test axes 'a' and 'b' are fine because Episode itself
    # imposes no axis-name vocabulary.
    report = build_report(eps)
    assert "a__b" in report.heatmap_2d
    h = report.heatmap_2d["a__b"]
    assert isinstance(h, Heatmap2D)
    assert h.axis_x == "a"
    assert h.axis_y == "b"
    assert h.x_values == [0.0, 1.0]
    assert h.y_values == [0.0, 1.0]
    # success_rate[y][x]:
    #   [b=0.0][a=0.0] = 1/2 = 0.5
    #   [b=0.0][a=1.0] = NaN (no episodes)
    #   [b=1.0][a=0.0] = NaN (no episodes)
    #   [b=1.0][a=1.0] = 1.0
    assert h.success_rate[0][0] == 0.5
    assert math.isnan(h.success_rate[0][1])
    assert math.isnan(h.success_rate[1][0])
    assert h.success_rate[1][1] == 1.0


# 11. Single-axis suite → empty heatmap_2d.
def test_heatmap_empty_for_single_axis_suite() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.3}),
        _ep(cell_index=1, episode_index=0, success=False, config={"lighting_intensity": 1.5}),
    ]
    report = build_report(eps)
    assert report.heatmap_2d == {}


# 12. Float normalization.
def test_float_normalization_collapses_one_part_in_1e15() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.3}),
        _ep(
            cell_index=0, episode_index=1, success=False, config={"lighting_intensity": 0.3 + 1e-15}
        ),
    ]
    report = build_report(eps)
    bd = report.per_axis[0]
    # Both episodes land in the same bucket → exactly one key, two episodes.
    assert len(bd.counts) == 1
    assert next(iter(bd.counts.values())) == 2
    assert next(iter(bd.successes.values())) == 1


# 13. cluster_multiple <= 0 raises.
def test_cluster_multiple_zero_or_negative_raises() -> None:
    eps = [_ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.5})]
    with pytest.raises(ValueError, match="cluster_multiple"):
        build_report(eps, cluster_multiple=0.0)
    with pytest.raises(ValueError, match="cluster_multiple"):
        build_report(eps, cluster_multiple=-1.0)


# 14. Determinism — equal Reports across two calls.
def test_build_report_is_deterministic() -> None:
    # Use a fully-populated heatmap grid so no NaN ruins equality.
    eps = _episodes_from_grid(
        axis_a="lighting_intensity",
        a_values=[0.0, 1.0, 2.0],
        axis_b="camera_offset_x",
        b_values=[-0.05, 0.0, 0.05],
        successes_per_cell=2,
        failures_per_cell=1,
    )
    a = build_report(eps)
    b = build_report(eps)
    assert a == b


# 15. Round-trip: model_dump → model_validate yields an equal model.
def test_round_trip_model_dump_validate() -> None:
    eps = _episodes_from_grid(
        axis_a="lighting_intensity",
        a_values=[0.0, 1.0, 2.0],
        axis_b="camera_offset_x",
        b_values=[-0.05, 0.0, 0.05],
        successes_per_cell=2,
        failures_per_cell=1,
    )
    report = build_report(eps)
    restored = Report.model_validate(report.model_dump())
    assert restored == report


# Bonus regression checks — keep schema invariants honest.
def test_extra_fields_forbidden_on_every_model() -> None:
    """ConfigDict(extra='forbid') should be set on every model in the package."""
    for model_cls in (Report, AxisBreakdown, CellBreakdown, FailureCluster, Heatmap2D):
        with pytest.raises(ValidationError):
            model_cls.model_validate({"__bogus_field__": 1})
