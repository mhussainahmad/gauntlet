"""Tests for the Wilson score interval helper (B-03).

Two layers under test:

* :mod:`gauntlet.report.wilson` — the pure CI math, including edge
  cases (n=0, k=0, k=n) and confidence-level validation.
* :func:`gauntlet.report.analyze.build_report` integration — the CI
  fields actually appear on every :class:`CellBreakdown`,
  :class:`AxisBreakdown` bucket, and :class:`FailureCluster` produced
  by the analyse layer.

These tests are intentionally narrow — the math is closed-form and
the tolerances are tight. Reference values were computed by hand from
the closed-form formula (Wilson, 1927) so a regression in the math
itself trips one of the exact-comparison tests.
"""

from __future__ import annotations

import math

import pytest

from gauntlet.report import build_report
from gauntlet.report.wilson import DEFAULT_CONFIDENCE, Z_95, wilson_interval
from gauntlet.runner.episode import Episode

# ────────────────────────────────────────────────────────────────────────
# Pure Wilson math
# ────────────────────────────────────────────────────────────────────────


def test_default_confidence_constant_is_ninety_five_percent() -> None:
    """The B-03 brief pins the default at 95% — the constant must agree."""
    assert DEFAULT_CONFIDENCE == 0.95


def test_z_95_constant_matches_normaldist_quantile() -> None:
    """The hard-coded Z_95 must match ``NormalDist().inv_cdf(0.975)`` exactly.

    If the constants ever drift apart, the cached fast path returns a
    different interval than the explicit-confidence path; that would be
    a silent correctness bug.
    """
    from statistics import NormalDist

    assert NormalDist().inv_cdf(0.975) == pytest.approx(Z_95, abs=1e-12)


def test_n_zero_returns_none_pair() -> None:
    """``n = 0`` is undefined; the helper returns ``(None, None)``."""
    assert wilson_interval(0, 0) == (None, None)


def test_known_value_5_of_10_at_95_percent() -> None:
    """Reference: 5/10 at 95% Wilson → roughly [0.2366, 0.7634].

    Computed by hand from the closed-form formula with z=1.96. Keeps
    the regression budget tight enough to catch a sign flip or a wrong
    z constant.
    """
    lo, hi = wilson_interval(5, 10)
    assert lo is not None
    assert hi is not None
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_all_failures_n_one_low_is_zero_high_is_finite() -> None:
    """k=0, n=1 → low clamps to 0.0, high stays well below 1.0.

    The naive normal approximation would return ``[0, 0]`` here; Wilson
    gives a non-degenerate upper bound that matters when a single all-
    failed cell is the loudest signal in the report.
    """
    lo, hi = wilson_interval(0, 1)
    assert lo == 0.0
    assert hi is not None
    assert 0.0 < hi < 1.0


def test_all_successes_n_one_high_is_one_low_is_finite() -> None:
    """k=1, n=1 → high clamps to 1.0, low stays well above 0.0."""
    lo, hi = wilson_interval(1, 1)
    assert hi == 1.0
    assert lo is not None
    assert 0.0 < lo < 1.0


def test_all_failures_n_ten_low_is_zero_high_below_one() -> None:
    """k=0, n=10 → low clamps to 0.0; high is roughly 0.28."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi is not None
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_all_successes_n_ten_high_is_one_low_above_zero() -> None:
    """k=10, n=10 → high clamps to 1.0; low is roughly 0.72."""
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo is not None
    assert lo == pytest.approx(0.7225, abs=1e-3)


def test_interval_widens_as_n_shrinks() -> None:
    """The CI on 1/2 is much wider than the CI on 100/200 at the same rate."""
    lo_small, hi_small = wilson_interval(1, 2)
    lo_big, hi_big = wilson_interval(100, 200)
    assert lo_small is not None and hi_small is not None
    assert lo_big is not None and hi_big is not None
    width_small = hi_small - lo_small
    width_big = hi_big - lo_big
    assert width_small > width_big
    # Both intervals should bracket the point estimate 0.5.
    assert lo_small < 0.5 < hi_small
    assert lo_big < 0.5 < hi_big


def test_higher_confidence_widens_interval() -> None:
    """99% Wilson at the same (k, n) is strictly wider than 95% Wilson."""
    lo95, hi95 = wilson_interval(5, 10, confidence=0.95)
    lo99, hi99 = wilson_interval(5, 10, confidence=0.99)
    assert lo95 is not None and hi95 is not None
    assert lo99 is not None and hi99 is not None
    assert lo99 < lo95
    assert hi99 > hi95


def test_explicit_confidence_at_95_matches_default_path() -> None:
    """Passing confidence=0.95 explicitly takes the NormalDist branch.

    The result must exactly match the cached ``Z_95`` fast path; if it
    doesn't, the two code paths have diverged and downstream consumers
    would see different brackets depending on how they invoked the
    helper.
    """
    fast = wilson_interval(5, 10)
    explicit = wilson_interval(5, 10, confidence=DEFAULT_CONFIDENCE)
    assert fast == explicit


def test_negative_inputs_raise() -> None:
    """Defensive validation — n < 0 or k < 0 or k > n must raise."""
    with pytest.raises(ValueError, match="n must be >= 0"):
        wilson_interval(0, -1)
    with pytest.raises(ValueError, match="0 <= successes <= n"):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError, match="0 <= successes <= n"):
        wilson_interval(6, 5)


def test_confidence_out_of_range_raises() -> None:
    """confidence must be in (0, 1); 0 / 1 / >1 / negative all raise."""
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(5, 10, confidence=0.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(5, 10, confidence=1.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(5, 10, confidence=1.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_interval(5, 10, confidence=-0.1)


def test_interval_clamped_to_unit_range() -> None:
    """Floating-point round-off must never produce a low<0 or high>1."""
    for k, n in [(0, 1), (1, 1), (0, 1000), (1000, 1000), (3, 5)]:
        lo, hi = wilson_interval(k, n)
        assert lo is not None and hi is not None
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= hi


# ────────────────────────────────────────────────────────────────────────
# Integration with build_report
# ────────────────────────────────────────────────────────────────────────


_SUITE = "wilson-ci-test-suite"


def _ep(*, cell_index: int, episode_index: int, success: bool, config: dict[str, float]) -> Episode:
    return Episode(
        suite_name=_SUITE,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=0,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def test_build_report_attaches_ci_to_every_per_cell_row() -> None:
    """Every :class:`CellBreakdown` must carry ci_low and ci_high."""
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}),
        _ep(cell_index=0, episode_index=1, success=False, config={"x": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"x": 1.0}),
    ]
    report = build_report(eps)
    for cell in report.per_cell:
        assert cell.ci_low is not None
        assert cell.ci_high is not None
        assert 0.0 <= cell.ci_low <= cell.success_rate <= cell.ci_high <= 1.0


def test_build_report_attaches_ci_to_every_axis_bucket() -> None:
    """Every :class:`AxisBreakdown` must carry per-bucket ci_low / ci_high
    in the same key shape as ``rates``.
    """
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"lighting": 0.3}),
        _ep(cell_index=0, episode_index=1, success=True, config={"lighting": 0.3}),
        _ep(cell_index=0, episode_index=2, success=False, config={"lighting": 0.3}),
        _ep(cell_index=0, episode_index=3, success=False, config={"lighting": 0.3}),
        _ep(cell_index=1, episode_index=0, success=True, config={"lighting": 1.5}),
        _ep(cell_index=1, episode_index=1, success=False, config={"lighting": 1.5}),
        _ep(cell_index=1, episode_index=2, success=False, config={"lighting": 1.5}),
        _ep(cell_index=1, episode_index=3, success=False, config={"lighting": 1.5}),
    ]
    report = build_report(eps)
    [bd] = report.per_axis
    assert set(bd.ci_low.keys()) == set(bd.rates.keys())
    assert set(bd.ci_high.keys()) == set(bd.rates.keys())
    for k, rate in bd.rates.items():
        lo = bd.ci_low[k]
        hi = bd.ci_high[k]
        assert lo is not None and hi is not None
        assert 0.0 <= lo <= rate <= hi <= 1.0


def test_build_report_attaches_ci_to_every_failure_cluster() -> None:
    """Every :class:`FailureCluster` must carry ci_low/ci_high on failure_rate."""
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
    for cluster in report.failure_clusters:
        assert cluster.ci_low is not None
        assert cluster.ci_high is not None
        assert 0.0 <= cluster.ci_low <= cluster.failure_rate <= cluster.ci_high <= 1.0


def test_build_report_default_confidence_matches_wilson_helper() -> None:
    """The interval baked into the report at default confidence must
    match the standalone helper exactly — no double-rounding, no
    confidence-level drift between helper and analyse layer.
    """
    eps = [
        _ep(cell_index=0, episode_index=i, success=(i < 5), config={"x": 0.5}) for i in range(10)
    ]
    report = build_report(eps)
    [cell] = report.per_cell
    expected_lo, expected_hi = wilson_interval(5, 10)
    assert cell.ci_low == expected_lo
    assert cell.ci_high == expected_hi


def test_build_report_higher_confidence_widens_per_cell_ci() -> None:
    """``confidence=0.99`` must widen every per-cell CI vs. the 95% default."""
    eps = [
        _ep(cell_index=0, episode_index=i, success=(i % 2 == 0), config={"x": 0.5})
        for i in range(8)
    ]
    r95 = build_report(eps, confidence=0.95)
    r99 = build_report(eps, confidence=0.99)
    for c95, c99 in zip(r95.per_cell, r99.per_cell, strict=True):
        assert c99.ci_low is not None and c99.ci_high is not None
        assert c95.ci_low is not None and c95.ci_high is not None
        assert c99.ci_low <= c95.ci_low
        assert c99.ci_high >= c95.ci_high


def test_build_report_rejects_confidence_out_of_range() -> None:
    """``confidence`` must be in (0, 1) — same contract as the helper."""
    eps = [_ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0})]
    with pytest.raises(ValueError, match="confidence must be in"):
        build_report(eps, confidence=0.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        build_report(eps, confidence=1.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        build_report(eps, confidence=1.1)


def test_old_report_json_without_ci_fields_still_validates() -> None:
    """Backwards-compat: a report.json written before B-03 lacks the CI
    fields entirely. The schema's defaults (None / empty dict) must let
    such a payload round-trip through ``Report.model_validate``.
    """
    from gauntlet.report import Report

    payload = {
        "suite_name": "legacy-suite",
        "n_episodes": 2,
        "n_success": 1,
        "per_axis": [
            {
                "name": "lighting",
                "rates": {"0.3": 1.0, "1.5": 0.0},
                "counts": {"0.3": 1, "1.5": 1},
                "successes": {"0.3": 1, "1.5": 0},
            }
        ],
        "per_cell": [
            {
                "cell_index": 0,
                "perturbation_config": {"lighting": 0.3},
                "n_episodes": 1,
                "n_success": 1,
                "success_rate": 1.0,
            },
            {
                "cell_index": 1,
                "perturbation_config": {"lighting": 1.5},
                "n_episodes": 1,
                "n_success": 0,
                "success_rate": 0.0,
            },
        ],
        "failure_clusters": [],
        "heatmap_2d": {},
        "overall_success_rate": 0.5,
        "overall_failure_rate": 0.5,
        "cluster_multiple": 2.0,
    }
    report = Report.model_validate(payload)
    # Defaults populated.
    for cell in report.per_cell:
        assert cell.ci_low is None
        assert cell.ci_high is None
    [bd] = report.per_axis
    assert bd.ci_low == {}
    assert bd.ci_high == {}


def test_round_trip_dump_validate_preserves_ci_fields() -> None:
    """Round-trip: a freshly built report's CIs survive dump → validate."""
    eps = [_ep(cell_index=0, episode_index=i, success=(i < 3), config={"x": 0.5}) for i in range(5)]
    report = build_report(eps)
    restored = type(report).model_validate(report.model_dump())
    for orig, back in zip(report.per_cell, restored.per_cell, strict=True):
        assert orig.ci_low == back.ci_low
        assert orig.ci_high == back.ci_high


def test_ci_low_high_are_json_finite() -> None:
    """Ensure the rendered JSON never carries a NaN/Inf for a populated bucket.

    The HTML embed walker replaces NaN/Inf with null, but the CI math
    should never produce a non-finite value to begin with — defensive
    invariant covering future refactors.
    """
    eps = [_ep(cell_index=0, episode_index=i, success=(i < 1), config={"x": 0.5}) for i in range(3)]
    report = build_report(eps)
    for cell in report.per_cell:
        assert cell.ci_low is not None and math.isfinite(cell.ci_low)
        assert cell.ci_high is not None and math.isfinite(cell.ci_high)
