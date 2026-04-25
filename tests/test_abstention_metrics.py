"""Tests for B-04 calibration-aware abstention scoring.

Covers the public surface of :mod:`gauntlet.report.abstention`:

* The "B-01 never ran" gate — function returns ``None`` when no
  episode carries ``failure_score``.
* AURC convention sanity — perfect detector trends high, useless
  detector trends near the dataset success rate.
* Abstention-corrected success rate is computed only over the kept
  set, with ``None`` returned when the kept set is empty.
* Correctly / falsely-abstained tallies match a hand-verified case.
* JSON round-trip of the schema field through ``Report.model_validate``.
* HTML render only fires the "Abstention scoring" section when the
  metrics field is populated.

Pure numpy + pytest. No Runner, no torch.
"""

from __future__ import annotations

import json

import numpy as np

from gauntlet.report import (
    AbstentionMetrics,
    Report,
    build_report,
    compute_abstention_metrics,
    render_html,
)
from gauntlet.runner.episode import Episode

_SUITE = "test-suite-abstention"


def _ep(
    *,
    cell_index: int = 0,
    episode_index: int,
    success: bool,
    failure_score: float | None = None,
    failure_alarm: bool | None = None,
    config: dict[str, float] | None = None,
) -> Episode:
    """Episode fixture keyed on the B-01 fields under test."""
    return Episode(
        suite_name=_SUITE,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=episode_index,
        perturbation_config=config if config is not None else {"axis_a": float(cell_index)},
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
        failure_score=failure_score,
        failure_alarm=failure_alarm,
    )


# ─────────────────────────────────────────────────────────────────
# The gate.
# ─────────────────────────────────────────────────────────────────


def test_returns_none_when_no_episode_carries_score() -> None:
    """B-01 never ran → function returns ``None``."""
    eps = [
        _ep(episode_index=i, success=(i % 2 == 0), failure_score=None, failure_alarm=None)
        for i in range(6)
    ]
    assert compute_abstention_metrics(eps) is None


def test_returns_metrics_when_at_least_one_episode_has_score() -> None:
    """Mixed scored/unscored is a valid input — we report on the scored subset."""
    eps = [
        _ep(episode_index=0, success=True, failure_score=0.5, failure_alarm=False),
        _ep(episode_index=1, success=False, failure_score=None, failure_alarm=None),
    ]
    metrics = compute_abstention_metrics(eps)
    assert metrics is not None
    assert metrics.n_with_score == 1


# ─────────────────────────────────────────────────────────────────
# AURC convention.
# ─────────────────────────────────────────────────────────────────


def test_perfect_detector_yields_high_aurc() -> None:
    """Detector score perfectly ranks successes before failures → AURC ≈ 1."""
    # 10 episodes, 50/50 success. Successes get the LOWEST scores so
    # they appear at the most-confident end of the sweep.
    eps: list[Episode] = []
    for i in range(5):
        eps.append(
            _ep(episode_index=i, success=True, failure_score=0.1 + i * 0.01, failure_alarm=False)
        )
    for i in range(5, 10):
        eps.append(
            _ep(episode_index=i, success=False, failure_score=0.9 + i * 0.01, failure_alarm=True)
        )
    metrics = compute_abstention_metrics(eps)
    assert metrics is not None
    # On a 10-episode 50/50 split where the detector ranks all 5
    # successes ahead of all 5 failures, the integral comes out a
    # touch above 0.84 (accuracy is 1.0 over the first half of
    # coverage and decays to 0.5 over the second half). We assert a
    # comfortable margin above the useless-detector baseline (0.5)
    # rather than pin the exact number — the convention check is
    # "perfect ≫ random", not "perfect = 1.0".
    assert metrics.aurc > 0.8, f"perfect detector AURC was {metrics.aurc:.3f}"


def test_useless_detector_yields_aurc_near_baseline() -> None:
    """Random scores uncorrelated with success → AURC ≈ overall success rate."""
    rng = np.random.default_rng(2026)
    n = 200
    successes = rng.uniform(size=n) < 0.5
    scores = rng.uniform(size=n)
    eps = [
        _ep(
            episode_index=i,
            success=bool(successes[i]),
            failure_score=float(scores[i]),
            failure_alarm=False,
        )
        for i in range(n)
    ]
    metrics = compute_abstention_metrics(eps)
    assert metrics is not None
    overall = float(successes.mean())
    # Useless detector's AURC equals the overall accuracy in
    # expectation — we allow a modest band around it.
    assert abs(metrics.aurc - overall) < 0.1, (
        f"useless detector AURC {metrics.aurc:.3f} drifted from baseline {overall:.3f}"
    )


# ─────────────────────────────────────────────────────────────────
# Abstention-corrected success rate + confusion counts.
# ─────────────────────────────────────────────────────────────────


def test_corrected_success_rate_and_confusion_counts() -> None:
    """Hand-verified case: 6 episodes, mixed outcomes and alarms."""
    # Layout (success, alarm):
    #   ep0: succ, no alarm  → kept, success
    #   ep1: succ, no alarm  → kept, success
    #   ep2: fail, alarm     → correctly abstained
    #   ep3: fail, alarm     → correctly abstained
    #   ep4: succ, alarm     → falsely abstained
    #   ep5: fail, no alarm  → kept, failure (missed)
    eps = [
        _ep(episode_index=0, success=True, failure_score=0.1, failure_alarm=False),
        _ep(episode_index=1, success=True, failure_score=0.2, failure_alarm=False),
        _ep(episode_index=2, success=False, failure_score=1.5, failure_alarm=True),
        _ep(episode_index=3, success=False, failure_score=1.7, failure_alarm=True),
        _ep(episode_index=4, success=True, failure_score=1.2, failure_alarm=True),
        _ep(episode_index=5, success=False, failure_score=0.4, failure_alarm=False),
    ]
    metrics = compute_abstention_metrics(eps)
    assert metrics is not None
    assert metrics.n_with_score == 6
    assert metrics.n_correctly_abstained == 2
    assert metrics.n_falsely_abstained == 1
    # Kept set: ep0, ep1, ep5 → 2 / 3 successful.
    assert metrics.abstention_corrected_success_rate is not None
    assert abs(metrics.abstention_corrected_success_rate - (2.0 / 3.0)) < 1e-9


def test_corrected_rate_none_when_every_episode_alarms() -> None:
    """Empty kept set → corrected rate is ``None`` (not a divide-by-zero)."""
    eps = [
        _ep(episode_index=i, success=(i % 2 == 0), failure_score=2.0, failure_alarm=True)
        for i in range(4)
    ]
    metrics = compute_abstention_metrics(eps)
    assert metrics is not None
    assert metrics.abstention_corrected_success_rate is None


# ─────────────────────────────────────────────────────────────────
# Schema round-trip + report integration.
# ─────────────────────────────────────────────────────────────────


def test_report_json_round_trip_preserves_metrics() -> None:
    """``Report.model_validate(report.model_dump_json())`` is byte-stable."""
    eps = [
        _ep(episode_index=0, success=True, failure_score=0.1, failure_alarm=False),
        _ep(episode_index=1, success=False, failure_score=1.5, failure_alarm=True),
        _ep(episode_index=2, success=True, failure_score=0.2, failure_alarm=False),
    ]
    report = build_report(eps)
    assert report.abstention_metrics is not None
    assert report.abstention_metrics.n_with_score == 3
    payload = report.model_dump_json()
    revived = Report.model_validate(json.loads(payload))
    assert revived.abstention_metrics == report.abstention_metrics


def test_build_report_leaves_field_none_without_scores() -> None:
    """Default-path episodes (no B-01 telemetry) → ``abstention_metrics is None``."""
    eps = [
        _ep(episode_index=0, success=True),
        _ep(episode_index=1, success=False),
        _ep(episode_index=2, success=True),
    ]
    report = build_report(eps)
    assert report.abstention_metrics is None


# ─────────────────────────────────────────────────────────────────
# HTML render gating.
# ─────────────────────────────────────────────────────────────────


def test_html_renders_section_only_when_metrics_present() -> None:
    """The "Abstention scoring" section appears only with B-01 telemetry."""
    eps_without = [
        _ep(episode_index=0, success=True),
        _ep(episode_index=1, success=False),
    ]
    html_off = render_html(build_report(eps_without))
    assert "Abstention scoring" not in html_off

    eps_with = [
        _ep(episode_index=0, success=True, failure_score=0.1, failure_alarm=False),
        _ep(episode_index=1, success=False, failure_score=1.5, failure_alarm=True),
    ]
    html_on = render_html(build_report(eps_with))
    assert "Abstention scoring" in html_on
    assert "AURC" in html_on
    assert "Correctly abstained" in html_on


def test_abstention_metrics_is_frozen() -> None:
    """``AbstentionMetrics`` is frozen — accidental mutation must fail."""
    metrics = AbstentionMetrics(
        n_with_score=3,
        aurc=0.7,
        abstention_corrected_success_rate=0.8,
        n_correctly_abstained=1,
        n_falsely_abstained=0,
    )
    import pydantic

    try:
        metrics.aurc = 0.5  # type: ignore[misc]
    except (pydantic.ValidationError, TypeError, AttributeError):
        return
    msg = "expected mutation of frozen model to raise"
    raise AssertionError(msg)
