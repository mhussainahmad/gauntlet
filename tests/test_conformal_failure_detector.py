"""Tests for the B-01 conformal failure-prediction detector.

Covers fit/score, the in-distribution vs OOD asymmetry, the
``alpha`` knob's empirical false-alarm-rate behaviour, JSON
serialisation, the greedy-policy ``None`` propagation, and the
``gauntlet monitor conformal fit`` / ``score`` CLI roundtrip.

Pure numpy + pytest + Typer's CliRunner. No torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.monitor.conformal import ConformalFailureDetector
from gauntlet.runner.episode import Episode


def _make_episode(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    seed: int = 0,
    success: bool = True,
    action_variance: float | None = 0.1,
) -> Episode:
    """Construct a minimal Episode keyed off a single ``action_variance``."""
    return Episode(
        suite_name="test-suite",
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config={},
        success=success,
        terminated=True,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
        action_variance=action_variance,
    )


# ----------------------------------------------------------------------------
# fit() basics.
# ----------------------------------------------------------------------------


class TestFit:
    def test_fit_computes_finite_threshold_above_calibration_max(self) -> None:
        # With method="higher" and rank == n at small n, the threshold is
        # the calibration set's maximum; downstream tests assert this.
        rng = np.random.default_rng(0)
        variances = rng.uniform(0.01, 0.1, size=200).tolist()
        episodes = [
            _make_episode(episode_index=i, action_variance=v) for i, v in enumerate(variances)
        ]

        det = ConformalFailureDetector.fit(episodes, alpha=0.1)

        assert det.n_calibration == 200
        assert det.alpha == 0.1
        # The conformal quantile must lie within the empirical range.
        assert min(variances) <= det.threshold <= max(variances)

    def test_fit_skips_episodes_with_none_action_variance(self) -> None:
        # Greedy-policy episodes are skipped, not error.
        episodes = [
            _make_episode(episode_index=0, action_variance=0.05),
            _make_episode(episode_index=1, action_variance=None),
            _make_episode(episode_index=2, action_variance=0.10),
        ] * 50
        det = ConformalFailureDetector.fit(episodes, alpha=0.1)
        # 2/3 of 150 are usable.
        assert det.n_calibration == 100

    def test_fit_rejects_all_none_calibration_set(self) -> None:
        episodes = [_make_episode(action_variance=None)] * 10
        with pytest.raises(ValueError, match="action_variance"):
            ConformalFailureDetector.fit(episodes, alpha=0.05)

    def test_fit_rejects_alpha_outside_unit_interval(self) -> None:
        episodes = [_make_episode(action_variance=0.05)] * 50
        with pytest.raises(ValueError, match="alpha"):
            ConformalFailureDetector.fit(episodes, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ConformalFailureDetector.fit(episodes, alpha=1.0)

    def test_fit_rejects_calibration_too_small_for_alpha(self) -> None:
        # alpha=0.01 needs n >= 99 for ceil((n+1)*0.99) <= n.
        episodes = [_make_episode(action_variance=v) for v in np.linspace(0, 1, 5)]
        with pytest.raises(ValueError, match="too small"):
            ConformalFailureDetector.fit(episodes, alpha=0.01)


# ----------------------------------------------------------------------------
# score() — in-distribution vs OOD.
# ----------------------------------------------------------------------------


class TestScore:
    def test_in_distribution_episode_does_not_alarm(self) -> None:
        rng = np.random.default_rng(1)
        cal = [
            _make_episode(episode_index=i, action_variance=float(v))
            for i, v in enumerate(rng.uniform(0.05, 0.15, size=200))
        ]
        det = ConformalFailureDetector.fit(cal, alpha=0.1)

        candidate = _make_episode(action_variance=0.05)  # well inside the range
        score, alarm = det.score(candidate)

        assert score is not None
        assert alarm is False
        assert score < 1.0

    def test_ood_episode_triggers_alarm(self) -> None:
        rng = np.random.default_rng(2)
        cal = [
            _make_episode(episode_index=i, action_variance=float(v))
            for i, v in enumerate(rng.uniform(0.05, 0.15, size=200))
        ]
        det = ConformalFailureDetector.fit(cal, alpha=0.1)

        # Way above the calibration max -> guaranteed alarm.
        candidate = _make_episode(action_variance=10.0)
        score, alarm = det.score(candidate)

        assert score is not None
        assert alarm is True
        assert score > 1.0

    def test_score_propagates_none_action_variance(self) -> None:
        cal = [_make_episode(action_variance=0.1)] * 100
        det = ConformalFailureDetector.fit(cal, alpha=0.1)

        score, alarm = det.score(_make_episode(action_variance=None))
        assert score is None
        assert alarm is None


# ----------------------------------------------------------------------------
# alpha controls the false-alarm rate.
# ----------------------------------------------------------------------------


class TestFalseAlarmRate:
    def test_smaller_alpha_yields_fewer_false_alarms(self) -> None:
        # Draw calibration + held-out test from the same distribution;
        # the empirical false-alarm rate on the test set should track
        # alpha (Romano-style finite-sample guarantee).
        rng = np.random.default_rng(7)
        cal_var = rng.normal(loc=0.5, scale=0.1, size=500)
        test_var = rng.normal(loc=0.5, scale=0.1, size=500)
        cal = [
            _make_episode(episode_index=i, action_variance=float(v)) for i, v in enumerate(cal_var)
        ]
        test = [
            _make_episode(episode_index=i, action_variance=float(v)) for i, v in enumerate(test_var)
        ]

        rates = {}
        for alpha in (0.01, 0.1, 0.3):
            det = ConformalFailureDetector.fit(cal, alpha=alpha)
            n_alarm = sum(1 for ep in test if det.score(ep)[1])
            rates[alpha] = n_alarm / len(test)

        # Monotone in alpha (smaller alpha → fewer alarms).
        assert rates[0.01] <= rates[0.1] <= rates[0.3]
        # Each rate is loosely bounded by alpha (allow finite-sample slack).
        assert rates[0.01] <= 0.05
        assert rates[0.1] <= 0.2


# ----------------------------------------------------------------------------
# Persistence.
# ----------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        cal = [_make_episode(episode_index=i, action_variance=0.1 + 0.001 * i) for i in range(100)]
        det = ConformalFailureDetector.fit(cal, alpha=0.1)

        restored = ConformalFailureDetector.from_dict(det.to_dict())

        assert restored.alpha == det.alpha
        assert restored.threshold == det.threshold
        assert restored.n_calibration == det.n_calibration

    def test_save_load_roundtrip_via_filesystem(self, tmp_path: Path) -> None:
        cal = [_make_episode(episode_index=i, action_variance=0.1 + 0.001 * i) for i in range(100)]
        det = ConformalFailureDetector.fit(cal, alpha=0.1)
        out = tmp_path / "detector.json"

        det.save(out)
        restored = ConformalFailureDetector.load(out)

        # Both produce identical scores on the same episode.
        ep = _make_episode(action_variance=0.5)
        assert det.score(ep) == restored.score(ep)
        # And the on-disk JSON is well-formed and human-readable.
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert {"schema_version", "alpha", "threshold", "n_calibration"} <= payload.keys()


# ----------------------------------------------------------------------------
# CLI roundtrip.
# ----------------------------------------------------------------------------


class TestCliRoundtrip:
    def test_fit_then_score_populates_episode_fields(self, tmp_path: Path) -> None:
        runner = CliRunner()

        # Calibration: 100 successful in-distribution episodes.
        rng = np.random.default_rng(3)
        cal_path = tmp_path / "calibration.json"
        cal_episodes = [
            _make_episode(episode_index=i, success=True, action_variance=float(v))
            for i, v in enumerate(rng.uniform(0.05, 0.15, size=100))
        ]
        cal_path.write_text(
            json.dumps([ep.model_dump(mode="json") for ep in cal_episodes]),
            encoding="utf-8",
        )

        # Candidates: one in-distribution, one OOD, one greedy (None).
        cand_path = tmp_path / "candidates.json"
        candidates = [
            _make_episode(episode_index=0, action_variance=0.05),
            _make_episode(episode_index=1, action_variance=10.0),
            _make_episode(episode_index=2, action_variance=None),
        ]
        cand_path.write_text(
            json.dumps([ep.model_dump(mode="json") for ep in candidates]),
            encoding="utf-8",
        )

        det_path = tmp_path / "detector.json"
        scored_path = tmp_path / "scored.json"

        # Fit.
        result = runner.invoke(
            app,
            [
                "monitor",
                "conformal",
                "fit",
                str(cal_path),
                "--out",
                str(det_path),
                "--alpha",
                "0.1",
            ],
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        assert det_path.is_file()

        # Score.
        result = runner.invoke(
            app,
            [
                "monitor",
                "conformal",
                "score",
                str(cand_path),
                "--detector",
                str(det_path),
                "--out",
                str(scored_path),
            ],
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")

        scored = json.loads(scored_path.read_text(encoding="utf-8"))
        # In-distribution episode -> no alarm.
        assert scored[0]["failure_alarm"] is False
        assert scored[0]["failure_score"] is not None and scored[0]["failure_score"] < 1.0
        # OOD episode -> alarm.
        assert scored[1]["failure_alarm"] is True
        assert scored[1]["failure_score"] > 1.0
        # Greedy-policy episode -> both fields stay None.
        assert scored[2]["failure_alarm"] is None
        assert scored[2]["failure_score"] is None
