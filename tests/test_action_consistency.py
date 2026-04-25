"""B-18 action mode-collapse metric tests.

Covers the SamplablePolicy Protocol, the RandomPolicy.act_n
state-preserving sampler, the worker's variance accumulator, the
Episode.action_variance schema field, the FailureCluster.mean_action_variance
aggregation, and the --measure-action-consistency CLI plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.policy import (
    Observation,
    Policy,
    RandomPolicy,
    ResettablePolicy,
    SamplablePolicy,
    ScriptedPolicy,
)
from gauntlet.report.analyze import build_report
from gauntlet.report.schema import FailureCluster
from gauntlet.runner.episode import Episode

_EMPTY_OBS: Observation = {"state": np.zeros(3, dtype=np.float64)}


# ----------------------------------------------------------------------------
# Protocol / SamplablePolicy detection.
# ----------------------------------------------------------------------------


class TestSamplablePolicyProtocol:
    def test_random_policy_is_samplable(self) -> None:
        p = RandomPolicy(action_dim=4, seed=0)
        assert isinstance(p, SamplablePolicy)
        # Must also still satisfy the existing Protocols — no regression.
        assert isinstance(p, Policy)
        assert isinstance(p, ResettablePolicy)

    def test_scripted_policy_is_not_samplable(self) -> None:
        # Greedy / open-loop policies do not implement act_n by design;
        # the column is None for them. This is the documented B-18
        # asymmetry anti-feature.
        p = ScriptedPolicy()
        assert not isinstance(p, SamplablePolicy)
        # Existing protocols still hold.
        assert isinstance(p, Policy)
        assert isinstance(p, ResettablePolicy)

    def test_protocol_excludes_arbitrary_object(self) -> None:
        class _NotASampler:
            def act(self, obs: Observation) -> np.ndarray:  # pragma: no cover
                return np.zeros(3, dtype=np.float64)

        assert not isinstance(_NotASampler(), SamplablePolicy)


# ----------------------------------------------------------------------------
# RandomPolicy.act_n behaviour.
# ----------------------------------------------------------------------------


class TestRandomPolicyActN:
    def test_returns_n_samples(self) -> None:
        p = RandomPolicy(action_dim=5, seed=0)
        samples = list(p.act_n(_EMPTY_OBS, n=8))
        assert len(samples) == 8
        for a in samples:
            assert a.shape == (5,)
            assert a.dtype == np.float64

    def test_samples_are_independent(self) -> None:
        # Sampleable policies must produce N *different* draws — that is
        # the point of the variance metric. An accidental "[self.act] *
        # n" implementation would pass shape/dtype checks but report
        # zero variance.
        p = RandomPolicy(action_dim=4, seed=42)
        samples = list(p.act_n(_EMPTY_OBS, n=8))
        # No two consecutive draws should be byte-identical.
        for i in range(len(samples) - 1):
            assert not np.array_equal(samples[i], samples[i + 1])

    def test_act_n_does_not_advance_internal_rng(self) -> None:
        # B-18 determinism contract — measurement must NOT shift the
        # rollout's RNG stream, otherwise Episode.seed no longer
        # reproduces the rollout.
        p = RandomPolicy(action_dim=4, seed=123)
        baseline = p.act(_EMPTY_OBS)

        p2 = RandomPolicy(action_dim=4, seed=123)
        # Burn n=8 draws via act_n then call act — must match baseline
        # because act_n snapshots/restores the bit-generator state.
        _ = p2.act_n(_EMPTY_OBS, n=8)
        after_measure = p2.act(_EMPTY_OBS)

        np.testing.assert_array_equal(baseline, after_measure)

    def test_rejects_n_below_one(self) -> None:
        p = RandomPolicy(action_dim=3, seed=0)
        with pytest.raises(ValueError, match=r"n must be >= 1"):
            list(p.act_n(_EMPTY_OBS, n=0))


# ----------------------------------------------------------------------------
# Episode schema field.
# ----------------------------------------------------------------------------


class TestEpisodeActionVarianceField:
    def test_default_is_none_for_backwards_compat(self) -> None:
        # Pre-B-18 Episode dicts that lack action_variance must still
        # validate. The default keeps the contract.
        ep = Episode(
            suite_name="s",
            cell_index=0,
            episode_index=0,
            seed=1,
            perturbation_config={},
            success=True,
            terminated=True,
            truncated=False,
            step_count=1,
            total_reward=0.0,
        )
        assert ep.action_variance is None

    def test_round_trips_through_json(self) -> None:
        ep = Episode(
            suite_name="s",
            cell_index=0,
            episode_index=0,
            seed=1,
            perturbation_config={"a": 0.5, "b": 1.0},
            success=False,
            terminated=False,
            truncated=True,
            step_count=10,
            total_reward=0.0,
            action_variance=0.0123,
        )
        as_json = ep.model_dump_json()
        restored = Episode.model_validate_json(as_json)
        assert restored.action_variance == pytest.approx(0.0123)


# ----------------------------------------------------------------------------
# FailureCluster aggregation in build_report.
# ----------------------------------------------------------------------------


def _make_episode(
    *,
    cell_index: int,
    episode_index: int,
    perturbation: dict[str, float],
    success: bool,
    action_variance: float | None,
) -> Episode:
    return Episode(
        suite_name="s",
        cell_index=cell_index,
        episode_index=episode_index,
        seed=1234 + cell_index * 100 + episode_index,
        perturbation_config=perturbation,
        success=success,
        terminated=True,
        truncated=False,
        step_count=10,
        total_reward=0.0,
        action_variance=action_variance,
    )


class TestFailureClusterAggregation:
    def test_mean_action_variance_averaged_over_reporting_episodes(self) -> None:
        # Construct a 2-axis suite where one (a, b) cell fails 4/4 (a
        # cluster). Variances 0.1, 0.3, 0.2, 0.4 → mean 0.25.
        episodes: list[Episode] = []
        for ep_idx, var in enumerate([0.1, 0.3, 0.2, 0.4]):
            episodes.append(
                _make_episode(
                    cell_index=0,
                    episode_index=ep_idx,
                    perturbation={"a": 1.0, "b": 2.0},
                    success=False,
                    action_variance=var,
                )
            )
        # Add one passing cell on a different (a, b) so the baseline
        # failure rate is non-zero but well below 1.0 — and the
        # cluster's lift clearly exceeds 2x.
        for ep_idx in range(4):
            episodes.append(
                _make_episode(
                    cell_index=1,
                    episode_index=ep_idx,
                    perturbation={"a": 0.0, "b": 0.0},
                    success=True,
                    action_variance=0.05,
                )
            )

        report = build_report(episodes, min_cluster_size=2)
        assert report.failure_clusters, "expected at least one cluster"
        cluster = report.failure_clusters[0]
        assert cluster.mean_action_variance == pytest.approx(0.25)

    def test_mean_action_variance_none_when_no_episode_reports(self) -> None:
        # Same shape but every episode reports action_variance=None
        # (greedy policy or unmeasured run). The cluster mean MUST be
        # None — not 0.0 — to honour the documented anti-feature.
        episodes: list[Episode] = []
        for ep_idx in range(4):
            episodes.append(
                _make_episode(
                    cell_index=0,
                    episode_index=ep_idx,
                    perturbation={"a": 1.0, "b": 2.0},
                    success=False,
                    action_variance=None,
                )
            )
        for ep_idx in range(4):
            episodes.append(
                _make_episode(
                    cell_index=1,
                    episode_index=ep_idx,
                    perturbation={"a": 0.0, "b": 0.0},
                    success=True,
                    action_variance=None,
                )
            )

        report = build_report(episodes, min_cluster_size=2)
        assert report.failure_clusters
        for cluster in report.failure_clusters:
            assert cluster.mean_action_variance is None

    def test_failure_cluster_default_mean_action_variance_is_none(self) -> None:
        # Pre-B-18 report.json files lack the field; default keeps them
        # validatable.
        cluster = FailureCluster(
            axes={"a": 1.0, "b": 2.0},
            n_episodes=4,
            n_success=0,
            failure_rate=1.0,
            lift=4.0,
        )
        assert cluster.mean_action_variance is None


# ----------------------------------------------------------------------------
# CLI flag plumbing — exercises typer registration without running a suite.
# ----------------------------------------------------------------------------


class TestCliFlagPlumbing:
    def test_run_help_advertises_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Widen the rich-rendered help so the long --measure-action-consistency
        # flag does not get word-wrapped across the assertion target.
        monkeypatch.setenv("COLUMNS", "200")
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--measure-action-consistency" in result.output


# ----------------------------------------------------------------------------
# Integration: variance > 0 for Random, 0/None for greedy.
# Exercises the worker code path end-to-end with a tiny suite.
# ----------------------------------------------------------------------------


def _write_suite(tmp_path: Path) -> Path:
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        """
name: b18-test
env: tabletop
seed: 1
episodes_per_cell: 1
axes:
  lighting_intensity:
    low: 0.3
    high: 1.0
    steps: 2
""".strip()
    )
    return suite_yaml


def _read_episodes(out_dir: Path) -> list[dict]:
    payload = json.loads((out_dir / "episodes.json").read_text())
    assert isinstance(payload, list)
    return payload


class TestEndToEndCli:
    def test_random_policy_yields_positive_action_variance(self, tmp_path: Path) -> None:
        suite_yaml = _write_suite(tmp_path)
        out = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_yaml),
                "--policy",
                "random",
                "--out",
                str(out),
                "--no-html",
                "--measure-action-consistency",
                "--env-max-steps",
                "12",
            ],
        )
        assert result.exit_code == 0, result.output
        episodes = _read_episodes(out)
        assert episodes, "expected at least one episode"
        # At least one measured step happened (stride=5, max_steps=12 →
        # steps 0, 5, 10) → variance reported.
        variances = [ep.get("action_variance") for ep in episodes]
        assert any(v is not None and v > 0.0 for v in variances), variances

    def test_scripted_policy_yields_none_variance(self, tmp_path: Path) -> None:
        suite_yaml = _write_suite(tmp_path)
        out = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_yaml),
                "--policy",
                "scripted",
                "--out",
                str(out),
                "--no-html",
                "--measure-action-consistency",
                "--env-max-steps",
                "12",
            ],
        )
        assert result.exit_code == 0, result.output
        episodes = _read_episodes(out)
        assert episodes
        # Greedy policy → action_variance must be None on every
        # episode, never 0.0 (the anti-feature: honest absence beats
        # false zero).
        for ep in episodes:
            assert ep.get("action_variance") is None
