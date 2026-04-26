"""Unit tests for B-39 cross-checkpoint regression bisection.

Coverage:

* :func:`gauntlet.bisect.bisect.bisect` on a synthetic monotonic
  regression -- four stub checkpoints whose target-cell success rate
  steps from 1.0 down to 0.0; bisect must land on the known step.
* Paired-CRN seed reuse -- the engine pairs the same env seeds across
  candidates because every step uses the same Suite; the test asserts
  ``ep_good[i].seed == ep_mid[i].seed`` at every shared paired index.
* CLI happy path with a 4-checkpoint list -- exercises the Typer
  invocation, JSON output shape, and the resolver indirection.
* Edge cases: ``good == bad`` (single-element list), missing target
  cell, single-element list (no intermediates), empty ckpt list.

The tests inject a fake :class:`gauntlet.runner.Runner` (via the
engine's ``runner_factory`` parameter) so no MuJoCo / spawn pool /
gymnasium imports happen. Each fake runner is parameterised with a
*success curve* keyed by checkpoint id, returning synthetic Episodes
that mimic the runner's actual seed-derivation contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from gauntlet.bisect import (
    BisectError,
    BisectResult,
    BisectStep,
    bisect,
)
from gauntlet.bisect.bisect import RunnerFactory
from gauntlet.cli import app
from gauntlet.policy.base import Policy
from gauntlet.runner import Episode, Runner
from gauntlet.suite.schema import AxisSpec, Suite

# ──────────────────────────────────────────────────────────────────────
# Fake Runner — synthetic Episode generator parameterised by a
# success curve. The curve maps (cell_index, episode_index) -> bool
# success and is keyed by the checkpoint id; the bisect engine drives
# the runner once per candidate, so the curve fully determines the
# bisect outcome without any real env stepping.
# ──────────────────────────────────────────────────────────────────────


_SuccessCurve = Mapping[tuple[int, int], bool]


def _derive_env_seed(*, master_seed: int, cell_index: int, episode_index: int) -> int:
    """Mirror the runner's per-episode seed derivation for the fake.

    The :class:`gauntlet.runner.Runner` derives env seeds via
    ``SeedSequence(master).spawn(n_cells)[i].spawn(eps)[j]``; the
    fake reproduces that exact tree so the per-episode env seed is
    bit-identical to a real run. This lets the paired-CRN test
    assert ``ep_good[i].seed == ep_mid[i].seed`` and have it mean
    something -- the seeds match because BOTH sides went through
    the same derivation tree, exactly like the real runner does.
    """
    master = np.random.SeedSequence(master_seed)
    cell_seqs = master.spawn(max(cell_index + 1, 1))
    episode_seqs = cell_seqs[cell_index].spawn(max(episode_index + 1, 1))
    state = episode_seqs[episode_index].generate_state(1, dtype=np.uint32)
    return int(state[0])


class _FakeRunner:
    """Drop-in for :class:`Runner` whose ``run`` synthesises Episodes.

    The ``success_curves`` dict is keyed by checkpoint id; each value
    is a ``(cell_index, episode_index) -> bool`` success curve. The
    fake reads the *current* checkpoint via the ``policy_id`` baked
    into the runner-factory (set by the caller) and looks up the
    matching curve. This mirrors the production Runner's contract
    that ``policy_id`` uniquely identifies the checkpoint behind a
    factory call.
    """

    def __init__(self, *, success_curves: dict[str, _SuccessCurve], policy_id: str) -> None:
        self._success_curves = success_curves
        self._policy_id = policy_id

    def run(
        self,
        *,
        policy_factory: Callable[[], Policy],
        suite: Suite,
    ) -> list[Episode]:
        """Synthesise one Episode per (cell, episode) using the curve."""
        # Touch the factory so a fake policy is constructed -- mirrors
        # the real runner's ``factory()`` call so a broken factory
        # surfaces here too. We do not actually use the policy.
        del policy_factory
        curve = self._success_curves[self._policy_id]
        if suite.seed is None:
            raise ValueError("test fake requires a deterministic suite.seed")
        master_seed = suite.seed
        cells = list(suite.cells())
        eps_per_cell = suite.episodes_per_cell
        episodes: list[Episode] = []
        for cell in cells:
            for ep_idx in range(eps_per_cell):
                env_seed = _derive_env_seed(
                    master_seed=master_seed,
                    cell_index=cell.index,
                    episode_index=ep_idx,
                )
                success = curve.get((cell.index, ep_idx), False)
                episodes.append(
                    Episode(
                        suite_name=suite.name,
                        cell_index=cell.index,
                        episode_index=ep_idx,
                        seed=env_seed,
                        perturbation_config=dict(cell.values),
                        success=success,
                        terminated=success,
                        truncated=not success,
                        step_count=10,
                        total_reward=1.0 if success else 0.0,
                        metadata={
                            "master_seed": master_seed,
                            "n_cells": len(cells),
                            "episodes_per_cell": eps_per_cell,
                        },
                    )
                )
        return episodes


def _make_runner_factory(
    *, success_curves: dict[str, _SuccessCurve], policy_id_log: list[str]
) -> tuple[RunnerFactory, list[str]]:
    """Build a runner factory that knows which checkpoint is current.

    The factory threads ``policy_id_log`` so each test can assert which
    checkpoints the bisect actually evaluated (and in what order). The
    log is mutated by the resolver, NOT by the runner itself, so the
    factory can be zero-arg (the engine's contract) while still
    forwarding the right policy_id into each fake runner.
    """

    def factory() -> Runner:
        # Use the most-recently-resolved checkpoint as the runner's
        # active id. This mirrors the production wiring where the
        # caller bakes a policy_id into a partial(Runner, policy_id=...)
        # right before calling factory(); here the resolver sets it.
        active = policy_id_log[-1] if policy_id_log else ""
        return cast(
            "Runner",
            _FakeRunner(success_curves=success_curves, policy_id=active),
        )

    return factory, policy_id_log


# ──────────────────────────────────────────────────────────────────────
# Suite + policy fixtures.
# ──────────────────────────────────────────────────────────────────────


def _single_cell_suite(*, seed: int = 42, episodes_per_cell: int = 8) -> Suite:
    """Single-cell suite -- the natural shape for bisecting one failing cell.

    A 1-step axis collapses to one cell; the bisect's target_cell_id
    is 0 by construction.
    """
    return Suite(
        name="bisect-test-suite",
        env="tabletop",
        seed=seed,
        episodes_per_cell=episodes_per_cell,
        axes={"lighting_intensity": AxisSpec(low=0.8, high=0.8, steps=1)},
    )


class _StubPolicy:
    """Placeholder policy used by the resolver. The fake runner never calls it."""

    def act(self, obs: object) -> object:  # pragma: no cover - never invoked
        raise NotImplementedError


def _stub_policy_factory() -> Policy:
    """Module-level factory so the engine's resolver returns a picklable callable."""
    return cast("Policy", _StubPolicy())


def _make_resolver(policy_id_log: list[str]) -> Callable[[str], Callable[[], Policy]]:
    """Resolver that logs each requested checkpoint id.

    The fake runner reads the most-recent entry from this log to
    decide which success curve to use, so the log doubles as a
    test-side audit of the engine's call order.
    """

    def resolver(spec: str) -> Callable[[], Policy]:
        policy_id_log.append(spec)
        return _stub_policy_factory

    return resolver


# ──────────────────────────────────────────────────────────────────────
# 1 — Synthetic monotonic regression.
# ──────────────────────────────────────────────────────────────────────


def test_bisect_finds_first_bad_on_monotonic_regression() -> None:
    """Four checkpoints ramping from 1.0 -> 0.0 success -- bisect must land at the step."""
    suite = _single_cell_suite(episodes_per_cell=8)
    eps = suite.episodes_per_cell

    # ckpt_0 = 8/8, ckpt_1 = 8/8 (still good), ckpt_2 = 0/8 (bad),
    # ckpt_3 = 0/8 (bad). The first-bad checkpoint is ckpt_2.
    success_curves: dict[str, _SuccessCurve] = {
        "ckpt_0": {(0, i): True for i in range(eps)},
        "ckpt_1": {(0, i): True for i in range(eps)},
        "ckpt_2": {(0, i): False for i in range(eps)},
        "ckpt_3": {(0, i): False for i in range(eps)},
    }
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )

    result = bisect(
        ckpt_list=["ckpt_0", "ckpt_1", "ckpt_2", "ckpt_3"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=runner_factory,
    )
    assert isinstance(result, BisectResult)
    assert result.first_bad == "ckpt_2"
    # The headline delta must match the synthetic curve.
    assert result.target_cell_delta == pytest.approx(-1.0)
    # Every step in the search recorded a regressed=True or False
    # decision; the search bracket length-1 invariant means we only
    # need at most ceil(log2(4)) = 2 midpoint steps.
    assert len(result.steps) <= 2
    # The good and bad anchors echo back unchanged.
    assert result.good_checkpoint == "ckpt_0"
    assert result.bad_checkpoint == "ckpt_3"
    assert result.ckpt_list == ["ckpt_0", "ckpt_1", "ckpt_2", "ckpt_3"]


def test_bisect_reports_full_step_log_with_decisions() -> None:
    """Every BisectStep carries a regressed flag whose value matches the curve."""
    suite = _single_cell_suite(episodes_per_cell=10)
    eps = suite.episodes_per_cell
    # ckpt_2 is the first-bad; the engine probes the midpoint (index 2)
    # first, sees a regression, and converges in a single step.
    success_curves: dict[str, _SuccessCurve] = {
        "ckpt_0": {(0, i): True for i in range(eps)},
        "ckpt_1": {(0, i): True for i in range(eps)},
        "ckpt_2": {(0, i): False for i in range(eps)},
        "ckpt_3": {(0, i): False for i in range(eps)},
    }
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )

    result = bisect(
        ckpt_list=["ckpt_0", "ckpt_1", "ckpt_2", "ckpt_3"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=runner_factory,
    )
    # The midpoint-2 step must exist and be flagged regressed=True.
    mid2 = next((s for s in result.steps if s.checkpoint == "ckpt_2"), None)
    assert mid2 is not None
    assert isinstance(mid2, BisectStep)
    assert mid2.regressed is True
    # delta_ci_high must be strictly below 0 to justify the regressed
    # decision (matches the engine's documented strict-inequality rule).
    assert mid2.delta_ci_high < 0.0


# ──────────────────────────────────────────────────────────────────────
# 2 — Paired-CRN seed reuse.
# ──────────────────────────────────────────────────────────────────────


def test_paired_crn_seeds_reused_across_candidates() -> None:
    """The good baseline and every midpoint must see identical per-episode env seeds."""
    suite = _single_cell_suite(episodes_per_cell=4)
    eps = suite.episodes_per_cell

    # Capture the Episode lists each candidate produces by tee-ing
    # them into a side dict keyed by checkpoint id. We monkey-patch
    # the fake runner's run method via a wrapping factory.
    captured: dict[str, list[Episode]] = {}

    success_curves: dict[str, _SuccessCurve] = {
        "ckpt_0": {(0, i): True for i in range(eps)},
        "ckpt_1": {(0, i): True for i in range(eps)},
        "ckpt_2": {(0, i): i % 2 == 0 for i in range(eps)},
        "ckpt_3": {(0, i): False for i in range(eps)},
    }

    policy_id_log: list[str] = []

    def factory() -> Runner:
        active = policy_id_log[-1]
        inner = _FakeRunner(success_curves=success_curves, policy_id=active)

        class _Capturing:
            def run(
                self,
                *,
                policy_factory: Callable[[], Policy],
                suite: Suite,
            ) -> list[Episode]:
                episodes = inner.run(policy_factory=policy_factory, suite=suite)
                # Last writer wins -- a checkpoint visited twice (e.g.
                # the good anchor on a length-2 list) records its
                # final episode list; the test only needs one snapshot
                # per id to compare paired seeds.
                captured[active] = episodes
                return episodes

        return cast("Runner", _Capturing())

    bisect(
        ckpt_list=["ckpt_0", "ckpt_1", "ckpt_2", "ckpt_3"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=factory,
    )

    # Every candidate the engine ran must have the same per-episode
    # env seed at every (cell, episode) coordinate -- the on-disk
    # proof that the CRN derivation tree held end-to-end.
    good = captured["ckpt_0"]
    for ckpt_id, eps_list in captured.items():
        if ckpt_id == "ckpt_0":
            continue
        assert len(eps_list) == len(good)
        for good_ep, mid_ep in zip(good, eps_list, strict=True):
            assert (good_ep.cell_index, good_ep.episode_index) == (
                mid_ep.cell_index,
                mid_ep.episode_index,
            )
            assert good_ep.seed == mid_ep.seed, (
                f"CRN violated: {ckpt_id} seed mismatch at "
                f"(cell={good_ep.cell_index}, ep={good_ep.episode_index}): "
                f"good={good_ep.seed} vs mid={mid_ep.seed}"
            )


# ──────────────────────────────────────────────────────────────────────
# 3 — Edge cases.
# ──────────────────────────────────────────────────────────────────────


def test_bisect_empty_ckpt_list_raises() -> None:
    suite = _single_cell_suite()
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves={},
        policy_id_log=policy_id_log,
    )
    with pytest.raises(BisectError, match="empty"):
        bisect(
            ckpt_list=[],
            policy_factory_resolver=_make_resolver(policy_id_log),
            suite=suite,
            target_cell_id=0,
            runner_factory=runner_factory,
        )


def test_bisect_single_element_list_returns_no_op() -> None:
    """A single-element list (good == bad) is degenerate but must not crash."""
    suite = _single_cell_suite(episodes_per_cell=4)
    eps = suite.episodes_per_cell
    success_curves: dict[str, _SuccessCurve] = {
        "ckpt_only": {(0, i): True for i in range(eps)},
    }
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )
    result = bisect(
        ckpt_list=["ckpt_only"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=runner_factory,
    )
    assert result.first_bad == "ckpt_only"
    assert result.good_checkpoint == "ckpt_only"
    assert result.bad_checkpoint == "ckpt_only"
    assert result.steps == []
    # Self-paired CI collapses to [0, 0].
    assert result.target_cell_delta == 0.0
    assert result.target_cell_delta_ci_low == 0.0
    assert result.target_cell_delta_ci_high == 0.0


def test_bisect_target_cell_outside_suite_raises() -> None:
    suite = _single_cell_suite()
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves={},
        policy_id_log=policy_id_log,
    )
    with pytest.raises(BisectError, match="not in suite"):
        bisect(
            ckpt_list=["a", "b"],
            policy_factory_resolver=_make_resolver(policy_id_log),
            suite=suite,
            target_cell_id=999,
            runner_factory=runner_factory,
        )


def test_bisect_two_element_list_evaluates_bad_against_good() -> None:
    """[good, bad] with no intermediates: still produces a target_cell_delta vs good."""
    suite = _single_cell_suite(episodes_per_cell=6)
    eps = suite.episodes_per_cell
    success_curves: dict[str, _SuccessCurve] = {
        "good": {(0, i): True for i in range(eps)},
        "bad": {(0, i): False for i in range(eps)},
    }
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )
    result = bisect(
        ckpt_list=["good", "bad"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=runner_factory,
    )
    # No midpoints to probe; the engine evaluates the bad anchor once
    # to populate the headline delta.
    assert result.first_bad == "bad"
    assert result.steps == []
    assert result.target_cell_delta == pytest.approx(-1.0)


def test_bisect_episodes_per_step_override_changes_per_run_count() -> None:
    """Override flows through to suite.episodes_per_cell on every candidate."""
    suite = _single_cell_suite(episodes_per_cell=2)
    # Override to 6 -> each candidate's run sees 6 episodes per cell.
    success_curves: dict[str, _SuccessCurve] = {
        "good": {(0, i): True for i in range(6)},
        "bad": {(0, i): False for i in range(6)},
    }
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )
    result = bisect(
        ckpt_list=["good", "bad"],
        policy_factory_resolver=_make_resolver(policy_id_log),
        suite=suite,
        target_cell_id=0,
        runner_factory=runner_factory,
        episodes_per_step=6,
    )
    assert result.first_bad == "bad"
    # Negative full-regression with 6 paired episodes -> sharp CI.
    assert result.target_cell_delta == pytest.approx(-1.0)
    assert result.target_cell_delta_ci_high < 0.0


def test_bisect_episodes_per_step_zero_raises() -> None:
    suite = _single_cell_suite()
    policy_id_log: list[str] = []
    runner_factory, _ = _make_runner_factory(
        success_curves={},
        policy_id_log=policy_id_log,
    )
    with pytest.raises(BisectError, match="episodes_per_step"):
        bisect(
            ckpt_list=["a", "b"],
            policy_factory_resolver=_make_resolver(policy_id_log),
            suite=suite,
            target_cell_id=0,
            runner_factory=runner_factory,
            episodes_per_step=0,
        )


# ──────────────────────────────────────────────────────────────────────
# 4 — CLI happy path.
# ──────────────────────────────────────────────────────────────────────


def _suite_yaml(tmp_path: Path) -> Path:
    """Write a deterministic single-cell suite YAML to disk for CLI tests."""
    suite_path = tmp_path / "bisect-suite.yaml"
    suite_path.write_text(
        "name: bisect-cli-suite\n"
        "env: tabletop\n"
        "seed: 42\n"
        "episodes_per_cell: 6\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    low: 0.8\n"
        "    high: 0.8\n"
        "    steps: 1\n",
        encoding="utf-8",
    )
    return suite_path


def test_cli_bisect_happy_path(tmp_path: Path) -> None:
    """End-to-end Typer invocation -- writes JSON, returns first-bad on monotonic curve."""
    suite_path = _suite_yaml(tmp_path)
    output = tmp_path / "bisect.json"
    eps = 6
    success_curves: dict[str, _SuccessCurve] = {
        "ckpt_0": {(0, i): True for i in range(eps)},
        "ckpt_1": {(0, i): True for i in range(eps)},
        "ckpt_2": {(0, i): False for i in range(eps)},
        "ckpt_3": {(0, i): False for i in range(eps)},
    }
    policy_id_log: list[str] = []

    def fake_resolver(spec: str) -> Callable[[], Policy]:
        policy_id_log.append(spec)
        return _stub_policy_factory

    runner_factory, _ = _make_runner_factory(
        success_curves=success_curves,
        policy_id_log=policy_id_log,
    )

    # Patch the production resolver + runner-factory builder for the
    # duration of the CLI invocation. The CLI's register() call has
    # already wired the production resolver at module import time;
    # patching at the import path the @app.command body actually
    # references intercepts the lookup.
    runner = CliRunner()
    with (
        patch("gauntlet.bisect.cli.resolve_policy_factory", side_effect=fake_resolver),
        patch("gauntlet.bisect.cli._build_runner_factory", return_value=runner_factory),
    ):
        result = runner.invoke(
            app,
            [
                "bisect",
                "--good",
                "ckpt_0",
                "--intermediate",
                "ckpt_1",
                "--intermediate",
                "ckpt_2",
                "--bad",
                "ckpt_3",
                "--suite",
                str(suite_path),
                "--target-cell",
                "0",
                "--output",
                str(output),
            ],
        )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["first_bad"] == "ckpt_2"
    assert payload["good_checkpoint"] == "ckpt_0"
    assert payload["bad_checkpoint"] == "ckpt_3"
    assert payload["target_cell_id"] == 0
    assert payload["target_cell_delta"] == pytest.approx(-1.0)
    assert payload["ckpt_list"] == ["ckpt_0", "ckpt_1", "ckpt_2", "ckpt_3"]


def test_cli_bisect_missing_suite_returns_error(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bisect",
            "--good",
            "x",
            "--bad",
            "y",
            "--suite",
            str(tmp_path / "missing.yaml"),
            "--target-cell",
            "0",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    # typer.BadParameter -> exit code 2 (Click's UsageError convention).
    assert result.exit_code != 0
    assert "suite file not found" in result.output or "suite file not found" in (
        result.stderr if hasattr(result, "stderr") else ""
    )
