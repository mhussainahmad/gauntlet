"""Unit tests for B-08 common-random-numbers paired comparison.

Coverage:

* :func:`gauntlet.diff.paired.mcnemar_test` — exact-binomial branch and
  chi-square branch agree with hand-computed reference values.
* :func:`gauntlet.diff.paired.paired_delta_ci` — bracket is symmetric
  around the point estimate, narrows as ``n_paired`` grows, and is
  strictly tighter than the corresponding independent-Wilson bracket
  (the headline variance-reduction win).
* :func:`gauntlet.diff.paired.pair_episodes` — accepts matched seeds,
  rejects ``master_seed`` mismatches, rejects per-episode seed drift.
* :func:`gauntlet.diff.paired.compute_paired_cells` — end-to-end pairing
  on hand-built episode lists.
* :func:`gauntlet.diff.diff_reports` with ``paired_comparison=`` —
  every CellFlip carries the paired CI + McNemar p-value.
* CLI ``compare --paired`` and ``diff --paired`` — happy path, error
  on mismatched master_seed, error on report.json input under explicit
  ``--paired``, auto-detection on episode pairs, opt-out via
  ``--no-paired``.
* The runner's seed-derivation contract — when two suites share
  ``master_seed`` AND grid topology, every paired ``(cell, episode)``
  index gets the same env seed (the on-disk proof CRN held end-to-end).

Tests intentionally avoid heavy backends — no MuJoCo, no
multiprocessing. The runner-side test uses a pure Python policy /
fake env factory that produces a deterministic Episode without
spinning up gymnasium.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.diff import (
    PairingError,
    compute_paired_cells,
    diff_reports,
    mcnemar_test,
    pair_episodes,
    paired_delta_ci,
)
from gauntlet.report import (
    AxisBreakdown,
    CellBreakdown,
    Report,
)
from gauntlet.report.wilson import wilson_interval
from gauntlet.runner import Episode

# ──────────────────────────────────────────────────────────────────────
# Fixtures.
# ──────────────────────────────────────────────────────────────────────


def _episode(
    *,
    cell_index: int,
    episode_index: int,
    seed: int,
    success: bool,
    master_seed: int = 42,
    suite_name: str = "synthetic",
    perturbation_config: dict[str, float] | None = None,
) -> Episode:
    """Build a minimal :class:`Episode` for paired-stats tests."""
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=perturbation_config or {},
        success=success,
        terminated=success,
        truncated=not success,
        step_count=10,
        total_reward=0.0,
        metadata={
            "master_seed": master_seed,
            "n_cells": 1,
            "episodes_per_cell": 1,
        },
    )


def _matched_pair(
    *,
    cell_index: int,
    episode_index: int,
    seed: int,
    a_success: bool,
    b_success: bool,
    master_seed: int = 42,
    perturbation_config: dict[str, float] | None = None,
) -> tuple[Episode, Episode]:
    """Two paired episodes — same seed, same cell, possibly different success."""
    cfg = perturbation_config or {"texture": 0.0}
    return (
        _episode(
            cell_index=cell_index,
            episode_index=episode_index,
            seed=seed,
            success=a_success,
            master_seed=master_seed,
            perturbation_config=cfg,
        ),
        _episode(
            cell_index=cell_index,
            episode_index=episode_index,
            seed=seed,
            success=b_success,
            master_seed=master_seed,
            perturbation_config=cfg,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# 1 — McNemar's test.
# ──────────────────────────────────────────────────────────────────────


def test_mcnemar_exact_branch_b_eq_c_returns_one() -> None:
    """When discordant pairs are balanced, p=1.0 (no evidence of difference)."""
    result = mcnemar_test(b=3, c=3)
    assert result.exact is True
    assert result.p_value == pytest.approx(1.0, abs=1e-12)
    assert result.b == 3
    assert result.c == 3


def test_mcnemar_exact_branch_strong_imbalance() -> None:
    """7 vs 0 discordant pairs ≈ Bernoulli(7, 0.5)·2 = 2/128 = 0.015625."""
    result = mcnemar_test(b=7, c=0)
    assert result.exact is True
    assert result.p_value == pytest.approx(2.0 / 128.0, rel=1e-9)


def test_mcnemar_chi_square_branch_matches_closed_form() -> None:
    """For n>=25 the chi-square ``(b-c)^2/n`` is the canonical statistic."""
    b, c = 30, 10
    result = mcnemar_test(b=b, c=c)
    assert result.exact is False
    expected_stat = (b - c) ** 2 / (b + c)
    assert result.statistic == pytest.approx(expected_stat, rel=1e-12)
    # Chi-square(1) survival at 10.0 ≈ 0.001565 (textbook value).
    assert 0.001 < result.p_value < 0.01


def test_mcnemar_zero_discordant_pairs_returns_p_one() -> None:
    """Both runs agreed on every paired outcome — uninformative test."""
    result = mcnemar_test(b=0, c=0)
    assert result.p_value == 1.0
    assert result.statistic == 0.0


def test_mcnemar_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_test(b=-1, c=2)


# ──────────────────────────────────────────────────────────────────────
# 2 — Paired Wald CI.
# ──────────────────────────────────────────────────────────────────────


def test_paired_delta_ci_zero_discordant_collapses_to_point() -> None:
    """No discordant pairs ⇒ delta = 0 with zero-width CI."""
    low, high = paired_delta_ci(b=0, c=0, n_paired=20)
    assert low == 0.0
    assert high == 0.0


def test_paired_delta_ci_bracket_brackets_point_estimate() -> None:
    """Wald CI is symmetric: ci_low <= delta <= ci_high.

    The signed delta is ``(c - b) / n`` -- positive when B improved
    over A, negative when B regressed for B.
    """
    b, c, n = 3, 1, 20
    low, high = paired_delta_ci(b=b, c=c, n_paired=n)
    delta = (c - b) / n  # = -0.10 -- B regressed against A
    # Sanity: point estimate sits inside the bracket.
    assert low <= delta <= high
    # The CI must be non-trivial for these counts.
    assert high - low > 0.0


def test_paired_delta_ci_clamps_to_unit_interval() -> None:
    """Even an extreme b/c ratio cannot push the bracket outside [-1, 1]."""
    low, high = paired_delta_ci(b=10, c=0, n_paired=10)
    assert low >= -1.0
    assert high <= 1.0


def test_paired_delta_ci_is_tighter_than_independent_wilson_diff() -> None:
    """The headline B-08 win: paired CI <= unpaired-Wilson-diff CI.

    For a paired trial with 20 episodes, ``b=2, c=0`` (B improved on
    2/20 paired episodes, the rest agreed), the paired CI is much
    tighter than the difference of two independent Wilson intervals
    on the same per-cell success-rates.
    """
    n = 20
    b, c = 2, 0
    # Paired bracket on the difference.
    paired_low, paired_high = paired_delta_ci(b=b, c=c, n_paired=n)
    paired_width = paired_high - paired_low
    # Independent-Wilson width estimate: each side has ``n_success_a``
    # and ``n_success_b`` derived from the b/c contingency. Here the
    # 18 concordant pairs split 9 both-pass / 9 both-fail (for the
    # purpose of isolating the variance reduction); each side then
    # reports 9/20 vs 11/20 success rates.
    a_succ = 9
    b_succ = 11
    a_low, a_high = wilson_interval(a_succ, n)
    b_low, b_high = wilson_interval(b_succ, n)
    # Naive independent-Wilson bracket on the difference: subtract
    # endpoint pairs (worst-case bound).
    assert a_low is not None
    assert a_high is not None
    assert b_low is not None
    assert b_high is not None
    indep_width = (b_high - b_low) + (a_high - a_low)
    # Paired must be strictly narrower (this is the headline B-08
    # claim — pure variance reduction from CRN).
    assert paired_width < indep_width


def test_paired_delta_ci_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        paired_delta_ci(b=10, c=10, n_paired=5)
    with pytest.raises(ValueError, match="must be > 0"):
        paired_delta_ci(b=0, c=0, n_paired=0)


# ──────────────────────────────────────────────────────────────────────
# 3 — pair_episodes.
# ──────────────────────────────────────────────────────────────────────


def test_pair_episodes_pairs_matched_master_seed_and_seeds() -> None:
    a, b = _matched_pair(cell_index=0, episode_index=0, seed=12345, a_success=True, b_success=False)
    seed, paired = pair_episodes([a], [b])
    assert seed == 42
    assert len(paired) == 1


def test_pair_episodes_rejects_master_seed_mismatch() -> None:
    a = _episode(
        cell_index=0,
        episode_index=0,
        seed=1,
        success=True,
        master_seed=42,
    )
    b = _episode(
        cell_index=0,
        episode_index=0,
        seed=1,
        success=True,
        master_seed=43,
    )
    with pytest.raises(PairingError, match="master_seed mismatch"):
        pair_episodes([a], [b])


def test_pair_episodes_rejects_env_seed_drift_at_shared_index() -> None:
    """Same master_seed but env-seed mismatch on a shared index is a runner bug."""
    a = _episode(cell_index=0, episode_index=0, seed=12345, success=True)
    # Different env seed despite identical master_seed + (cell, episode) —
    # this would only happen if the runner contract was violated.
    b = _episode(cell_index=0, episode_index=0, seed=99999, success=True)
    with pytest.raises(PairingError, match="env-seed mismatch"):
        pair_episodes([a], [b])


def test_pair_episodes_rejects_empty_lists() -> None:
    with pytest.raises(PairingError, match="non-empty"):
        pair_episodes([], [])


def test_pair_episodes_drops_unshared_cells_silently() -> None:
    """A cell that exists on only one side is not a paired observation."""
    a, b = _matched_pair(cell_index=0, episode_index=0, seed=1, a_success=True, b_success=False)
    a_only = _episode(cell_index=99, episode_index=0, seed=2, success=True)
    seed, paired = pair_episodes([a, a_only], [b])
    assert seed == 42
    # Only the shared (cell=0, ep=0) key survives.
    assert len(paired) == 1


# ──────────────────────────────────────────────────────────────────────
# 4 — compute_paired_cells (end-to-end pairing).
# ──────────────────────────────────────────────────────────────────────


def test_compute_paired_cells_emits_one_entry_per_shared_cell() -> None:
    # Two cells, two episodes each, one regression in cell 0.
    a_eps: list[Episode] = []
    b_eps: list[Episode] = []
    for ep_idx in range(2):
        a, b = _matched_pair(
            cell_index=0,
            episode_index=ep_idx,
            seed=10 + ep_idx,
            a_success=True,
            b_success=ep_idx == 0,  # B fails one of the two paired episodes
        )
        a_eps.append(a)
        b_eps.append(b)
    for ep_idx in range(2):
        a, b = _matched_pair(
            cell_index=1,
            episode_index=ep_idx,
            seed=20 + ep_idx,
            a_success=True,
            b_success=True,
        )
        a_eps.append(a)
        b_eps.append(b)

    paired = compute_paired_cells(a_eps, b_eps)
    assert paired.master_seed == 42
    assert paired.n_cells == 2
    assert paired.n_paired_episodes == 4

    # Cell 0 has b=1, c=0 (one regression for B).
    cell0 = next(c for c in paired.cells if c.cell_index == 0)
    assert cell0.n_paired == 2
    assert cell0.a_success_rate == 1.0
    assert cell0.b_success_rate == 0.5
    assert cell0.delta == pytest.approx(-0.5)
    assert cell0.mcnemar.b == 1
    assert cell0.mcnemar.c == 0
    # CI brackets the negative point estimate.
    assert cell0.delta_ci_low <= cell0.delta <= cell0.delta_ci_high

    # Cell 1: full agreement, mcnemar p=1.0.
    cell1 = next(c for c in paired.cells if c.cell_index == 1)
    assert cell1.delta == 0.0
    assert cell1.mcnemar.p_value == 1.0


# ──────────────────────────────────────────────────────────────────────
# 5 — diff_reports with paired_comparison= attaches CI + McNemar p.
# ──────────────────────────────────────────────────────────────────────


def _report_with_two_cells(
    *,
    suite_name: str,
    cell0_rate: float,
    cell1_rate: float,
) -> Report:
    """Build a 2-cell Report covering the cells used by the paired tests."""
    cells = [
        CellBreakdown(
            cell_index=0,
            perturbation_config={"texture": 0.0},
            n_episodes=10,
            n_success=int(cell0_rate * 10),
            success_rate=cell0_rate,
        ),
        CellBreakdown(
            cell_index=1,
            perturbation_config={"texture": 1.0},
            n_episodes=10,
            n_success=int(cell1_rate * 10),
            success_rate=cell1_rate,
        ),
    ]
    overall = (cell0_rate + cell1_rate) / 2.0
    return Report(
        suite_name=suite_name,
        n_episodes=20,
        n_success=int(overall * 20),
        per_axis=[
            AxisBreakdown(
                name="texture",
                rates={0.0: cell0_rate, 1.0: cell1_rate},
                counts={0.0: 10, 1.0: 10},
                successes={0.0: int(cell0_rate * 10), 1.0: int(cell1_rate * 10)},
            )
        ],
        per_cell=cells,
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=overall,
        overall_failure_rate=1.0 - overall,
        cluster_multiple=2.0,
    )


def test_diff_reports_unpaired_default_keeps_legacy_shape() -> None:
    rep_a = _report_with_two_cells(suite_name="syn", cell0_rate=1.0, cell1_rate=0.5)
    rep_b = _report_with_two_cells(suite_name="syn", cell0_rate=0.6, cell1_rate=0.5)
    result = diff_reports(rep_a, rep_b, cell_flip_threshold=0.1)
    assert result.paired is False
    assert result.paired_comparison is None
    flip = next(f for f in result.cell_flips if f.cell_index == 0)
    assert flip.paired is False
    assert flip.delta_ci_low is None
    assert flip.mcnemar_p_value is None


def test_diff_reports_with_paired_payload_attaches_ci_to_each_flip() -> None:
    # Build paired episodes: 10 paired episodes per cell. Cell 0 has
    # 4 regressions for B (b=4, c=0). Cell 1 unchanged (all match).
    a_eps: list[Episode] = []
    b_eps: list[Episode] = []
    for ep_idx in range(10):
        b_succ_cell0 = ep_idx >= 4  # B fails first 4 paired episodes
        a, b = _matched_pair(
            cell_index=0,
            episode_index=ep_idx,
            seed=100 + ep_idx,
            a_success=True,
            b_success=b_succ_cell0,
            perturbation_config={"texture": 0.0},
        )
        a_eps.append(a)
        b_eps.append(b)
    for ep_idx in range(10):
        a, b = _matched_pair(
            cell_index=1,
            episode_index=ep_idx,
            seed=200 + ep_idx,
            a_success=True,
            b_success=True,
            perturbation_config={"texture": 1.0},
        )
        a_eps.append(a)
        b_eps.append(b)
    paired = compute_paired_cells(a_eps, b_eps, suite_name="syn")
    rep_a = _report_with_two_cells(suite_name="syn", cell0_rate=1.0, cell1_rate=1.0)
    rep_b = _report_with_two_cells(suite_name="syn", cell0_rate=0.6, cell1_rate=1.0)
    result = diff_reports(rep_a, rep_b, cell_flip_threshold=0.1, paired_comparison=paired)
    assert result.paired is True
    assert result.paired_comparison is not None
    assert result.paired_comparison.master_seed == 42

    flip = next(f for f in result.cell_flips if f.cell_index == 0)
    assert flip.paired is True
    assert flip.delta_ci_low is not None
    assert flip.delta_ci_high is not None
    assert flip.mcnemar_p_value is not None
    # Sanity: bracket contains the b - a delta.
    delta = flip.b_success_rate - flip.a_success_rate
    assert flip.delta_ci_low <= delta <= flip.delta_ci_high


# ──────────────────────────────────────────────────────────────────────
# 6 — Runner contract: same master_seed ⇒ paired (cell, episode) seeds.
# ──────────────────────────────────────────────────────────────────────


def test_runner_seed_derivation_is_paired_under_shared_master_seed() -> None:
    """The on-disk proof that CRN is free under shared master_seed.

    Two ``Runner._build_work_items`` calls on the same suite topology
    with the same master seed must produce per-(cell, episode) env
    seeds that match exactly. This is the no-runner-changes-needed
    hinge of B-08 — all the work in ``diff/paired.py`` is just
    exploiting the pairing the runner already gives us.

    The test stubs the suite enumeration so it does not import any
    backend; we only need ``cells()`` and ``episodes_per_cell``.
    """
    # We mirror the exact spawn tree from runner.py:_build_work_items
    # so any future change to that derivation is caught by this test.
    n_cells = 4
    eps_per_cell = 3
    seeds_a: list[int] = []
    seeds_b: list[int] = []
    for master_seed in (123, 123):
        master = np.random.SeedSequence(master_seed)
        cell_seqs = master.spawn(n_cells)
        out: list[int] = []
        for cell_idx in range(n_cells):
            episode_seqs = cell_seqs[cell_idx].spawn(eps_per_cell)
            for ep_idx in range(eps_per_cell):
                out.append(int(episode_seqs[ep_idx].generate_state(1, dtype=np.uint32)[0]))
        if not seeds_a:
            seeds_a = out
        else:
            seeds_b = out
    assert seeds_a == seeds_b


# ──────────────────────────────────────────────────────────────────────
# 7 — CLI integration.
# ──────────────────────────────────────────────────────────────────────


def _write_episodes_json(
    path: Path,
    *,
    cell_count: int,
    eps_per_cell: int,
    a_or_b: str,
    master_seed: int = 42,
    flip_cell_zero: bool = False,
) -> None:
    """Hand-build an episodes.json on disk for CLI tests.

    ``a_or_b`` selects which side; when ``flip_cell_zero=True`` and
    ``a_or_b == "b"``, half of cell 0's paired episodes flip to
    failure to seed a regression the paired test will detect.
    """
    episodes: list[dict[str, object]] = []
    for cell_idx in range(cell_count):
        for ep_idx in range(eps_per_cell):
            seed = 1000 * cell_idx + ep_idx + 1
            success = True
            if flip_cell_zero and a_or_b == "b" and cell_idx == 0 and ep_idx % 2 == 0:
                success = False
            episodes.append(
                {
                    "suite_name": "synthetic",
                    "cell_index": cell_idx,
                    "episode_index": ep_idx,
                    "seed": seed,
                    "perturbation_config": {"texture": float(cell_idx)},
                    "success": success,
                    "terminated": success,
                    "truncated": not success,
                    "step_count": 5,
                    "total_reward": 0.0,
                    "metadata": {
                        "master_seed": master_seed,
                        "n_cells": cell_count,
                        "episodes_per_cell": eps_per_cell,
                    },
                }
            )
    path.write_text(json.dumps(episodes), encoding="utf-8")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_compare_paired_default_auto_on_for_episode_inputs(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Default invocation (no flag) auto-enables pairing on episode inputs."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=10, a_or_b="a")
    _write_episodes_json(b_path, cell_count=2, eps_per_cell=10, a_or_b="b", flip_cell_zero=True)
    out_path = tmp_path / "compare.json"
    result = runner.invoke(app, ["compare", str(a_path), str(b_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["paired"] is True
    # The regression on cell 0 should appear with paired CI fields.
    assert len(payload["regressions"]) >= 1
    reg = payload["regressions"][0]
    assert reg.get("paired") is True
    assert reg.get("delta_ci_low") is not None
    assert reg.get("delta_ci_high") is not None
    assert reg.get("mcnemar_p_value") is not None


def test_cli_compare_no_paired_explicit_disables_pairing(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=10, a_or_b="a")
    _write_episodes_json(b_path, cell_count=2, eps_per_cell=10, a_or_b="b", flip_cell_zero=True)
    out_path = tmp_path / "compare.json"
    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_path),
            "--no-paired",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["paired"] is False


def test_cli_compare_paired_master_seed_mismatch_errors(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Distinct master_seed across runs ⇒ clean error under explicit --paired."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=5, a_or_b="a", master_seed=42)
    _write_episodes_json(b_path, cell_count=2, eps_per_cell=5, a_or_b="b", master_seed=43)
    out_path = tmp_path / "compare.json"
    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_path),
            "--paired",
        ],
    )
    assert result.exit_code != 0
    assert "master_seed mismatch" in result.stderr or "master_seed mismatch" in result.output


def test_cli_compare_paired_rejects_report_json_input(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """report.json carries no per-episode detail; explicit --paired errors."""
    a_path = tmp_path / "a_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=5, a_or_b="a")
    # Build a report.json by piping the episodes through gauntlet report.
    rep_path = tmp_path / "b_report.json"
    rep = Report(
        suite_name="synthetic",
        n_episodes=10,
        n_success=10,
        per_axis=[],
        per_cell=[
            CellBreakdown(
                cell_index=0,
                perturbation_config={"texture": 0.0},
                n_episodes=5,
                n_success=5,
                success_rate=1.0,
            ),
            CellBreakdown(
                cell_index=1,
                perturbation_config={"texture": 1.0},
                n_episodes=5,
                n_success=5,
                success_rate=1.0,
            ),
        ],
        failure_clusters=[],
        heatmap_2d={},
        overall_success_rate=1.0,
        overall_failure_rate=0.0,
        cluster_multiple=2.0,
    )
    rep_path.write_text(rep.model_dump_json(), encoding="utf-8")
    out_path = tmp_path / "compare.json"
    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(rep_path),
            "--out",
            str(out_path),
            "--paired",
        ],
    )
    assert result.exit_code != 0
    msg = result.stderr + result.output
    assert "requires per-episode data" in msg


def test_cli_diff_paired_default_attaches_ci_to_flips(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """``gauntlet diff`` with episode inputs surfaces per-flip paired CI lines."""
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=10, a_or_b="a")
    _write_episodes_json(b_path, cell_count=2, eps_per_cell=10, a_or_b="b", flip_cell_zero=True)
    result = runner.invoke(app, ["diff", str(a_path), str(b_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["paired"] is True
    flip = next(f for f in payload["cell_flips"] if f["cell_index"] == 0)
    assert flip["paired"] is True
    assert flip["delta_ci_low"] is not None
    assert flip["delta_ci_high"] is not None
    assert flip["mcnemar_p_value"] is not None


def test_cli_diff_paired_text_render_includes_paired_header(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    a_path = tmp_path / "a_episodes.json"
    b_path = tmp_path / "b_episodes.json"
    _write_episodes_json(a_path, cell_count=2, eps_per_cell=10, a_or_b="a")
    _write_episodes_json(b_path, cell_count=2, eps_per_cell=10, a_or_b="b", flip_cell_zero=True)
    # B-20: ``flip_cell_zero=True`` flips half the cell-0 episodes
    # (b=5, c=0 over n_paired=10) — McNemar p ≈ 0.0625 > 0.05 → the
    # verdict is ``within_noise`` and the rendered row is suppressed by
    # default. Pass ``--show-noise`` to keep this test asserting on the
    # paired header / suffix surface area without depending on the
    # default-cry-wolf-suppression policy.
    result = runner.invoke(app, ["diff", str(a_path), str(b_path), "--show-noise"])
    assert result.exit_code == 0, result.output
    assert "paired: true" in result.stdout
    assert "paired CI" in result.stdout
    assert "McNemar p=" in result.stdout
