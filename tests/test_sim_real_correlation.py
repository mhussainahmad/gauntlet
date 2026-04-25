"""Tests for the B-28 sim-vs-real correlation report.

Pins the public surface of :mod:`gauntlet.aggregate.sim_real`:

* matched-on-everything pair → correlation 1.0
* perfect anti-correlation → correlation -1.0
* mismatched cells are skipped (counted in unmatched totals)
* JSON round-trip is byte-stable
* ``Episode.source`` defaults to ``None`` (legacy compat)
* CLI subcommand writes ``sim_real_report.json`` and emits the top-N table
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.aggregate import (
    SimRealReport,
    compute_sim_real_correlation,
)
from gauntlet.aggregate.sim_real import pair_episodes
from gauntlet.cli import app
from gauntlet.runner.episode import Episode

SUITE_HASH_A = "a" * 64
SUITE_HASH_B = "b" * 64


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    source: str | None = None,
    suite_hash: str | None = SUITE_HASH_A,
    seed: int = 0,
    suite_name: str = "tiny",
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=5,
        total_reward=1.0 if success else 0.0,
        suite_hash=suite_hash,
        source=source,  # type: ignore[arg-type]
    )


def _write_episodes(directory: Path, episodes: list[Episode]) -> None:
    """Mirror the on-disk shape produced by ``gauntlet run --out``."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = [ep.model_dump(mode="json") for ep in episodes]
    (directory / "episodes.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_paired_dirs(
    tmp_path: Path,
    sim_eps: list[Episode],
    real_eps: list[Episode],
) -> tuple[Path, Path]:
    sim_dir = tmp_path / "sim"
    real_dir = tmp_path / "real"
    _write_episodes(sim_dir / "run-001", sim_eps)
    _write_episodes(real_dir / "run-001", real_eps)
    return sim_dir, real_dir


# ---------------------------------------------------------------------------
# Schema / defaults.
# ---------------------------------------------------------------------------


def test_episode_source_defaults_to_none() -> None:
    """Legacy episodes.json files (no ``source`` key) load cleanly."""
    ep = _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0})
    assert ep.source is None


def test_episode_source_accepts_sim_and_real() -> None:
    sim = _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}, source="sim")
    real = _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}, source="real")
    assert sim.source == "sim"
    assert real.source == "real"


# ---------------------------------------------------------------------------
# Pairing semantics.
# ---------------------------------------------------------------------------


def test_pair_skips_episodes_with_missing_suite_hash() -> None:
    """Match key requires ``suite_hash``; ``None`` on either side → unmatched."""
    sim = [
        _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}),
        _ep(cell_index=0, episode_index=1, success=True, config={"x": 0.0}, suite_hash=None),
    ]
    real = [
        _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}),
        _ep(cell_index=0, episode_index=1, success=True, config={"x": 0.0}, suite_hash=None),
    ]
    pairs, n_unmatched_sim, n_unmatched_real = pair_episodes(sim, real)
    assert len(pairs) == 1
    assert n_unmatched_sim == 1
    assert n_unmatched_real == 1


# ---------------------------------------------------------------------------
# Correlation values.
# ---------------------------------------------------------------------------


def test_identical_sim_real_yields_perfect_correlation(tmp_path: Path) -> None:
    """Identical sim and real (per-cell rates vary) → correlation 1.0."""
    sim_eps: list[Episode] = []
    real_eps: list[Episode] = []
    # 4 cells, 4 episodes each, varying success rate by cell so the
    # axis bucketing has variance to correlate against.
    for cell_idx, x_val in enumerate((0.0, 0.25, 0.5, 0.75)):
        for ep_i in range(4):
            # cell success rate matches the axis value (so x=0.0 → 0/4,
            # x=0.75 → 3/4) — sim and real agree exactly.
            should_succeed = ep_i < round(x_val * 4)
            sim_eps.append(
                _ep(
                    cell_index=cell_idx,
                    episode_index=ep_i,
                    success=should_succeed,
                    config={"x": x_val},
                    source="sim",
                )
            )
            real_eps.append(
                _ep(
                    cell_index=cell_idx,
                    episode_index=ep_i,
                    success=should_succeed,
                    config={"x": x_val},
                    source="real",
                )
            )

    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim_eps, real_eps)
    report = compute_sim_real_correlation(sim_dir, real_dir)

    assert report.n_paired_total == 16
    assert report.n_unmatched_sim == 0
    assert report.n_unmatched_real == 0
    assert math.isclose(report.overall_correlation, 1.0, abs_tol=1e-9)
    axis = report.per_axis["x"]
    assert math.isclose(axis.correlation, 1.0, abs_tol=1e-9)
    assert math.isclose(axis.transferability_score, 1.0, abs_tol=1e-9)
    assert math.isclose(axis.sim_mean, axis.real_mean, abs_tol=1e-9)


def test_anti_correlated_sim_real_yields_negative_one(tmp_path: Path) -> None:
    """Sim succeeds where real fails (and vice versa) → correlation -1.0."""
    sim_eps: list[Episode] = []
    real_eps: list[Episode] = []
    for cell_idx, x_val in enumerate((0.0, 0.25, 0.5, 0.75)):
        for ep_i in range(4):
            sim_should = ep_i < round(x_val * 4)
            # Real is the inverse: high x_val → low real success.
            real_should = ep_i < round((1.0 - x_val) * 4)
            sim_eps.append(
                _ep(
                    cell_index=cell_idx,
                    episode_index=ep_i,
                    success=sim_should,
                    config={"x": x_val},
                )
            )
            real_eps.append(
                _ep(
                    cell_index=cell_idx,
                    episode_index=ep_i,
                    success=real_should,
                    config={"x": x_val},
                )
            )

    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim_eps, real_eps)
    report = compute_sim_real_correlation(sim_dir, real_dir)

    assert math.isclose(report.overall_correlation, -1.0, abs_tol=1e-9)
    axis = report.per_axis["x"]
    assert math.isclose(axis.correlation, -1.0, abs_tol=1e-9)
    # R^2 of a perfect anti-fit is still 1.0.
    assert math.isclose(axis.transferability_score, 1.0, abs_tol=1e-9)


def test_mismatched_cells_are_skipped(tmp_path: Path) -> None:
    """Episodes whose key (suite_hash, cell, ep) doesn't appear on the other side are dropped."""
    sim_eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"x": 1.0}),  # no real partner
    ]
    real_eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0}),
        _ep(cell_index=2, episode_index=0, success=True, config={"x": 2.0}),  # no sim partner
    ]
    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim_eps, real_eps)
    report = compute_sim_real_correlation(sim_dir, real_dir)

    assert report.n_paired_total == 1
    assert report.n_unmatched_sim == 1
    assert report.n_unmatched_real == 1


def test_single_cell_pair_yields_nan_correlation(tmp_path: Path) -> None:
    """n<2 cells → degenerate Pearson; report NaN, don't crash."""
    sim = [_ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0})]
    real = [_ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0})]
    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim, real)
    report = compute_sim_real_correlation(sim_dir, real_dir)
    assert report.n_paired_total == 1
    assert math.isnan(report.overall_correlation)
    assert math.isnan(report.per_axis["x"].correlation)


# ---------------------------------------------------------------------------
# JSON round-trip.
# ---------------------------------------------------------------------------


def test_report_json_roundtrips(tmp_path: Path) -> None:
    sim_eps = [
        _ep(cell_index=c, episode_index=0, success=bool(c % 2), config={"x": float(c)})
        for c in range(4)
    ]
    real_eps = [
        _ep(cell_index=c, episode_index=0, success=bool(c % 2), config={"x": float(c)})
        for c in range(4)
    ]
    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim_eps, real_eps)
    report = compute_sim_real_correlation(sim_dir, real_dir)
    payload = report.model_dump(mode="json")
    restored = SimRealReport.model_validate(payload)
    assert restored.model_dump(mode="json") == payload


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_aggregate_sim_real_writes_report(tmp_path: Path, cli_runner: CliRunner) -> None:
    # Vary success rate by cell so per-cell rates have variance — keeps
    # Pearson finite so the output JSON re-validates without quirking
    # around the NaN-to-null normalisation in ``_write_json``.
    sim_eps = [
        _ep(cell_index=c, episode_index=0, success=bool(c % 2), config={"x": float(c)})
        for c in range(4)
    ]
    real_eps = [
        _ep(cell_index=c, episode_index=0, success=bool(c % 2), config={"x": float(c)})
        for c in range(4)
    ]
    sim_dir, real_dir = _build_paired_dirs(tmp_path, sim_eps, real_eps)
    out_dir = tmp_path / "out"

    result = cli_runner.invoke(
        app,
        ["aggregate-sim-real", str(sim_dir), str(real_dir), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    json_path = out_dir / "sim_real_report.json"
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    restored = SimRealReport.model_validate(payload)
    assert restored.n_paired_total == 4
    # Stderr (output for CliRunner) contains the top-N preamble.
    assert "paired episodes:" in result.output
    assert "top" in result.output and "axes by |correlation|" in result.output


def test_cli_aggregate_sim_real_missing_dir_errors(
    tmp_path: Path,
    cli_runner: CliRunner,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    _write_episodes(
        real_dir / "run-001",
        [_ep(cell_index=0, episode_index=0, success=True, config={"x": 0.0})],
    )
    missing = tmp_path / "does-not-exist"
    out_dir = tmp_path / "out"

    result = cli_runner.invoke(
        app,
        ["aggregate-sim-real", str(missing), str(real_dir), "--out", str(out_dir)],
    )
    assert result.exit_code != 0
    assert "sim directory not found" in result.output
