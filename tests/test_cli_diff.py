"""CLI tests for ``gauntlet diff`` — see ``polish/gauntlet-diff``.

The tests drive the real typer ``app`` via :class:`typer.testing.CliRunner`
against synthetic ``report.json`` fixtures written into ``tmp_path``.
No Runner / MuJoCo dependency: the differ is pure-Python and the CLI
auto-detects ``report.json`` by top-level dict shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.report import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Report,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────────────────────────────
# Synthetic report fixtures.
# ──────────────────────────────────────────────────────────────────────


def _axis(name: str, rates: dict[float, float]) -> AxisBreakdown:
    counts = dict.fromkeys(rates, 10)
    successes = {value: round(rate * 10) for value, rate in rates.items()}
    return AxisBreakdown(name=name, rates=dict(rates), counts=counts, successes=successes)


def _cell(
    *, cell_index: int, perturbation_config: dict[str, float], success_rate: float
) -> CellBreakdown:
    n_episodes = 10
    return CellBreakdown(
        cell_index=cell_index,
        perturbation_config=dict(perturbation_config),
        n_episodes=n_episodes,
        n_success=round(success_rate * n_episodes),
        success_rate=success_rate,
    )


def _cluster(*, axes: dict[str, float], failure_rate: float, lift: float) -> FailureCluster:
    n_episodes = 5
    return FailureCluster(
        axes=dict(axes),
        n_episodes=n_episodes,
        n_success=round((1.0 - failure_rate) * n_episodes),
        failure_rate=failure_rate,
        lift=lift,
    )


def _report(
    *,
    suite_name: str = "synthetic",
    overall_success_rate: float,
    n_episodes: int = 100,
    per_axis: list[AxisBreakdown] | None = None,
    per_cell: list[CellBreakdown] | None = None,
    failure_clusters: list[FailureCluster] | None = None,
) -> Report:
    return Report(
        suite_name=suite_name,
        n_episodes=n_episodes,
        n_success=round(overall_success_rate * n_episodes),
        per_axis=per_axis if per_axis is not None else [],
        per_cell=per_cell if per_cell is not None else [],
        failure_clusters=failure_clusters if failure_clusters is not None else [],
        heatmap_2d={},
        overall_success_rate=overall_success_rate,
        overall_failure_rate=1.0 - overall_success_rate,
        cluster_multiple=2.0,
    )


def _write_report(path: Path, report: Report) -> None:
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def _write_two_reports(tmp_path: Path) -> tuple[Path, Path]:
    """Build a believable two-checkpoint scenario as report.json files."""
    a = _report(
        overall_success_rate=0.8,
        per_axis=[_axis("lighting_intensity", {0.3: 0.95, 1.5: 0.65})],
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config={"lighting_intensity": 0.3},
                success_rate=0.95,
            ),
            _cell(
                cell_index=1,
                perturbation_config={"lighting_intensity": 1.5},
                success_rate=0.65,
            ),
        ],
        failure_clusters=[
            _cluster(
                axes={"lighting_intensity": 1.5, "object_mass": 2.0},
                failure_rate=0.6,
                lift=2.0,
            )
        ],
    )
    b = _report(
        overall_success_rate=0.55,
        per_axis=[_axis("lighting_intensity", {0.3: 0.9, 1.5: 0.20})],
        per_cell=[
            _cell(
                cell_index=0,
                perturbation_config={"lighting_intensity": 0.3},
                success_rate=0.9,
            ),
            _cell(
                cell_index=1,
                perturbation_config={"lighting_intensity": 1.5},
                success_rate=0.20,
            ),
        ],
        failure_clusters=[
            _cluster(
                axes={"lighting_intensity": 1.5, "object_mass": 2.0},
                failure_rate=0.95,
                lift=4.0,
            )
        ],
    )
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write_report(a_path, a)
    _write_report(b_path, b)
    return a_path, b_path


# ──────────────────────────────────────────────────────────────────────
# 1 — `gauntlet --help` lists `diff` alongside `compare`.
# ──────────────────────────────────────────────────────────────────────


def test_top_level_help_lists_diff(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    assert "diff" in result.stdout
    # And `compare` is still there — we did not replace it.
    assert "compare" in result.stdout


def test_diff_help_shows_options(runner: CliRunner) -> None:
    # Force a wide terminal so typer/rich does not line-wrap long flag
    # names (``--cluster-intensify-threshold`` is 30 chars and would
    # overflow the default 80-col CliRunner pty otherwise).
    result = runner.invoke(app, ["diff", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.stderr
    for token in (
        "--json",
        "--cell-flip-threshold",
        "--cluster-intensify-threshold",
    ):
        assert token in result.stdout


# ──────────────────────────────────────────────────────────────────────
# 2 — Default text rendering on stdout, summary on stderr.
# ──────────────────────────────────────────────────────────────────────


def test_diff_text_output_to_stdout_with_summary_on_stderr(
    runner: CliRunner, tmp_path: Path
) -> None:
    a_path, b_path = _write_two_reports(tmp_path)

    result = runner.invoke(app, ["diff", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.stderr

    # Stdout: text rendering markers.
    assert "--- a:" in result.stdout
    assert "+++ b:" in result.stdout
    assert "@@ overall @@" in result.stdout
    assert "lighting_intensity" in result.stdout
    # The regressed cell is surfaced.
    assert "cell 1" in result.stdout
    assert "regressed" in result.stdout
    # Cluster intensification surfaces (lift went 2.0x -> 4.0x; default
    # threshold is 0.5, delta is 2.0).
    assert "object_mass" in result.stdout

    # Stderr: one-line summary.
    assert "Diffed" in result.stderr
    assert "overall" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 3 — `--json` emits the ReportDiff payload on stdout.
# ──────────────────────────────────────────────────────────────────────


def test_diff_json_output_validates(runner: CliRunner, tmp_path: Path) -> None:
    a_path, b_path = _write_two_reports(tmp_path)
    result = runner.invoke(app, ["diff", str(a_path), str(b_path), "--json"])
    assert result.exit_code == 0, result.stderr

    payload: dict[str, Any] = json.loads(result.stdout)
    assert payload["a_label"] == str(a_path)
    assert payload["b_label"] == str(b_path)
    assert payload["a_suite_name"] == "synthetic"
    assert payload["b_suite_name"] == "synthetic"
    assert isinstance(payload["axis_deltas"], dict)
    assert "lighting_intensity" in payload["axis_deltas"]
    # The regressed cell exceeds the default 0.10 threshold.
    flips = payload["cell_flips"]
    assert any(f["direction"] == "regressed" for f in flips)
    # cluster_intensified populated by the two-report scenario.
    assert payload["cluster_intensified"]


# ──────────────────────────────────────────────────────────────────────
# 4 — Threshold flags actually gate output.
# ──────────────────────────────────────────────────────────────────────


def test_diff_cell_flip_threshold_suppresses_small_changes(
    runner: CliRunner, tmp_path: Path
) -> None:
    a_path, b_path = _write_two_reports(tmp_path)
    # Lift the threshold above the regression we baked in (-0.45) so it
    # disappears.
    result = runner.invoke(
        app,
        ["diff", str(a_path), str(b_path), "--json", "--cell-flip-threshold", "0.99"],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cell_flips"] == []


def test_diff_cluster_intensify_threshold_suppresses_intensification(
    runner: CliRunner, tmp_path: Path
) -> None:
    a_path, b_path = _write_two_reports(tmp_path)
    # Lift the threshold above the +2.0x lift growth.
    result = runner.invoke(
        app,
        [
            "diff",
            str(a_path),
            str(b_path),
            "--json",
            "--cluster-intensify-threshold",
            "5.0",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cluster_intensified"] == []


# ──────────────────────────────────────────────────────────────────────
# 5 — Cross-suite warning + success.
# ──────────────────────────────────────────────────────────────────────


def test_diff_warns_on_cross_suite(runner: CliRunner, tmp_path: Path) -> None:
    a = _report(suite_name="suite-alpha", overall_success_rate=0.8)
    b = _report(suite_name="suite-beta", overall_success_rate=0.7)
    a_path = tmp_path / "alpha.json"
    b_path = tmp_path / "beta.json"
    _write_report(a_path, a)
    _write_report(b_path, b)

    result = runner.invoke(app, ["diff", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.stderr
    assert "warning" in result.stderr.lower()
    assert "suite-alpha" in result.stderr
    assert "suite-beta" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 6 — Missing input file is a clean exit-1.
# ──────────────────────────────────────────────────────────────────────


def test_diff_missing_file_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    a_path = tmp_path / "missing.json"
    b_path = tmp_path / "also_missing.json"
    result = runner.invoke(app, ["diff", str(a_path), str(b_path)])
    assert result.exit_code != 0
    assert "not found" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 7 — episodes.json input is auto-detected (parity with `compare`).
# ──────────────────────────────────────────────────────────────────────


def test_diff_accepts_episodes_json_input(runner: CliRunner, tmp_path: Path) -> None:
    """Both report.json and episodes.json must work, mirroring `compare`."""
    from gauntlet.runner.episode import Episode

    def _ep(*, cell_index: int, success: bool, config: dict[str, float]) -> Episode:
        return Episode(
            suite_name="epi-suite",
            cell_index=cell_index,
            episode_index=0,
            seed=cell_index,
            perturbation_config=dict(config),
            success=success,
            terminated=success,
            truncated=False,
            step_count=5,
            total_reward=1.0 if success else 0.0,
        )

    episodes_a = [
        _ep(cell_index=0, success=True, config={"axis_a": 0.0}),
        _ep(cell_index=1, success=True, config={"axis_a": 1.0}),
    ]
    episodes_b = [
        _ep(cell_index=0, success=True, config={"axis_a": 0.0}),
        _ep(cell_index=1, success=False, config={"axis_a": 1.0}),
    ]
    a_path = tmp_path / "episodes_a.json"
    b_path = tmp_path / "episodes_b.json"
    a_path.write_text(
        json.dumps([ep.model_dump(mode="json") for ep in episodes_a], indent=2),
        encoding="utf-8",
    )
    b_path.write_text(
        json.dumps([ep.model_dump(mode="json") for ep in episodes_b], indent=2),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["diff", str(a_path), str(b_path), "--json"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["a_suite_name"] == "epi-suite"
    # cell 1 went success → failure (delta -1.0, well above the 0.10 default).
    assert any(f["direction"] == "regressed" for f in payload["cell_flips"])
