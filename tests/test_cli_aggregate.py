"""CLI tests for ``gauntlet aggregate`` — see Phase 3 Task 19 / RFC 019.

Pin the typer surface and the on-disk artefact contract: the
subcommand reads every ``report.json`` recursively under DIR and
writes ``<out>/fleet_report.json`` (always) and
``<out>/fleet_report.html`` (when ``--html``, the default).

The Runner / suite layers are NOT exercised here — the
``gauntlet.aggregate`` module is the pipeline. CliRunner drives the
real ``app`` against fixtures built directly from
:func:`gauntlet.report.build_report`, mirroring the layout that
``gauntlet run --out <dir>`` produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.aggregate import FleetReport
from gauntlet.cli import app
from gauntlet.report import Report, build_report
from gauntlet.runner.episode import Episode

# ---------------------------------------------------------------------------
# Shared fixtures — build mini reports without touching MuJoCo.
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    seed: int = 0,
    suite_name: str = "tiny-fleet",
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
    )


def _grid_report(
    *,
    suite_name: str = "tiny-fleet",
    failing_cluster: tuple[float, float] | None,
) -> Report:
    """3x3 grid * 5 episodes = 45 episodes; one cell is 100% failure if set."""
    eps: list[Episode] = []
    cell_idx = 0
    for lighting in (0.3, 0.6, 0.9):
        for texture in (0.0, 1.0, 2.0):
            for ep_i in range(5):
                config = {"lighting_intensity": lighting, "object_texture": texture}
                if failing_cluster is not None and (lighting, texture) == failing_cluster:
                    success = False
                else:
                    success = True
                eps.append(
                    _ep(
                        suite_name=suite_name,
                        cell_index=cell_idx,
                        episode_index=ep_i,
                        success=success,
                        config=config,
                        seed=cell_idx * 100 + ep_i,
                    )
                )
            cell_idx += 1
    return build_report(eps)


def _write_run(base: Path, rep: Report, *, run_name: str) -> Path:
    """Mirror the on-disk layout produced by ``gauntlet run --out``."""
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "report.json").write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir / "report.json"


# ---------------------------------------------------------------------------
# Happy-path artefact contract.
# ---------------------------------------------------------------------------


def test_aggregate_writes_fleet_json_and_html(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Default invocation drops both fleet_report.json and fleet_report.html."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-a")
    _write_run(runs, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    out = tmp_path / "out"

    result = runner.invoke(app, ["aggregate", str(runs), "--out", str(out)])

    assert result.exit_code == 0, result.stderr
    fleet_json = out / "fleet_report.json"
    fleet_html = out / "fleet_report.html"
    assert fleet_json.is_file()
    assert fleet_html.is_file()

    payload = json.loads(fleet_json.read_text(encoding="utf-8"))
    fleet = FleetReport.model_validate(payload)
    assert fleet.n_runs == 2
    # ``run-a`` sorts before ``run-b`` so its report.json appears first.
    assert fleet.runs[0].source_file == "run-a/report.json"
    assert fleet.runs[1].source_file == "run-b/report.json"

    body = fleet_html.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>")
    assert "Persistent failure clusters" in body


def test_aggregate_no_html_skips_html(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """``--no-html`` writes JSON only — useful for CI artefacts."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cluster=None), run_name="run-a")
    out = tmp_path / "out"

    result = runner.invoke(app, ["aggregate", str(runs), "--out", str(out), "--no-html"])

    assert result.exit_code == 0, result.stderr
    assert (out / "fleet_report.json").is_file()
    assert not (out / "fleet_report.html").exists()


def test_aggregate_threshold_changes_cluster_count(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """At threshold 0.5, the 2/4-run cluster is in; at 0.51 it's out."""
    runs = tmp_path / "runs"
    runs.mkdir()
    shared = (0.3, 0.0)
    _write_run(runs, _grid_report(failing_cluster=shared), run_name="run-a")
    _write_run(runs, _grid_report(failing_cluster=shared), run_name="run-b")
    _write_run(runs, _grid_report(failing_cluster=None), run_name="run-c")
    _write_run(runs, _grid_report(failing_cluster=None), run_name="run-d")

    out_lo = tmp_path / "out-lo"
    out_hi = tmp_path / "out-hi"

    res_lo = runner.invoke(
        app,
        [
            "aggregate",
            str(runs),
            "--out",
            str(out_lo),
            "--no-html",
            "--persistence-threshold",
            "0.5",
        ],
    )
    res_hi = runner.invoke(
        app,
        [
            "aggregate",
            str(runs),
            "--out",
            str(out_hi),
            "--no-html",
            "--persistence-threshold",
            "0.51",
        ],
    )
    assert res_lo.exit_code == 0, res_lo.stderr
    assert res_hi.exit_code == 0, res_hi.stderr

    fleet_lo = FleetReport.model_validate_json(
        (out_lo / "fleet_report.json").read_text(encoding="utf-8"),
    )
    fleet_hi = FleetReport.model_validate_json(
        (out_hi / "fleet_report.json").read_text(encoding="utf-8"),
    )
    assert len(fleet_lo.persistent_failure_clusters) >= 1
    assert len(fleet_hi.persistent_failure_clusters) == 0
    assert fleet_lo.persistence_threshold == pytest.approx(0.5)
    assert fleet_hi.persistence_threshold == pytest.approx(0.51)


def test_aggregate_creates_missing_out_dir(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cluster=None), run_name="run-a")
    out = tmp_path / "deep" / "nested" / "out"
    assert not out.exists()

    result = runner.invoke(app, ["aggregate", str(runs), "--out", str(out), "--no-html"])

    assert result.exit_code == 0, result.stderr
    assert (out / "fleet_report.json").is_file()


# ---------------------------------------------------------------------------
# Error envelopes — clean stderr, non-zero exit code.
# ---------------------------------------------------------------------------


def test_aggregate_missing_directory_errors_cleanly(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"
    out = tmp_path / "out"
    result = runner.invoke(app, ["aggregate", str(missing), "--out", str(out)])
    assert result.exit_code != 0
    assert "directory not found" in result.stderr


def test_aggregate_empty_directory_errors_cleanly(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    result = runner.invoke(app, ["aggregate", str(empty), "--out", str(out)])
    assert result.exit_code != 0
    assert "no report.json" in result.stderr


def test_aggregate_malformed_report_errors_cleanly(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    bad_dir = runs / "broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "report.json").write_text("{not valid json", encoding="utf-8")
    out = tmp_path / "out"
    result = runner.invoke(app, ["aggregate", str(runs), "--out", str(out)])
    assert result.exit_code != 0
    assert "broken/report.json" in result.stderr


def test_aggregate_threshold_out_of_range_errors_cleanly(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """typer's ``min``/``max`` on the option keeps invalid values out."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(failing_cluster=None), run_name="run-a")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "aggregate",
            str(runs),
            "--out",
            str(out),
            "--persistence-threshold",
            "1.5",
        ],
    )
    assert result.exit_code != 0
