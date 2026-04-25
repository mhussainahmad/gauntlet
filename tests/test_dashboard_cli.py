"""CLI tests for ``gauntlet dashboard`` — Phase 3 Task 20 / RFC 020 §3.2.

Drives the typer ``app`` end-to-end against a synthetic directory of
``report.json`` files. Reuses the fixture pattern from
:mod:`tests.test_dashboard` (mini reports built directly via
:func:`gauntlet.report.build_report` — no MuJoCo / pybullet) and the
embedded-JSON parse-back trick from
``test_build_dashboard_embeds_data_block`` so the CLI assertions match
what the Python API contract already pins.

The Python API itself is exercised by :mod:`tests.test_dashboard`; this
module pins only the typer surface (subgroup discovery, flags,
artefact emission, error envelopes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.report import Report, build_report
from gauntlet.runner.episode import Episode

# ---------------------------------------------------------------------------
# Shared fixtures — mirror tests/test_dashboard.py.
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
    suite_env: str | None = "tabletop",
    failing_cluster: tuple[float, float] | None = None,
) -> Report:
    """3 x 3 grid x 5 episodes = 45 episodes per report."""
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
    return build_report(eps, suite_env=suite_env)


def _write_run(base: Path, rep: Report, *, run_name: str) -> Path:
    """Mirror the on-disk layout produced by ``gauntlet run --out``."""
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    report_json = run_dir / "report.json"
    report_json.write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report_json


# ---------------------------------------------------------------------------
# `dashboard --help` / subcommand discovery.
# ---------------------------------------------------------------------------


def test_top_level_help_lists_dashboard(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    assert "dashboard" in result.stdout


def test_dashboard_help_lists_build(runner: CliRunner) -> None:
    result = runner.invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0, result.stderr
    assert "build" in result.stdout


def test_dashboard_build_help_shows_options(runner: CliRunner) -> None:
    result = runner.invoke(app, ["dashboard", "build", "--help"])
    assert result.exit_code == 0, result.stderr
    for token in ("RUNS_DIR", "--out", "--title"):
        assert token in result.stdout


# ---------------------------------------------------------------------------
# `dashboard build` — happy path.
# ---------------------------------------------------------------------------


def test_dashboard_build_writes_index_and_assets(runner: CliRunner, tmp_path: Path) -> None:
    """Default invocation drops index.html + dashboard.js + dashboard.css."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(), run_name="run-a")
    _write_run(runs, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    out = tmp_path / "out"

    result = runner.invoke(app, ["dashboard", "build", str(runs), "--out", str(out)])

    assert result.exit_code == 0, result.stderr
    assert (out / "index.html").is_file()
    assert (out / "dashboard.js").is_file()
    assert (out / "dashboard.css").is_file()
    # Stderr summary names the artefact and the run count.
    assert "index.html" in result.stderr
    assert "2 runs" in result.stderr


def test_dashboard_build_index_html_embeds_expected_json(runner: CliRunner, tmp_path: Path) -> None:
    """The embedded ``<script id="dashboard-data">`` literal re-parses cleanly."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(), run_name="run-a")
    _write_run(runs, _grid_report(failing_cluster=(0.3, 0.0)), run_name="run-b")
    out = tmp_path / "out"

    result = runner.invoke(app, ["dashboard", "build", str(runs), "--out", str(out)])

    assert result.exit_code == 0, result.stderr
    body = (out / "index.html").read_text(encoding="utf-8")
    assert '<script id="dashboard-data" type="application/json">' in body

    start = body.index('id="dashboard-data"')
    payload_start = body.index(">", start) + 1
    payload_end = body.index("</script>", payload_start)
    parsed = json.loads(body[payload_start:payload_end])
    assert parsed["summary"]["n_runs"] == 2
    assert parsed["summary"]["n_total_episodes"] == 90
    run_ids = {r["run_id"] for r in parsed["runs"]}
    assert run_ids == {"run-a", "run-b"}


def test_dashboard_build_title_flows_through(runner: CliRunner, tmp_path: Path) -> None:
    """``--title`` is rendered into the page (autoescaped by Jinja)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(), run_name="run-a")
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "dashboard",
            "build",
            str(runs),
            "--out",
            str(out),
            "--title",
            "Nightly Sweep",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = (out / "index.html").read_text(encoding="utf-8")
    assert "Nightly Sweep" in body


def test_dashboard_build_creates_missing_out_dir(runner: CliRunner, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, _grid_report(), run_name="run-a")
    out = tmp_path / "deep" / "nested" / "out"
    assert not out.exists()

    result = runner.invoke(app, ["dashboard", "build", str(runs), "--out", str(out)])

    assert result.exit_code == 0, result.stderr
    assert (out / "index.html").is_file()


# ---------------------------------------------------------------------------
# Error envelopes — clean stderr, non-zero exit code.
# ---------------------------------------------------------------------------


def test_dashboard_build_missing_runs_dir_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    out = tmp_path / "out"
    result = runner.invoke(app, ["dashboard", "build", str(missing), "--out", str(out)])
    assert result.exit_code != 0
    assert "runs-dir not found" in result.stderr


def test_dashboard_build_empty_runs_dir_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    """Empty directory -> builder raises ValueError -> clean CLI error."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    result = runner.invoke(app, ["dashboard", "build", str(empty), "--out", str(out)])
    assert result.exit_code != 0
    # ``build_dashboard`` raises ``no report.json files found under ...``.
    assert "no report.json" in result.stderr
