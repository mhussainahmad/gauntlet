"""CLI tests — see ``GAUNTLET_SPEC.md`` §5 task 9.

Drives the typer ``app`` end-to-end against tiny synthetic suites. The
Runner / Report layers are NOT mocked; the test pipeline is the real
pipeline (just with ``--env-max-steps 5`` and 1-episode cells to keep
each scenario sub-second).

All factory helpers used through ``--policy`` live at module scope so
the dotted import path ``tests.test_cli:_test_random_factory`` resolves
when typer's CliRunner invokes the resolver in-process.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.policy.random import RandomPolicy
from gauntlet.runner.episode import Episode

# ----------------------------------------------------------------------
# CliRunner fixture — typer 0.12+ keeps stdout / stderr separate by
# default, so no extra constructor args are needed to assert on them.
# ----------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Module-level factory referenced by test 7 via ``--policy
# tests.test_cli:_test_random_factory``. Must be top-level so importlib
# can find it.
def _test_random_factory() -> RandomPolicy:
    """Zero-arg policy factory exercised via the module-spec policy form."""
    return RandomPolicy(action_dim=7)


# ----------------------------------------------------------------------
# YAML helpers — keep test suites tiny and fast.
# ----------------------------------------------------------------------


_TINY_SUITE_YAML = textwrap.dedent(
    """\
    name: tiny-test-suite
    env: tabletop
    episodes_per_cell: 1
    seed: 7
    axes:
      lighting_intensity:
        values: [0.5]
    """
)


def _write_tiny_suite(tmp_path: Path) -> Path:
    """Write a 1-cell x 1-episode suite YAML and return its path."""
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_TINY_SUITE_YAML, encoding="utf-8")
    return suite_path


def _write_two_cell_suite(tmp_path: Path) -> Path:
    """Write a 2-cell x 1-episode suite (used for compare tests)."""
    yaml = textwrap.dedent(
        """\
        name: tiny-cmp-suite
        env: tabletop
        episodes_per_cell: 1
        seed: 11
        axes:
          lighting_intensity:
            values: [0.3, 1.5]
        """
    )
    suite_path = tmp_path / "suite_cmp.yaml"
    suite_path.write_text(yaml, encoding="utf-8")
    return suite_path


# ----------------------------------------------------------------------
# Synthetic Episode helpers — used to build episodes.json files for the
# `report` and `compare` tests without touching MuJoCo.
# ----------------------------------------------------------------------


def _ep(
    *,
    suite_name: str,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    seed: int = 0,
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


def _write_episodes_json(path: Path, episodes: list[Episode]) -> None:
    payload: list[dict[str, Any]] = [ep.model_dump(mode="json") for ep in episodes]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# 1 — top-level help.
# ──────────────────────────────────────────────────────────────────────


def test_help_lists_three_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    for cmd in ("run", "report", "compare"):
        assert cmd in result.stdout


# ──────────────────────────────────────────────────────────────────────
# 2 — `run --help` exposes the documented options.
# ──────────────────────────────────────────────────────────────────────


def test_run_help_shows_options(runner: CliRunner) -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, result.stderr
    for token in ("--policy", "--out", "--n-workers", "--seed-override", "--no-html"):
        assert token in result.stdout


# ──────────────────────────────────────────────────────────────────────
# 3 — `run` end-to-end on a tiny synthetic suite.
# ──────────────────────────────────────────────────────────────────────


def test_run_end_to_end_writes_three_artefacts(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "random",
            "--out",
            str(out_dir),
            "-w",
            "1",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr

    episodes_path = out_dir / "episodes.json"
    report_path = out_dir / "report.json"
    html_path = out_dir / "report.html"

    assert episodes_path.is_file()
    assert report_path.is_file()
    assert html_path.is_file()

    # episodes.json: list of 1 dict (1 cell x 1 episode).
    episodes_payload = json.loads(episodes_path.read_text(encoding="utf-8"))
    assert isinstance(episodes_payload, list)
    assert len(episodes_payload) == 1
    assert episodes_payload[0]["suite_name"] == "tiny-test-suite"

    # report.json: dict, parses cleanly (no NaN literals).
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report_payload, dict)
    assert report_payload["suite_name"] == "tiny-test-suite"
    assert report_payload["n_episodes"] == 1

    # HTML: starts with the doctype.
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    # Stderr summary line is present.
    assert "Wrote 1 episodes" in result.stderr
    assert str(out_dir) in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 4 — `--no-html` skips the HTML artefact only.
# ──────────────────────────────────────────────────────────────────────


def test_run_no_html_skips_html_file(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out_nohtml"
    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            "random",
            "-o",
            str(out_dir),
            "--no-html",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert (out_dir / "episodes.json").is_file()
    assert (out_dir / "report.json").is_file()
    assert not (out_dir / "report.html").exists()


# ──────────────────────────────────────────────────────────────────────
# 5 — `--seed-override` propagates into Episode metadata.
# ──────────────────────────────────────────────────────────────────────


def test_run_seed_override_propagates_to_metadata(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out_seed"
    override = 42

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            "random",
            "-o",
            str(out_dir),
            "--seed-override",
            str(override),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr

    episodes_payload = json.loads((out_dir / "episodes.json").read_text(encoding="utf-8"))
    assert episodes_payload[0]["metadata"]["master_seed"] == override


# ──────────────────────────────────────────────────────────────────────
# 6 — Unknown policy spec is a clean exit-1 error.
# ──────────────────────────────────────────────────────────────────────


def test_run_unknown_policy_spec_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out_bad_policy"
    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            "definitely-not-a-policy",
            "-o",
            str(out_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "definitely-not-a-policy" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 7 — `module:attr` policy spec resolves to a module-level factory.
# ──────────────────────────────────────────────────────────────────────


def test_run_module_attr_policy_spec_succeeds(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out_modattr"
    spec = "tests.test_cli:_test_random_factory"
    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            spec,
            "-o",
            str(out_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert (out_dir / "episodes.json").is_file()


# ──────────────────────────────────────────────────────────────────────
# 8 — `report` rebuilds HTML from an episodes.json.
# ──────────────────────────────────────────────────────────────────────


def test_report_from_episodes_json_writes_html(runner: CliRunner, tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.json"
    episodes = [
        _ep(
            suite_name="report-from-eps",
            cell_index=0,
            episode_index=0,
            success=True,
            config={"lighting_intensity": 0.5},
        ),
    ]
    _write_episodes_json(episodes_path, episodes)

    result = runner.invoke(app, ["report", str(episodes_path)])
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ──────────────────────────────────────────────────────────────────────
# 9 — `report` on a pre-built report.json works.
# ──────────────────────────────────────────────────────────────────────


def test_report_from_report_json_writes_html(runner: CliRunner, tmp_path: Path) -> None:
    # Step 1: produce a real report.json via the `run` pipeline.
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out_for_rerender"
    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            "random",
            "-o",
            str(out_dir),
            "--no-html",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    report_json = out_dir / "report.json"
    assert report_json.is_file()

    # Step 2: feed it back into `report`. Output goes next to the input
    # by default; assert it lands where we expect.
    custom_html = tmp_path / "rerendered.html"
    result2 = runner.invoke(app, ["report", str(report_json), "--out", str(custom_html)])
    assert result2.exit_code == 0, result2.stderr
    assert custom_html.is_file()
    assert custom_html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# ──────────────────────────────────────────────────────────────────────
# 10 — `report` auto-detects list vs dict at the top level.
# ──────────────────────────────────────────────────────────────────────


def test_report_auto_detects_list_vs_dict(runner: CliRunner, tmp_path: Path) -> None:
    # List path (episodes.json).
    eps_path = tmp_path / "eps.json"
    _write_episodes_json(
        eps_path,
        [
            _ep(
                suite_name="autodetect",
                cell_index=0,
                episode_index=0,
                success=True,
                config={"lighting_intensity": 0.5},
            ),
        ],
    )
    eps_html = tmp_path / "eps.html"
    r1 = runner.invoke(app, ["report", str(eps_path), "--out", str(eps_html)])
    assert r1.exit_code == 0, r1.stderr
    assert eps_html.is_file()

    # Dict path (build a real report.json via the analyze layer).
    from gauntlet.report import build_report  # local import to avoid top-level coupling

    rep = build_report(
        [
            _ep(
                suite_name="autodetect",
                cell_index=0,
                episode_index=0,
                success=True,
                config={"lighting_intensity": 0.5},
            ),
        ]
    )
    rep_path = tmp_path / "rep.json"
    rep_path.write_text(rep.model_dump_json(indent=2), encoding="utf-8")

    rep_html = tmp_path / "rep.html"
    r2 = runner.invoke(app, ["report", str(rep_path), "--out", str(rep_html)])
    assert r2.exit_code == 0, r2.stderr
    assert rep_html.is_file()


# ──────────────────────────────────────────────────────────────────────
# 11 — `report` on a missing file → exit 1, clean message.
# ──────────────────────────────────────────────────────────────────────


def test_report_missing_file_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    result = runner.invoke(app, ["report", str(missing)])
    assert result.exit_code != 0
    assert "not found" in result.stderr.lower() or "no such" in result.stderr.lower()


# ──────────────────────────────────────────────────────────────────────
# 12 — `compare` writes a structurally-correct compare.json.
# ──────────────────────────────────────────────────────────────────────


def test_compare_writes_compare_json_with_expected_structure(
    runner: CliRunner, tmp_path: Path
) -> None:
    # Run A: 5 episodes, all success, single cell.
    eps_a = [
        _ep(
            suite_name="cmp-suite",
            cell_index=0,
            episode_index=i,
            success=True,
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    # Run B: 5 episodes, all FAIL, same single cell. Delta is -1.0.
    eps_b = [
        _ep(
            suite_name="cmp-suite",
            cell_index=0,
            episode_index=i,
            success=False,
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write_episodes_json(a_path, eps_a)
    _write_episodes_json(b_path, eps_b)
    out_path = tmp_path / "compare.json"

    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_path),
            "--threshold",
            "0.1",
            "--min-cell-size",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert out_path.is_file()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["a"]["name"] == "cmp-suite"
    assert payload["b"]["name"] == "cmp-suite"
    assert payload["a"]["overall_success_rate"] == 1.0
    assert payload["b"]["overall_success_rate"] == 0.0
    assert payload["delta_success_rate"] == pytest.approx(-1.0)
    # The huge regression must show up on the regressions list (negative delta).
    assert len(payload["regressions"]) == 1
    assert payload["regressions"][0]["delta"] < 0
    assert payload["regressions"][0]["axis_combination"] == {"lighting_intensity": 0.5}
    assert payload["improvements"] == []


# ──────────────────────────────────────────────────────────────────────
# 13 — Mismatched suite names: still works, but warns to stderr.
# ──────────────────────────────────────────────────────────────────────


def test_compare_cross_suite_warns_but_succeeds(runner: CliRunner, tmp_path: Path) -> None:
    eps_a = [
        _ep(
            suite_name="suite-A",
            cell_index=0,
            episode_index=i,
            success=True,
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    eps_b = [
        _ep(
            suite_name="suite-B",
            cell_index=0,
            episode_index=i,
            success=True,
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write_episodes_json(a_path, eps_a)
    _write_episodes_json(b_path, eps_b)
    out_path = tmp_path / "compare.json"

    result = runner.invoke(app, ["compare", str(a_path), str(b_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.stderr
    assert "warning" in result.stderr.lower()
    assert "suite-A" in result.stderr
    assert "suite-B" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# 14 — `--threshold` filters out small differences.
# ──────────────────────────────────────────────────────────────────────


def test_compare_threshold_filters_small_deltas(runner: CliRunner, tmp_path: Path) -> None:
    # Build A: 4/5 success at lighting=0.5  (rate=0.8).
    eps_a = [
        _ep(
            suite_name="thr-suite",
            cell_index=0,
            episode_index=i,
            success=(i < 4),
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    # Build B: 3/5 success at the same cell      (rate=0.6, delta=-0.2).
    eps_b = [
        _ep(
            suite_name="thr-suite",
            cell_index=0,
            episode_index=i,
            success=(i < 3),
            config={"lighting_intensity": 0.5},
        )
        for i in range(5)
    ]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    _write_episodes_json(a_path, eps_a)
    _write_episodes_json(b_path, eps_b)

    # threshold=0.1 → -0.2 IS a regression.
    out_low = tmp_path / "cmp_low.json"
    r_low = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_low),
            "--threshold",
            "0.1",
            "--min-cell-size",
            "5",
        ],
    )
    assert r_low.exit_code == 0, r_low.stderr
    payload_low = json.loads(out_low.read_text(encoding="utf-8"))
    assert len(payload_low["regressions"]) == 1

    # threshold=0.5 → -0.2 is too small, regressions list is empty.
    out_high = tmp_path / "cmp_high.json"
    r_high = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out_high),
            "--threshold",
            "0.5",
            "--min-cell-size",
            "5",
        ],
    )
    assert r_high.exit_code == 0, r_high.stderr
    payload_high = json.loads(out_high.read_text(encoding="utf-8"))
    assert payload_high["regressions"] == []


# ──────────────────────────────────────────────────────────────────────
# 15 — Phase 2 Task 5: `compare` cross-backend guard (RFC-005 §11.3 / §12 Q2).
# ──────────────────────────────────────────────────────────────────────


def _write_report_json_with_env(
    path: Path,
    *,
    suite_name: str,
    suite_env: str | None,
) -> None:
    """Write a minimal report.json with a specific ``suite_env`` field.

    The compare CLI accepts either a report dict or an episode list;
    we write a report directly so the ``suite_env`` is preserved
    through the load path (building from episodes currently passes
    ``suite_env=None``).
    """
    from gauntlet.report import build_report

    ep = _ep(
        suite_name=suite_name,
        cell_index=0,
        episode_index=0,
        success=True,
        config={"lighting_intensity": 0.5},
    )
    report = build_report([ep], suite_env=suite_env)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def test_compare_rejects_cross_backend_without_opt_in(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Cross-backend compare without ``--allow-cross-backend`` must exit
    non-zero and mention the drift-vs-regression distinction from §7.4.
    """
    a_path = tmp_path / "report_a.json"
    b_path = tmp_path / "report_b.json"
    _write_report_json_with_env(a_path, suite_name="s", suite_env="tabletop")
    _write_report_json_with_env(
        b_path, suite_name="s", suite_env="tabletop-pybullet"
    )

    result = runner.invoke(
        app,
        ["compare", str(a_path), str(b_path), "--out", str(tmp_path / "c.json")],
    )
    assert result.exit_code != 0
    assert "cross-backend" in result.stderr.lower()
    assert "tabletop-pybullet" in result.stderr
    assert "--allow-cross-backend" in result.stderr


def test_compare_allow_cross_backend_emits_warning_and_succeeds(
    runner: CliRunner, tmp_path: Path
) -> None:
    """With ``--allow-cross-backend`` the command proceeds but prints a
    loud warning about simulator drift.
    """
    a_path = tmp_path / "report_a.json"
    b_path = tmp_path / "report_b.json"
    _write_report_json_with_env(a_path, suite_name="s", suite_env="tabletop")
    _write_report_json_with_env(
        b_path, suite_name="s", suite_env="tabletop-pybullet"
    )
    out = tmp_path / "c.json"

    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out),
            "--allow-cross-backend",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert "warning" in result.stderr.lower()
    assert "cross-backend" in result.stderr.lower()
    assert "tabletop" in result.stderr
    assert out.is_file()


def test_compare_same_backend_no_cross_warning(
    runner: CliRunner, tmp_path: Path
) -> None:
    """When both reports share a suite_env, the cross-backend guard is
    silent.
    """
    a_path = tmp_path / "report_a.json"
    b_path = tmp_path / "report_b.json"
    _write_report_json_with_env(a_path, suite_name="s", suite_env="tabletop")
    _write_report_json_with_env(b_path, suite_name="s", suite_env="tabletop")

    result = runner.invoke(
        app,
        ["compare", str(a_path), str(b_path), "--out", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 0, result.stderr
    assert "cross-backend" not in result.stderr.lower()


def test_compare_legacy_reports_without_suite_env_still_work(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Phase-1 reports have no ``suite_env`` field; they must still
    compare cleanly without triggering the cross-backend guard.
    """
    a_path = tmp_path / "report_a.json"
    b_path = tmp_path / "report_b.json"
    _write_report_json_with_env(a_path, suite_name="s", suite_env=None)
    _write_report_json_with_env(b_path, suite_name="s", suite_env=None)

    result = runner.invoke(
        app,
        ["compare", str(a_path), str(b_path), "--out", str(tmp_path / "c.json")],
    )
    assert result.exit_code == 0, result.stderr
    assert "cross-backend" not in result.stderr.lower()
