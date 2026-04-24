"""CLI error-path coverage backfill — Phase 2.5 Task 11.

Targets the under-covered error branches of ``gauntlet.cli``'s ``run`` /
``report`` / ``compare`` subcommands. The happy paths and several errors
are already exercised by ``tests/test_cli.py``; this file fills in the
JSON-parsing failure modes, suite-load failure modes, and the small
formatting branches the existing suite did not reach.

All tests run in the default (torch-free) job under ``CliRunner`` and
take well under 5 s each.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from gauntlet.cli import (
    _build_compare,
    _episodes_from_dicts,
    _fmt_signed_pct,
    _load_report_or_episodes,
    _read_json,
    app,
)
from gauntlet.runner.episode import Episode


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _ep(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    success: bool = True,
    config: dict[str, float] | None = None,
    suite_name: str = "err-suite",
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=0,
        perturbation_config=dict(config or {"lighting_intensity": 0.5}),
        success=success,
        terminated=success,
        truncated=False,
        step_count=5,
        total_reward=1.0 if success else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────
# _fmt_signed_pct — zero-delta branch (line 119 / delta.zero style).
# ──────────────────────────────────────────────────────────────────────


def test_fmt_signed_pct_zero_delta_uses_zero_style() -> None:
    """Exact-zero deltas resolve to the dim ``delta.zero`` rich style."""
    rendered = _fmt_signed_pct(0.0)
    assert "delta.zero" in rendered
    assert "+0.0%" in rendered


def test_fmt_signed_pct_positive_delta_uses_up_style() -> None:
    rendered = _fmt_signed_pct(0.05)
    assert "delta.up" in rendered
    assert "+5.0%" in rendered


def test_fmt_signed_pct_negative_delta_uses_down_style() -> None:
    rendered = _fmt_signed_pct(-0.05)
    assert "delta.down" in rendered
    assert "-5.0%" in rendered


# ──────────────────────────────────────────────────────────────────────
# _read_json — JSONDecodeError branch (lines 144-145).
# ──────────────────────────────────────────────────────────────────────


def test_read_json_invalid_json_raises_typer_exit(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{", encoding="utf-8")
    with pytest.raises(typer.Exit):
        _read_json(bad)


def test_read_json_missing_file_raises_typer_exit(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(typer.Exit):
        _read_json(missing)


# ──────────────────────────────────────────────────────────────────────
# _load_report_or_episodes — build_report ValueError + dict / list
# validation paths (lines 160-161, 165-167, 167-170).
# ──────────────────────────────────────────────────────────────────────


def test_load_report_or_episodes_rejects_scalar_top_level(tmp_path: Path) -> None:
    """A top-level scalar (string / int) is neither list nor dict and is
    rejected with the ``must be a list (episodes) or dict (report)`` hint.
    """
    p = tmp_path / "scalar.json"
    p.write_text(json.dumps("just a string"), encoding="utf-8")
    with pytest.raises(typer.Exit):
        _load_report_or_episodes(p)


def test_load_report_or_episodes_invalid_report_dict(tmp_path: Path) -> None:
    """A dict that is not a valid Report payload fails with the
    ``not a valid report.json`` hint."""
    p = tmp_path / "bad_report.json"
    p.write_text(json.dumps({"not": "a report"}), encoding="utf-8")
    with pytest.raises(typer.Exit):
        _load_report_or_episodes(p)


def test_load_report_or_episodes_empty_list_fails_build_report(tmp_path: Path) -> None:
    """An empty list reaches build_report which raises ValueError; the
    CLI translates it to ``cannot build report``."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(typer.Exit):
        _load_report_or_episodes(p)


# ──────────────────────────────────────────────────────────────────────
# _episodes_from_dicts — not-a-dict and validation error branches.
# ──────────────────────────────────────────────────────────────────────


def test_episodes_from_dicts_rejects_non_dict_item(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        _episodes_from_dicts(["string-not-dict"], source=tmp_path / "x.json")


def test_episodes_from_dicts_rejects_invalid_episode(tmp_path: Path) -> None:
    """A dict that does not satisfy Episode's schema raises a clean
    typer.Exit (wrapping pydantic's ValidationError)."""
    with pytest.raises(typer.Exit):
        _episodes_from_dicts([{"foo": "bar"}], source=tmp_path / "x.json")


# ──────────────────────────────────────────────────────────────────────
# `run` failure modes — suite path / YAML / OS errors (313, 317-320).
# ──────────────────────────────────────────────────────────────────────


def test_run_missing_suite_file_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "no-suite.yaml"
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(missing), "-p", "random", "-o", str(out_dir), "--env-max-steps", "5"],
    )
    assert result.exit_code != 0
    assert "suite file not found" in result.stderr


def test_run_invalid_suite_yaml_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    """A YAML file whose top-level is a scalar fails the loader with a
    ``ValueError`` that the CLI rewraps as ``invalid suite YAML``.
    """
    bad = tmp_path / "scalar.yaml"
    bad.write_text("just-a-scalar\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(bad), "-p", "random", "-o", str(out_dir), "--env-max-steps", "5"],
    )
    assert result.exit_code != 0
    assert "invalid suite YAML" in result.stderr


def test_run_pydantic_validation_error_in_suite(runner: CliRunner, tmp_path: Path) -> None:
    """A YAML mapping missing required fields fails Suite.model_validate
    with a ``ValidationError`` that the CLI rewraps as ``invalid suite YAML``.
    """
    bad = tmp_path / "missing_keys.yaml"
    bad.write_text("name: foo\n", encoding="utf-8")  # no env / axes / etc.
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(bad), "-p", "random", "-o", str(out_dir), "--env-max-steps", "5"],
    )
    assert result.exit_code != 0
    assert "invalid suite YAML" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# `report` — auto-detect failure modes (uses _load_report_or_episodes).
# ──────────────────────────────────────────────────────────────────────


def test_report_invalid_json_in_input(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("nonsense{{{", encoding="utf-8")
    result = runner.invoke(app, ["report", str(bad)])
    assert result.exit_code != 0
    assert "invalid JSON" in result.stderr


def test_report_top_level_scalar_rejected(runner: CliRunner, tmp_path: Path) -> None:
    p = tmp_path / "scalar.json"
    p.write_text(json.dumps(42), encoding="utf-8")
    result = runner.invoke(app, ["report", str(p)])
    assert result.exit_code != 0
    assert "must be a list" in result.stderr


def test_report_creates_missing_parent_dir_for_out(runner: CliRunner, tmp_path: Path) -> None:
    """``--out`` pointing at a non-existent subdirectory must mkdir -p
    it (line 399 branch)."""
    eps_path = tmp_path / "eps.json"
    eps_path.write_text(
        json.dumps([_ep(success=True).model_dump(mode="json")]),
        encoding="utf-8",
    )
    nested = tmp_path / "nested" / "deeper" / "out.html"
    result = runner.invoke(app, ["report", str(eps_path), "--out", str(nested)])
    assert result.exit_code == 0, result.stderr
    assert nested.is_file()


# ──────────────────────────────────────────────────────────────────────
# `compare` — improvements branch + parent-dir branch (453, 578).
# ──────────────────────────────────────────────────────────────────────


def test_compare_records_improvements_when_b_better_than_a(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Symmetric to the existing ``regressions`` test — when run B is
    strictly better than A by more than the threshold, the improvements
    list is populated and the regressions list is empty.
    """
    eps_a = [
        _ep(
            cell_index=0,
            episode_index=i,
            success=False,
            config={"lighting_intensity": 0.5},
            suite_name="cmp",
        )
        for i in range(5)
    ]
    eps_b = [
        _ep(
            cell_index=0,
            episode_index=i,
            success=True,
            config={"lighting_intensity": 0.5},
            suite_name="cmp",
        )
        for i in range(5)
    ]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps([e.model_dump(mode="json") for e in eps_a]), encoding="utf-8")
    b_path.write_text(json.dumps([e.model_dump(mode="json") for e in eps_b]), encoding="utf-8")
    out = tmp_path / "improve.json"

    result = runner.invoke(
        app,
        [
            "compare",
            str(a_path),
            str(b_path),
            "--out",
            str(out),
            "--threshold",
            "0.1",
            "--min-cell-size",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["improvements"]) == 1
    assert payload["improvements"][0]["delta"] > 0
    assert payload["regressions"] == []
    assert "improvements: " in result.stderr


def test_compare_creates_missing_parent_dir_for_out(runner: CliRunner, tmp_path: Path) -> None:
    """``--out`` to a non-existent subdirectory must mkdir -p it (line 578)."""
    eps_a = [_ep(suite_name="s") for _ in range(5)]
    eps_b = [_ep(suite_name="s") for _ in range(5)]
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps([e.model_dump(mode="json") for e in eps_a]), encoding="utf-8")
    b_path.write_text(json.dumps([e.model_dump(mode="json") for e in eps_b]), encoding="utf-8")
    nested = tmp_path / "nested" / "deeper" / "compare.json"

    result = runner.invoke(
        app,
        ["compare", str(a_path), str(b_path), "--out", str(nested)],
    )
    assert result.exit_code == 0, result.stderr
    assert nested.is_file()


# ──────────────────────────────────────────────────────────────────────
# _build_compare — direct unit test for skip-by-min-cell-size branch.
# ──────────────────────────────────────────────────────────────────────


def test_build_compare_skips_cells_below_min_size() -> None:
    """A cell with fewer than ``min_cell_size`` episodes on either side
    is excluded from both regressions and improvements regardless of
    its delta magnitude.
    """
    from gauntlet.report import build_report

    eps_a = [_ep(cell_index=0, episode_index=i, success=True, suite_name="s") for i in range(2)]
    eps_b = [_ep(cell_index=0, episode_index=i, success=False, suite_name="s") for i in range(2)]
    rep_a = build_report(eps_a)
    rep_b = build_report(eps_b)

    payload: dict[str, Any] = _build_compare(rep_a, rep_b, threshold=0.1, min_cell_size=5)
    assert payload["regressions"] == []
    assert payload["improvements"] == []


# ──────────────────────────────────────────────────────────────────────
# `run` end-to-end with ``--record-trajectories`` — exercises the
# trajectory_dir branch in execute_one (worker) at the same time.
# ──────────────────────────────────────────────────────────────────────


def test_run_with_record_trajectories_writes_npz(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        textwrap.dedent(
            """\
            name: traj-test
            env: tabletop
            episodes_per_cell: 1
            seed: 5
            axes:
              lighting_intensity:
                values: [0.5]
            """
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    traj_dir = tmp_path / "trajectories"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "-p",
            "random",
            "-o",
            str(out_dir),
            "--record-trajectories",
            str(traj_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    npz_files = list(traj_dir.glob("*.npz"))
    assert len(npz_files) == 1
