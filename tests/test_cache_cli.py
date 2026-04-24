"""CLI tests for the ``--cache-dir`` / ``--no-cache`` / ``--cache-stats`` flags.

Drives the typer ``app`` end-to-end against a tiny synthetic suite so the
flag wiring is exercised against the real Runner -> EpisodeCache path.
``--env-max-steps`` is set throughout to keep each invocation sub-second.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app

# ----------------------------------------------------------------------
# CliRunner fixture — typer 0.12+ keeps stdout / stderr separate by
# default.
# ----------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# Single-axis 2-cell suite so two episodes exercise the cache.
_TINY_SUITE_YAML = textwrap.dedent(
    """\
    name: tiny-cache-suite
    env: tabletop
    episodes_per_cell: 1
    seed: 13
    axes:
      lighting_intensity:
        low: 0.5
        high: 1.0
        steps: 2
    """
)


def _write_tiny_suite(tmp_path: Path) -> Path:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_TINY_SUITE_YAML, encoding="utf-8")
    return suite_path


# ----------------------------------------------------------------------
# 1. --cache-dir on a cold cache -> all misses, all puts.
# ----------------------------------------------------------------------


def test_cli_cache_dir_first_run_is_all_misses(tmp_path: Path, runner: CliRunner) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--no-html",
            "--env-max-steps",
            "5",
            "--cache-dir",
            str(cache_dir),
            "--cache-stats",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # Two cells x 1 episode -> 2 puts, 0 hits, 2 misses on a cold cache.
    assert "cache: hits=0 misses=2 puts=2" in result.stderr
    # Cache directory was populated.
    written = list(cache_dir.rglob("*.json"))
    assert len(written) == 2


# ----------------------------------------------------------------------
# 2. Second invocation against the same cache -> all hits.
# ----------------------------------------------------------------------


def test_cli_cache_dir_second_run_is_all_hits(tmp_path: Path, runner: CliRunner) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    common_args = [
        "run",
        str(suite_path),
        "--policy",
        "scripted",
        "--out",
        str(out_dir),
        "--no-html",
        "--env-max-steps",
        "5",
        "--cache-dir",
        str(cache_dir),
        "--cache-stats",
    ]

    first = runner.invoke(app, common_args)
    assert first.exit_code == 0, first.stderr
    assert "cache: hits=0 misses=2 puts=2" in first.stderr

    # Snapshot first-run episodes.json for byte-identity check below.
    first_episodes = json.loads((out_dir / "episodes.json").read_text(encoding="utf-8"))

    second = runner.invoke(app, common_args)
    assert second.exit_code == 0, second.stderr
    # Cold cache -> hot cache: 2 hits, 0 misses, 0 puts.
    assert "cache: hits=2 misses=0 puts=0" in second.stderr

    # Episodes round-trip identically.
    second_episodes = json.loads((out_dir / "episodes.json").read_text(encoding="utf-8"))
    assert first_episodes == second_episodes


# ----------------------------------------------------------------------
# 3. --no-cache overrides --cache-dir.
# ----------------------------------------------------------------------


def test_cli_no_cache_overrides_cache_dir(tmp_path: Path, runner: CliRunner) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--no-html",
            "--env-max-steps",
            "5",
            "--cache-dir",
            str(cache_dir),
            "--no-cache",
            "--cache-stats",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # --no-cache disables caching entirely -> zeros across the board.
    assert "cache: hits=0 misses=0 puts=0" in result.stderr
    # Cache directory must NOT have been created (no puts happened).
    assert not cache_dir.exists()


# ----------------------------------------------------------------------
# 4. --cache-dir without --env-max-steps must error cleanly.
# ----------------------------------------------------------------------


def test_cli_cache_dir_requires_env_max_steps(tmp_path: Path, runner: CliRunner) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--no-html",
            "--cache-dir",
            str(cache_dir),
            # NO --env-max-steps
        ],
    )
    assert result.exit_code != 0
    assert "--cache-dir requires --env-max-steps" in result.stderr


# ----------------------------------------------------------------------
# 5. Default (no --cache-dir) emits no cache stats line.
# ----------------------------------------------------------------------


def test_cli_default_emits_no_cache_stats_line(tmp_path: Path, runner: CliRunner) -> None:
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "scripted",
            "--out",
            str(out_dir),
            "--no-html",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # Without --cache-stats the summary must not mention the cache.
    assert "cache:" not in result.stderr
