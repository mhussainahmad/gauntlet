"""Tests for B-22 — episode-level seed manifest + ``gauntlet repro``.

Each test runs ``gauntlet run`` end-to-end on a tiny synthetic suite
(``--env-max-steps 5``, 1-cell x 1-episode) and asserts the new
``repro.json`` artefact and ``gauntlet repro <id>`` subcommand behave
to spec. The Runner / Report layers are NOT mocked — the test pipeline
is the real pipeline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.runner import (
    Episode,
    capture_gauntlet_version,
    capture_git_commit,
    compute_suite_hash,
)
from gauntlet.suite import load_suite

# ----------------------------------------------------------------------
# Fixtures + helpers — mirror tests/test_cli.py wiring.
# ----------------------------------------------------------------------


@pytest.fixture
def cli() -> CliRunner:
    return CliRunner()


_TINY_SUITE_YAML = textwrap.dedent(
    """\
    name: tiny-repro-suite
    env: tabletop
    episodes_per_cell: 1
    seed: 7
    axes:
      lighting_intensity:
        values: [0.5]
    """
)


def _write_tiny_suite(tmp_path: Path) -> Path:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(_TINY_SUITE_YAML, encoding="utf-8")
    return suite_path


def _run_once(cli: CliRunner, tmp_path: Path) -> tuple[Path, Path]:
    """Run ``gauntlet run`` once and return (suite_path, out_dir)."""
    suite_path = _write_tiny_suite(tmp_path)
    out_dir = tmp_path / "out"
    result = cli.invoke(
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
            "--no-html",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    return suite_path, out_dir


# ----------------------------------------------------------------------
# 1 — repro.json is written next to episodes.json with the right shape.
# ----------------------------------------------------------------------


def test_repro_json_written_with_expected_shape(cli: CliRunner, tmp_path: Path) -> None:
    suite_path, out_dir = _run_once(cli, tmp_path)

    repro_path = out_dir / "repro.json"
    assert repro_path.is_file(), "gauntlet run must emit repro.json"

    payload = json.loads(repro_path.read_text(encoding="utf-8"))
    # Top-level keys.
    for key in (
        "schema_version",
        "suite_path",
        "suite_name",
        "suite_env",
        "suite_hash",
        "policy",
        "seed_override",
        "env_max_steps",
        "episodes",
    ):
        assert key in payload, f"repro.json missing top-level key {key!r}"

    assert payload["schema_version"] == 1
    assert payload["suite_name"] == "tiny-repro-suite"
    assert payload["suite_env"] == "tabletop"
    assert payload["policy"] == "random"
    assert payload["env_max_steps"] == 5
    assert payload["seed_override"] is None
    assert payload["suite_path"] == str(suite_path)

    # Per-episode entries: 1 cell x 1 episode = 1 entry.
    eps = payload["episodes"]
    assert isinstance(eps, list) and len(eps) == 1
    entry = eps[0]
    for key in (
        "episode_id",
        "cell_index",
        "episode_index",
        "env_seed",
        "policy_seed",
        "axis_config",
        "gauntlet_version",
        "suite_hash",
        "git_commit",
    ):
        assert key in entry, f"per-episode entry missing key {key!r}"
    assert entry["episode_id"] == "cell_0_episode_0"
    assert entry["cell_index"] == 0
    assert entry["episode_index"] == 0
    assert entry["axis_config"] == {"lighting_intensity": 0.5}
    # Both streams derive from the same SeedSequence node, so env_seed
    # and policy_seed are mirrored in the manifest.
    assert entry["env_seed"] == entry["policy_seed"]


# ----------------------------------------------------------------------
# 2 — provenance fields populated on every Episode.
# ----------------------------------------------------------------------


def test_provenance_fields_populated_on_episodes(cli: CliRunner, tmp_path: Path) -> None:
    _, out_dir = _run_once(cli, tmp_path)
    episodes_payload = json.loads((out_dir / "episodes.json").read_text(encoding="utf-8"))
    assert len(episodes_payload) == 1
    ep_dict = episodes_payload[0]

    # gauntlet_version should be the installed distribution version
    # (Episode is produced by the live Runner inside the test process).
    expected_version = capture_gauntlet_version()
    assert ep_dict["gauntlet_version"] == expected_version
    assert isinstance(ep_dict["gauntlet_version"], str)

    # suite_hash matches the canonical hash of the suite Pydantic model.
    suite = load_suite(Path(json.loads((out_dir / "repro.json").read_text())["suite_path"]))
    assert ep_dict["suite_hash"] == compute_suite_hash(suite)

    # git_commit is either a 40-char SHA hex string or None — both are
    # legal depending on whether tests run inside a git checkout. We
    # accept either rather than baking a CI assumption.
    commit = ep_dict["git_commit"]
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


# ----------------------------------------------------------------------
# 3 — repro.json is byte-deterministic under sort_keys.
# ----------------------------------------------------------------------


def test_repro_json_keys_sorted_for_determinism(cli: CliRunner, tmp_path: Path) -> None:
    _, out_dir = _run_once(cli, tmp_path)
    raw = (out_dir / "repro.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    # Re-encode with sort_keys=True; the file must round-trip identically
    # if the writer used sort_keys (it does — see _write_repro_json).
    expected = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    )
    assert raw == expected
    # Stronger signal: hash stability across two re-encodings.
    h1 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert h1 == h2


# ----------------------------------------------------------------------
# 4 — `gauntlet repro <id>` matches on a fresh checkout (same code).
# ----------------------------------------------------------------------


def test_repro_subcommand_matches_original_episode(cli: CliRunner, tmp_path: Path) -> None:
    _, out_dir = _run_once(cli, tmp_path)
    result = cli.invoke(
        app,
        [
            "repro",
            "cell_0_episode_0",
            "--out",
            str(out_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert "repro match" in result.stderr
    assert "cell_0_episode_0" in result.stderr


# ----------------------------------------------------------------------
# 5 — `gauntlet repro` detects a tampered episode and exits non-zero.
# ----------------------------------------------------------------------


def test_repro_subcommand_detects_tampered_episode(cli: CliRunner, tmp_path: Path) -> None:
    _, out_dir = _run_once(cli, tmp_path)

    # Forge a mismatch by mutating the recorded total_reward in
    # episodes.json — the freshly-rolled repro Episode will produce the
    # honest reward and the diff-summary check must fire.
    eps_path = out_dir / "episodes.json"
    eps = json.loads(eps_path.read_text(encoding="utf-8"))
    eps[0]["total_reward"] = eps[0]["total_reward"] + 999.0
    eps[0]["step_count"] = eps[0]["step_count"] + 1
    eps_path.write_text(json.dumps(eps, indent=2) + "\n", encoding="utf-8")

    result = cli.invoke(
        app,
        [
            "repro",
            "cell_0_episode_0",
            "--out",
            str(out_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert "repro mismatch" in result.stderr
    assert "total_reward" in result.stderr
    assert "step_count" in result.stderr


# ----------------------------------------------------------------------
# 6 — `gauntlet repro` reports unknown episode-id with a friendly hint.
# ----------------------------------------------------------------------


def test_repro_subcommand_unknown_episode_id_lists_available(
    cli: CliRunner, tmp_path: Path
) -> None:
    _, out_dir = _run_once(cli, tmp_path)
    result = cli.invoke(
        app,
        [
            "repro",
            "cell_99_episode_99",
            "--out",
            str(out_dir),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "cell_99_episode_99" in result.stderr
    assert "available" in result.stderr
    # The available preview must include at least the one real id.
    assert "cell_0_episode_0" in result.stderr


# ----------------------------------------------------------------------
# 7 — capture_git_commit returns None outside a git checkout.
# ----------------------------------------------------------------------


def test_capture_git_commit_returns_none_outside_git(tmp_path: Path) -> None:
    # Simulate a non-git directory by patching subprocess.run to raise
    # FileNotFoundError (no git binary on PATH) — provenance.py must
    # catch and degrade to None rather than propagate.
    with mock.patch(
        "gauntlet.runner.provenance.subprocess.run",
        side_effect=FileNotFoundError("no git"),
    ):
        assert capture_git_commit(cwd=tmp_path) is None

    # Also exercise the non-zero return-code branch directly.
    fake_completed = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=128,
        stdout="",
        stderr="fatal: not a git repository",
    )
    with mock.patch("gauntlet.runner.provenance.subprocess.run", return_value=fake_completed):
        assert capture_git_commit(cwd=tmp_path) is None


# ----------------------------------------------------------------------
# 8 — Episode default values for the new provenance fields are None.
# ----------------------------------------------------------------------


def test_episode_provenance_fields_default_to_none() -> None:
    ep = Episode(
        suite_name="x",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={},
        success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=1.0,
    )
    assert ep.gauntlet_version is None
    assert ep.suite_hash is None
    assert ep.git_commit is None
