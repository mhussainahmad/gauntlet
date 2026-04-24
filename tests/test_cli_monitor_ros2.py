"""CLI coverage backfill for ``monitor`` / ``ros2`` / ``replay`` subcommands.

Phase 2.5 Task 11. Targets the argument-validation and lazy-import
error branches in ``gauntlet.cli`` that the existing tests skipped:

* ``monitor train`` / ``monitor score`` reject missing files / dirs and
  surface a clean install hint when torch is absent.
* ``ros2 publish`` ``--dry-run`` runs without rclpy, and the non-dry-run
  path surfaces the install hint when rclpy is absent.
* ``ros2 record`` rejects empty topics / negative durations and surfaces
  the install hint when rclpy is absent.
* ``replay`` validates ``--episode-id`` shape, rejects duplicate
  ``--override`` axes, surfaces ``PolicySpecError``, and creates missing
  output directories.

All tests run in the default torch-/rclpy-free job. We use
``monkeypatch.setitem(sys.modules, ...)`` to simulate missing extras.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import _parse_episode_id, app
from gauntlet.runner.episode import Episode


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _ep(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    suite_name: str = "monitor-suite",
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=0,
        perturbation_config={"lighting_intensity": 0.5},
        success=True,
        terminated=True,
        truncated=False,
        step_count=5,
        total_reward=1.0,
    )


def _write_eps(path: Path, eps: list[Episode]) -> None:
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in eps]) + "\n",
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────
# `monitor train` argument-validation paths.
# ──────────────────────────────────────────────────────────────────────


def test_monitor_train_missing_trajectory_dir(runner: CliRunner, tmp_path: Path) -> None:
    """``monitor train`` exits non-zero with a clear message when the
    trajectory directory is absent."""
    missing = tmp_path / "no-such-dir"
    out = tmp_path / "ae"
    result = runner.invoke(
        app, ["monitor", "train", str(missing), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "trajectory dir not found" in result.stderr


def test_monitor_train_torch_absent_surfaces_install_hint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``import torch`` fails, the lazy import inside the command
    is caught and rewrapped as a clean ``error: ... uv sync --extra
    monitor ...`` line on stderr (lines 668-689 / _fail_monitor_extra)."""
    # Create a real (empty) trajectory dir so we get past the file check.
    traj = tmp_path / "trajectories"
    traj.mkdir()
    out = tmp_path / "ae"

    # Force the lazy ``from gauntlet.monitor.train import train_ae`` to
    # blow up by removing the cached module and pre-poisoning torch.
    monkeypatch.setitem(sys.modules, "torch", None)
    for cached in ("gauntlet.monitor.train", "gauntlet.monitor.ae"):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    result = runner.invoke(
        app, ["monitor", "train", str(traj), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "uv sync --extra monitor" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# `monitor score` argument-validation paths.
# ──────────────────────────────────────────────────────────────────────


def test_monitor_score_missing_episodes_file(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "no-eps.json"
    traj = tmp_path / "traj"
    traj.mkdir()
    ae = tmp_path / "ae"
    ae.mkdir()
    result = runner.invoke(
        app,
        [
            "monitor",
            "score",
            str(missing),
            str(traj),
            "--ae",
            str(ae),
            "--out",
            str(tmp_path / "drift.json"),
        ],
    )
    assert result.exit_code != 0
    assert "episodes file not found" in result.stderr


def test_monitor_score_missing_trajectory_dir(runner: CliRunner, tmp_path: Path) -> None:
    eps = tmp_path / "eps.json"
    _write_eps(eps, [_ep()])
    missing_traj = tmp_path / "no-traj"
    ae = tmp_path / "ae"
    ae.mkdir()
    result = runner.invoke(
        app,
        [
            "monitor",
            "score",
            str(eps),
            str(missing_traj),
            "--ae",
            str(ae),
            "--out",
            str(tmp_path / "drift.json"),
        ],
    )
    assert result.exit_code != 0
    assert "trajectory dir not found" in result.stderr


def test_monitor_score_missing_ae_dir(runner: CliRunner, tmp_path: Path) -> None:
    eps = tmp_path / "eps.json"
    _write_eps(eps, [_ep()])
    traj = tmp_path / "traj"
    traj.mkdir()
    missing_ae = tmp_path / "no-ae"
    result = runner.invoke(
        app,
        [
            "monitor",
            "score",
            str(eps),
            str(traj),
            "--ae",
            str(missing_ae),
            "--out",
            str(tmp_path / "drift.json"),
        ],
    )
    assert result.exit_code != 0
    assert "AE checkpoint dir not found" in result.stderr


def test_monitor_score_torch_absent_surfaces_install_hint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eps = tmp_path / "eps.json"
    _write_eps(eps, [_ep()])
    traj = tmp_path / "traj"
    traj.mkdir()
    ae = tmp_path / "ae"
    ae.mkdir()

    monkeypatch.setitem(sys.modules, "torch", None)
    for cached in ("gauntlet.monitor.score", "gauntlet.monitor.ae"):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    result = runner.invoke(
        app,
        [
            "monitor",
            "score",
            str(eps),
            str(traj),
            "--ae",
            str(ae),
            "--out",
            str(tmp_path / "drift.json"),
        ],
    )
    assert result.exit_code != 0
    assert "uv sync --extra monitor" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# `ros2 publish` — dry-run happy path + arg validation + no-rclpy.
# ──────────────────────────────────────────────────────────────────────


def test_ros2_publish_dry_run_works_without_rclpy(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--dry-run`` short-circuits the rclpy import entirely; only the
    pydantic-only schema module is loaded. Works on a default install."""
    eps_path = tmp_path / "eps.json"
    _write_eps(eps_path, [_ep(cell_index=0), _ep(cell_index=1)])

    result = runner.invoke(
        app,
        ["ros2", "publish", str(eps_path), "--topic", "/g/e", "--dry-run"],
    )
    assert result.exit_code == 0, result.stderr
    # Each payload echoed as one line of JSON.
    assert result.stderr.count('"schema_version":"v1"') == 2
    assert "Dry-run" in result.stderr


def test_ros2_publish_missing_episodes_file(runner: CliRunner, tmp_path: Path) -> None:
    missing = tmp_path / "no-eps.json"
    result = runner.invoke(
        app,
        ["ros2", "publish", str(missing), "--topic", "/g/e", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "episodes file not found" in result.stderr


def test_ros2_publish_empty_topic_rejected(runner: CliRunner, tmp_path: Path) -> None:
    eps_path = tmp_path / "eps.json"
    _write_eps(eps_path, [_ep()])
    result = runner.invoke(
        app,
        ["ros2", "publish", str(eps_path), "--topic", "", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "non-empty" in result.stderr


def test_ros2_publish_non_list_episodes_payload(
    runner: CliRunner, tmp_path: Path
) -> None:
    eps_path = tmp_path / "eps.json"
    eps_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    result = runner.invoke(
        app, ["ros2", "publish", str(eps_path), "--topic", "/g/e", "--dry-run"]
    )
    assert result.exit_code != 0
    assert "expected a list" in result.stderr


def test_ros2_publish_non_dry_run_without_rclpy_surfaces_hint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-dry-run path lazily imports ``gauntlet.ros2.publisher``
    which raises ``ImportError`` with the rclpy install hint when rclpy
    is absent — the CLI rewraps that as a clean error."""
    eps_path = tmp_path / "eps.json"
    _write_eps(eps_path, [_ep()])

    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.delitem(sys.modules, "gauntlet.ros2.publisher", raising=False)

    result = runner.invoke(
        app, ["ros2", "publish", str(eps_path), "--topic", "/g/e"]
    )
    assert result.exit_code != 0
    assert "rclpy" in result.stderr.lower()


# ──────────────────────────────────────────────────────────────────────
# `ros2 record` — argument validation + no-rclpy.
# ──────────────────────────────────────────────────────────────────────


def test_ros2_record_empty_topic_rejected(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["ros2", "record", "--topic", "", "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "non-empty" in result.stderr


def test_ros2_record_negative_duration_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app,
        [
            "ros2",
            "record",
            "--topic",
            "/foo",
            "--out",
            str(out),
            "--duration",
            "-1.0",
        ],
    )
    assert result.exit_code != 0
    assert ">= 0" in result.stderr


def test_ros2_record_without_rclpy_surfaces_hint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out.jsonl"
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.delitem(sys.modules, "gauntlet.ros2.recorder", raising=False)

    result = runner.invoke(
        app,
        [
            "ros2",
            "record",
            "--topic",
            "/foo",
            "--out",
            str(out),
            "--duration",
            "1.0",
        ],
    )
    assert result.exit_code != 0
    assert "rclpy" in result.stderr.lower()


# ──────────────────────────────────────────────────────────────────────
# `replay` — _parse_episode_id branches + duplicate override + bad
# policy spec branch (lines 781-784, 904-906, 911-912, 866).
# ──────────────────────────────────────────────────────────────────────


def test_parse_episode_id_non_integer_halves() -> None:
    import typer

    with pytest.raises(typer.Exit):
        _parse_episode_id("foo:bar")


def test_parse_episode_id_negative_halves() -> None:
    import typer

    with pytest.raises(typer.Exit):
        _parse_episode_id("-1:0")
    with pytest.raises(typer.Exit):
        _parse_episode_id("0:-1")


def test_parse_episode_id_too_many_colons() -> None:
    import typer

    with pytest.raises(typer.Exit):
        _parse_episode_id("1:2:3")


def _write_two_by_three_yaml(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            name: replay-2x3
            env: tabletop
            seed: 2024
            episodes_per_cell: 2
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 2
              camera_offset_x:
                low: -0.05
                high: 0.05
                steps: 3
            """
        ),
        encoding="utf-8",
    )


def _two_by_three_episodes(tmp_path: Path) -> Path:
    """Build a real 12-episode set via the Runner so replay has something
    to chew on. Single small invocation; reused by several tests."""
    from gauntlet.policy.scripted import ScriptedPolicy
    from gauntlet.runner import Runner
    from gauntlet.suite.schema import AxisSpec, Suite

    def _env() -> object:
        from gauntlet.env.tabletop import TabletopEnv

        return TabletopEnv(max_steps=20)

    suite = Suite(
        name="replay-2x3",
        env="tabletop",
        seed=2024,
        episodes_per_cell=2,
        axes={
            "lighting_intensity": AxisSpec(low=0.3, high=1.5, steps=2),
            "camera_offset_x": AxisSpec(low=-0.05, high=0.05, steps=3),
        },
    )
    runner = Runner(n_workers=1, env_factory=_env)  # type: ignore[arg-type]
    eps = runner.run(policy_factory=ScriptedPolicy, suite=suite)
    out = tmp_path / "eps.json"
    _write_eps(out, eps)
    return out


def test_replay_cli_non_list_episodes_payload(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A top-level dict in the episodes JSON is not allowed (line 866)."""
    eps_path = tmp_path / "eps.json"
    eps_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite)

    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
        ],
    )
    assert result.exit_code != 0
    assert "expected a list" in result.stderr


def test_replay_cli_missing_suite_file(runner: CliRunner, tmp_path: Path) -> None:
    eps_path = _two_by_three_episodes(tmp_path)
    missing = tmp_path / "no-suite.yaml"
    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(missing),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "suite file not found" in result.stderr


def test_replay_cli_invalid_suite_yaml(runner: CliRunner, tmp_path: Path) -> None:
    eps_path = _two_by_three_episodes(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("scalar-not-mapping\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(bad),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "invalid suite YAML" in result.stderr


def test_replay_cli_duplicate_override_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    eps_path = _two_by_three_episodes(tmp_path)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--override",
            "lighting_intensity=0.6",
            "--override",
            "lighting_intensity=0.7",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "more than once" in result.stderr


def test_replay_cli_bad_policy_spec(runner: CliRunner, tmp_path: Path) -> None:
    eps_path = _two_by_three_episodes(tmp_path)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)
    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "no-such-policy",
            "--episode-id",
            "0:0",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "no-such-policy" in result.stderr


def test_replay_cli_creates_missing_out_parent_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    eps_path = _two_by_three_episodes(tmp_path)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)
    nested = tmp_path / "deeply" / "nested" / "replay.json"

    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--out",
            str(nested),
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert nested.is_file()


def test_replay_cli_available_ids_preview_truncates_with_more_marker(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``_available_ids_preview`` lists the first 10 ids and appends
    ``... (N more)`` when there are more (line 793->795 branch)."""
    eps_path = _two_by_three_episodes(tmp_path)  # 12 episodes
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "99:99",
            "--env-max-steps",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "more" in result.stderr  # the truncation marker
