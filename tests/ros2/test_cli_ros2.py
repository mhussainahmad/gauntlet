"""CLI tests for `gauntlet ros2 publish` / `gauntlet ros2 record` — RFC-010 §9.

All ``@pytest.mark.ros2``. Exercises the typer subcommands with the
rclpy MagicMock seeded by :mod:`tests.ros2.conftest`. The publisher /
recorder seams are patched per-test on the bound module symbols.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.runner import Episode

pytestmark = pytest.mark.ros2


def _runner() -> CliRunner:
    """CliRunner — typer 0.12+ keeps stdout / stderr separate by default."""
    return CliRunner()


def _make_episode(**overrides: object) -> Episode:
    base: dict[str, object] = {
        "suite_name": "tabletop-smoke-v1",
        "cell_index": 0,
        "episode_index": 0,
        "seed": 42,
        "perturbation_config": {"lighting_intensity": 1.0},
        "success": True,
        "terminated": True,
        "truncated": False,
        "step_count": 12,
        "total_reward": 0.95,
        "metadata": {"master_seed": 1234},
    }
    base.update(overrides)
    return Episode.model_validate(base)


def _write_episodes_json(path: Path, episodes: list[Episode]) -> None:
    path.write_text(
        json.dumps([ep.model_dump(mode="json") for ep in episodes]) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def rclpy_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the publisher AND recorder bound rclpy with a fresh MagicMock."""
    from gauntlet.ros2 import publisher as pub_mod
    from gauntlet.ros2 import recorder as rec_mod

    fresh = MagicMock(name="rclpy")
    fresh.ok.return_value = False
    fresh.init.return_value = None
    fresh.shutdown.return_value = None
    fresh.spin_once.return_value = None
    fresh.create_node.return_value = MagicMock(name="Node")
    monkeypatch.setattr(pub_mod, "rclpy", fresh)
    monkeypatch.setattr(rec_mod, "rclpy", fresh)
    return fresh


# ──────────────────────────────────────────────────────────────────────
# `gauntlet ros2 publish`
# ──────────────────────────────────────────────────────────────────────


class TestRos2PublishCli:
    def test_dry_run_prints_payloads_without_calling_rclpy(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        episodes_path = tmp_path / "episodes.json"
        episodes = [_make_episode(cell_index=0), _make_episode(cell_index=1)]
        _write_episodes_json(episodes_path, episodes)

        result = _runner().invoke(
            app,
            ["ros2", "publish", str(episodes_path), "--topic", "/g/e", "--dry-run"],
        )

        assert result.exit_code == 0, result.stderr
        # Each payload should appear on stderr as one JSON object.
        stderr = result.stderr
        assert stderr.count('"schema_version":"v1"') == 2
        assert "Dry-run" in stderr
        # The dry-run path must NOT call rclpy.init.
        rclpy_mock.init.assert_not_called()
        rclpy_mock.create_node.assert_not_called()

    def test_publish_calls_publisher_for_each_episode(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        episodes_path = tmp_path / "episodes.json"
        episodes = [
            _make_episode(cell_index=0, episode_index=0),
            _make_episode(cell_index=0, episode_index=1),
            _make_episode(cell_index=1, episode_index=0),
        ]
        _write_episodes_json(episodes_path, episodes)

        result = _runner().invoke(
            app,
            ["ros2", "publish", str(episodes_path), "--topic", "/g/e"],
        )

        assert result.exit_code == 0, result.stderr
        node = rclpy_mock.create_node.return_value
        mock_pub = node.create_publisher.return_value
        assert mock_pub.publish.call_count == 3

    def test_publish_rejects_missing_episodes_file(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        result = _runner().invoke(
            app,
            ["ros2", "publish", str(tmp_path / "missing.json"), "--topic", "/g/e"],
        )
        assert result.exit_code != 0
        assert "not found" in result.stderr

    def test_publish_rejects_empty_topic(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        episodes_path = tmp_path / "episodes.json"
        _write_episodes_json(episodes_path, [_make_episode()])

        result = _runner().invoke(
            app,
            ["ros2", "publish", str(episodes_path), "--topic", ""],
        )
        assert result.exit_code != 0
        assert "topic" in result.stderr.lower()

    def test_publish_rejects_episodes_json_with_dict_top_level(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps({"not_a_list": True}), encoding="utf-8")
        result = _runner().invoke(
            app,
            ["ros2", "publish", str(path), "--topic", "/g/e"],
        )
        assert result.exit_code != 0
        assert "expected a list" in result.stderr

    def test_publish_summary_reports_episode_count(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        episodes_path = tmp_path / "episodes.json"
        _write_episodes_json(episodes_path, [_make_episode(), _make_episode(seed=7)])

        result = _runner().invoke(
            app,
            ["ros2", "publish", str(episodes_path), "--topic", "/g/e"],
        )
        assert result.exit_code == 0, result.stderr
        assert "Published 2 episodes" in result.stderr

    def test_publish_help_renders(self) -> None:
        """`gauntlet ros2 publish --help` lists every flag."""
        result = _runner().invoke(app, ["ros2", "publish", "--help"])
        assert result.exit_code == 0
        for fragment in ("--topic", "--node-name", "--dry-run", "--qos-depth"):
            assert fragment in result.stdout


# ──────────────────────────────────────────────────────────────────────
# `gauntlet ros2 record`
# ──────────────────────────────────────────────────────────────────────


class TestRos2RecordCli:
    def test_record_writes_output_file_under_short_duration(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        out = tmp_path / "trajectory.jsonl"
        result = _runner().invoke(
            app,
            [
                "ros2",
                "record",
                "--topic",
                "/sensor/joints",
                "--out",
                str(out),
                "--duration",
                "0.05",
            ],
        )
        assert result.exit_code == 0, result.stderr
        assert out.exists()
        # No real callbacks fired (rclpy mocked), so file is empty.
        assert out.read_text(encoding="utf-8") == ""
        assert "Recorded 0 messages" in result.stderr

    def test_record_creates_subscription_with_topic(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        out = tmp_path / "trajectory.jsonl"
        _runner().invoke(
            app,
            [
                "ros2",
                "record",
                "--topic",
                "/sensor/joints",
                "--out",
                str(out),
                "--duration",
                "0.01",
            ],
        )
        node = rclpy_mock.create_node.return_value
        node.create_subscription.assert_called_once()
        args = node.create_subscription.call_args.args
        assert args[1] == "/sensor/joints"

    def test_record_rejects_negative_duration(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        result = _runner().invoke(
            app,
            [
                "ros2",
                "record",
                "--topic",
                "/foo",
                "--out",
                str(tmp_path / "out.jsonl"),
                "--duration",
                "-1.0",
            ],
        )
        assert result.exit_code != 0
        assert "duration" in result.stderr.lower()

    def test_record_help_renders(self) -> None:
        result = _runner().invoke(app, ["ros2", "record", "--help"])
        assert result.exit_code == 0
        for fragment in ("--topic", "--out", "--duration"):
            assert fragment in result.stdout


# ──────────────────────────────────────────────────────────────────────
# Subcommand group help
# ──────────────────────────────────────────────────────────────────────


class TestRos2GroupHelp:
    def test_top_level_ros2_help_lists_publish_and_record(self) -> None:
        result = _runner().invoke(app, ["ros2", "--help"])
        assert result.exit_code == 0
        assert "publish" in result.stdout
        assert "record" in result.stdout
