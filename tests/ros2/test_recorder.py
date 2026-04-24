"""Tests for :class:`Ros2RolloutRecorder` — RFC-010 §9 cases 8-10.

All ``@pytest.mark.ros2``. The ``rclpy`` / ``std_msgs.msg`` modules are
seeded as :class:`MagicMock` instances by :mod:`tests.ros2.conftest`, so
``import gauntlet.ros2.recorder`` resolves at collection time. The
tests then patch the specific seam they exercise — most importantly
``Node.create_subscription``, which we capture and invoke manually to
drive the recorder's callback.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gauntlet.ros2.recorder import Ros2RolloutRecorder

pytestmark = pytest.mark.ros2


@pytest.fixture
def rclpy_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the recorder's bound rclpy with a fresh MagicMock per test."""
    from gauntlet.ros2 import recorder as rec_mod

    fresh = MagicMock(name="rclpy")
    fresh.ok.return_value = False
    fresh.init.return_value = None
    fresh.shutdown.return_value = None
    fresh.spin_once.return_value = None
    fresh.create_node.return_value = MagicMock(name="Node")
    monkeypatch.setattr(rec_mod, "rclpy", fresh)
    return fresh


def _captured_callback(rclpy_mock: MagicMock) -> Callable[[Any], None]:
    """Return the callable handed to ``Node.create_subscription``."""
    node = rclpy_mock.create_node.return_value
    create_sub = node.create_subscription
    create_sub.assert_called_once()
    callback = create_sub.call_args.args[2]
    assert callable(callback)
    return callback  # type: ignore[no-any-return]


class TestConstructorValidation:
    def test_rejects_empty_topic(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="topic"):
            Ros2RolloutRecorder(topic="", out_path=tmp_path / "out.jsonl")

    def test_rejects_empty_node_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="node_name"):
            Ros2RolloutRecorder(
                topic="/foo",
                out_path=tmp_path / "out.jsonl",
                node_name="",
            )

    def test_rejects_negative_duration(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="duration_s"):
            Ros2RolloutRecorder(
                topic="/foo",
                out_path=tmp_path / "out.jsonl",
                duration_s=-1.0,
            )

    def test_rejects_zero_qos_depth(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="qos_depth"):
            Ros2RolloutRecorder(
                topic="/foo",
                out_path=tmp_path / "out.jsonl",
                qos_depth=0,
            )


class TestRos2RolloutRecorder:
    def test_enter_creates_node_and_subscription(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "out.jsonl"
        with Ros2RolloutRecorder(topic="/sensor/joints", out_path=out_path):
            rclpy_mock.create_node.assert_called_once_with("gauntlet_rollout_recorder")
            node = rclpy_mock.create_node.return_value
            node.create_subscription.assert_called_once()
            args = node.create_subscription.call_args.args
            assert args[1] == "/sensor/joints"
            assert args[3] == 10  # default qos_depth

    def test_enter_initialises_rclpy_only_when_not_running(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        rclpy_mock.ok.return_value = True
        with Ros2RolloutRecorder(topic="/foo", out_path=tmp_path / "out.jsonl"):
            rclpy_mock.init.assert_not_called()

    def test_callback_writes_jsonl_line(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        out_path = tmp_path / "out.jsonl"
        with Ros2RolloutRecorder(topic="/sensor/joints", out_path=out_path) as recorder:
            cb = _captured_callback(rclpy_mock)
            cb(MagicMock(__str__=lambda self: "msg-1"))
            cb(MagicMock(__str__=lambda self: "msg-2"))
            assert recorder.n_received == 2

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["topic"] == "/sensor/joints"
        assert first["data"] == "msg-1"
        assert isinstance(first["timestamp"], float)
        second = json.loads(lines[1])
        assert second["data"] == "msg-2"

    def test_exit_destroys_subscription_node_and_shuts_down(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        rclpy_mock.ok.side_effect = [False, True]
        out_path = tmp_path / "out.jsonl"
        with Ros2RolloutRecorder(topic="/foo", out_path=out_path):
            pass
        node = rclpy_mock.create_node.return_value
        node.destroy_subscription.assert_called_once()
        node.destroy_node.assert_called_once_with()
        rclpy_mock.shutdown.assert_called_once_with()

    def test_exit_is_idempotent(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        out_path = tmp_path / "out.jsonl"
        recorder = Ros2RolloutRecorder(topic="/foo", out_path=out_path)
        recorder.__enter__()
        recorder.__exit__(None, None, None)
        recorder.__exit__(None, None, None)  # no-op
        node = rclpy_mock.create_node.return_value
        node.destroy_node.assert_called_once_with()

    def test_exit_skips_shutdown_when_rclpy_already_down(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        rclpy_mock.ok.return_value = False
        with Ros2RolloutRecorder(topic="/foo", out_path=tmp_path / "out.jsonl"):
            pass
        rclpy_mock.shutdown.assert_not_called()

    def test_spin_until_done_polls_rclpy_spin_once(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        with Ros2RolloutRecorder(
            topic="/foo",
            out_path=tmp_path / "out.jsonl",
            duration_s=0.05,
        ) as recorder:
            n = recorder.spin_until_done()
        # spin_once should have been polled at least once during the
        # short duration window.
        assert rclpy_mock.spin_once.call_count >= 1
        assert n == 0

    def test_spin_until_done_terminates_within_duration(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        """duration_s=0.05 should return almost immediately, not hang."""
        import time

        with Ros2RolloutRecorder(
            topic="/foo",
            out_path=tmp_path / "out.jsonl",
            duration_s=0.05,
        ) as recorder:
            t0 = time.monotonic()
            recorder.spin_until_done()
            elapsed = time.monotonic() - t0
        # Generous upper bound — _SPIN_TIMEOUT_S is 0.1 s; at most one
        # spin loop overruns the deadline.
        assert elapsed < 1.0

    def test_spin_until_done_outside_context_raises(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        recorder = Ros2RolloutRecorder(
            topic="/foo",
            out_path=tmp_path / "out.jsonl",
        )
        with pytest.raises(RuntimeError, match="outside the context manager"):
            recorder.spin_until_done()

    def test_callback_after_exit_is_noop(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        """Defensive: a stray callback after __exit__ does not crash.

        rclpy could in principle deliver a queued message between
        callback registration and subscription teardown; the recorder
        treats post-exit callbacks as no-ops rather than raising.
        """
        out_path = tmp_path / "out.jsonl"
        with Ros2RolloutRecorder(topic="/foo", out_path=out_path) as recorder:
            cb = _captured_callback(rclpy_mock)
        # Now outside the context manager — calling cb must not crash.
        cb(MagicMock(__str__=lambda self: "after-exit"))
        # File was closed cleanly with the in-context content.
        assert out_path.read_text(encoding="utf-8") == ""
        assert recorder.n_received == 0

    def test_topic_property_echoes_constructor_arg(
        self, rclpy_mock: MagicMock, tmp_path: Path
    ) -> None:
        recorder = Ros2RolloutRecorder(
            topic="/robot/joint_states",
            out_path=tmp_path / "out.jsonl",
        )
        assert recorder.topic == "/robot/joint_states"

    def test_out_parent_dir_is_created(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "deeper" / "out.jsonl"
        with Ros2RolloutRecorder(topic="/foo", out_path=nested):
            pass
        assert nested.exists()

    def test_reuse_after_close_raises(self, rclpy_mock: MagicMock, tmp_path: Path) -> None:
        recorder = Ros2RolloutRecorder(
            topic="/foo",
            out_path=tmp_path / "out.jsonl",
        )
        with recorder:
            pass
        with pytest.raises(RuntimeError, match="already closed"):
            recorder.__enter__()
