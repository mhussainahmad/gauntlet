"""Tests for :class:`Ros2EpisodePublisher` — RFC-010 §9 cases 5-7.

All ``@pytest.mark.ros2``. The ``rclpy`` / ``std_msgs.msg`` modules are
seeded as :class:`MagicMock` instances by :mod:`tests.ros2.conftest`, so
``import gauntlet.ros2.publisher`` resolves at collection time. The
tests then patch the specific seam they exercise (``rclpy.init``,
``rclpy.ok``, ``rclpy.create_node``, ``Node.create_publisher``,
``Publisher.publish``) on the per-test mocks.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from gauntlet.ros2.publisher import Ros2EpisodePublisher
from gauntlet.ros2.schema import Ros2EpisodePayload
from gauntlet.runner import Episode

pytestmark = pytest.mark.ros2


def _make_episode(**overrides: object) -> Episode:
    """Sensible-default Episode for the publisher tests."""
    base: dict[str, object] = {
        "suite_name": "tabletop-smoke-v1",
        "cell_index": 1,
        "episode_index": 0,
        "seed": 42,
        "perturbation_config": {"lighting_intensity": 1.0},
        "success": True,
        "terminated": True,
        "truncated": False,
        "step_count": 12,
        "total_reward": 0.75,
        "metadata": {"master_seed": 1234},
    }
    base.update(overrides)
    return Episode.model_validate(base)


@pytest.fixture
def rclpy_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the rclpy module with a fresh MagicMock for one test.

    The conftest seeds a package-level mock, but each test wants a
    clean call-history. We import the publisher's bound ``rclpy``
    symbol and patch ``rclpy.init`` / ``rclpy.ok`` /
    ``rclpy.create_node`` / ``rclpy.shutdown`` on the per-test mock.
    """
    from gauntlet.ros2 import publisher as pub_mod

    fresh = MagicMock(name="rclpy")
    fresh.ok.return_value = False  # default: not initialised
    fresh.init.return_value = None
    fresh.shutdown.return_value = None
    fresh.create_node.return_value = MagicMock(name="Node")
    monkeypatch.setattr(pub_mod, "rclpy", fresh)
    return fresh


def _publisher_returned_by(rclpy_mock: MagicMock) -> Any:
    """Helper: pull the mock-publisher returned by ``Node.create_publisher``."""
    node = rclpy_mock.create_node.return_value
    return node.create_publisher.return_value


class TestRos2EpisodePublisher:
    def test_constructor_initialises_rclpy_when_not_running(self, rclpy_mock: MagicMock) -> None:
        rclpy_mock.ok.return_value = False
        Ros2EpisodePublisher(topic="/gauntlet/test")
        rclpy_mock.init.assert_called_once_with()

    def test_constructor_skips_init_when_rclpy_already_running(self, rclpy_mock: MagicMock) -> None:
        """RFC §11: nested publishers must not re-init rclpy."""
        rclpy_mock.ok.return_value = True
        Ros2EpisodePublisher(topic="/gauntlet/test")
        rclpy_mock.init.assert_not_called()

    def test_constructor_creates_node_with_default_name(self, rclpy_mock: MagicMock) -> None:
        Ros2EpisodePublisher(topic="/gauntlet/test")
        rclpy_mock.create_node.assert_called_once_with("gauntlet_episode_publisher")

    def test_constructor_creates_publisher_with_topic_and_qos_depth(
        self, rclpy_mock: MagicMock
    ) -> None:
        Ros2EpisodePublisher(topic="/gauntlet/test", qos_depth=5)
        node = rclpy_mock.create_node.return_value
        node.create_publisher.assert_called_once()
        # Args: msg_type=String, topic, qos_depth.
        call_args = node.create_publisher.call_args
        # The first positional is the std_msgs.msg.String mock; we
        # check topic and qos depth.
        assert call_args.args[1] == "/gauntlet/test"
        assert call_args.args[2] == 5

    def test_constructor_rejects_empty_topic(self, rclpy_mock: MagicMock) -> None:
        with pytest.raises(ValueError, match="topic"):
            Ros2EpisodePublisher(topic="")

    def test_constructor_rejects_empty_node_name(self, rclpy_mock: MagicMock) -> None:
        with pytest.raises(ValueError, match="node_name"):
            Ros2EpisodePublisher(topic="/gauntlet/test", node_name="")

    def test_constructor_rejects_zero_qos_depth(self, rclpy_mock: MagicMock) -> None:
        with pytest.raises(ValueError, match="qos_depth"):
            Ros2EpisodePublisher(topic="/gauntlet/test", qos_depth=0)

    def test_publish_episode_emits_json_payload(self, rclpy_mock: MagicMock) -> None:
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        episode = _make_episode()

        returned_payload = publisher.publish_episode(episode)

        mock_pub = _publisher_returned_by(rclpy_mock)
        mock_pub.publish.assert_called_once()
        msg = mock_pub.publish.call_args.args[0]
        # The msg.data must be the JSON serialisation of the payload.
        as_dict = json.loads(msg.data)
        restored = Ros2EpisodePayload.model_validate(as_dict)

        assert restored == returned_payload
        assert restored.suite_name == episode.suite_name
        assert restored.cell_index == episode.cell_index
        assert restored.episode_index == episode.episode_index
        assert restored.success is episode.success
        assert restored.schema_version == "v1"

    def test_publish_episode_returns_constructed_payload(self, rclpy_mock: MagicMock) -> None:
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        episode = _make_episode(seed=7, total_reward=0.42)

        payload = publisher.publish_episode(episode)

        assert isinstance(payload, Ros2EpisodePayload)
        assert payload.seed == 7
        assert payload.total_reward == pytest.approx(0.42)

    def test_publish_episode_after_close_raises(self, rclpy_mock: MagicMock) -> None:
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        publisher.close()
        with pytest.raises(RuntimeError, match="after close"):
            publisher.publish_episode(_make_episode())

    def test_close_destroys_publisher_node_and_shuts_down(self, rclpy_mock: MagicMock) -> None:
        rclpy_mock.ok.side_effect = [False, True]  # init -> closed
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        node = rclpy_mock.create_node.return_value
        mock_pub = _publisher_returned_by(rclpy_mock)

        publisher.close()

        node.destroy_publisher.assert_called_once_with(mock_pub)
        node.destroy_node.assert_called_once_with()
        rclpy_mock.shutdown.assert_called_once_with()

    def test_close_is_idempotent(self, rclpy_mock: MagicMock) -> None:
        rclpy_mock.ok.side_effect = [False, True, False]
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        node = rclpy_mock.create_node.return_value

        publisher.close()
        publisher.close()  # second call: no-op

        node.destroy_node.assert_called_once_with()
        rclpy_mock.shutdown.assert_called_once_with()

    def test_close_skips_shutdown_when_rclpy_already_down(self, rclpy_mock: MagicMock) -> None:
        """rclpy.shutdown is only called when rclpy.ok() is True at close.

        Pins the "nested publishers don't shut down rclpy out from
        under each other" guarantee from the docstring.
        """
        rclpy_mock.ok.side_effect = [False, False]  # init checks: not running, then still not
        publisher = Ros2EpisodePublisher(topic="/gauntlet/test")
        publisher.close()
        rclpy_mock.shutdown.assert_not_called()

    def test_context_manager_closes_on_exit(self, rclpy_mock: MagicMock) -> None:
        rclpy_mock.ok.side_effect = [False, True]
        with Ros2EpisodePublisher(topic="/gauntlet/test") as publisher:
            publisher.publish_episode(_make_episode())
        node = rclpy_mock.create_node.return_value
        node.destroy_node.assert_called_once_with()

    def test_topic_and_node_name_properties_echo_constructor_args(
        self, rclpy_mock: MagicMock
    ) -> None:
        publisher = Ros2EpisodePublisher(topic="/foo/bar", node_name="my_node")
        assert publisher.topic == "/foo/bar"
        assert publisher.node_name == "my_node"
