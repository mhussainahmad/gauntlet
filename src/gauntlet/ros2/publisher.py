"""ROS 2 episode publisher — RFC-010 §7.

Module-scope ``try: import rclpy`` guard. Raises a clean
:class:`ImportError` with the apt / Docker install hint when ``rclpy``
is unavailable, before any class statement is parsed. Mirrors
:mod:`gauntlet.monitor.ae` (the closest precedent for a heavy-dep guard).

The class composes — does NOT subclass :class:`rclpy.node.Node`. With
``rclpy.*`` declared ``ignore_missing_imports`` in :file:`pyproject.toml`,
:class:`Node` types as :class:`Any`; subclassing :class:`Any` would trip
mypy ``--strict``'s ``disallow_subclassing_any``. Composition holds the
node internally as ``self._node: Any`` and avoids the override entirely.

Lifecycle (RFC-010 §7):

1. Constructor calls :func:`rclpy.init` only if :func:`rclpy.ok` is
   False (idempotent on multi-publisher processes).
2. :meth:`Ros2EpisodePublisher.publish_episode` serialises an Episode
   via :meth:`Ros2EpisodePayload.from_episode().model_dump_json()` and
   publishes the result on the configured topic as
   ``std_msgs.msg.String``.
3. :meth:`Ros2EpisodePublisher.close` destroys the node and calls
   :func:`rclpy.shutdown`. Idempotent — safe to call twice.
"""

from __future__ import annotations

# rclpy is NOT distributed via PyPI in its official form (RFC-010 §3 /
# exploration §Q2). The lazy-import guard below raises a clean
# ImportError with the apt / Docker install hint when rclpy is absent;
# this is the same shape the monitor torch modules use.
_ROS2_INSTALL_HINT = (
    "gauntlet.ros2.publisher requires the rclpy Python bindings, which are NOT "
    "distributed via PyPI in their official form. Install ROS 2 (Humble or "
    "Jazzy) via your system package manager, e.g.:\n"
    "    sudo apt install ros-humble-rclpy   # Ubuntu 22.04\n"
    "    sudo apt install ros-jazzy-rclpy    # Ubuntu 24.04\n"
    "or run inside an official ROS 2 Docker image, e.g.:\n"
    "    docker run -it osrf/ros:humble-desktop\n"
    "Then source the relevant setup.bash before importing gauntlet.ros2.publisher."
)

try:
    import rclpy
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
    raise ImportError(_ROS2_INSTALL_HINT) from exc

from typing import Any

from gauntlet.ros2.schema import Ros2EpisodePayload
from gauntlet.runner import Episode

__all__ = ["Ros2EpisodePublisher"]


_DEFAULT_TOPIC = "/gauntlet/episodes"
_DEFAULT_NODE_NAME = "gauntlet_episode_publisher"
_DEFAULT_QOS_DEPTH = 10


class Ros2EpisodePublisher:
    """Publish :class:`Episode` summaries to a ROS 2 topic.

    The wire format is JSON serialised inside ``std_msgs/msg/String.data``
    (RFC-010 §5). One call to :meth:`publish_episode` produces one
    message; the full Episode field surface is round-tripped via
    :class:`Ros2EpisodePayload`.

    Args:
        topic: Topic to publish on. Defaults to ``/gauntlet/episodes``.
        node_name: ROS 2 node name. Defaults to
            ``"gauntlet_episode_publisher"``. Two publishers in the
            same process must use distinct names.
        qos_depth: ``QoSProfile`` history depth. Defaults to 10 — the
            same default :func:`rclpy.create_publisher` would pick if
            the user passed an integer in place of a profile.

    Example:

        >>> publisher = Ros2EpisodePublisher(topic="/gauntlet/episodes")
        >>> try:
        ...     for ep in episodes:
        ...         publisher.publish_episode(ep)
        ... finally:
        ...     publisher.close()
    """

    def __init__(
        self,
        *,
        topic: str = _DEFAULT_TOPIC,
        node_name: str = _DEFAULT_NODE_NAME,
        qos_depth: int = _DEFAULT_QOS_DEPTH,
    ) -> None:
        if not topic:
            raise ValueError("topic must be a non-empty string")
        if not node_name:
            raise ValueError("node_name must be a non-empty string")
        if qos_depth < 1:
            raise ValueError(f"qos_depth must be >= 1; got {qos_depth}")

        self._topic = topic
        self._node_name = node_name
        self._qos_depth = qos_depth
        self._closed = False

        # rclpy.init is global. Guard with rclpy.ok() so a second
        # publisher inside the same process is a no-op rather than a
        # double-init error (RFC-010 §7).
        if not rclpy.ok():
            rclpy.init()

        self._node: Any = rclpy.create_node(node_name)
        self._publisher: Any = self._node.create_publisher(String, topic, qos_depth)

    # ----------------------------------------------------------------- public API

    @property
    def topic(self) -> str:
        """The topic this publisher is bound to."""
        return self._topic

    @property
    def node_name(self) -> str:
        """The ROS 2 node name this publisher is bound to."""
        return self._node_name

    def publish_episode(self, episode: Episode) -> Ros2EpisodePayload:
        """Serialise *episode* and publish it on the configured topic.

        Returns the constructed :class:`Ros2EpisodePayload` so callers
        can audit / log what was sent without re-deriving it from the
        Episode. The returned payload is the same one whose JSON
        representation appears in the published ``String.data``.
        """
        if self._closed:
            raise RuntimeError("publish_episode called after close(); construct a new publisher")
        payload = Ros2EpisodePayload.from_episode(episode)
        msg = String()
        msg.data = payload.model_dump_json()
        self._publisher.publish(msg)
        return payload

    def close(self) -> None:
        """Tear down the publisher, destroy the node, and shut down rclpy.

        Idempotent — safe to call twice. ``rclpy.shutdown`` is only
        invoked when ``rclpy.ok()`` is True at call time, so nested
        publishers don't shut down rclpy out from under each other.
        """
        if self._closed:
            return
        self._closed = True
        # Destroy the publisher then the node.
        try:
            self._node.destroy_publisher(self._publisher)
        finally:
            self._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    # ----------------------------------------------------------------- context manager

    def __enter__(self) -> Ros2EpisodePublisher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
