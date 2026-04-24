"""ROS 2 rollout recorder — RFC-010 §6.

Module-scope ``try: import rclpy`` guard. Raises a clean
:class:`ImportError` with the apt / Docker install hint when ``rclpy``
is unavailable, before any class statement is parsed. Mirrors
:mod:`gauntlet.ros2.publisher` (the sibling module).

The :class:`Ros2RolloutRecorder` is a context manager: ``__enter__``
initialises rclpy + creates a node + creates a subscription on the
configured topic; the subscription's callback appends one JSONL line
per received message to the configured output path. ``__exit__``
destroys the subscription, the node, and shuts down rclpy.

Spinning policy (RFC-010 §6): :meth:`spin_until_done` polls
:func:`rclpy.spin_once` with a small timeout in a loop until either
the configured duration elapses or a ``KeyboardInterrupt`` is raised.
The output file is flushed on every callback so a SIGINT does not
lose received data.

Output format (RFC-010 §6 / §11): JSON-lines. One line per received
message: ``{"timestamp": <float>, "topic": <str>, "data": <str>}``.
``data`` is :func:`str` of the message payload — lossy but generic;
a future RFC may add typed message-class introspection.
"""

from __future__ import annotations

# rclpy is NOT distributed via PyPI in its official form (RFC-010 §3 /
# exploration §Q2). The lazy-import guard below raises a clean
# ImportError with the apt / Docker install hint when rclpy is absent.
_ROS2_INSTALL_HINT = (
    "gauntlet.ros2.recorder requires the rclpy Python bindings, which are NOT "
    "distributed via PyPI in their official form. Install ROS 2 (Humble or "
    "Jazzy) via your system package manager, e.g.:\n"
    "    sudo apt install ros-humble-rclpy   # Ubuntu 22.04\n"
    "    sudo apt install ros-jazzy-rclpy    # Ubuntu 24.04\n"
    "or run inside an official ROS 2 Docker image, e.g.:\n"
    "    docker run -it osrf/ros:humble-desktop\n"
    "Then source the relevant setup.bash before importing gauntlet.ros2.recorder."
)

try:
    import rclpy
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
    raise ImportError(_ROS2_INSTALL_HINT) from exc

import json
import time
from io import TextIOWrapper
from pathlib import Path
from typing import Any

__all__ = ["Ros2RolloutRecorder"]


_DEFAULT_NODE_NAME = "gauntlet_rollout_recorder"
_DEFAULT_QOS_DEPTH = 10
_DEFAULT_DURATION_S = 30.0
_SPIN_TIMEOUT_S = 0.1


class Ros2RolloutRecorder:
    """Context-manager subscriber that dumps received messages to JSONL.

    Args:
        topic: Topic to subscribe to (required).
        out_path: JSONL output path (required). Parent directory is
            created on enter.
        message_type: Concrete ROS 2 message class to subscribe with.
            Defaults to :class:`std_msgs.msg.String`. Users wanting to
            capture e.g. ``sensor_msgs.msg.JointState`` import the
            class and pass it explicitly.
        node_name: ROS 2 node name. Defaults to
            ``"gauntlet_rollout_recorder"``. Distinct names per
            recorder if you nest them.
        duration_s: Soft cap for :meth:`spin_until_done`. ``0.0``
            means "spin forever (until KeyboardInterrupt)". Defaults
            to 30.0 seconds.
        qos_depth: ``QoSProfile`` history depth. Defaults to 10.

    Example:

        >>> with Ros2RolloutRecorder(
        ...     topic="/robot/joint_states",
        ...     out_path=Path("trajectory.jsonl"),
        ...     duration_s=30.0,
        ... ) as recorder:
        ...     recorder.spin_until_done()
    """

    def __init__(
        self,
        *,
        topic: str,
        out_path: Path,
        message_type: type = String,
        node_name: str = _DEFAULT_NODE_NAME,
        duration_s: float = _DEFAULT_DURATION_S,
        qos_depth: int = _DEFAULT_QOS_DEPTH,
    ) -> None:
        if not topic:
            raise ValueError("topic must be a non-empty string")
        if not node_name:
            raise ValueError("node_name must be a non-empty string")
        if duration_s < 0.0:
            raise ValueError(f"duration_s must be >= 0; got {duration_s}")
        if qos_depth < 1:
            raise ValueError(f"qos_depth must be >= 1; got {qos_depth}")

        self._topic = topic
        self._out_path = out_path
        self._message_type = message_type
        self._node_name = node_name
        self._duration_s = duration_s
        self._qos_depth = qos_depth

        self._node: Any = None
        self._subscription: Any = None
        self._fh: TextIOWrapper | None = None
        self._n_received = 0
        self._closed = False

    # ----------------------------------------------------------------- public API

    @property
    def topic(self) -> str:
        """The topic this recorder is subscribed to."""
        return self._topic

    @property
    def n_received(self) -> int:
        """Number of messages received since :meth:`__enter__`."""
        return self._n_received

    def spin_until_done(self) -> int:
        """Poll :func:`rclpy.spin_once` until duration elapses.

        Returns the total number of messages received across the
        recorder's lifetime. ``KeyboardInterrupt`` flushes the JSONL
        file and re-raises so a user can ``Ctrl-C`` cleanly.
        """
        if self._fh is None or self._node is None:
            raise RuntimeError("spin_until_done called outside the context manager")

        # ``None`` deadline means spin forever (until KeyboardInterrupt).
        deadline = time.monotonic() + self._duration_s if self._duration_s > 0.0 else None

        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                rclpy.spin_once(self._node, timeout_sec=_SPIN_TIMEOUT_S)
        except KeyboardInterrupt:
            if self._fh is not None:
                self._fh.flush()
            raise
        return self._n_received

    # ----------------------------------------------------------------- internals

    def _on_message(self, msg: Any) -> None:
        """Subscription callback — append one JSONL line per message."""
        if self._fh is None:
            # Defensive: a callback after exit (impossible under our
            # context manager, but rclpy could in principle deliver)
            # is a no-op rather than a crash.
            return
        line = {
            "timestamp": time.time(),
            "topic": self._topic,
            "data": str(msg),
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self._n_received += 1

    # ----------------------------------------------------------------- context manager

    def __enter__(self) -> Ros2RolloutRecorder:
        if self._closed:
            raise RuntimeError("recorder already closed; construct a new one")
        # Set up the output file first so any subsequent rclpy failure
        # leaves no stray node behind.
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._out_path.open("w", encoding="utf-8")

        if not rclpy.ok():
            rclpy.init()

        try:
            self._node = rclpy.create_node(self._node_name)
            self._subscription = self._node.create_subscription(
                self._message_type,
                self._topic,
                self._on_message,
                self._qos_depth,
            )
        except Exception:
            # Clean up the file handle if rclpy node/subscription
            # creation throws after we've opened the file.
            self._fh.close()
            self._fh = None
            if rclpy.ok():
                rclpy.shutdown()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._node is not None and self._subscription is not None:
                self._node.destroy_subscription(self._subscription)
            if self._node is not None:
                self._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
