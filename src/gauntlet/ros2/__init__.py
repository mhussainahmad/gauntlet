"""ROS 2 integration — RFC-010.

Step 4 surface: only the rclpy-free :class:`Ros2EpisodePayload`. The
rclpy-backed publisher / recorder land in subsequent steps and are
routed through a lazy ``__getattr__`` that raises a clean ImportError
with the apt / Docker install hint when ``rclpy`` is missing.

The ``[ros2]`` extra in :file:`pyproject.toml` is empty on purpose —
``rclpy`` is NOT distributed via PyPI in its official form (RFC-010 §3).
Users install via ``apt install ros-<distro>-rclpy`` or ``docker run
osrf/ros:<distro>-desktop`` and source the relevant ``setup.bash``
before invoking ``gauntlet ros2 publish``.
"""

from __future__ import annotations

from gauntlet.ros2.schema import Ros2EpisodePayload as Ros2EpisodePayload

__all__ = ["Ros2EpisodePayload"]
