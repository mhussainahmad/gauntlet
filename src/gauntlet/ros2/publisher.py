"""ROS 2 episode publisher — RFC-010 §7.

Module-scope ``try: import rclpy`` guard. Raises a clean
:class:`ImportError` with the apt / Docker install hint when ``rclpy``
is unavailable, before any class statement is parsed. Mirrors
:mod:`gauntlet.monitor.ae` (the closest precedent for a heavy-dep guard).

The actual :class:`Ros2EpisodePublisher` body lands in step 6 — this
step (step 5) only ships the import guard so the install-hint contract
is verified by :mod:`tests.ros2.test_import_guards` before the class
itself exists.

The class composes — does NOT subclass :class:`rclpy.node.Node`. With
``rclpy.*`` declared ``ignore_missing_imports`` in :file:`pyproject.toml`,
:class:`Node` types as :class:`Any`; subclassing :class:`Any` would trip
mypy ``--strict``'s ``disallow_subclassing_any``. Composition holds the
node internally as ``self._node: Any`` and avoids the override entirely.
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
    import rclpy  # noqa: F401 — module-scope guard; the actual import sites
    # land in step 6 (this commit ships only the guard contract).
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
    raise ImportError(_ROS2_INSTALL_HINT) from exc


__all__ = ["Ros2EpisodePublisher"]


class Ros2EpisodePublisher:
    """Skeleton — body lands in step 6.

    The class shell exists so the :func:`gauntlet.ros2.__getattr__`
    lazy-loader (and its ``TYPE_CHECKING`` re-export in
    :mod:`gauntlet.ros2`) resolve to a real symbol once the import
    guard has cleared. Calling any method raises
    :class:`NotImplementedError` until step 6 fills in the body.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Ros2EpisodePublisher body is wired in Phase 2 Task 10 step 6.")
