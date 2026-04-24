"""ROS 2 rollout recorder — RFC-010 §6.

Module-scope ``try: import rclpy`` guard. Raises a clean
:class:`ImportError` with the apt / Docker install hint when ``rclpy``
is unavailable, before any class statement is parsed. Mirrors
:mod:`gauntlet.ros2.publisher` (the sibling module).

The actual :class:`Ros2RolloutRecorder` body lands in step 7 — this
step (step 5) only ships the import guard so the install-hint contract
is verified before the class itself exists.
"""

from __future__ import annotations

# Same install hint as the publisher — kept duplicated rather than
# routed through the package __init__ to avoid an import-cycle with the
# ``__getattr__`` lazy loader.
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
    import rclpy  # noqa: F401 — module-scope guard; the actual import sites
    # land in step 7 (this commit ships only the guard contract).
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
    raise ImportError(_ROS2_INSTALL_HINT) from exc


__all__ = ["Ros2RolloutRecorder"]


class Ros2RolloutRecorder:
    """Skeleton — body lands in step 7.

    The class shell exists so the :func:`gauntlet.ros2.__getattr__`
    lazy-loader (and its ``TYPE_CHECKING`` re-export in
    :mod:`gauntlet.ros2`) resolve to a real symbol once the import
    guard has cleared. Calling any method raises
    :class:`NotImplementedError` until step 7 fills in the body.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Ros2RolloutRecorder body is wired in Phase 2 Task 10 step 7.")
