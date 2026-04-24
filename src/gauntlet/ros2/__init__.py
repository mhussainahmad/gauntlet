"""ROS 2 integration — RFC-010.

The public surface is bimodal, mirroring :mod:`gauntlet.monitor`:

* **rclpy-free**: :class:`Ros2EpisodePayload` is re-exported eagerly. It
  only needs pydantic, so ``from gauntlet.ros2 import Ros2EpisodePayload``
  works on the default (no-extras) install path.
* **rclpy-backed**: :class:`Ros2EpisodePublisher` and
  :class:`Ros2RolloutRecorder` are lazily resolved via module-level
  ``__getattr__``. On a machine without a real ROS 2 install the import
  path fires only when the user asks for them, and raises a clean
  :class:`ImportError` that points at the apt / Docker install path.

This keeps ``import gauntlet.ros2`` cheap and safe on the default
install path while still letting users write
``from gauntlet.ros2 import Ros2EpisodePublisher`` once they've
``apt install``-ed (or Docker-run) ROS 2.

The ``[ros2]`` extra in :file:`pyproject.toml` is empty on purpose —
``rclpy`` is NOT distributed via PyPI in its official form (RFC-010 §3).
The extra exists so ``uv sync --extra ros2`` is a recognised invocation
and so the ``ros2-dev`` dev-group has a partner extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gauntlet.ros2.schema import Ros2EpisodePayload as Ros2EpisodePayload

if TYPE_CHECKING:
    # These symbols exist only when rclpy is available. The stubs here
    # make static type-checkers happy regardless of whether the runtime
    # ROS 2 install is present.
    from gauntlet.ros2.publisher import Ros2EpisodePublisher as Ros2EpisodePublisher
    from gauntlet.ros2.recorder import Ros2RolloutRecorder as Ros2RolloutRecorder

__all__ = [
    "Ros2EpisodePayload",
    "Ros2EpisodePublisher",
    "Ros2RolloutRecorder",
]


_ROS2_INSTALL_HINT = (
    "gauntlet.ros2.{attr} requires the rclpy Python bindings, which are NOT "
    "distributed via PyPI in their official form. Install ROS 2 (Humble or "
    "Jazzy) via your system package manager, e.g.:\n"
    "    sudo apt install ros-humble-rclpy   # Ubuntu 22.04\n"
    "    sudo apt install ros-jazzy-rclpy    # Ubuntu 24.04\n"
    "or run inside an official ROS 2 Docker image, e.g.:\n"
    "    docker run -it osrf/ros:humble-desktop\n"
    "Then source the relevant setup.bash before invoking gauntlet ros2."
)


# Lazy re-exports for the rclpy-backed symbols. Importing
# ``gauntlet.ros2.publisher`` / ``.recorder`` raises the install-hint
# ``ImportError`` when rclpy is missing; routing those imports through
# ``__getattr__`` keeps a bare ``import gauntlet.ros2`` rclpy-free.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Ros2EpisodePublisher": ("gauntlet.ros2.publisher", "Ros2EpisodePublisher"),
    "Ros2RolloutRecorder": ("gauntlet.ros2.recorder", "Ros2RolloutRecorder"),
}


def __getattr__(name: str) -> Any:
    """Route rclpy-requiring symbols to their lazy-import module."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        try:
            import importlib

            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(_ROS2_INSTALL_HINT.format(attr=name)) from exc
        return getattr(module, attr_name)
    raise AttributeError(f"module 'gauntlet.ros2' has no attribute {name!r}")
