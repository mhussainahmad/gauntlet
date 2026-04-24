"""Import-guard contract for the ros2 modules — RFC-010 §9.

Marked ``@pytest.mark.ros2`` because the test re-patches the rclpy
module cache, which has process-wide effects we don't want spilling
into the default torch-free job. The guard contract itself is:

1. ``from gauntlet.ros2 import Ros2EpisodePayload`` works without
   rclpy (covered by tests/ros2/test_schema.py — runs in default job).
2. ``from gauntlet.ros2 import Ros2EpisodePublisher`` (or
   :class:`Ros2RolloutRecorder`) raises a clean :class:`ImportError`
   pointing at the apt / Docker install path when rclpy is missing.
3. ``import gauntlet.ros2.publisher`` directly raises the same
   ImportError when rclpy is missing.

We exercise rclpy absence by setting ``sys.modules["rclpy"] = None``
and flushing the cached module entries — same shape the
:mod:`tests.test_import_guards` ``torch_absent`` fixture uses for the
monitor extras guards.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.ros2


@pytest.fixture
def rclpy_absent(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Simulate rclpy not being installed for the import-guard tests.

    The conftest.py at the package level seeds ``sys.modules["rclpy"]``
    with a :class:`MagicMock` so test collection works at all. This
    fixture flushes that mock + the cached gauntlet.ros2 modules and
    sets ``sys.modules["rclpy"] = None``, which makes the next
    ``import rclpy`` raise :class:`ImportError`.
    """
    to_flush = [
        "rclpy",
        "rclpy.node",
        "rclpy.qos",
        "gauntlet.ros2.publisher",
        "gauntlet.ros2.recorder",
    ]
    for name in to_flush:
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name, raising=False)
    # Setting to None makes ``import rclpy`` raise ImportError without
    # actually uninstalling the package.
    monkeypatch.setitem(sys.modules, "rclpy", None)
    yield


def test_publisher_import_raises_install_hint_when_rclpy_missing(
    rclpy_absent: None,
) -> None:
    """``import gauntlet.ros2.publisher`` fails loudly with the install hint."""
    with pytest.raises(ImportError, match="apt install ros-humble-rclpy"):
        importlib.import_module("gauntlet.ros2.publisher")


def test_recorder_import_raises_install_hint_when_rclpy_missing(
    rclpy_absent: None,
) -> None:
    """``import gauntlet.ros2.recorder`` fails loudly with the install hint."""
    with pytest.raises(ImportError, match="apt install ros-jazzy-rclpy"):
        importlib.import_module("gauntlet.ros2.recorder")


def test_publisher_install_hint_mentions_docker_alternative(
    rclpy_absent: None,
) -> None:
    """Hint must point at both apt and the official Docker image."""
    with pytest.raises(ImportError, match="osrf/ros:humble-desktop"):
        importlib.import_module("gauntlet.ros2.publisher")


def test_lazy_reexport_raises_install_hint_at_attribute_access(
    rclpy_absent: None,
) -> None:
    """``gauntlet.ros2.Ros2EpisodePublisher`` via ``__getattr__`` fails loudly.

    Critical for the bimodal-surface contract from RFC-010 §4 / the
    package docstring: ``import gauntlet.ros2`` must NOT raise (the
    schema is rclpy-free), but attribute access for the rclpy-backed
    symbols must surface the install hint.
    """
    pkg = importlib.import_module("gauntlet.ros2")
    # The schema re-export keeps working — pure pydantic, no rclpy.
    assert pkg.Ros2EpisodePayload is not None
    with pytest.raises(ImportError, match="apt install ros-humble-rclpy"):
        pkg.__getattr__("Ros2EpisodePublisher")
    with pytest.raises(ImportError, match="apt install ros-jazzy-rclpy"):
        pkg.__getattr__("Ros2RolloutRecorder")


def test_unknown_attribute_via_getattr_raises_attributeerror() -> None:
    """``__getattr__`` does not silently swallow typos."""
    import gauntlet.ros2 as pkg

    with pytest.raises(AttributeError, match="NotARealSymbol"):
        pkg.__getattr__("NotARealSymbol")
