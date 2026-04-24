"""Test scaffolding for the ros2 test suite — RFC-010 §9.

The hard fact: CI (and most developer machines) does not have ``rclpy``
installed. ``rclpy`` is NOT distributed via PyPI in its official form
(RFC-010 §3 / exploration §Q2) — install paths are
``apt install ros-<distro>-rclpy`` or ``docker run osrf/ros:<distro>-
desktop``. The ``ros2-tests`` CI job intentionally does NOT install ROS
2; it relies on the mocks seeded here.

This conftest seeds ``sys.modules`` with :class:`unittest.mock.MagicMock`
instances for ``rclpy``, ``rclpy.node``, ``std_msgs``, ``std_msgs.msg``,
``sensor_msgs``, ``sensor_msgs.msg``, ``geometry_msgs``,
``geometry_msgs.msg`` BEFORE pytest collects any sibling test module.
That way, the module-scope ``try: import rclpy`` guard in
:mod:`gauntlet.ros2.publisher` / :mod:`gauntlet.ros2.recorder` resolves
against the mock, and ``from gauntlet.ros2.publisher import ...`` at
the top of a test file no longer blows up at collection time.

The :mod:`tests.ros2.test_import_guards` test module re-patches
``sys.modules["rclpy"] = None`` inside a fixture (with explicit module
cache flushes) to verify the install-hint :class:`ImportError` contract;
that fixture is scoped to a single test, leaving the package-level
mocks intact for the rest of the suite.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Seed the rclpy / std_msgs / sensor_msgs / geometry_msgs surface with
# MagicMocks at conftest import time. Pytest imports conftest before
# any test module in the same directory, so this fires before the
# ``from gauntlet.ros2.publisher import ...`` statement at the top of
# test_publisher.py / test_recorder.py / test_cli_ros2.py.
#
# We only seed entries that are NOT already present — if a developer
# really has rclpy installed locally, we leave their real install in
# place so the tests exercise reality.
_MOCKED_MODULES = (
    "rclpy",
    "rclpy.node",
    "rclpy.qos",
    "std_msgs",
    "std_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
)

for _name in _MOCKED_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock(name=_name)
