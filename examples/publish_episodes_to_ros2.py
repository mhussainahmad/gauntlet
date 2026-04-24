"""Publish a prior run's episodes to a ROS 2 topic.

Usage:

    # First, do a regular gauntlet run to produce episodes.json:
    uv run gauntlet run examples/suites/tabletop-smoke.yaml \\
        --policy random --out out/

    # Then, with ROS 2 (humble or jazzy) sourced on PATH:
    uv run python examples/publish_episodes_to_ros2.py \\
        --episodes out/episodes.json --topic /gauntlet/episodes

Without ROS 2 installed the script still imports cleanly: the ``--mock``
flag (default off) seeds a :class:`unittest.mock.MagicMock` for ``rclpy``
and the standard message packages so users on a plain torch-free /
rclpy-free machine can preview the JSON payloads end-to-end without
installing ROS 2. ``--dry-run`` is the equivalent CLI flag on
``gauntlet ros2 publish``.

Mirrors ``examples/replay_failure.py`` in shape — short, README-anchored,
factory at module scope so the script doubles as documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

from gauntlet.runner import Episode


def _seed_rclpy_mocks() -> None:
    """Seed ``sys.modules`` with mocks for the ROS 2 import surface.

    Useful for previewing the published payload shape on a machine
    without ROS 2 installed. The publisher module's module-scope
    ``try: import rclpy`` resolves against the mock instead of failing
    with the install-hint :class:`ImportError`. The mock's ``publish``
    captures call args so the demo script can echo what would have been
    sent.
    """
    for name in (
        "rclpy",
        "rclpy.node",
        "rclpy.qos",
        "std_msgs",
        "std_msgs.msg",
        "sensor_msgs",
        "sensor_msgs.msg",
        "geometry_msgs",
        "geometry_msgs.msg",
    ):
        if name not in sys.modules:
            sys.modules[name] = MagicMock(name=name)


def main(*, episodes_path: Path, topic: str, mock: bool) -> None:
    if mock:
        _seed_rclpy_mocks()

    # Lazy import — when --mock is False and rclpy isn't installed the
    # ImportError fires here with the apt / Docker install hint.
    from gauntlet.ros2 import Ros2EpisodePublisher

    raw = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes = [Episode.model_validate(item) for item in raw]
    print(f"Loaded {len(episodes)} episodes from {episodes_path}")

    publisher = Ros2EpisodePublisher(topic=topic)
    try:
        for ep in episodes:
            payload = publisher.publish_episode(ep)
            print(
                f"  cell={ep.cell_index} ep={ep.episode_index} "
                f"success={ep.success} -> "
                f"{payload.model_dump_json()}"
            )
    finally:
        publisher.close()
    print(f"Published {len(episodes)} payloads to {topic}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Publish a prior run's episodes to a ROS 2 topic.",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Path to episodes.json (output of gauntlet run).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/gauntlet/episodes",
        help="ROS 2 topic to publish on. Defaults to /gauntlet/episodes.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Seed sys.modules with rclpy mocks so the script imports "
            "cleanly without a real ROS 2 install. Useful for previewing "
            "the JSON payload shape."
        ),
    )
    args = parser.parse_args()
    main(episodes_path=args.episodes, topic=args.topic, mock=args.mock)
