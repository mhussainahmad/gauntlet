"""Pydantic message schema for the ROS 2 integration — RFC-010 §5.

ROS-2-free. The :class:`Ros2EpisodePayload` model is the JSON payload
serialised inside ``std_msgs/msg/String`` and published on the
``/gauntlet/episodes`` topic (or whichever topic the user picks). It is
also consumed by the recorder when round-tripping captured messages.

Public surface:

* :class:`Ros2EpisodePayload` — one Episode summary per published
  message. Construct via :meth:`Ros2EpisodePayload.from_episode`.

The schema is torch-free, rclpy-free, and importable in the default
CI job — it is the only ``gauntlet.ros2`` symbol that does not require
the apt/Docker ROS 2 install.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauntlet.runner import Episode

__all__ = ["Ros2EpisodePayload"]


class Ros2EpisodePayload(BaseModel):
    """JSON payload published as ``std_msgs/msg/String.data`` per Episode.

    Every field is JSON-safe. ``schema_version`` is hard-coded to
    ``"v1"`` and exists so future-RFC schema changes don't silently
    break old consumers — see RFC-010 §11. A v2 will be a
    coordinated bump on both the publisher and the schema.

    Attributes:
        schema_version: Always ``"v1"`` in this RFC.
        suite_name: Echoed from :attr:`Episode.suite_name`.
        cell_index: Echoed from :attr:`Episode.cell_index`.
        episode_index: Echoed from :attr:`Episode.episode_index`.
        seed: Echoed from :attr:`Episode.seed`.
        perturbation_config: Echoed from
            :attr:`Episode.perturbation_config`.
        success: Echoed from :attr:`Episode.success`.
        terminated: Echoed from :attr:`Episode.terminated`.
        truncated: Echoed from :attr:`Episode.truncated`.
        step_count: Echoed from :attr:`Episode.step_count`.
        total_reward: Echoed from :attr:`Episode.total_reward`.
        metadata: Echoed from :attr:`Episode.metadata`. Empty dict
            when the producing Episode had no metadata.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["v1"]

    suite_name: str
    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]

    success: bool
    terminated: bool
    truncated: bool
    step_count: int
    total_reward: float

    metadata: dict[str, float | int | str | bool]

    @classmethod
    def from_episode(cls, episode: Episode) -> Ros2EpisodePayload:
        """Build a payload from an :class:`Episode` — pure mapping.

        No coercion happens here beyond Pydantic's own validation. The
        Episode's fields are JSON-safe primitives by construction
        (``ConfigDict(extra="forbid")`` on Episode + the typed
        ``metadata: dict[str, float | int | str | bool]`` annotation),
        so this is a 1:1 re-shape with ``schema_version`` stamped on.
        """
        return cls(
            schema_version="v1",
            suite_name=episode.suite_name,
            cell_index=episode.cell_index,
            episode_index=episode.episode_index,
            seed=episode.seed,
            perturbation_config=dict(episode.perturbation_config),
            success=episode.success,
            terminated=episode.terminated,
            truncated=episode.truncated,
            step_count=episode.step_count,
            total_reward=episode.total_reward,
            metadata=dict(episode.metadata),
        )
