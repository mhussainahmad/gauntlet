"""Schema round-trip tests for :class:`Ros2EpisodePayload` — RFC-010 §9.

Pure-pydantic. NO ``ros2`` marker — the schema module is rclpy-free
and runs in the default torch-free CI job. These tests pin three
properties of the v1 message contract:

1. The ``Episode -> Ros2EpisodePayload -> JSON -> Ros2EpisodePayload``
   round-trip is lossless for the canonical Episode shape.
2. ``ConfigDict(extra="forbid")`` rejects unknown fields — silent
   schema additions are a contract violation, not a feature.
3. ``Episode.metadata`` round-trips through the payload regardless of
   which primitive types it carries (str / int / float / bool).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gauntlet.ros2.schema import Ros2EpisodePayload
from gauntlet.runner import Episode


def _make_episode(**overrides: object) -> Episode:
    """Build a typed Episode with sensible defaults for round-trip tests."""
    base: dict[str, object] = {
        "suite_name": "tabletop-smoke-v1",
        "cell_index": 3,
        "episode_index": 1,
        "seed": 42,
        "perturbation_config": {"lighting_intensity": 1.0, "camera_offset_x": -0.02},
        "success": True,
        "terminated": True,
        "truncated": False,
        "step_count": 17,
        "total_reward": 0.95,
        "metadata": {"master_seed": 1234, "n_cells": 6, "episodes_per_cell": 4},
    }
    base.update(overrides)
    return Episode.model_validate(base)


class TestRos2EpisodePayloadSchema:
    def test_round_trip_episode_to_payload_to_json(self) -> None:
        """Episode -> payload -> JSON -> payload is lossless for v1."""
        episode = _make_episode()
        payload = Ros2EpisodePayload.from_episode(episode)
        as_json = payload.model_dump_json()
        restored = Ros2EpisodePayload.model_validate_json(as_json)

        assert restored == payload
        assert restored.schema_version == "v1"
        assert restored.suite_name == episode.suite_name
        assert restored.cell_index == episode.cell_index
        assert restored.episode_index == episode.episode_index
        assert restored.seed == episode.seed
        assert restored.success is episode.success
        assert restored.terminated is episode.terminated
        assert restored.truncated is episode.truncated
        assert restored.step_count == episode.step_count
        assert restored.total_reward == pytest.approx(episode.total_reward)
        assert restored.perturbation_config == dict(episode.perturbation_config)
        assert restored.metadata == dict(episode.metadata)

    def test_extra_fields_forbidden(self) -> None:
        """Unknown fields in the JSON payload raise ValidationError.

        Pins the ``ConfigDict(extra="forbid")`` contract — silent
        schema additions are a contract violation per RFC-010 §5.
        """
        payload = Ros2EpisodePayload.from_episode(_make_episode())
        raw = json.loads(payload.model_dump_json())
        raw["unexpected_future_field"] = "surprise"
        with pytest.raises(ValidationError):
            Ros2EpisodePayload.model_validate(raw)

    def test_metadata_round_trips_primitive_mix(self) -> None:
        """Mixed-primitive Episode.metadata survives the payload round-trip."""
        episode = _make_episode(
            metadata={
                "master_seed": 7,
                "label": "smoke",
                "is_baseline": True,
                "wall_clock_s": 0.123,
            }
        )
        payload = Ros2EpisodePayload.from_episode(episode)
        restored = Ros2EpisodePayload.model_validate_json(payload.model_dump_json())

        assert restored.metadata == {
            "master_seed": 7,
            "label": "smoke",
            "is_baseline": True,
            "wall_clock_s": pytest.approx(0.123),
        }

    def test_schema_version_pinned_to_v1(self) -> None:
        """A payload with the wrong schema_version literal is rejected."""
        payload = Ros2EpisodePayload.from_episode(_make_episode())
        raw = json.loads(payload.model_dump_json())
        raw["schema_version"] = "v2"
        with pytest.raises(ValidationError):
            Ros2EpisodePayload.model_validate(raw)

    def test_empty_metadata_round_trips_as_empty_dict(self) -> None:
        """Episode with an empty ``metadata`` round-trips as ``{}``."""
        episode = _make_episode(metadata={})
        payload = Ros2EpisodePayload.from_episode(episode)
        restored = Ros2EpisodePayload.model_validate_json(payload.model_dump_json())
        assert restored.metadata == {}


class TestPackageImportSurface:
    def test_top_level_import_does_not_load_publisher_or_recorder(self) -> None:
        """``import gauntlet.ros2`` is rclpy-free.

        The eager re-export is only :class:`Ros2EpisodePayload` (pure
        pydantic). The publisher / recorder modules — which carry the
        module-scope ``try: import rclpy`` guard — are routed through
        ``__getattr__`` and must NOT be loaded by a bare
        ``import gauntlet.ros2``. This is the bimodal-surface contract
        from RFC-010 §4 and is the property that keeps the default
        torch-free / rclpy-free CI job green.
        """
        import sys

        # Flush any prior load of the rclpy-backed modules so the
        # assertion below tests the clean import path. (Other tests in
        # the suite may have triggered the lazy loader.)
        for cached in ("gauntlet.ros2", "gauntlet.ros2.publisher", "gauntlet.ros2.recorder"):
            sys.modules.pop(cached, None)

        import gauntlet.ros2 as pkg

        assert pkg.Ros2EpisodePayload is Ros2EpisodePayload
        # The bimodal-surface contract: the rclpy-backed modules must
        # NOT have been imported by ``import gauntlet.ros2``.
        assert "gauntlet.ros2.publisher" not in sys.modules
        assert "gauntlet.ros2.recorder" not in sys.modules
