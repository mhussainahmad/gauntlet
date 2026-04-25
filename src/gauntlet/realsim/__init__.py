"""Real-to-sim scene reconstruction stub — see ``GAUNTLET_SPEC.md`` §7
and ``docs/phase3-rfc-021-real-to-sim-stub.md``.

Public surface (grown across the implementation commits):

* :class:`Scene`, :class:`CameraFrame`, :class:`CameraIntrinsics`,
  :class:`Pose` — pydantic models for the reconstruction input set.

Subsequent commits expand the surface with the ingestion pipeline
(``ingest_frames``), the on-disk round-trip (``save_scene`` /
``load_scene``), and the renderer registry / Protocol
(``RealSimRenderer`` + ``register_renderer`` / ``get_renderer``).
"""

from __future__ import annotations

from gauntlet.realsim.schema import POSE_BOTTOM_ROW_TOLERANCE as POSE_BOTTOM_ROW_TOLERANCE
from gauntlet.realsim.schema import SCENE_SCHEMA_VERSION as SCENE_SCHEMA_VERSION
from gauntlet.realsim.schema import CameraFrame as CameraFrame
from gauntlet.realsim.schema import CameraIntrinsics as CameraIntrinsics
from gauntlet.realsim.schema import Pose as Pose
from gauntlet.realsim.schema import Scene as Scene

__all__ = [
    "POSE_BOTTOM_ROW_TOLERANCE",
    "SCENE_SCHEMA_VERSION",
    "CameraFrame",
    "CameraIntrinsics",
    "Pose",
    "Scene",
]
