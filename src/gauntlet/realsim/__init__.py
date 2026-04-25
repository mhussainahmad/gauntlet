"""Real-to-sim scene reconstruction stub — see ``GAUNTLET_SPEC.md`` §7
and ``docs/phase3-rfc-021-real-to-sim-stub.md``.

Public surface (grown across the implementation commits):

* :class:`Scene`, :class:`CameraFrame`, :class:`CameraIntrinsics`,
  :class:`Pose` — pydantic models for the reconstruction input set.
* :func:`ingest_frames` — validate a directory of frames + a
  calibration spec and produce a :class:`Scene`.
* :func:`save_scene` / :func:`load_scene` — round-trip on-disk
  manifest persistence.

Subsequent commits add the renderer registry / Protocol
(``RealSimRenderer`` + ``register_renderer`` / ``get_renderer``) and
the CLI wiring.
"""

from __future__ import annotations

from gauntlet.realsim.io import MANIFEST_FILENAME as MANIFEST_FILENAME
from gauntlet.realsim.io import SceneIOError as SceneIOError
from gauntlet.realsim.io import load_scene as load_scene
from gauntlet.realsim.io import save_scene as save_scene
from gauntlet.realsim.pipeline import IMAGE_MAGIC_BYTES as IMAGE_MAGIC_BYTES
from gauntlet.realsim.pipeline import IngestionError as IngestionError
from gauntlet.realsim.pipeline import ingest_frames as ingest_frames
from gauntlet.realsim.schema import POSE_BOTTOM_ROW_TOLERANCE as POSE_BOTTOM_ROW_TOLERANCE
from gauntlet.realsim.schema import SCENE_SCHEMA_VERSION as SCENE_SCHEMA_VERSION
from gauntlet.realsim.schema import CameraFrame as CameraFrame
from gauntlet.realsim.schema import CameraIntrinsics as CameraIntrinsics
from gauntlet.realsim.schema import Pose as Pose
from gauntlet.realsim.schema import Scene as Scene

__all__ = [
    "IMAGE_MAGIC_BYTES",
    "MANIFEST_FILENAME",
    "POSE_BOTTOM_ROW_TOLERANCE",
    "SCENE_SCHEMA_VERSION",
    "CameraFrame",
    "CameraIntrinsics",
    "IngestionError",
    "Pose",
    "Scene",
    "SceneIOError",
    "ingest_frames",
    "load_scene",
    "save_scene",
]
