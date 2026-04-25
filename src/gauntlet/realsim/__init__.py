"""Real-to-sim scene reconstruction stub — see ``GAUNTLET_SPEC.md`` §7
and ``docs/phase3-rfc-021-real-to-sim-stub.md``.

Public surface:

* :class:`Scene`, :class:`CameraFrame`, :class:`CameraIntrinsics`,
  :class:`Pose` — pydantic models for the reconstruction input set.
* :func:`ingest_frames` — validate a directory of frames + a
  calibration spec and produce a :class:`Scene`.
* :func:`save_scene` / :func:`load_scene` — round-trip on-disk
  manifest persistence.
* :class:`RealSimRenderer` — :class:`typing.Protocol` for a renderer
  plugin to implement against the schema. The first concrete
  implementation (gaussian splatting) is deferred — see RFC §1 / §2.
* :func:`register_renderer` / :func:`get_renderer` /
  :func:`list_renderers` — module-local renderer registry. Stays
  local rather than going through :mod:`gauntlet.plugins` so the
  plugin module's pinned public surface does not grow for an
  extension point that has zero in-tree consumers (RFC §4.6).

The CLI surface (``gauntlet realsim ingest`` / ``gauntlet realsim
info``) is registered by :mod:`gauntlet.cli` against this package.
"""

from __future__ import annotations

from gauntlet.realsim.io import MANIFEST_FILENAME as MANIFEST_FILENAME
from gauntlet.realsim.io import SceneIOError as SceneIOError
from gauntlet.realsim.io import load_scene as load_scene
from gauntlet.realsim.io import save_scene as save_scene
from gauntlet.realsim.pipeline import IMAGE_MAGIC_BYTES as IMAGE_MAGIC_BYTES
from gauntlet.realsim.pipeline import IngestionError as IngestionError
from gauntlet.realsim.pipeline import ingest_frames as ingest_frames
from gauntlet.realsim.renderer import RealSimRenderer as RealSimRenderer
from gauntlet.realsim.renderer import RendererFactory as RendererFactory
from gauntlet.realsim.renderer import RendererRegistryError as RendererRegistryError
from gauntlet.realsim.renderer import get_renderer as get_renderer
from gauntlet.realsim.renderer import list_renderers as list_renderers
from gauntlet.realsim.renderer import register_renderer as register_renderer
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
    "RealSimRenderer",
    "RendererFactory",
    "RendererRegistryError",
    "Scene",
    "SceneIOError",
    "get_renderer",
    "ingest_frames",
    "list_renderers",
    "load_scene",
    "register_renderer",
    "save_scene",
]
