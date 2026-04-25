"""Pydantic models for real-to-sim scene reconstruction inputs.

See ``docs/phase3-rfc-021-real-to-sim-stub.md`` and ``GAUNTLET_SPEC.md``
§7 ("Real-to-sim scene reconstruction (gaussian splatting from robot
cameras into an eval environment)").

The schema is the *input* contract for a future gaussian-splatting
renderer — it does not own the renderer itself (see
:mod:`gauntlet.realsim.renderer`). A future contributor (or
third-party plugin) can implement :class:`RealSimRenderer` against
these types without changing the schema.

Design notes:

* Every model uses ``ConfigDict(extra="forbid", ser_json_inf_nan="strings")``,
  matching the rest of gauntlet's pydantic surface (report / aggregate /
  monitor schemas). Silent additions are a contract violation; non-finite
  floats round-trip through JSON cleanly.
* :class:`Pose` carries a 4x4 row-major matrix as ``list[list[float]]``.
  See RFC §4.1 for why we picked this over quaternion + translation
  (NeRFStudio / COLMAP convention; single round-trip artefact; finite
  validation surface).
* :class:`CameraIntrinsics` is shared by id, not embedded per-frame —
  see RFC §4.2.
* :attr:`Scene.version` starts at 1; bump on any incompatible schema
  move so older readers fail fast on unknown future versions instead
  of silently misparsing.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "POSE_BOTTOM_ROW_TOLERANCE",
    "SCENE_SCHEMA_VERSION",
    "CameraFrame",
    "CameraIntrinsics",
    "Pose",
    "Scene",
]


# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------


#: Current on-disk :class:`Scene` schema version. Bump on any
#: incompatible field move (renames, type narrowings, semantic shifts).
#: Additive optional fields with defaults do *not* bump this — they
#: remain backwards-compatible for older manifests.
SCENE_SCHEMA_VERSION: int = 1


#: Absolute tolerance for the ``[0, 0, 0, 1]`` bottom row of a pose
#: matrix. Robotics pipelines emit slightly off poses (numerical drift,
#: per-frame state estimation jitter); 1e-6 accommodates that without
#: accepting a clearly broken matrix.
POSE_BOTTOM_ROW_TOLERANCE: float = 1e-6


# ---------------------------------------------------------------------------
# CameraIntrinsics.
# ---------------------------------------------------------------------------


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics + optional radial-tangential distortion.

    All values are in pixels except :attr:`distortion`, which is a
    list of OpenCV-order coefficients ``[k1, k2, p1, p2, k3, ...]``.
    A typical robot wrist camera has zero distortion; we keep
    :attr:`distortion` defaulted to an empty list rather than five
    zeros so the manifest is honest about whether the calibration
    pipeline produced distortion coefficients at all.

    The renderer is responsible for applying distortion at render
    time. The schema does not preprocess frames (see RFC §2 — no
    image preprocessing during ingest).
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float] = Field(default_factory=list)

    @field_validator("fx", "fy")
    @classmethod
    def _focal_length_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"focal length must be a positive finite float; got {v!r}")
        return v

    @field_validator("cx", "cy")
    @classmethod
    def _principal_point_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"principal point must be a finite float; got {v!r}")
        return v

    @field_validator("width", "height")
    @classmethod
    def _resolution_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"image dimension must be > 0; got {v!r}")
        return v

    @field_validator("distortion")
    @classmethod
    def _distortion_finite(cls, v: list[float]) -> list[float]:
        for i, coeff in enumerate(v):
            if not math.isfinite(coeff):
                raise ValueError(f"distortion[{i}] must be finite; got {coeff!r}")
        return v


# ---------------------------------------------------------------------------
# Pose.
# ---------------------------------------------------------------------------


class Pose(BaseModel):
    """A 4x4 row-major rigid transform from camera frame to world frame.

    Storage is ``list[list[float]]`` of shape (4, 4). Validation:

    * exactly 4 rows;
    * each row exactly 4 columns;
    * all 16 entries finite floats;
    * bottom row equals ``[0, 0, 0, 1]`` within
      :data:`POSE_BOTTOM_ROW_TOLERANCE`.

    The 3x3 rotation sub-block is **not** validated for orthonormality
    — robotics pipelines routinely emit slightly-off poses (numerical
    drift, per-frame state estimation jitter) and the renderer is
    responsible for renormalising. A user who wants a strict check
    can build it on top of the schema; we do not impose one.

    Why row-major 4x4 over quaternion+translation: see
    ``docs/phase3-rfc-021-real-to-sim-stub.md`` §4.1.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    matrix: list[list[float]]

    @model_validator(mode="after")
    def _validate_matrix_shape(self) -> Pose:
        m = self.matrix
        if len(m) != 4:
            raise ValueError(f"pose matrix must have 4 rows; got {len(m)}")
        for i, row in enumerate(m):
            if len(row) != 4:
                raise ValueError(f"pose matrix row {i} must have 4 columns; got {len(row)}")
            for j, value in enumerate(row):
                if not math.isfinite(value):
                    raise ValueError(
                        f"pose matrix entry [{i}][{j}] must be a finite float; got {value!r}"
                    )
        # Bottom-row check: [0, 0, 0, 1] within tolerance.
        bottom = m[3]
        expected = (0.0, 0.0, 0.0, 1.0)
        for j, (got, want) in enumerate(zip(bottom, expected, strict=True)):
            if abs(got - want) > POSE_BOTTOM_ROW_TOLERANCE:
                raise ValueError(
                    f"pose matrix bottom row must be {list(expected)} within "
                    f"{POSE_BOTTOM_ROW_TOLERANCE}; got [3][{j}]={got!r}"
                )
        return self


# ---------------------------------------------------------------------------
# CameraFrame.
# ---------------------------------------------------------------------------


class CameraFrame(BaseModel):
    """One row of the manifest — a single timestamped camera frame.

    :attr:`path` is the relative POSIX path inside the scene directory
    pointing at the frame container (PNG, JPEG, PPM). It is *not*
    absolute — :class:`Scene` is portable across machines.

    :attr:`intrinsics_id` keys into :attr:`Scene.intrinsics`. The
    pipeline rejects a manifest that references an unknown id at
    validation time (see :func:`gauntlet.realsim.pipeline.ingest_frames`).
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    path: str
    timestamp: float
    intrinsics_id: str
    pose: Pose

    @field_validator("path")
    @classmethod
    def _path_is_relative_posix(cls, v: str) -> str:
        if not v:
            raise ValueError("frame path must be a non-empty string")
        if v.startswith("/"):
            raise ValueError(f"frame path must be relative; got absolute {v!r}")
        if "\\" in v:
            raise ValueError(f"frame path must use POSIX separators; got {v!r}")
        if ".." in v.split("/"):
            raise ValueError(f"frame path must not contain '..' segments; got {v!r}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _timestamp_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"timestamp must be a finite float; got {v!r}")
        return v

    @field_validator("intrinsics_id")
    @classmethod
    def _intrinsics_id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("intrinsics_id must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Scene.
# ---------------------------------------------------------------------------


class Scene(BaseModel):
    """A real-to-sim scene reconstruction input set.

    Built by :func:`gauntlet.realsim.ingest_frames` from a directory of
    raw frames + a calibration JSON; persisted to disk via
    :func:`gauntlet.realsim.save_scene`; round-tripped through
    :func:`gauntlet.realsim.load_scene`.

    Field order: metadata first, then the breakdown surface. ``frames``
    is the headline — a renderer reads it row-by-row.

    :attr:`version` is the on-disk schema version (currently
    :data:`SCENE_SCHEMA_VERSION`). Older readers reject unknown future
    versions; newer readers accept older versions when the schema is
    additive-only since the bump.

    :attr:`source` is a freeform tag (robot id, log id, customer scene
    id, etc.). It is metadata only — no validation beyond "string or
    None".
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    version: int = SCENE_SCHEMA_VERSION
    source: str | None = None
    intrinsics: dict[str, CameraIntrinsics] = Field(default_factory=dict)
    frames: list[CameraFrame] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _version_known(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"scene version must be >= 1; got {v}")
        if v > SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"scene version {v} is newer than this gauntlet (max "
                f"{SCENE_SCHEMA_VERSION}); upgrade gauntlet to read this scene"
            )
        return v

    @model_validator(mode="after")
    def _validate_intrinsics_refs(self) -> Scene:
        known = set(self.intrinsics.keys())
        for i, frame in enumerate(self.frames):
            if frame.intrinsics_id not in known:
                raise ValueError(
                    f"frame[{i}] (path={frame.path!r}) references unknown "
                    f"intrinsics_id {frame.intrinsics_id!r}; known ids: "
                    f"{sorted(known) if known else '[]'}"
                )
        return self
