"""Ingestion pipeline: directory of robot frames + calibration -> :class:`Scene`.

See ``docs/phase3-rfc-021-real-to-sim-stub.md`` §4.4 for the
image-validation strategy (magic bytes only, no Pillow dependency)
and §4.2 for the intrinsics-by-id model.

The pipeline is the first half of the round-trip. The second half —
:func:`gauntlet.realsim.save_scene` / :func:`load_scene` — lives in
:mod:`gauntlet.realsim.io`.

Public surface:

* :func:`ingest_frames` — validate a directory of frames + a
  calibration spec and produce a :class:`Scene`.

The calibration spec mirrors the on-disk manifest's vocabulary so the
two share a single mental model. Differences:

* The calibration spec **does not** carry ``version`` / ``source``;
  those come from the function arguments / module defaults.
* Frame paths in the calibration spec are *relative to* the
  ``frames_dir`` argument; :func:`ingest_frames` resolves them at
  validation time and rejects any path that escapes
  ``frames_dir`` (defence in depth on top of the schema-level
  ``..`` check).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gauntlet.realsim.schema import (
    SCENE_SCHEMA_VERSION,
    CameraFrame,
    CameraIntrinsics,
    Pose,
    Scene,
)

__all__ = [
    "IMAGE_MAGIC_BYTES",
    "IngestionError",
    "ingest_frames",
]


# ---------------------------------------------------------------------------
# Image magic-byte recognition (no Pillow dependency).
# ---------------------------------------------------------------------------


#: Mapping of human-readable container name -> magic-byte prefix.
#: Used by :func:`_recognise_image` to validate frame headers without
#: importing Pillow / imageio. Each entry is the *minimum* prefix that
#: distinguishes the container; longer prefixes (e.g. EXIF JPEG) still
#: match.
IMAGE_MAGIC_BYTES: dict[str, bytes] = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "ppm-binary": b"P6\n",
    "ppm-ascii": b"P3\n",
}


def _recognise_image(payload: bytes) -> str | None:
    """Return the container name if the bytes match a known magic prefix.

    Returns ``None`` for unrecognised payloads. The caller decides how
    to surface the rejection — the pipeline includes the offending
    file path + the first eight bytes in the error message.
    """
    for name, magic in IMAGE_MAGIC_BYTES.items():
        if payload.startswith(magic):
            return name
    return None


# ---------------------------------------------------------------------------
# Calibration spec parsing.
# ---------------------------------------------------------------------------


class IngestionError(ValueError):
    """Raised when ``ingest_frames`` fails validation.

    Subclasses :class:`ValueError` so existing CLI ``except ValueError``
    handlers (e.g. ``gauntlet.cli._fail``) catch it without a code
    change.
    """


def _load_calibration(calib: Path | dict[str, Any]) -> dict[str, Any]:
    """Coerce the ``calib`` argument into a plain dict.

    Accepts a :class:`pathlib.Path` (read + parse JSON) or a pre-loaded
    ``dict`` (notebook / library use). Any other type is rejected
    early so a typo produces a clear message.
    """
    if isinstance(calib, dict):
        return dict(calib)
    if isinstance(calib, Path):
        if not calib.is_file():
            raise IngestionError(f"calibration file not found: {calib}")
        try:
            raw = json.loads(calib.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestionError(f"{calib}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
        if not isinstance(raw, dict):
            raise IngestionError(
                f"{calib}: top-level JSON must be an object; got {type(raw).__name__}"
            )
        return raw
    raise IngestionError(
        f"calib must be a Path or dict; got {type(calib).__name__}",
    )


def _parse_intrinsics(
    raw: object,
) -> dict[str, CameraIntrinsics]:
    """Parse the ``intrinsics`` block of a calibration spec.

    The input must be a dict mapping ``id -> intrinsics-dict``. Each
    value is fed through ``CameraIntrinsics.model_validate``; on
    failure we re-raise as :class:`IngestionError` with the offending
    id baked into the message so the user can pinpoint the bad block
    in a multi-camera calibration JSON.
    """
    if not isinstance(raw, dict):
        raise IngestionError(
            f"calib['intrinsics'] must be an object; got {type(raw).__name__}",
        )
    if not raw:
        raise IngestionError("calib['intrinsics'] must define at least one camera")
    out: dict[str, CameraIntrinsics] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise IngestionError(
                f"calib['intrinsics'] keys must be non-empty strings; got {key!r}",
            )
        try:
            out[key] = CameraIntrinsics.model_validate(value)
        except ValidationError as exc:
            raise IngestionError(
                f"calib['intrinsics'][{key!r}]: {exc}",
            ) from exc
    return out


def _parse_pose(raw: object, *, frame_index: int) -> Pose:
    """Coerce ``raw`` into a :class:`Pose`.

    Accepts:

    * A ``list[list[float]]`` (the canonical shape) — wrapped in a
      ``{"matrix": ...}`` dict before validation.
    * A ``dict`` with key ``"matrix"`` — passed through.

    Anything else is rejected with the offending frame index so the
    caller can find the bad row in a long calibration file.
    """
    if isinstance(raw, list):
        payload: dict[str, Any] = {"matrix": raw}
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise IngestionError(
            f"calib['frames'][{frame_index}]['pose'] must be a list or dict; "
            f"got {type(raw).__name__}",
        )
    try:
        return Pose.model_validate(payload)
    except ValidationError as exc:
        raise IngestionError(
            f"calib['frames'][{frame_index}]['pose']: {exc}",
        ) from exc


def _parse_frames(
    raw: object,
    *,
    intrinsics_ids: set[str],
) -> list[CameraFrame]:
    """Parse the ``frames`` array of a calibration spec."""
    if not isinstance(raw, list):
        raise IngestionError(
            f"calib['frames'] must be a list; got {type(raw).__name__}",
        )
    out: list[CameraFrame] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise IngestionError(
                f"calib['frames'][{i}] must be an object; got {type(item).__name__}",
            )
        # Pull pose out so we can give a frame-indexed error if it's
        # the wrong type before pydantic re-wraps.
        pose_raw = item.get("pose")
        if pose_raw is None:
            raise IngestionError(f"calib['frames'][{i}] is missing 'pose'")
        pose = _parse_pose(pose_raw, frame_index=i)

        # Build a CameraFrame from the rest. Copy item so we don't
        # mutate the user's calibration dict.
        frame_payload = dict(item)
        frame_payload["pose"] = pose
        try:
            frame = CameraFrame.model_validate(frame_payload)
        except ValidationError as exc:
            raise IngestionError(f"calib['frames'][{i}]: {exc}") from exc
        if frame.intrinsics_id not in intrinsics_ids:
            raise IngestionError(
                f"calib['frames'][{i}] references unknown intrinsics_id "
                f"{frame.intrinsics_id!r}; known ids: "
                f"{sorted(intrinsics_ids) if intrinsics_ids else '[]'}",
            )
        out.append(frame)
    return out


# ---------------------------------------------------------------------------
# Frame-file validation.
# ---------------------------------------------------------------------------


def _validate_frame_file(frames_dir: Path, frame: CameraFrame) -> Path:
    """Resolve the frame's ``path`` against ``frames_dir`` and validate it.

    Checks performed:

    1. Path resolves under ``frames_dir`` (no escape via symlink or
       odd casing). The schema-level ``..`` check on
       :class:`CameraFrame.path` catches the obvious case; this is
       defence in depth.
    2. File exists and is a regular file.
    3. First 8 bytes match a known image magic prefix
       (:data:`IMAGE_MAGIC_BYTES`). Rejection includes the path and
       the first 8 bytes hex so the user can diagnose stub /
       truncated files.

    Returns the resolved absolute path. The caller copies / symlinks
    from there during :func:`gauntlet.realsim.save_scene`.
    """
    candidate = (frames_dir / frame.path).resolve()
    try:
        # Python 3.11: ``Path.is_relative_to`` is the right tool.
        if not candidate.is_relative_to(frames_dir.resolve()):
            raise IngestionError(
                f"frame path {frame.path!r} resolves outside frames_dir "
                f"({candidate} not under {frames_dir.resolve()})",
            )
    except ValueError as exc:  # pragma: no cover -- different drives on Windows
        raise IngestionError(
            f"frame path {frame.path!r} could not be resolved against frames_dir: {exc}",
        ) from exc

    if not candidate.is_file():
        raise IngestionError(
            f"frame file not found: {frame.path!r} (resolved to {candidate})",
        )

    head = candidate.read_bytes()[:8]
    if _recognise_image(head) is None:
        raise IngestionError(
            f"frame {frame.path!r} is not a recognised image container "
            f"(first 8 bytes: {head.hex(' ')!r}); supported: "
            f"{sorted(IMAGE_MAGIC_BYTES)}",
        )
    return candidate


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def ingest_frames(
    frames_dir: Path,
    calib: Path | dict[str, Any],
    *,
    source: str | None = None,
) -> Scene:
    """Validate a directory of frames + a calibration spec.

    Args:
        frames_dir: directory holding the raw frame files. Must exist.
            Frame paths declared in ``calib`` are resolved relative to
            this directory.
        calib: calibration spec. Either a :class:`pathlib.Path` to a
            JSON file or a pre-loaded ``dict`` (the latter is convenient
            for notebook use). Required keys are ``"intrinsics"`` and
            ``"frames"``; both are described in
            ``docs/phase3-rfc-021-real-to-sim-stub.md`` §7.
        source: freeform metadata tag (robot id, log id, etc.). Stored
            on :attr:`Scene.source`. Defaults to ``None``.

    Returns:
        A fully validated :class:`Scene`. Frame paths inside the
        returned :class:`Scene` are unchanged (the calibration's
        relative paths) so the result is portable; the on-disk layout
        is materialised by :func:`gauntlet.realsim.save_scene`.

    Raises:
        IngestionError: on any validation failure. The error message
            names the offending field / file so the user can pinpoint
            the bad row without re-reading the entire calibration.
    """
    if not frames_dir.is_dir():
        raise IngestionError(f"frames_dir not found: {frames_dir}")

    calib_dict = _load_calibration(calib)

    intrinsics_raw = calib_dict.get("intrinsics")
    if intrinsics_raw is None:
        raise IngestionError("calib is missing 'intrinsics'")
    intrinsics = _parse_intrinsics(intrinsics_raw)

    frames_raw = calib_dict.get("frames")
    if frames_raw is None:
        raise IngestionError("calib is missing 'frames'")
    frames = _parse_frames(frames_raw, intrinsics_ids=set(intrinsics.keys()))

    # Image-byte validation runs after metadata validation so the
    # user sees the metadata error first when both are wrong (typical
    # human workflow: fix the JSON, then re-run).
    for frame in frames:
        _validate_frame_file(frames_dir, frame)

    return Scene(
        version=SCENE_SCHEMA_VERSION,
        source=source,
        intrinsics=intrinsics,
        frames=frames,
    )
