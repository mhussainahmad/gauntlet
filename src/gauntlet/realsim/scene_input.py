"""Raw single-capture scene-input parser — Phase 3 Task 18.

This module sits one layer **below** :mod:`gauntlet.realsim.schema` /
:mod:`gauntlet.realsim.pipeline`. The schema layer owns the validated,
manifest-backed :class:`gauntlet.realsim.Scene` artefact (per-camera-id
intrinsics, POSIX-relative frame paths, round-tripped through
``manifest.json``). This module owns the *raw parse* of an external
capture export — a COLMAP-style or Polycam-style directory dropped
verbatim by the capture pipeline.

Why two layers, not one:

* The :class:`Scene` schema is a *curated* gauntlet artefact. It rejects
  malformed data fast, hides field-level provenance, and exists to feed
  a renderer or a sweep.
* :class:`RealSceneInput` is a *forensic* surface: it preserves the
  raw extrinsics list (as ``np.ndarray``), points at depth-map and
  point-cloud sidecars by absolute path, and surfaces a list of
  validation *warnings* (not hard errors) so a caller can decide
  whether the export is good enough to feed into ingest.

The (future) bridge between the two lives in
:mod:`gauntlet.realsim.scene_to_axis`, which converts a parsed
:class:`RealSceneInput` into a list of
:class:`gauntlet.suite.schema.ExtrinsicsValue`-shaped dicts for the
``camera_extrinsics`` axis (B-42).

Renderer scope (very explicit): this module ships the **input
pipeline only**. The actual gaussian-splatting / NeRF / mesh renderer
is a future PR — see
:class:`gauntlet.realsim.scene_to_axis.RendererNotImplementedError`.

Hard requirements (Phase 3 T18):

* :func:`gauntlet.security.safe_join` is used at every path-boundary —
  no raw ``Path / user_supplied_str`` joins. The capture-dir layout is
  parsed from JSON the user supplied, so every JSON-derived filename
  is a defence-in-depth target.
* :func:`validate_scene_input` collects **all** warnings, not just the
  first. Callers loop over the returned list to decide whether to
  proceed.
* The dataclass is frozen and ``to_dict`` produces a JSON-serialisable
  representation (``np.ndarray`` -> nested ``list[list[float]]``).
  ``dataclasses.asdict`` is *not* a stable serialiser for this type —
  use :meth:`RealSceneInput.to_dict`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gauntlet.security import PathTraversalError, safe_join

__all__ = [
    "INTRINSICS_REQUIRED_KEYS",
    "RealSceneInput",
    "RealSceneInputError",
    "load_real_scene",
    "validate_scene_input",
]


#: Minimum set of pinhole-intrinsics keys we demand at parse time.
#: Distortion coefficients are *optional* (matches
#: :class:`gauntlet.realsim.CameraIntrinsics`); a capture pipeline that
#: emits zero distortion legitimately omits the field.
INTRINSICS_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"fx", "fy", "cx", "cy", "width", "height"},
)


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class RealSceneInputError(ValueError):
    """Raised on hard parse failures by :func:`load_real_scene`.

    Subclasses :class:`ValueError` so existing CLI ``except ValueError``
    handlers (e.g. ``gauntlet.cli._fail``) catch it without a code
    change. Soft-state findings (intrinsics missing optional fields,
    extrinsics not contiguous, etc.) are returned by
    :func:`validate_scene_input` as warning strings instead of raising.
    """


# ---------------------------------------------------------------------------
# Dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealSceneInput:
    """Raw, parsed view of a single capture-dir export.

    Field semantics:

    * :attr:`capture_dir` — resolved absolute path to the export root.
      Every path-typed field below resolves under this directory via
      :func:`gauntlet.security.safe_join`; absolute paths are explicitly
      rejected to keep the artefact portable.
    * :attr:`intrinsics` — flat ``dict[str, float]`` of pinhole
      intrinsics. Keys are the spec-mandated
      :data:`INTRINSICS_REQUIRED_KEYS` plus any optional distortion
      scalars (``k1``, ``k2``, ``p1``, ``p2``, ``k3``) the export
      emitted. Width and height land here as floats — the parser does
      *not* coerce to int because some COLMAP exports emit fractional
      principal points and we want the round-trip to be lossless.
    * :attr:`extrinsics_per_frame` — list of 4x4 ``np.ndarray`` rigid
      transforms (camera -> world), one per frame, in the JSON's
      declared order. Bottom-row sanity is checked at parse time but
      the rotation sub-block is *not* re-orthonormalised.
    * :attr:`depth_maps` — optional list of resolved absolute paths to
      per-frame depth-map sidecars (typically ``.exr`` or ``.npy``).
      ``None`` when the export does not include depth. When present,
      length must match :attr:`extrinsics_per_frame`.
    * :attr:`point_cloud` — optional resolved absolute path to a single
      aggregate point-cloud sidecar (``.ply`` / ``.npy``). ``None`` when
      not provided.
    * :attr:`metadata` — freeform dict copied verbatim from the
      export's ``metadata`` block (capture device, software version,
      timestamps, etc.). Empty dict when absent. Typed as
      ``dict[str, object]`` rather than ``dict[str, Any]`` so mypy
      ``--strict`` keeps the explicit-Any ban honest; downstream
      consumers narrow values themselves.

    The dataclass is ``frozen=True`` so handles can be passed across
    threads / suite-loaders without surprise mutation; in-place changes
    raise ``dataclasses.FrozenInstanceError``.

    Serialisation: use :meth:`to_dict` — it converts ``np.ndarray``
    extrinsics to nested ``list[list[float]]`` and stringifies path
    fields. ``dataclasses.asdict`` is **not** a stable serialiser for
    this type because it leaves ``np.ndarray`` verbatim, which is not
    JSON-encodable.
    """

    capture_dir: Path
    intrinsics: dict[str, float]
    extrinsics_per_frame: list[NDArray[np.float64]]
    depth_maps: list[Path] | None = None
    point_cloud: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of this scene input.

        ``np.ndarray`` extrinsics become ``list[list[float]]`` and every
        :class:`pathlib.Path` becomes a POSIX string. The point of this
        method (vs ``dataclasses.asdict``) is round-trip faithfulness:
        the output is a plain dict that ``json.dumps`` accepts without
        a custom encoder.
        """
        return {
            "capture_dir": str(self.capture_dir),
            "intrinsics": dict(self.intrinsics),
            "extrinsics_per_frame": [m.tolist() for m in self.extrinsics_per_frame],
            "depth_maps": (
                [str(p) for p in self.depth_maps] if self.depth_maps is not None else None
            ),
            "point_cloud": str(self.point_cloud) if self.point_cloud is not None else None,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _read_json(capture_dir: Path, filename: str) -> object:
    """Load *filename* from *capture_dir* with safe_join + JSON parsing.

    Raises :class:`RealSceneInputError` on missing file, traversal
    attempt, or invalid JSON. The wrapping is deliberate so callers see
    one error class regardless of failure mode.
    """
    try:
        path = safe_join(capture_dir, filename)
    except PathTraversalError as exc:
        raise RealSceneInputError(
            f"capture-dir entry {filename!r} escapes the capture directory: {exc}",
        ) from exc
    if not path.is_file():
        raise RealSceneInputError(
            f"required capture-dir file not found: {filename!r} (resolved to {path})",
        )
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealSceneInputError(
            f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})",
        ) from exc
    return loaded


def _parse_intrinsics(raw: object) -> dict[str, float]:
    """Coerce ``intrinsics.json`` content into ``dict[str, float]``.

    Required keys (:data:`INTRINSICS_REQUIRED_KEYS`) are enforced here;
    optional distortion keys pass through if present and are dropped if
    not. Anything non-finite or non-numeric raises.
    """
    if not isinstance(raw, dict):
        raise RealSceneInputError(
            f"intrinsics.json top-level must be an object; got {type(raw).__name__}",
        )
    missing = INTRINSICS_REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise RealSceneInputError(
            f"intrinsics.json missing required keys: {sorted(missing)}",
        )
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise RealSceneInputError(
                f"intrinsics.json keys must be strings; got {key!r}",
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RealSceneInputError(
                f"intrinsics.json[{key!r}] must be a number; got {type(value).__name__}",
            )
        coerced = float(value)
        if coerced != coerced or coerced in (float("inf"), float("-inf")):
            # NaN compares unequal to itself; the inf check covers ±inf.
            raise RealSceneInputError(
                f"intrinsics.json[{key!r}] must be finite; got {value!r}",
            )
        out[key] = coerced
    return out


def _parse_extrinsics(raw: object) -> list[NDArray[np.float64]]:
    """Coerce ``extrinsics.json`` content into a list of 4x4 ndarrays.

    The on-disk format is ``list[list[list[float]]]`` — a top-level
    list of 4x4 row-major matrices. Each entry is converted to
    ``np.ndarray`` of dtype ``float64`` and shape-checked. Bottom-row
    sanity ([0, 0, 0, 1] within 1e-6) is enforced here so an obviously
    broken export is rejected up front.
    """
    if not isinstance(raw, list):
        raise RealSceneInputError(
            f"extrinsics.json top-level must be a list of 4x4 matrices; got {type(raw).__name__}",
        )
    if not raw:
        raise RealSceneInputError("extrinsics.json must declare at least one frame")
    out: list[NDArray[np.float64]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, list):
            raise RealSceneInputError(
                f"extrinsics.json[{i}] must be a 4x4 list-of-lists; got {type(item).__name__}",
            )
        try:
            arr = np.asarray(item, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RealSceneInputError(
                f"extrinsics.json[{i}] could not be coerced to a numeric matrix: {exc}",
            ) from exc
        if arr.shape != (4, 4):
            raise RealSceneInputError(
                f"extrinsics.json[{i}] must be 4x4; got shape {arr.shape}",
            )
        if not bool(np.all(np.isfinite(arr))):
            raise RealSceneInputError(
                f"extrinsics.json[{i}] contains non-finite entries",
            )
        bottom_expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if not bool(np.allclose(arr[3], bottom_expected, atol=1e-6)):
            raise RealSceneInputError(
                f"extrinsics.json[{i}] bottom row must be [0, 0, 0, 1]; got {arr[3].tolist()}",
            )
        out.append(arr)
    return out


def _resolve_optional_path(capture_dir: Path, rel: object, *, label: str) -> Path | None:
    """safe_join *rel* under *capture_dir* if *rel* is a string, else None.

    ``label`` is the human-readable context name for the error message.
    Returns ``None`` when *rel* is ``None`` or absent.
    """
    if rel is None:
        return None
    if not isinstance(rel, str):
        raise RealSceneInputError(
            f"{label} must be a string path or null; got {type(rel).__name__}",
        )
    if not rel:
        raise RealSceneInputError(f"{label} must be a non-empty string")
    try:
        return safe_join(capture_dir, rel)
    except PathTraversalError as exc:
        raise RealSceneInputError(
            f"{label} {rel!r} escapes the capture directory: {exc}",
        ) from exc


def _resolve_depth_maps(capture_dir: Path, raw: object) -> list[Path] | None:
    """Resolve a list of relative depth-map paths via safe_join.

    Accepts:

    * ``None`` / absent -> returns ``None``.
    * ``list[str]`` -> resolved list (does not require existence; the
      validator surfaces missing files as warnings, not hard errors).
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise RealSceneInputError(
            f"depth_maps must be a list of relative paths or null; got {type(raw).__name__}",
        )
    out: list[Path] = []
    for i, entry in enumerate(raw):
        resolved = _resolve_optional_path(
            capture_dir,
            entry,
            label=f"depth_maps[{i}]",
        )
        if resolved is None:
            # ``_resolve_optional_path`` returns None only when the input
            # is None; inside a list that's a structural error.
            raise RealSceneInputError(
                f"depth_maps[{i}] must be a string path; got null",
            )
        out.append(resolved)
    return out


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def load_real_scene(capture_dir: Path) -> RealSceneInput:
    """Parse a structured capture-dir export into a :class:`RealSceneInput`.

    Expected on-disk shape::

        <capture_dir>/
            intrinsics.json    # required
            extrinsics.json    # required: list[list[list[float]]] (N x 4 x 4)
            manifest.json      # optional: declares depth_maps / point_cloud / metadata
            depth_maps/        # optional: per-frame depth-map sidecars
                <frame>.exr
            points.ply         # optional: aggregate point cloud

    The two ``.json`` manifests carry path strings (for ``depth_maps``
    and ``point_cloud``) and freeform metadata. Path resolution goes
    through :func:`gauntlet.security.safe_join` at every boundary so a
    malicious export with ``../etc/passwd`` in any path-typed field is
    rejected with a :class:`RealSceneInputError`.

    The convention is a relaxed superset of the COLMAP / NeRFStudio
    layout — pre-existing ``transforms.json`` exports can be converted
    to the gauntlet layout in a few lines (see :doc:`docs/realsim`).

    Args:
        capture_dir: directory holding the capture export. Must exist.

    Returns:
        A populated :class:`RealSceneInput`. Soft validation findings
        (e.g. depth maps declared but files missing on disk, extrinsics
        list non-contiguous) are *not* raised here — call
        :func:`validate_scene_input` on the returned object to surface
        them.

    Raises:
        RealSceneInputError: on missing required files, traversal
            attempts, JSON-decode failures, or schema-shape violations
            (intrinsics missing required keys, extrinsics not 4x4,
            non-finite values, etc.).
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise RealSceneInputError(f"capture_dir not found: {capture_dir}")
    capture_dir = capture_dir.resolve()

    # Required: intrinsics + extrinsics. Both surface as RealSceneInputError
    # on absence (different from ``depth_maps`` / ``points.ply`` which are
    # schema-optional).
    intrinsics_raw = _read_json(capture_dir, "intrinsics.json")
    intrinsics = _parse_intrinsics(intrinsics_raw)

    extrinsics_raw = _read_json(capture_dir, "extrinsics.json")
    if isinstance(extrinsics_raw, dict) and "frames" in extrinsics_raw:
        extrinsics_payload: object = extrinsics_raw["frames"]
    else:
        extrinsics_payload = extrinsics_raw
    extrinsics = _parse_extrinsics(extrinsics_payload)

    # Optional sidecars manifest (``manifest.json``) describing depth
    # maps, point cloud, and metadata. We use ``safe_join`` to detect
    # the file rather than a raw ``Path /``: same threat model.
    sidecars: dict[str, object] = {}
    sidecars_path = safe_join(capture_dir, "manifest.json")
    if sidecars_path.is_file():
        sidecars_raw = _read_json(capture_dir, "manifest.json")
        if not isinstance(sidecars_raw, dict):
            raise RealSceneInputError(
                f"manifest.json top-level must be an object; got {type(sidecars_raw).__name__}",
            )
        # Re-bind via dict() so we own the mapping (don't mutate the
        # caller's parsed JSON).
        sidecars = dict(sidecars_raw)

    depth_maps = _resolve_depth_maps(capture_dir, sidecars.get("depth_maps"))
    point_cloud = _resolve_optional_path(
        capture_dir,
        sidecars.get("point_cloud"),
        label="point_cloud",
    )

    metadata_raw = sidecars.get("metadata", {})
    if not isinstance(metadata_raw, dict):
        raise RealSceneInputError(
            f"manifest.json['metadata'] must be an object; got {type(metadata_raw).__name__}",
        )
    metadata: dict[str, object] = dict(metadata_raw)

    return RealSceneInput(
        capture_dir=capture_dir,
        intrinsics=intrinsics,
        extrinsics_per_frame=extrinsics,
        depth_maps=depth_maps,
        point_cloud=point_cloud,
        metadata=metadata,
    )


def validate_scene_input(scene: RealSceneInput) -> list[str]:
    """Surface soft validation warnings for a parsed :class:`RealSceneInput`.

    Returns a list of human-readable warning strings. An **empty list
    means valid**. Hard schema violations are caught earlier by
    :func:`load_real_scene` and surfaced as :class:`RealSceneInputError`;
    this function is for the soft, advisory checks a downstream
    consumer (CLI, dashboard, sweep loader) wants to surface to the
    user without aborting.

    All findings are reported in one pass (no early-return) so the user
    sees the full picture in one go. Order: intrinsics findings,
    extrinsics findings, depth-map findings, point-cloud findings.

    Currently surfaces:

    * intrinsics: optional distortion field present but mixed-length
      (e.g. ``k1`` without ``k2``).
    * extrinsics: fewer than 2 frames (a single-pose export is parseable
      but not useful).
    * depth maps: declared but length does not match
      :attr:`RealSceneInput.extrinsics_per_frame`.
    * depth maps: declared but a referenced file is missing on disk.
    * point cloud: declared but the file is missing on disk.
    * point cloud: depth maps present but no aggregate point cloud
      (advisory — most renderers prefer the aggregate when it exists).
    """
    warnings: list[str] = []

    # --- intrinsics -------------------------------------------------------
    distortion_keys = ("k1", "k2", "p1", "p2", "k3")
    present = [k for k in distortion_keys if k in scene.intrinsics]
    if present and len(present) < len(distortion_keys):
        missing = [k for k in distortion_keys if k not in scene.intrinsics]
        warnings.append(
            f"intrinsics declares partial distortion ({present!r}); "
            f"missing {missing!r} — renderer will assume zero for those",
        )

    # --- extrinsics -------------------------------------------------------
    if len(scene.extrinsics_per_frame) < 2:
        warnings.append(
            f"extrinsics_per_frame has only {len(scene.extrinsics_per_frame)} "
            f"frame(s); a multi-view reconstruction needs >= 2",
        )

    # --- depth maps -------------------------------------------------------
    if scene.depth_maps is not None:
        if len(scene.depth_maps) != len(scene.extrinsics_per_frame):
            warnings.append(
                f"depth_maps has {len(scene.depth_maps)} entries but "
                f"extrinsics_per_frame has {len(scene.extrinsics_per_frame)}; "
                f"renderer expects 1:1 alignment",
            )
        for i, dm in enumerate(scene.depth_maps):
            if not dm.is_file():
                warnings.append(
                    f"depth_maps[{i}] declared but file is missing on disk: {dm}",
                )

    # --- point cloud ------------------------------------------------------
    if scene.point_cloud is not None and not scene.point_cloud.is_file():
        warnings.append(
            f"point_cloud declared but file is missing on disk: {scene.point_cloud}",
        )
    if scene.depth_maps is not None and scene.depth_maps and scene.point_cloud is None:
        warnings.append(
            "depth_maps declared but no point_cloud aggregate; renderers "
            "typically prefer the aggregate when one exists",
        )

    return warnings
