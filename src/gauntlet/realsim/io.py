"""On-disk persistence for :class:`Scene` — round-trip via ``manifest.json``.

The scene directory layout produced by :func:`save_scene` and consumed
by :func:`load_scene`:

    <scene-dir>/
        manifest.json          # serialised :class:`Scene`
        frames/
            0001.png           # frame files referenced by the manifest
            0002.png
            ...

Every :attr:`CameraFrame.path` in the manifest is relative to the
scene directory root (POSIX separator), so the directory tarballs and
unpacks on a different machine without rewriting paths.

See ``docs/phase3-rfc-021-real-to-sim-stub.md`` §4.5 for the
"structural, not byte-identical" round-trip contract.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import ValidationError

from gauntlet.realsim.schema import Scene

__all__ = [
    "MANIFEST_FILENAME",
    "SceneIOError",
    "load_scene",
    "save_scene",
]


#: Canonical filename for the manifest written into a scene directory.
#: Pinned at module scope so downstream consumers can reference it
#: without hard-coding the string.
MANIFEST_FILENAME: str = "manifest.json"


class SceneIOError(ValueError):
    """Raised when :func:`save_scene` / :func:`load_scene` fails.

    Subclasses :class:`ValueError` so existing CLI ``except ValueError``
    handlers (e.g. ``gauntlet.cli._fail``) catch it without a code
    change.
    """


# ---------------------------------------------------------------------------
# save_scene.
# ---------------------------------------------------------------------------


def save_scene(
    scene: Scene,
    scene_dir: Path,
    *,
    frames_dir: Path | None = None,
    symlink: bool = False,
    overwrite: bool = False,
) -> Path:
    """Materialise *scene* on disk under *scene_dir*.

    Writes ``<scene-dir>/manifest.json`` and copies (or symlinks)
    each referenced frame from *frames_dir* into *scene_dir* using
    the manifest's relative paths.

    Args:
        scene: the :class:`Scene` to persist.
        scene_dir: output directory; created if missing.
        frames_dir: directory the manifest's frame paths are relative
            to *on the input side*. Defaults to *scene_dir* — useful
            when the user is re-saving an already-on-disk scene
            in-place.
        symlink: when True, symlink frames instead of copying. Saves
            disk for large dumps at the cost of cross-machine
            portability. Default False.
        overwrite: when True, allow writing into a non-empty
            *scene_dir*. Default False — refuses to overwrite an
            existing manifest so a typo doesn't clobber a sibling
            scene.

    Returns:
        The absolute path to the written manifest.

    Raises:
        SceneIOError: on any IO / validation failure. Wraps the
            underlying error so callers can ``except SceneIOError``
            uniformly.
    """
    scene_dir = Path(scene_dir)
    if frames_dir is None:
        frames_dir = scene_dir
    frames_dir = Path(frames_dir)

    scene_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = scene_dir / MANIFEST_FILENAME

    if manifest_path.exists() and not overwrite:
        raise SceneIOError(
            f"manifest already exists at {manifest_path}; pass overwrite=True to replace",
        )

    # Resolve once so we can detect in-place re-saves (input == output)
    # and skip the copy step for those frames — copying a file onto
    # itself raises ``SameFileError`` on most platforms, which would
    # surface as a confusing IO error otherwise.
    scene_dir_abs = scene_dir.resolve()
    frames_dir_abs = frames_dir.resolve()

    for frame in scene.frames:
        src = (frames_dir_abs / frame.path).resolve()
        dst = (scene_dir_abs / frame.path).resolve()
        if src == dst:
            # In-place re-save: nothing to copy.
            continue
        if not src.is_file():
            raise SceneIOError(
                f"frame {frame.path!r} not found in frames_dir: {src}",
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Remove any stale destination so symlink / copy never has to
        # negotiate with a previous file. ``missing_ok`` is 3.8+.
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            if symlink:
                dst.symlink_to(src)
            else:
                # ``copy2`` preserves mtime, which the dashboard's
                # time-series view (RFC 020 §6) uses as its x-axis when
                # the user later runs ``gauntlet dashboard build``.
                shutil.copy2(src, dst)
        except OSError as exc:
            raise SceneIOError(
                f"failed to {'symlink' if symlink else 'copy'} {src} -> {dst}: {exc}",
            ) from exc

    payload = scene.model_dump(mode="json")
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# load_scene.
# ---------------------------------------------------------------------------


def load_scene(scene_dir: Path) -> Scene:
    """Load a :class:`Scene` from *scene_dir*.

    Reads ``<scene-dir>/manifest.json`` and validates it through
    :class:`Scene`. Does **not** revalidate every frame's image
    bytes — the manifest is the source of truth at load time, mirroring
    how the rest of gauntlet treats serialised pydantic artefacts
    (e.g. :func:`gauntlet.report.Report.model_validate`).

    Args:
        scene_dir: directory holding the manifest and frame files.

    Returns:
        A :class:`Scene` matching the on-disk manifest.

    Raises:
        SceneIOError: when the manifest is missing, malformed JSON, or
            fails :class:`Scene` validation.
    """
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SceneIOError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SceneIOError(
            f"{manifest_path}: invalid JSON ({exc.msg} at line {exc.lineno})",
        ) from exc
    try:
        return Scene.model_validate(raw)
    except ValidationError as exc:
        raise SceneIOError(f"{manifest_path}: not a valid scene manifest: {exc}") from exc
