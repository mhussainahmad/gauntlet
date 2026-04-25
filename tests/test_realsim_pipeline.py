"""Pipeline + IO tests for the real-to-sim stub — Phase 3 Task 18.

Drives :func:`ingest_frames` over a synthetic directory of 1x1 PNG
frames + a calibration JSON, then exercises the
:func:`save_scene` / :func:`load_scene` round-trip. No Pillow
dependency: the PNG bytes are hand-built from the spec.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from gauntlet.realsim import (
    SCENE_SCHEMA_VERSION,
    IngestionError,
    Scene,
    SceneIOError,
    ingest_frames,
    load_scene,
    save_scene,
)
from gauntlet.realsim.pipeline import IMAGE_MAGIC_BYTES, _recognise_image

# ---------------------------------------------------------------------------
# Test-helper: hand-rolled 1x1 PNG so we don't pull in Pillow.
# ---------------------------------------------------------------------------


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _build_minimal_png() -> bytes:
    """Build the smallest valid RGB PNG (1x1, single red pixel)."""
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 1x1, bit-depth 8, colour-type 2 (RGB), default compression /
    # filter / interlace.
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: filter byte 0 followed by one RGB pixel.
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


PNG_BLOB = _build_minimal_png()


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _good_intrinsics_dict() -> dict[str, Any]:
    return {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
    }


def _write_calib(
    tmp_path: Path,
    *,
    n_frames: int = 2,
    intrinsics_id: str = "wrist",
    bad_id_for_frame: int | None = None,
) -> tuple[Path, Path]:
    """Materialise a frames dir + calibration JSON in *tmp_path*.

    Returns ``(frames_dir, calib_path)``. Each frame is a hand-built
    1x1 PNG so no Pillow / ffmpeg is required.
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames: list[dict[str, Any]] = []
    for i in range(n_frames):
        rel = f"{i:04d}.png"
        (frames_dir / rel).write_bytes(PNG_BLOB)
        ref = "ghost" if bad_id_for_frame is not None and i == bad_id_for_frame else intrinsics_id
        frames.append(
            {
                "path": rel,
                "timestamp": float(i),
                "intrinsics_id": ref,
                "pose": _identity_matrix(),
            }
        )
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps(
            {
                "intrinsics": {intrinsics_id: _good_intrinsics_dict()},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return frames_dir, calib_path


# ---------------------------------------------------------------------------
# _recognise_image.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload_prefix", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "jpeg"),
        (b"P6\n# binary ppm\n", "ppm-binary"),
        (b"P3\n# ascii ppm\n", "ppm-ascii"),
    ],
)
def test_recognise_image_known_containers(payload_prefix: bytes, expected: str) -> None:
    assert _recognise_image(payload_prefix) == expected


def test_recognise_image_unknown_returns_none() -> None:
    assert _recognise_image(b"\x00\x00\x00\x00\x00\x00\x00\x00") is None
    assert _recognise_image(b"") is None


def test_image_magic_bytes_constants_exist() -> None:
    """The public constant carries every container the pipeline accepts."""
    assert {"png", "jpeg", "ppm-binary", "ppm-ascii"} <= set(IMAGE_MAGIC_BYTES)


# ---------------------------------------------------------------------------
# ingest_frames — happy path.
# ---------------------------------------------------------------------------


def test_ingest_frames_happy_path(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=3)
    scene = ingest_frames(frames_dir, calib_path, source="customer-A")
    assert isinstance(scene, Scene)
    assert scene.version == SCENE_SCHEMA_VERSION
    assert scene.source == "customer-A"
    assert len(scene.frames) == 3
    assert scene.frames[0].path == "0000.png"
    assert scene.frames[2].timestamp == 2.0
    assert scene.intrinsics["wrist"].width == 640


def test_ingest_frames_accepts_dict_calib(tmp_path: Path) -> None:
    """A dict calib is convenient for notebook callers — RFC §7."""
    frames_dir, _ = _write_calib(tmp_path, n_frames=1)
    calib_dict = {
        "intrinsics": {"wrist": _good_intrinsics_dict()},
        "frames": [
            {
                "path": "0000.png",
                "timestamp": 0.0,
                "intrinsics_id": "wrist",
                "pose": _identity_matrix(),
            }
        ],
    }
    scene = ingest_frames(frames_dir, calib_dict)
    assert scene.source is None
    assert len(scene.frames) == 1


def test_ingest_frames_pose_as_dict_also_accepted(tmp_path: Path) -> None:
    """``pose`` may be a list (canonical) or a ``{"matrix": ...}`` dict."""
    frames_dir, _ = _write_calib(tmp_path, n_frames=1)
    calib_dict = {
        "intrinsics": {"wrist": _good_intrinsics_dict()},
        "frames": [
            {
                "path": "0000.png",
                "timestamp": 0.0,
                "intrinsics_id": "wrist",
                "pose": {"matrix": _identity_matrix()},
            }
        ],
    }
    scene = ingest_frames(frames_dir, calib_dict)
    assert scene.frames[0].pose.matrix[0][0] == 1.0


# ---------------------------------------------------------------------------
# ingest_frames — error paths.
# ---------------------------------------------------------------------------


def test_ingest_frames_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"
    with pytest.raises(IngestionError, match=r"frames_dir not found"):
        ingest_frames(missing, {"intrinsics": {}, "frames": []})


def test_ingest_frames_missing_calib_file_raises(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    with pytest.raises(IngestionError, match=r"calibration file not found"):
        ingest_frames(frames_dir, tmp_path / "no-calib.json")


def test_ingest_frames_invalid_calib_json_raises(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    bad = tmp_path / "calib.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(IngestionError, match=r"invalid JSON"):
        ingest_frames(frames_dir, bad)


def test_ingest_frames_calib_top_level_must_be_object(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    bad = tmp_path / "calib.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(IngestionError, match=r"top-level JSON must be an object"):
        ingest_frames(frames_dir, bad)


def test_ingest_frames_missing_intrinsics_block(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"frames": []}), encoding="utf-8")
    with pytest.raises(IngestionError, match=r"missing 'intrinsics'"):
        ingest_frames(frames_dir, calib)


def test_ingest_frames_missing_frames_block(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    calib = tmp_path / "calib.json"
    calib.write_text(
        json.dumps({"intrinsics": {"wrist": _good_intrinsics_dict()}}),
        encoding="utf-8",
    )
    with pytest.raises(IngestionError, match=r"missing 'frames'"):
        ingest_frames(frames_dir, calib)


def test_ingest_frames_empty_intrinsics_rejected(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    with pytest.raises(IngestionError, match=r"at least one camera"):
        ingest_frames(frames_dir, {"intrinsics": {}, "frames": []})


def test_ingest_frames_unknown_intrinsics_id_named_in_error(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=2, bad_id_for_frame=1)
    with pytest.raises(IngestionError) as exc:
        ingest_frames(frames_dir, calib_path)
    msg = str(exc.value)
    assert "ghost" in msg
    assert "frames'][1]" in msg


def test_ingest_frames_bad_intrinsics_block_includes_id(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    calib_dict = {
        "intrinsics": {
            "broken": {**_good_intrinsics_dict(), "fx": -10.0},
        },
        "frames": [],
    }
    with pytest.raises(IngestionError, match=r"intrinsics.*broken"):
        ingest_frames(frames_dir, calib_dict)


def test_ingest_frames_bad_pose_includes_frame_index(tmp_path: Path) -> None:
    frames_dir, _ = _write_calib(tmp_path, n_frames=2)
    bad_matrix = _identity_matrix()
    bad_matrix[3][3] = 999.0  # bottom-row tolerance violation
    calib_dict = {
        "intrinsics": {"wrist": _good_intrinsics_dict()},
        "frames": [
            {
                "path": "0000.png",
                "timestamp": 0.0,
                "intrinsics_id": "wrist",
                "pose": _identity_matrix(),
            },
            {
                "path": "0001.png",
                "timestamp": 1.0,
                "intrinsics_id": "wrist",
                "pose": bad_matrix,
            },
        ],
    }
    with pytest.raises(IngestionError) as exc:
        ingest_frames(frames_dir, calib_dict)
    assert "frames'][1]" in str(exc.value)


def test_ingest_frames_missing_frame_file(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Reference a frame file we never wrote.
    calib_dict = {
        "intrinsics": {"wrist": _good_intrinsics_dict()},
        "frames": [
            {
                "path": "missing.png",
                "timestamp": 0.0,
                "intrinsics_id": "wrist",
                "pose": _identity_matrix(),
            }
        ],
    }
    with pytest.raises(IngestionError) as exc:
        ingest_frames(frames_dir, calib_dict)
    assert "missing.png" in str(exc.value)


def test_ingest_frames_unrecognised_image_bytes(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Write a non-image file at the path the calib points at.
    (frames_dir / "0000.png").write_bytes(b"\x00\x00\x00\x00not an image")
    calib_dict = {
        "intrinsics": {"wrist": _good_intrinsics_dict()},
        "frames": [
            {
                "path": "0000.png",
                "timestamp": 0.0,
                "intrinsics_id": "wrist",
                "pose": _identity_matrix(),
            }
        ],
    }
    with pytest.raises(IngestionError) as exc:
        ingest_frames(frames_dir, calib_dict)
    assert "0000.png" in str(exc.value)
    assert "first 8 bytes" in str(exc.value)


def test_ingest_frames_bad_calib_type_argument(tmp_path: Path) -> None:
    """A non-Path / non-dict calib argument is rejected with a clear message."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    with pytest.raises(IngestionError, match=r"must be a Path or dict"):
        ingest_frames(frames_dir, ["not", "a", "dict"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# save_scene / load_scene round-trip.
# ---------------------------------------------------------------------------


def test_save_load_scene_round_trip(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=2)
    scene = ingest_frames(frames_dir, calib_path, source="round-trip")
    out = tmp_path / "scene-out"
    save_scene(scene, out, frames_dir=frames_dir)

    # Manifest + frames materialised on disk.
    assert (out / "manifest.json").is_file()
    assert (out / "0000.png").is_file()
    assert (out / "0001.png").is_file()

    rebuilt = load_scene(out)
    assert rebuilt == scene


def test_save_scene_refuses_overwrite_by_default(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=1)
    scene = ingest_frames(frames_dir, calib_path)
    out = tmp_path / "scene-out"
    save_scene(scene, out, frames_dir=frames_dir)
    with pytest.raises(SceneIOError, match=r"already exists"):
        save_scene(scene, out, frames_dir=frames_dir)


def test_save_scene_overwrite_true_replaces_manifest(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=1)
    scene = ingest_frames(frames_dir, calib_path)
    out = tmp_path / "scene-out"
    save_scene(scene, out, frames_dir=frames_dir)
    save_scene(scene, out, frames_dir=frames_dir, overwrite=True)
    rebuilt = load_scene(out)
    assert rebuilt == scene


def test_save_scene_symlink_mode_creates_symlinks(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=2)
    scene = ingest_frames(frames_dir, calib_path)
    out = tmp_path / "scene-out"
    save_scene(scene, out, frames_dir=frames_dir, symlink=True)
    linked = out / "0000.png"
    assert linked.is_symlink()
    assert linked.resolve() == (frames_dir / "0000.png").resolve()


def test_save_scene_missing_frame_in_frames_dir(tmp_path: Path) -> None:
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=1)
    scene = ingest_frames(frames_dir, calib_path)
    # Empty frames_dir on the save side -> save_scene cannot find the frame.
    empty_frames = tmp_path / "empty"
    empty_frames.mkdir()
    out = tmp_path / "scene-out"
    with pytest.raises(SceneIOError, match=r"not found in frames_dir"):
        save_scene(scene, out, frames_dir=empty_frames)


def test_load_scene_missing_manifest(tmp_path: Path) -> None:
    out = tmp_path / "empty-scene"
    out.mkdir()
    with pytest.raises(SceneIOError, match=r"manifest not found"):
        load_scene(out)


def test_load_scene_invalid_json(tmp_path: Path) -> None:
    out = tmp_path / "scene-out"
    out.mkdir()
    (out / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SceneIOError, match=r"invalid JSON"):
        load_scene(out)


def test_load_scene_invalid_schema(tmp_path: Path) -> None:
    out = tmp_path / "scene-out"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    with pytest.raises(SceneIOError, match=r"not a valid scene manifest"):
        load_scene(out)


def test_save_scene_in_place_re_save(tmp_path: Path) -> None:
    """save_scene with frames_dir == scene_dir doesn't try to copy files onto themselves."""
    frames_dir, calib_path = _write_calib(tmp_path, n_frames=1)
    scene = ingest_frames(frames_dir, calib_path)
    save_scene(scene, frames_dir, frames_dir=frames_dir)
    # Manifest landed alongside the original frame.
    assert (frames_dir / "manifest.json").is_file()
    assert (frames_dir / "0000.png").is_file()
    rebuilt = load_scene(frames_dir)
    assert rebuilt == scene
