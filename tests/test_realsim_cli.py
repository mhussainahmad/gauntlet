"""CLI tests for ``gauntlet realsim`` — Phase 3 Task 18.

Drives the typer ``app`` end-to-end against synthetic frames + a
calibration JSON. Reuses the hand-rolled 1x1 PNG strategy from
:mod:`tests.test_realsim_pipeline` so the suite has no Pillow / ffmpeg
dependency.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.realsim import MANIFEST_FILENAME, load_scene


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# 1x1 PNG (no Pillow). Same construction as test_realsim_pipeline.
# ---------------------------------------------------------------------------


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _build_minimal_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
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


def _materialise_inputs(tmp_path: Path, *, n_frames: int = 2) -> tuple[Path, Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames: list[dict[str, Any]] = []
    for i in range(n_frames):
        rel = f"{i:04d}.png"
        (frames_dir / rel).write_bytes(PNG_BLOB)
        frames.append(
            {
                "path": rel,
                "timestamp": float(i),
                "intrinsics_id": "wrist",
                "pose": _identity_matrix(),
            }
        )
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps(
            {
                "intrinsics": {"wrist": _good_intrinsics_dict()},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return frames_dir, calib_path


# ---------------------------------------------------------------------------
# `realsim --help` / subcommand discovery.
# ---------------------------------------------------------------------------


def test_top_level_help_lists_realsim(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    assert "realsim" in result.stdout


def test_realsim_help_lists_ingest_and_info(runner: CliRunner) -> None:
    result = runner.invoke(app, ["realsim", "--help"])
    assert result.exit_code == 0, result.stderr
    assert "ingest" in result.stdout
    assert "info" in result.stdout


def test_realsim_ingest_help_shows_options(runner: CliRunner) -> None:
    result = runner.invoke(app, ["realsim", "ingest", "--help"])
    assert result.exit_code == 0, result.stderr
    for token in ("--calib", "--out", "--source", "--symlink", "--overwrite"):
        assert token in result.stdout


# ---------------------------------------------------------------------------
# `realsim ingest` — happy path.
# ---------------------------------------------------------------------------


def test_realsim_ingest_writes_manifest_and_frames(runner: CliRunner, tmp_path: Path) -> None:
    frames_dir, calib_path = _materialise_inputs(tmp_path, n_frames=3)
    out_dir = tmp_path / "scene-out"

    result = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
            "--source",
            "customer-A-2026-04-23",
        ],
    )
    assert result.exit_code == 0, result.stderr

    # Manifest exists and round-trips.
    manifest_path = out_dir / MANIFEST_FILENAME
    assert manifest_path.is_file()
    scene = load_scene(out_dir)
    assert scene.source == "customer-A-2026-04-23"
    assert len(scene.frames) == 3
    # Frame files materialised by ``save_scene``.
    assert (out_dir / "0000.png").is_file()
    assert (out_dir / "0002.png").is_file()
    # Stderr summary names the manifest.
    assert "manifest.json" in result.stderr
    assert "3 frames" in result.stderr


def test_realsim_ingest_symlink_mode(runner: CliRunner, tmp_path: Path) -> None:
    frames_dir, calib_path = _materialise_inputs(tmp_path, n_frames=1)
    out_dir = tmp_path / "scene-out"

    result = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
            "--symlink",
        ],
    )
    assert result.exit_code == 0, result.stderr
    linked = out_dir / "0000.png"
    assert linked.is_symlink()
    assert linked.resolve() == (frames_dir / "0000.png").resolve()


def test_realsim_ingest_overwrite_replaces_existing_manifest(
    runner: CliRunner, tmp_path: Path
) -> None:
    frames_dir, calib_path = _materialise_inputs(tmp_path, n_frames=1)
    out_dir = tmp_path / "scene-out"

    first = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
        ],
    )
    assert first.exit_code == 0, first.stderr

    second = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
        ],
    )
    # Default -> refuse (because manifest exists).
    assert second.exit_code != 0
    assert "already exists" in second.stderr

    third = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
            "--overwrite",
        ],
    )
    assert third.exit_code == 0, third.stderr


# ---------------------------------------------------------------------------
# `realsim ingest` — error paths.
# ---------------------------------------------------------------------------


def test_realsim_ingest_missing_frames_dir_errors_cleanly(
    runner: CliRunner, tmp_path: Path
) -> None:
    _, calib_path = _materialise_inputs(tmp_path, n_frames=1)
    missing = tmp_path / "no-such-dir"
    result = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(missing),
            "--calib",
            str(calib_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "frames_dir not found" in result.stderr


def test_realsim_ingest_missing_calib_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    frames_dir, _ = _materialise_inputs(tmp_path, n_frames=1)
    result = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(tmp_path / "no-calib.json"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "calibration file not found" in result.stderr


def test_realsim_ingest_unknown_intrinsics_id_errors_cleanly(
    runner: CliRunner, tmp_path: Path
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "0000.png").write_bytes(PNG_BLOB)
    calib = tmp_path / "calib.json"
    calib.write_text(
        json.dumps(
            {
                "intrinsics": {"wrist": _good_intrinsics_dict()},
                "frames": [
                    {
                        "path": "0000.png",
                        "timestamp": 0.0,
                        "intrinsics_id": "ghost",
                        "pose": _identity_matrix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "ghost" in result.stderr


# ---------------------------------------------------------------------------
# `realsim info`.
# ---------------------------------------------------------------------------


def test_realsim_info_summary(runner: CliRunner, tmp_path: Path) -> None:
    # First ingest a scene so we have a manifest to inspect.
    frames_dir, calib_path = _materialise_inputs(tmp_path, n_frames=4)
    out_dir = tmp_path / "scene-out"
    ingest = runner.invoke(
        app,
        [
            "realsim",
            "ingest",
            str(frames_dir),
            "--calib",
            str(calib_path),
            "--out",
            str(out_dir),
            "--source",
            "log-2026-04-24",
        ],
    )
    assert ingest.exit_code == 0, ingest.stderr

    info = runner.invoke(app, ["realsim", "info", str(out_dir)])
    assert info.exit_code == 0, info.stderr
    body = info.stderr
    assert "version: 1" in body
    assert "log-2026-04-24" in body
    assert "intrinsics: 1" in body
    assert "wrist" in body
    assert "frames: 4" in body
    assert "time range:" in body


def test_realsim_info_missing_dir_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["realsim", "info", str(tmp_path / "missing")])
    assert result.exit_code != 0
    assert "scene_dir not found" in result.stderr


def test_realsim_info_missing_manifest_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    empty = tmp_path / "empty-scene"
    empty.mkdir()
    result = runner.invoke(app, ["realsim", "info", str(empty)])
    assert result.exit_code != 0
    assert "manifest not found" in result.stderr
