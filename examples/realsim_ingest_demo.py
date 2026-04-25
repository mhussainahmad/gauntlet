"""Real-to-sim ingest demo — synthetic frames + calibration -> :class:`Scene`.

Usage:
    uv run python examples/realsim_ingest_demo.py [--out OUT_DIR]

Demonstrates the input contract the future gaussian-splatting renderer
will consume: a directory of camera frames + a calibration spec
(intrinsics-by-id + per-frame poses + paths) goes in, a fully
validated :class:`gauntlet.realsim.Scene` comes out, and the round-trip
through :func:`save_scene` / :func:`load_scene` produces a portable
``manifest.json`` directory that can be tarballed onto a different
machine without rewriting any path.

Pipeline:

1. Materialise three hand-rolled 1x1 PNG frames into ``frames_dir``
   (no Pillow dependency — the bytes are built from the PNG spec).
2. Build a calibration dict in memory (the ``ingest_frames`` happy
   path also accepts a ``Path`` to a JSON file; we use the dict path
   here because it keeps the example self-contained).
3. Call :func:`gauntlet.realsim.ingest_frames` to validate everything
   and produce a :class:`Scene`.
4. Persist the scene to ``scene_dir`` via
   :func:`gauntlet.realsim.save_scene` — this writes ``manifest.json``
   and copies each frame into the destination using the manifest's
   relative paths.
5. Round-trip through :func:`gauntlet.realsim.load_scene` to confirm
   the on-disk artefact is consumable end-to-end.
6. Print a summary the future renderer would consume.

The renderer itself is deferred (see RFC 021 §1 / §2); this script
exercises the input contract that any concrete renderer will need to
honour.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from typing import Any

from gauntlet.realsim import (
    SCENE_SCHEMA_VERSION,
    Scene,
    ingest_frames,
    load_scene,
    save_scene,
)

__all__ = ["main"]


_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_OUT: Path = _REPO_ROOT / "out-realsim"


def _png_chunk(name: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk: length prefix + name + data + CRC."""
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _build_minimal_png() -> bytes:
    """Build the smallest valid RGB PNG (1x1, single red pixel).

    Mirrors the helper in ``tests/test_realsim_pipeline.py`` so the
    demo passes the same magic-byte validation the unit tests pin.
    No Pillow / imageio dependency: the bytes follow the PNG spec
    directly.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: 1x1, bit-depth 8, colour-type 2 (RGB), default
    # compression / filter / interlace.
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # IDAT: filter byte 0 followed by one RGB pixel (0xFF 0x00 0x00 = red).
    idat = _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _identity_pose() -> list[list[float]]:
    """4x4 identity transform — placeholder for a real per-frame pose."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _wrist_intrinsics() -> dict[str, Any]:
    """Pinhole intrinsics for a synthetic 640x480 wrist camera."""
    return {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
    }


def _materialise_frames(frames_dir: Path, n_frames: int) -> list[str]:
    """Write ``n_frames`` 1x1 PNGs into *frames_dir*; return the rel paths."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    blob = _build_minimal_png()
    rel_paths: list[str] = []
    for i in range(n_frames):
        rel = f"{i:04d}.png"
        (frames_dir / rel).write_bytes(blob)
        rel_paths.append(rel)
    return rel_paths


def _build_calibration(rel_paths: list[str]) -> dict[str, Any]:
    """Assemble the calibration dict ``ingest_frames`` consumes.

    Same vocabulary as the on-disk manifest (intrinsics-by-id +
    per-frame entries) so a real calibration JSON file would have the
    identical shape — see ``docs/phase3-rfc-021-real-to-sim-stub.md``
    §7 for the full schema.
    """
    return {
        "intrinsics": {"wrist": _wrist_intrinsics()},
        "frames": [
            {
                "path": rel,
                "timestamp": float(i),
                "intrinsics_id": "wrist",
                "pose": _identity_pose(),
            }
            for i, rel in enumerate(rel_paths)
        ],
    }


def main(*, out_dir: Path = _DEFAULT_OUT, n_frames: int = 3) -> None:
    """Run the ingest -> save -> load round-trip end-to-end.

    Args:
        out_dir: directory the demo writes into. ``out_dir/frames``
            holds the synthetic input frames; ``out_dir/scene`` holds
            the persisted manifest + frame copies. Created if missing.
        n_frames: how many synthetic frames to materialise. The
            default of 3 is enough to exercise the multi-frame path
            without making the example slow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    scene_dir = out_dir / "scene"

    # Wipe a previous scene_dir so save_scene's overwrite=False stays
    # honest across re-runs of the example.
    if scene_dir.exists():
        for child in sorted(scene_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        scene_dir.rmdir()

    # Step 1 + 2: synthesise frames + calibration.
    rel_paths = _materialise_frames(frames_dir, n_frames=n_frames)
    calib = _build_calibration(rel_paths)
    print(f"Materialised {n_frames} synthetic 1x1 PNG frames under {frames_dir}")

    # Step 3: ingest -> Scene. ``source`` is freeform metadata; in a
    # real pipeline this would be the robot id or log id the frames
    # came from.
    scene: Scene = ingest_frames(frames_dir, calib, source="example-robot-001")
    print(
        f"Ingested scene: schema v{scene.version}, source={scene.source!r}, "
        f"{len(scene.intrinsics)} intrinsics, {len(scene.frames)} frames."
    )

    # Step 4: persist to disk. ``frames_dir`` is the input root the
    # manifest's relative paths resolve against; ``scene_dir`` is the
    # output root that ends up portable.
    manifest_path = save_scene(scene, scene_dir, frames_dir=frames_dir)
    print(f"Wrote manifest -> {manifest_path}")

    # Step 5: round-trip — load_scene reads the manifest and validates
    # it through Scene; structural equality is the round-trip contract
    # (see RFC 021 §4.5).
    reloaded = load_scene(scene_dir)
    assert reloaded.version == scene.version
    assert reloaded.source == scene.source
    assert len(reloaded.frames) == len(scene.frames)
    assert set(reloaded.intrinsics.keys()) == set(scene.intrinsics.keys())
    print(
        f"Round-trip OK: load_scene({scene_dir}) returned a Scene with "
        f"{len(reloaded.frames)} frames matching the original."
    )

    # Step 6: summary — what a renderer plugin would consume next.
    first_frame = reloaded.frames[0]
    intrinsics = reloaded.intrinsics[first_frame.intrinsics_id]
    print(
        f"  schema_version: {SCENE_SCHEMA_VERSION} (current); "
        f"first frame: path={first_frame.path!r}, "
        f"timestamp={first_frame.timestamp:.3f}s, "
        f"intrinsics_id={first_frame.intrinsics_id!r} "
        f"({intrinsics.width}x{intrinsics.height} @ "
        f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f})."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-to-sim ingest demo: synthetic frames + calibration -> Scene.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=3,
        help="Number of synthetic frames to ingest (default: 3).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(out_dir=args.out, n_frames=args.n_frames)
