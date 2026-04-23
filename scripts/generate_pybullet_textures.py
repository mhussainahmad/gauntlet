"""One-shot utility — emits the two PNG textures shipped with the
PyBullet backend (RFC-005 §5.1 / §6.1).

Run once; the output files are committed to the repo. Regenerate only
if the design changes.

Usage:
    uv run python scripts/generate_pybullet_textures.py
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def _solid_rgba_png(r: int, g: int, b: int, a: int, *, size: int = 2) -> bytes:
    """Return a PNG byte string for a solid ``size x size`` RGBA image."""
    # PNG rows are each prefixed by a 1-byte filter code (0 = None).
    pixel = bytes([r, g, b, a])
    row = b"\x00" + pixel * size
    raw = row * size
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(
        ">IIBBBBB",
        size,  # width
        size,  # height
        8,  # bit depth
        6,  # colour type — 6 = RGBA
        0,  # compression method
        0,  # filter method
        0,  # interlace method
    )
    idat = zlib.compress(raw, level=9)
    return (
        signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    )


def main() -> int:
    assets = (
        Path(__file__).resolve().parent.parent / "src" / "gauntlet" / "env" / "pybullet" / "assets"
    )
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "cube_default.png").write_bytes(
        _solid_rgba_png(220, 40, 40, 255)  # red
    )
    (assets / "cube_alt.png").write_bytes(
        _solid_rgba_png(40, 180, 60, 255)  # green
    )
    print(f"wrote {assets}/cube_default.png")
    print(f"wrote {assets}/cube_alt.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
