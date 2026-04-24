"""Unit tests for :class:`gauntlet.runner.video.VideoWriter`.

Marked ``@pytest.mark.video`` — runs in the dedicated ``video-tests``
CI job (which installs the ``[video]`` extra). On the default torch-
/extras-free job these tests are deselected by marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from gauntlet.runner.video import VideoWriter, video_path_for

pytestmark = pytest.mark.video


# ────────────────────────────────────────────────────────────────────────
# video_path_for — pure-string contract
# ────────────────────────────────────────────────────────────────────────


def test_video_path_for_is_deterministic(tmp_path: Path) -> None:
    """Same (cell, episode, seed) -> same path; different seeds -> different paths."""
    p1 = video_path_for(tmp_path, cell_index=3, episode_index=7, seed=12345)
    p2 = video_path_for(tmp_path, cell_index=3, episode_index=7, seed=12345)
    assert p1 == p2
    assert p1.name == "episode_cell0003_ep0007_seed12345.mp4"

    # Different seed -> different filename so workers cannot collide.
    p3 = video_path_for(tmp_path, cell_index=3, episode_index=7, seed=99)
    assert p3 != p1


def test_video_path_for_zero_pads_indices(tmp_path: Path) -> None:
    """4-digit zero-pad on cell/episode keeps lexicographic sort honest."""
    p = video_path_for(tmp_path, cell_index=12, episode_index=0, seed=1)
    assert "cell0012_ep0000_seed1.mp4" in p.name


# ────────────────────────────────────────────────────────────────────────
# VideoWriter — config validation
# ────────────────────────────────────────────────────────────────────────


def test_video_writer_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=0)
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=-30)


def test_video_writer_rejects_non_int_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=30.0)  # type: ignore[arg-type]


def test_video_writer_fps_round_trips() -> None:
    w = VideoWriter(fps=24)
    assert w.fps == 24


# ────────────────────────────────────────────────────────────────────────
# VideoWriter.write — actual MP4 emission
# ────────────────────────────────────────────────────────────────────────


def _make_frames(t: int, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """Return *t* uint8 RGB frames with a moving square so x264 has signal.

    A solid colour compresses to a degenerate file size; the moving
    square keeps the encoder honest and matches what a real rollout
    produces.
    """
    frames: list[np.ndarray] = []
    for i in range(t):
        frame = np.full((h, w, 3), fill_value=20 + i * 5, dtype=np.uint8)
        # 8x8 white square that drifts diagonally.
        x = (i * 4) % (w - 8)
        y = (i * 4) % (h - 8)
        frame[y : y + 8, x : x + 8, :] = 255
        frames.append(frame)
    return frames


def test_video_writer_emits_nonempty_mp4(tmp_path: Path) -> None:
    """End-to-end: write 8 frames, assert the file exists and is non-empty."""
    out = tmp_path / "videos" / "test.mp4"  # parent dir does NOT exist
    writer = VideoWriter(fps=10)
    writer.write(out, _make_frames(8))

    assert out.exists(), "VideoWriter.write should have created the MP4"
    assert out.stat().st_size > 0, "MP4 should be non-empty"
    # Quick header sniff: every MP4 starts with an ``ftyp`` box at byte 4.
    with out.open("rb") as fh:
        header = fh.read(12)
    assert b"ftyp" in header, f"file does not look like an MP4: {header!r}"


def test_video_writer_creates_parent_dir(tmp_path: Path) -> None:
    """Belt-and-braces: the parent directory is created if missing."""
    nested = tmp_path / "a" / "b" / "c"
    out = nested / "x.mp4"
    assert not nested.exists()
    VideoWriter(fps=10).write(out, _make_frames(4))
    assert nested.is_dir()
    assert out.exists()


def test_video_writer_accepts_4d_array(tmp_path: Path) -> None:
    """A pre-stacked ``(T, H, W, 3)`` ndarray is accepted alongside lists."""
    arr = np.stack(_make_frames(4), axis=0)
    assert arr.ndim == 4 and arr.shape[-1] == 3
    out = tmp_path / "arr.mp4"
    VideoWriter(fps=10).write(out, arr)
    assert out.exists()


def test_video_writer_pads_odd_dimensions(tmp_path: Path) -> None:
    """yuv420p needs even H/W; 31x31 frames must still encode."""
    frames = _make_frames(4, h=31, w=31)
    out = tmp_path / "odd.mp4"
    VideoWriter(fps=10).write(out, frames)
    assert out.exists()


def test_video_writer_rejects_empty_frame_list(tmp_path: Path) -> None:
    out = tmp_path / "empty.mp4"
    with pytest.raises(ValueError, match="at least one frame"):
        VideoWriter(fps=10).write(out, [])


def test_video_writer_rejects_non_uint8(tmp_path: Path) -> None:
    """Float frames are a category error — caller must cast first."""
    bad = [np.zeros((16, 16, 3), dtype=np.float32) for _ in range(4)]
    out = tmp_path / "bad.mp4"
    # ``cast(Any, ...)`` bypasses the static type check so we can verify
    # the RUNTIME dtype validation path — that is the whole point of
    # this test and mirrors the pattern from tests/test_replay.py.
    with pytest.raises(ValueError, match="uint8"):
        VideoWriter(fps=10).write(out, cast(Any, bad))


def test_video_writer_rejects_wrong_shape(tmp_path: Path) -> None:
    """Shape (T, H, W) without an RGB channel is rejected."""
    bad = [np.zeros((16, 16), dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "bad.mp4"
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        VideoWriter(fps=10).write(out, cast(Any, bad))


def test_video_writer_rejects_zero_t_array(tmp_path: Path) -> None:
    """A pre-stacked array with T=0 is rejected with the same message."""
    arr = np.zeros((0, 16, 16, 3), dtype=np.uint8)
    out = tmp_path / "zero.mp4"
    with pytest.raises(ValueError, match="at least one frame"):
        VideoWriter(fps=10).write(out, arr)
