"""Branch-coverage backfill for :mod:`gauntlet.runner.video`.

Phase 2.5 Task 11. The end-to-end video tests live under
``tests/video/`` behind the ``video`` marker (deselected by default
because they require the ``[video]`` extra). This file covers the
same branches under the default test selection by mocking
``imageio.v3.imwrite`` — every code path inside
:meth:`gauntlet.runner.video.VideoWriter.write` runs without actually
encoding an MP4.

Targets ``video.py`` lines 122 (init guard), 128 (fps property), and
the full body of :meth:`write` (162-215): list/array normalisation,
dtype guard, shape guard, T=0 guard, parity-pad, parent-mkdir, and
the encode call itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.runner.video import VideoWriter, video_path_for

# ----------------------------------------------------------------------------
# video_path_for — pure deterministic filename builder.
# ----------------------------------------------------------------------------


def test_video_path_for_returns_canonical_filename(tmp_path: Path) -> None:
    p = video_path_for(tmp_path, cell_index=3, episode_index=7, seed=12345)
    assert p == tmp_path / "episode_cell0003_ep0007_seed12345.mp4"


def test_video_path_for_zero_pads_to_four_digits(tmp_path: Path) -> None:
    p = video_path_for(tmp_path, cell_index=0, episode_index=0, seed=1)
    assert p.name == "episode_cell0000_ep0000_seed1.mp4"


def test_video_path_for_distinguishes_seeds(tmp_path: Path) -> None:
    a = video_path_for(tmp_path, cell_index=1, episode_index=1, seed=10)
    b = video_path_for(tmp_path, cell_index=1, episode_index=1, seed=11)
    assert a != b


# ----------------------------------------------------------------------------
# VideoWriter.__init__ + .fps property — config validation (L121-128).
# ----------------------------------------------------------------------------


def test_video_writer_rejects_zero_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=0)


def test_video_writer_rejects_negative_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=-30)


def test_video_writer_rejects_float_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoWriter(fps=24.0)  # type: ignore[arg-type]


def test_video_writer_fps_property_round_trips() -> None:
    """L128: ``fps`` is a read-only property returning the constructor arg."""
    w = VideoWriter(fps=42)
    assert w.fps == 42


# ----------------------------------------------------------------------------
# VideoWriter.write — happy paths + every guard (L162-215). imageio is
# mocked so tests run in the default torch-/extras-free job.
# ----------------------------------------------------------------------------


@pytest.fixture
def _mock_imwrite(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``imageio.v3.imwrite`` regardless of whether the [video]
    extra is installed (real imageio) or not (mock-substitute).
    """
    mock = MagicMock()
    if "imageio" not in sys.modules:
        monkeypatch.setitem(sys.modules, "imageio", MagicMock())
    if "imageio.v3" not in sys.modules:
        monkeypatch.setitem(sys.modules, "imageio.v3", MagicMock())
    monkeypatch.setattr("imageio.v3.imwrite", mock)
    return mock


def _frames(t: int, h: int = 8, w: int = 8) -> list[np.ndarray]:
    return [np.full((h, w, 3), fill_value=i * 10, dtype=np.uint8) for i in range(t)]


def test_write_accepts_list_of_frames(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    out = tmp_path / "x.mp4"
    VideoWriter(fps=10).write(out, _frames(3))
    assert _mock_imwrite.call_count == 1
    args, kwargs = _mock_imwrite.call_args
    assert args[0] == out
    assert isinstance(args[1], np.ndarray)
    assert args[1].shape == (3, 8, 8, 3)
    assert args[1].dtype == np.uint8
    assert kwargs == {"fps": 10, "codec": "libx264", "macro_block_size": 1}


def test_write_accepts_4d_array(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    """A pre-stacked ``(T, H, W, 3)`` array bypasses the ``np.stack`` branch."""
    arr = np.stack(_frames(4), axis=0)
    out = tmp_path / "arr.mp4"
    VideoWriter(fps=8).write(out, arr)
    assert _mock_imwrite.call_count == 1
    written = _mock_imwrite.call_args.args[1]
    assert written.shape == (4, 8, 8, 3)


def test_write_creates_parent_directory(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    nested = tmp_path / "a" / "b" / "c"
    out = nested / "v.mp4"
    assert not nested.exists()
    VideoWriter(fps=10).write(out, _frames(2))
    assert nested.is_dir()


def test_write_pads_odd_height(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    """yuv420p needs even H — odd H is bottom-padded by one row."""
    odd = [np.full((9, 8, 3), fill_value=10, dtype=np.uint8) for _ in range(3)]
    VideoWriter(fps=10).write(tmp_path / "h.mp4", odd)
    written = _mock_imwrite.call_args.args[1]
    assert written.shape == (3, 10, 8, 3)


def test_write_pads_odd_width(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    odd = [np.full((8, 9, 3), fill_value=10, dtype=np.uint8) for _ in range(3)]
    VideoWriter(fps=10).write(tmp_path / "w.mp4", odd)
    written = _mock_imwrite.call_args.args[1]
    assert written.shape == (3, 8, 10, 3)


def test_write_pads_odd_height_and_width(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    odd = [np.full((9, 7, 3), fill_value=10, dtype=np.uint8) for _ in range(2)]
    VideoWriter(fps=10).write(tmp_path / "hw.mp4", odd)
    written = _mock_imwrite.call_args.args[1]
    assert written.shape == (2, 10, 8, 3)


def test_write_skips_padding_when_dimensions_already_even(
    tmp_path: Path, _mock_imwrite: MagicMock
) -> None:
    even = [np.full((8, 8, 3), fill_value=10, dtype=np.uint8) for _ in range(3)]
    VideoWriter(fps=10).write(tmp_path / "even.mp4", even)
    written = _mock_imwrite.call_args.args[1]
    # Same shape — no padding, no copy needed.
    assert written.shape == (3, 8, 8, 3)


def test_write_rejects_empty_list(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        VideoWriter(fps=10).write(tmp_path / "x.mp4", [])
    assert _mock_imwrite.call_count == 0


def test_write_rejects_zero_t_array(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    arr = np.zeros((0, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one frame"):
        VideoWriter(fps=10).write(tmp_path / "x.mp4", arr)


def test_write_rejects_non_uint8_dtype(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    bad = [np.zeros((8, 8, 3), dtype=np.float32) for _ in range(2)]
    with pytest.raises(ValueError, match="uint8"):
        VideoWriter(fps=10).write(tmp_path / "x.mp4", cast(Any, bad))


def test_write_rejects_2d_input(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    """``(T, H, W)`` without an RGB channel is rejected."""
    bad = [np.zeros((8, 8), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        VideoWriter(fps=10).write(tmp_path / "x.mp4", cast(Any, bad))


def test_write_rejects_non_rgb_channel(tmp_path: Path, _mock_imwrite: MagicMock) -> None:
    """Channel dim != 3 (e.g. RGBA) is rejected."""
    bad = [np.zeros((8, 8, 4), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\)"):
        VideoWriter(fps=10).write(tmp_path / "x.mp4", cast(Any, bad))
