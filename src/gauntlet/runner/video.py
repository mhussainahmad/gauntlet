"""MP4 video writer for the Polish "rollout video recording" feature.

Wraps the optional ``imageio[ffmpeg]`` backend behind a lazy import so
the default (extras-free) install never sees ``imageio`` at module
import time. The default ``Runner(record_video=False)`` code path does
not import this module — but a ``from gauntlet.runner.video import
VideoWriter`` from a torch-/extras-free interpreter is also safe; the
import succeeds and the lazy hint surfaces only when the user calls
:meth:`VideoWriter.write` without the ``[video]`` extra installed.

Memory model
------------
The Runner buffers ``obs["image"]`` arrays per step in the worker, then
hands the buffered list to :meth:`VideoWriter.write` after the rollout
completes. Peak memory **per in-flight episode per worker**:

    H * W * 3 * max_steps   bytes
    224 * 224 * 3 * 200   ~= 30 MB    (default tabletop)
    224 * 224 * 3 *  20   ~=  3 MB    (test fixtures)

For ``n_workers`` workers in flight at once, total peak is
``n_workers * H * W * 3 * max_steps`` bytes. ``record_only_failures=
True`` only suppresses the *write*; the buffer still grows during the
rollout because success is not known until the final ``info["success"]``
arrives.

Backwards-compat / library choice
---------------------------------
``imageio[ffmpeg]>=2.34,<3`` is the dependency. The ``[ffmpeg]`` extra
pulls ``imageio-ffmpeg``, which bundles a static ffmpeg binary — no
system ffmpeg install required. See the partner
``docs/polish-exploration-rollout-video.md`` for the av/OpenCV/manual-
ffmpeg trade-off analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing-only
    pass

__all__ = ["VideoWriter", "video_path_for"]


# Filename template used by the Runner. Centralised here so the test
# suite can pin the exact contract without duplicating the format
# string across modules. The cell/episode widths (``:04d``) match the
# trajectory NPZ convention so per-episode artifacts sort
# lexicographically together.
_VIDEO_FILENAME_TEMPLATE = "episode_cell{cell:04d}_ep{episode:04d}_seed{seed}.mp4"


def video_path_for(video_dir: Path, cell_index: int, episode_index: int, seed: int) -> Path:
    """Return the canonical MP4 path for a (cell, episode, seed).

    Deterministic: the same ``(cell_index, episode_index, seed)`` always
    maps to the same path. Workers cannot collide because every tuple
    is unique across the (cell, episode) lattice, and the seed is
    derived deterministically from the master seed.

    Args:
        video_dir: Directory the MP4 will live in. Must be created by
            the caller (the Runner does this once on entry).
        cell_index: :attr:`gauntlet.suite.SuiteCell.index`.
        episode_index: Zero-based ordinal within the cell.
        seed: The integer env seed for this rollout (echoed from
            :attr:`gauntlet.runner.Episode.seed`). Embedded in the
            filename so an MP4 carries enough identity to reproduce
            the rollout end-to-end via :mod:`gauntlet.replay` even
            after re-runs overwrite siblings.

    Returns:
        ``video_dir / "episode_cellNNNN_epNNNN_seedS.mp4"``.
    """
    return video_dir / _VIDEO_FILENAME_TEMPLATE.format(
        cell=cell_index, episode=episode_index, seed=seed
    )


class VideoWriter:
    """Lazy-import MP4 encoder wrapping :mod:`imageio.v3`.

    The class itself is import-safe in a torch-/extras-free interpreter
    (no ``imageio`` is touched at construction). The first call to
    :meth:`write` performs the lazy import and raises a clear
    ``ImportError`` with the exact install hint when the ``[video]``
    extra is missing:

        ``pip install "gauntlet[video]"``

    Lifecycle:

    * ``__init__`` validates the configuration. No I/O.
    * :meth:`write` encodes a frame list to an MP4 in one call. Single-
      shot — there is no resumable streaming API. The Runner builds
      the full per-episode buffer in memory and hands it over once.

    Args:
        fps: Output framerate. Must be a positive integer.

    Raises:
        ValueError: if ``fps`` is not a positive integer.
    """

    # libx264 is the default video codec — every browser, every macOS /
    # Linux / Windows player handles it. The ``yuv420p`` pixel format
    # is required for browser playback (Chromium / Safari refuse other
    # formats); we pass it via the ffmpeg backend's ``output_params``
    # pass-through because ``imageio[ffmpeg]`` does not expose a
    # dedicated ``pixel_format=`` kwarg on every release line. Both
    # settings are baked in to keep the per-episode write deterministic.
    _CODEC = "libx264"
    _OUTPUT_PARAMS: tuple[str, ...] = ("-pix_fmt", "yuv420p")

    def __init__(self, *, fps: int = 30) -> None:
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"fps must be a positive int; got {fps!r}")
        self._fps = fps

    @property
    def fps(self) -> int:
        """Configured framerate (read-only after construction)."""
        return self._fps

    def write(
        self,
        path: Path,
        frames: list[NDArray[np.uint8]] | NDArray[np.uint8],
    ) -> None:
        """Encode *frames* to an MP4 at *path* via the bundled ffmpeg.

        Performs the lazy ``imageio`` import; raises ``ImportError`` with
        the install hint if the ``[video]`` extra is missing. The parent
        directory is created if missing so the caller does not have to
        pre-mkdir per-episode (the Runner mkdirs once on entry; this is
        belt-and-braces for callers that bypass the Runner path).

        ``yuv420p`` requires every spatial dimension to be even; the
        method pads the array by one row / column with the bottom-right
        edge value when needed so a 223x223 image still encodes.

        Args:
            path: Output MP4 path. Parent dirs are created if missing.
            frames: List of ``(H, W, 3)`` uint8 arrays, or a single
                ``(T, H, W, 3)`` uint8 array. Must be non-empty (a
                zero-step rollout has no video to write — the Runner
                short-circuits before reaching this method, but the
                check is kept here for defence in depth).

        Raises:
            ImportError: if the ``[video]`` extra is missing. Message
                includes the exact ``pip install "gauntlet[video]"``
                hint.
            ValueError: if ``frames`` is empty, has an unsupported
                shape, or contains non-uint8 data.
        """
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                "rollout video recording requires the optional [video] extra. "
                'Install with `pip install "gauntlet[video]"` (pulls '
                "imageio[ffmpeg], which bundles a static ffmpeg binary — no "
                "system ffmpeg install required)."
            ) from exc

        # Normalise ``frames`` to a 4D ``(T, H, W, 3)`` uint8 array.
        if isinstance(frames, list):
            if not frames:
                raise ValueError(
                    "VideoWriter.write requires at least one frame; got an empty list."
                )
            stacked = np.stack(frames, axis=0)
        else:
            stacked = np.asarray(frames)
        if stacked.dtype != np.uint8:
            raise ValueError(
                f"frames must be uint8 RGB; got dtype {stacked.dtype}. "
                "Cast via ``np.asarray(arr, dtype=np.uint8)`` upstream."
            )
        if stacked.ndim != 4 or stacked.shape[-1] != 3:
            raise ValueError(
                "frames must be shape (T, H, W, 3); got "
                f"{stacked.shape}. Stack obs['image'] arrays per step."
            )
        if stacked.shape[0] == 0:
            raise ValueError("VideoWriter.write requires at least one frame; got T=0.")

        # libx264 + yuv420p requires even H and W; pad the bottom-right
        # edge with replicated values when needed. Cheap (one numpy
        # copy at most) and keeps the codec contract honest.
        _t, h, w, _c = stacked.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            stacked = np.pad(
                stacked,
                pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
                mode="edge",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        # imageio v3 accepts ``codec`` / ``fps`` / ``output_params``
        # as keyword extras; the ffmpeg backend forwards
        # ``output_params`` straight to the ffmpeg CLI so we can pin
        # the pixel format independently of the imageio release line.
        # ``macro_block_size=1`` disables imageio's default 16-pixel
        # block alignment (which would otherwise auto-resize tiny
        # frames and break the explicit yuv420p contract).
        _imwrite: Any = iio.imwrite
        _imwrite(
            path,
            stacked,
            fps=self._fps,
            codec=self._CODEC,
            output_params=list(self._OUTPUT_PARAMS),
            macro_block_size=1,
        )
