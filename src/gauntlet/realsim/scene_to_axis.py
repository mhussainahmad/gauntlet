"""Bridge from raw :class:`RealSceneInput` to the ``camera_extrinsics`` axis.

The B-42 ``camera_extrinsics`` axis (see
:func:`gauntlet.env.perturbation.axes.camera_extrinsics`) consumes a
list of structured 6-D pose deltas keyed by integer index. The on-disk
schema for one entry is
:class:`gauntlet.suite.schema.ExtrinsicsValue`::

    {translation: [dx, dy, dz], rotation: [drx, dry, drz]}

with ``translation`` in metres and ``rotation`` in radians (XYZ Euler,
MuJoCo / PyBullet camera convention).

This module converts a :class:`RealSceneInput` (which carries 4x4
camera-to-world rigid transforms in :attr:`extrinsics_per_frame`) into
that exact shape, sub-sampling to ``n_samples`` evenly-spaced frames
when the capture has more frames than the suite wants.

**Renderer scope (out of scope for T18):** the actual gaussian-splatting
/ NeRF / mesh renderer that *uses* a :class:`RealSceneInput` to paint
pixels is deferred to a future PR. The seam is
:class:`RendererNotImplementedError` — any future helper that pretends
to render must raise it. We deliberately do *not* ship a stub renderer
here: the safest signal to a future contributor that this slot is empty
is a single named exception and a doc string.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from gauntlet.realsim.scene_input import RealSceneInput

__all__ = [
    "RendererNotImplementedError",
    "rotation_matrix_to_xyz_euler",
    "scene_to_camera_extrinsics",
]


# ---------------------------------------------------------------------------
# Renderer seam.
# ---------------------------------------------------------------------------


class RendererNotImplementedError(NotImplementedError):
    """Raised by future renderer-shaped helpers that have not landed yet.

    T18 ships the **input pipeline only** — the gaussian-splatting /
    NeRF / mesh renderer that consumes a :class:`RealSceneInput` is a
    future PR. This exception is the explicit, named seam:

    * Any new helper added to this package that performs (or pretends
      to perform) actual rendering MUST raise
      :class:`RendererNotImplementedError` until the renderer ships.
    * The metadata test
      ``tests/test_realsim_scene_to_axis.py::test_no_accidental_renderer_landed``
      walks the public surface of :mod:`gauntlet.realsim` and asserts no
      symbol has accidentally grown a working ``render`` method.

    The exception subclasses :class:`NotImplementedError` so existing
    ``except NotImplementedError`` filters in plugin / runner glue catch
    it the same way they catch other deliberate "future seam" markers
    (see :class:`gauntlet.env.GauntletEnv` ``set_perturbation`` axes
    that raise on Genesis).
    """


# ---------------------------------------------------------------------------
# Rotation conversion.
# ---------------------------------------------------------------------------


def rotation_matrix_to_xyz_euler(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to ``(rx, ry, rz)`` XYZ Euler in radians.

    Convention: the matrix represents the rotation ``R = Rz @ Ry @ Rx``
    (extrinsic XYZ; equivalent to intrinsic ZYX). This matches the
    MuJoCo / PyBullet camera Euler convention used by the
    ``set_camera_extrinsics_list`` setter on
    :class:`gauntlet.env.tabletop.TabletopEnv`.

    Algorithm: the closed-form ``atan2``-based extraction. Stable for
    every rotation except the gimbal-lock pole at ``ry = ±π/2``, where
    we collapse ``rz`` to zero and read ``rx`` off the residual. This
    matches the SciPy ``Rotation.as_euler('xyz')`` behaviour at the
    pole; we don't import SciPy because (a) it is not in the core
    deps, and (b) six lines of math are clearer than dragging in a
    multi-megabyte dependency for one extraction.

    Args:
        rotation: a 3x3 ``float64`` ``np.ndarray`` representing a
            rotation. Orthonormality is **not** checked — the caller is
            expected to feed in the rotation block of a previously-
            validated 4x4 pose.

    Returns:
        Tuple ``(rx, ry, rz)`` of floats in radians, each in
        ``[-π, π]``.

    Raises:
        ValueError: if ``rotation`` does not have shape ``(3, 3)``.
    """
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix_to_xyz_euler expects a 3x3 matrix; got shape {rotation.shape}",
        )
    sy = -float(rotation[2, 0])
    # Clamp into [-1, 1] for numerical safety before asin.
    sy_clamped = max(-1.0, min(1.0, sy))
    ry = math.asin(sy_clamped)
    # Cosine of ry tells us whether we're at the gimbal-lock pole.
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        # Gimbal-lock fallback: collapse rz to zero, read rx off the
        # residual block. ``atan2(-r12, r11)`` is the well-known
        # degenerate-case extraction.
        rx = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        rz = 0.0
    return rx, ry, rz


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def scene_to_camera_extrinsics(
    scene: RealSceneInput,
    n_samples: int = 16,
) -> list[dict[str, list[float]]]:
    """Convert a :class:`RealSceneInput` into ``camera_extrinsics`` axis values.

    Output is a list of ``n_samples`` dicts, each shaped to match
    :class:`gauntlet.suite.schema.ExtrinsicsValue`::

        {"translation": [dx, dy, dz], "rotation": [drx, dry, drz]}

    Translation is the homogeneous translation column of the 4x4 pose;
    rotation is the XYZ-Euler decomposition of the rotation sub-block
    (radians, MuJoCo / PyBullet convention — see
    :func:`rotation_matrix_to_xyz_euler`).

    Sub-sampling: when ``len(scene.extrinsics_per_frame) > n_samples``,
    we pick ``n_samples`` indices evenly spaced through the capture
    (always including the first and last frames). When the capture has
    *fewer* frames than ``n_samples``, we return one entry per frame
    without duplication or interpolation — sweep loaders handle short
    lists fine and synthesising frames would fabricate data the capture
    does not contain.

    Args:
        scene: parsed capture-dir handle.
        n_samples: maximum number of axis entries to emit. Must be >= 1.

    Returns:
        List of ``ExtrinsicsValue``-shaped dicts, length
        ``min(n_samples, len(scene.extrinsics_per_frame))``.

    Raises:
        ValueError: if ``n_samples`` < 1 or the scene has zero frames
            (the latter shouldn't happen — :func:`load_real_scene` already
            rejects empty extrinsics — but we guard anyway).

    Note:
        This function does **not** invoke any renderer. It is a pure
        coordinate-system bridge: in (4x4 matrices) -> out (axis-shaped
        dicts). The renderer that *uses* the resulting axis values to
        paint pixels is deferred to a future PR (see
        :class:`RendererNotImplementedError`).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    n_frames = len(scene.extrinsics_per_frame)
    if n_frames == 0:
        raise ValueError("scene has zero extrinsics frames; nothing to convert")

    # Pick the sub-sampled indices. ``np.linspace`` over ints with
    # endpoint=True gives the conventional first / last-included
    # spacing; we round to ``int64`` and de-duplicate to handle the
    # ``n_samples > n_frames`` edge.
    n_out = min(n_samples, n_frames)
    raw_indices = np.linspace(0, n_frames - 1, num=n_out, dtype=np.int64)
    seen: set[int] = set()
    indices: list[int] = []
    for idx in raw_indices.tolist():
        idx_int = int(idx)
        if idx_int not in seen:
            seen.add(idx_int)
            indices.append(idx_int)

    out: list[dict[str, list[float]]] = []
    for idx in indices:
        pose = scene.extrinsics_per_frame[idx]
        # Translation column: [m[0][3], m[1][3], m[2][3]].
        translation = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
        rx, ry, rz = rotation_matrix_to_xyz_euler(pose[:3, :3])
        out.append(
            {
                "translation": translation,
                "rotation": [rx, ry, rz],
            },
        )
    return out
