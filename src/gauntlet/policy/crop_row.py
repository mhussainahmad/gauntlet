"""Reference policies for ``env: crop-row``.

:class:`CropRowClassicalPolicy` is a conventional vision pipeline for
row guidance, the kind that predates learned detectors and is still a
common baseline:

1. **Excess-green index** ``ExG = 2G - R - B`` separates vegetation from
   soil.
2. **Otsu threshold** on ExG over the ground part of the image, so the
   split adapts to exposure.
3. **Row tracking** from the bottom of the image upwards: start at the
   image centre (the camera sits over the row being followed) and
   follow the centroid of vegetation pixels inside a window that spans
   half a row spacing on the ground.
4. **Ground projection** of the tracked pixels through the calibrated
   camera (:meth:`gauntlet.env.crop_row.CropRowEnv.ground_coords`) and a
   least-squares line fit, giving the vehicle's lateral and heading
   error relative to the row.
5. **Steering** from those two estimates with :func:`row_steering`.

Its gains and thresholds were tuned on unperturbed scenes only.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from gauntlet.env.crop_row import CropRowEnv
from gauntlet.policy.base import Action, Observation

__all__ = ["CropRowClassicalPolicy", "row_steering"]

K_LATERAL: Final[float] = 3.0
K_HEADING: Final[float] = 5.0


def row_steering(lateral_error: float, heading_error: float) -> NDArray[np.float64]:
    """Steering action that drives both errors to zero.

    A linear state-feedback law on the kinematic model: with the env's
    speed and yaw-rate scale the closed loop is roughly critically
    damped at 1.5 rad/s.
    """
    steer = -(K_LATERAL * lateral_error + K_HEADING * heading_error)
    return np.array([float(np.clip(steer, -1.0, 1.0))], dtype=np.float64)


def _otsu(values: NDArray[np.float64]) -> float:
    hist, edges = np.histogram(values, bins=64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * np.square(mu0 - mu1)
    return float(centers[int(np.argmax(between))])


class CropRowClassicalPolicy:
    """Excess-green + Otsu + windowed row tracking + calibrated line fit."""

    MIN_EXG_CONTRAST: Final[float] = 12.0
    """Below this ExG spread the frame is treated as having no vegetation signal."""

    def __init__(self) -> None:
        self._size: tuple[int, int] | None = None
        self._last: tuple[float, float] = (0.0, 0.0)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        del rng
        self._last = (0.0, 0.0)

    def estimate(self, image: NDArray[np.uint8]) -> tuple[float, float] | None:
        """Return ``(lateral_error, heading_error)`` or ``None`` if no row was found."""
        h, w = image.shape[:2]
        if self._size != (h, w):
            self._size = (h, w)
            self._ground, self._fwd, self._left = CropRowEnv.ground_coords((h, w))
            # Pixels per half row spacing at each image row (for the tracking window).
            col_m = np.abs(np.diff(self._left, axis=1)).mean(axis=1)
            self._half_window = np.clip(
                0.5 * CropRowEnv.ROW_SPACING / np.maximum(col_m, 1e-6) * 0.5, 2, w / 2
            )
        img = image.astype(np.float64)
        exg = 2 * img[..., 1] - img[..., 0] - img[..., 2]
        rows = np.where(self._ground.all(axis=1) & (self._fwd.max(axis=1) < 8.0))[0]
        if rows.size < 4:
            return None
        region = exg[rows]
        if float(region.std()) < self.MIN_EXG_CONTRAST:
            return None
        mask = exg > _otsu(region.ravel())
        cols = np.arange(w, dtype=np.float64)
        u = w / 2.0
        fwd_pts: list[float] = []
        left_pts: list[float] = []
        for v in rows[::-1]:
            half = float(self._half_window[v])
            lo, hi = int(max(0, u - half)), int(min(w, u + half + 1))
            m = mask[v, lo:hi]
            if m.sum() >= 2:
                u = float(cols[lo:hi][m].mean())
                ui = round(u - 0.5)
                fwd_pts.append(float(self._fwd[v, ui]))
                left_pts.append(float(self._left[v, ui]))
        if len(fwd_pts) < 4:
            return None
        c1, c0 = np.polyfit(np.asarray(fwd_pts), np.asarray(left_pts), 1)
        return -float(c0), -float(np.arctan(c1))

    def act(self, obs: Observation) -> Action:
        est = self.estimate(np.asarray(obs["image"], dtype=np.uint8))
        if est is not None:
            self._last = est
        return row_steering(*self._last)
