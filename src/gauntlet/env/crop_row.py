"""Camera-guided crop-row following (``env: crop-row``).

A field vehicle drives along planted rows at constant speed; the policy
sees only a forward-looking camera image and outputs a steering command.
It is the closed-loop analogue of vision-based row guidance (autosteer
on a camera rather than GNSS): perception errors turn into lateral
drift, and drift past half a row spacing puts the wheels on the crop.

The scene is **procedurally generated**, not field imagery: a flat
ground plane, rows of plants at a fixed spacing, scattered weeds, soil
texture and a sky. It is rendered with a pinhole camera by inverse
ground-plane projection in numpy (no simulator, no GPU), so a frame
costs about a millisecond. The point is the evaluation method, not
photorealism: the same suite structure applies to a real perception
model replayed against logged field images.

Observation: ``{"image": uint8[H, W, 3]}`` only. Ground-truth lateral
and heading errors are published in ``info`` (for analysis and for
training a model from labelled frames) but never in the observation.

Action: ``[steer]`` in ``[-1, 1]``, a yaw-rate command scaled by
:attr:`CropRowEnv.MAX_YAW_RATE`. Positive steers left.

Success: the episode runs its full length without the vehicle leaving
the row corridor (``|lateral error| < CORRIDOR``) and ends within
:attr:`CropRowEnv.SETTLED_ERROR` of the row centre.

Perturbation axes (all applied on the next ``reset``):

* ``dust_density`` (0..1): airborne dust; tan haze that thickens with
  distance and in patches, with a loss of contrast and saturation.
* ``glare_intensity`` (0..1): low sun in front of the camera; a flare
  and veiling glare that wash the image out.
* ``motion_blur`` (0..1): blur along the image's vertical axis, the
  dominant direction of apparent motion when driving forward over
  rough ground; 1.0 is a 15-pixel streak at the default resolution.
* ``weed_density`` (0..1): green weeds between the rows, the classic
  confuser for vegetation-index row detectors.
* ``row_curvature`` (1/m): rows bend with this curvature (sign drawn
  from the seed), so a policy must track a turning row.
* ``lighting_intensity``: global illumination scale (shared with the
  tabletop envs).
* ``inference_delay_jitter`` is handled by the runner and applies here
  too; :attr:`CropRowEnv.control_dt` converts milliseconds to steps.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Final

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["CROP_ROW_AXES", "CropRowEnv"]

CROP_ROW_AXES: Final[frozenset[str]] = frozenset(
    {
        "dust_density",
        "glare_intensity",
        "motion_blur",
        "weed_density",
        "row_curvature",
        "lighting_intensity",
    }
)

_DEFAULTS: Final[dict[str, float]] = {
    "dust_density": 0.0,
    "glare_intensity": 0.0,
    "motion_blur": 0.0,
    "weed_density": 0.0,
    "row_curvature": 0.0,
    "lighting_intensity": 1.0,
}

_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "dust_density": (0.0, 1.0),
    "glare_intensity": (0.0, 1.0),
    "motion_blur": (0.0, 1.0),
    "weed_density": (0.0, 1.0),
    "row_curvature": (0.0, 0.1),
    "lighting_intensity": (0.05, 3.0),
}

_SOIL: Final = np.array([122.0, 92.0, 66.0])
_PLANT: Final = np.array([58.0, 140.0, 48.0])
_WEED: Final = np.array([92.0, 150.0, 60.0])
_SKY_TOP: Final = np.array([120.0, 160.0, 215.0])
_SKY_HORIZON: Final = np.array([205.0, 215.0, 225.0])
_DUST: Final = np.array([196.0, 176.0, 140.0])


def _hash01(*keys: NDArray[np.int64] | np.int64) -> NDArray[np.float64]:
    """Deterministic per-cell uniform [0, 1) from integer keys (splitmix64)."""
    x = np.zeros(np.broadcast(*keys).shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i, k in enumerate(keys):
            x ^= k.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15 + 2 * i + 1)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _value_noise(
    x: NDArray[np.float64], y: NDArray[np.float64], seed: np.int64
) -> NDArray[np.float64]:
    """Smooth [0, 1) value noise: smoothstep-interpolated hashes on a unit grid."""
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    ix, iy = x0.astype(np.int64), y0.astype(np.int64)
    s = np.full(ix.shape, seed, dtype=np.int64)
    n00, n10 = _hash01(ix, iy, s), _hash01(ix + 1, iy, s)
    n01, n11 = _hash01(ix, iy + 1, s), _hash01(ix + 1, iy + 1, s)
    top = n00 + (n10 - n00) * fx
    bot = n01 + (n11 - n01) * fx
    return np.asarray(top + (bot - top) * fy, dtype=np.float64)


class CropRowEnv(gym.Env[dict[str, NDArray[np.uint8]], NDArray[np.float64]]):
    """Follow a crop row from a forward camera image."""

    AXIS_NAMES: ClassVar[frozenset[str]] = CROP_ROW_AXES
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset(
        {"dust_density", "glare_intensity", "motion_blur", "lighting_intensity"}
    )

    ROW_SPACING: ClassVar[float] = 0.76
    """Row spacing in metres (30-inch rows)."""
    PLANT_SPACING: ClassVar[float] = 0.15
    SPEED: ClassVar[float] = 1.5
    """Forward speed, m/s."""
    DT: ClassVar[float] = 0.1
    """Control period, s."""
    MAX_YAW_RATE: ClassVar[float] = 0.5
    """Yaw rate at ``|steer| == 1``, rad/s."""
    CORRIDOR: ClassVar[float] = 0.25
    """Leaving ``|lateral error| < CORRIDOR`` ends the episode as a failure."""
    SETTLED_ERROR: ClassVar[float] = 0.08
    """Final ``|lateral error|`` needed for success."""
    CAMERA_HEIGHT: ClassVar[float] = 1.6
    CAMERA_PITCH: ClassVar[float] = math.radians(28.0)
    HFOV: ClassVar[float] = math.radians(75.0)

    def __init__(
        self,
        *,
        max_steps: int = 100,
        render_in_obs: bool = True,
        render_size: tuple[int, int] = (96, 128),
    ) -> None:
        """Construct the env.

        ``render_in_obs`` is accepted for parity with the other envs and
        ignored: the image is the only observation. ``render_size`` is
        ``(H, W)``.
        """
        del render_in_obs
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        h, w = render_size
        if h < 16 or w < 16:
            raise ValueError(f"render_size must be at least (16, 16); got {render_size}")
        self._max_steps = int(max_steps)
        self._h, self._w = int(h), int(w)
        self.observation_space = spaces.Dict(
            {"image": spaces.Box(0, 255, shape=(self._h, self._w, 3), dtype=np.uint8)}
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        self._precompute_rays()
        self._active: dict[str, float] = dict(_DEFAULTS)
        self._pending: dict[str, float] = {}
        self._rng = np.random.default_rng(0)
        self._scene_seed = 0
        self._curv = 0.0
        self._x = self._y = self._yaw = 0.0
        self._t = 0
        self._sun_u = 0.7
        self._max_err = 0.0

    # ------------------------------------------------------------ protocol

    @property
    def control_dt(self) -> float:
        """Seconds per control step (used by ``inference_delay_jitter``)."""
        return self.DT

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a perturbation for the next :meth:`reset`."""
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        v = float(value)
        lo, hi = _BOUNDS[name]
        if not math.isfinite(v) or v < lo or v > hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}]; got {value!r}")
        self._pending[name] = v

    def restore_baseline(self) -> None:
        """Drop every applied and queued perturbation."""
        self._active = dict(_DEFAULTS)
        self._pending = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.uint8]], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._active.update(self._pending)
        self._pending = {}
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        self._scene_seed = int(rng.integers(0, 2**31 - 1))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self._curv = sign * self._active["row_curvature"]
        self._x = 0.0
        self._y = float(rng.uniform(-0.08, 0.08))
        self._yaw = float(rng.uniform(-0.06, 0.06))
        self._sun_u = float(rng.uniform(0.55, 0.85))
        self._t = 0
        self._max_err = abs(self._lateral_error())
        return self._obs(), self._info(success=False)

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, NDArray[np.uint8]], float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (1,):
            raise ValueError(f"action must have shape (1,); got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action must be finite (no NaN/Inf); got {a}")
        steer = float(np.clip(a[0], -1.0, 1.0))
        self._yaw += steer * self.MAX_YAW_RATE * self.DT
        self._x += self.SPEED * self.DT * math.cos(self._yaw)
        self._y += self.SPEED * self.DT * math.sin(self._yaw)
        self._t += 1
        err = abs(self._lateral_error())
        self._max_err = max(self._max_err, err)
        off_row = err >= self.CORRIDOR
        truncated = self._t >= self._max_steps and not off_row
        success = truncated and err < self.SETTLED_ERROR
        reward = -err
        return self._obs(), reward, off_row, truncated, self._info(success=success)

    def close(self) -> None:
        """No resources to release."""

    # ------------------------------------------------------------- geometry

    def _row_offset(self, x: float | NDArray[np.float64]) -> Any:
        return 0.5 * self._curv * np.square(x)

    def _lateral_error(self) -> float:
        """Signed distance from the vehicle to the centre of row 0 (m, + = left)."""
        return float(self._y - self._row_offset(self._x))

    def _heading_error(self) -> float:
        return float(self._yaw - math.atan(self._curv * self._x))

    def _info(self, *, success: bool) -> dict[str, Any]:
        return {
            "success": bool(success),
            "step": self._t,
            "lateral_error": self._lateral_error(),
            "heading_error": self._heading_error(),
            "max_lateral_error": self._max_err,
        }

    # -------------------------------------------------------------- render

    @classmethod
    def ground_coords(
        cls, render_size: tuple[int, int]
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        """Per-pixel ground intersection for the env's calibrated camera.

        Returns ``(on_ground, forward_m, left_m)`` arrays of shape
        ``render_size``: whether the pixel's ray hits the ground, and
        where, in the vehicle frame. This is the camera calibration a
        perception stack would have; it carries no scene state.
        """
        h, w = render_size
        f = (w / 2) / math.tan(cls.HFOV / 2)
        u = np.arange(w) + 0.5 - w / 2
        v = np.arange(h) + 0.5 - h / 2
        uu, vv = np.meshgrid(u, v)
        # Camera frame: x right, y down, z forward; pitched down about x.
        c, s = math.cos(cls.CAMERA_PITCH), math.sin(cls.CAMERA_PITCH)
        dy = vv / f
        fwd = c - s * dy
        down = s + c * dy
        lat = -uu / f
        ground = down > 1e-3
        t = np.where(ground, cls.CAMERA_HEIGHT / np.where(ground, down, 1.0), 0.0)
        return ground, np.where(ground, fwd * t, 0.0), np.where(ground, lat * t, 0.0)

    def _precompute_rays(self) -> None:
        """Per-pixel ground intersection in the vehicle frame (fixed camera)."""
        h, w = self._h, self._w
        f = (w / 2) / math.tan(self.HFOV / 2)
        u = np.arange(w) + 0.5 - w / 2
        v = np.arange(h) + 0.5 - h / 2
        uu, vv = np.meshgrid(u, v)
        c, s = math.cos(self.CAMERA_PITCH), math.sin(self.CAMERA_PITCH)
        down = s + c * (vv / f)
        self._ground, self._gf, self._gl = self.ground_coords((h, w))
        self._dist = np.where(self._ground, np.hypot(self._gf, self._gl), 60.0)
        # Sky gradient by elevation of the ray.
        elev = np.clip(-down, 0.0, 1.0)
        self._sky = (
            _SKY_HORIZON[None, None, :]
            + (_SKY_TOP - _SKY_HORIZON)[None, None, :] * np.sqrt(elev)[..., None]
        )
        self._uu, self._vv = uu, vv

    def _obs(self) -> dict[str, NDArray[np.uint8]]:
        return {"image": self._render()}

    def _render(self) -> NDArray[np.uint8]:
        p = self._active
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        wx = self._x + cy * self._gf - sy * self._gl
        wy = self._y + sy * self._gf + cy * self._gl
        seed = np.int64(self._scene_seed)

        # Soil with a 4 cm texture.
        cx = np.floor(wx / 0.04).astype(np.int64)
        cyy = np.floor(wy / 0.04).astype(np.int64)
        soil_n = _hash01(cx, cyy, seed + 7)
        img = _SOIL[None, None, :] * (0.82 + 0.3 * soil_n)[..., None]

        # Crop plants along each row.
        ry = wy - self._row_offset(wx)
        k = np.round(ry / self.ROW_SPACING).astype(np.int64)
        d = ry - k * self.ROW_SPACING
        j = np.round(wx / self.PLANT_SPACING).astype(np.int64)
        jit = (_hash01(k, j, seed + 1) - 0.5) * 0.06
        rad = 0.055 + 0.04 * _hash01(k, j, seed + 2)
        present = _hash01(k, j, seed + 3) > 0.08
        s = wx - j * self.PLANT_SPACING - jit
        plant = present & (np.square(s / rad) + np.square(d / (rad * 1.1)) < 1.0)
        shade = 0.75 + 0.45 * _hash01(np.floor(wx / 0.02).astype(np.int64), cyy, seed + 4)
        img = np.where(plant[..., None], _PLANT[None, None, :] * shade[..., None], img)

        # Weeds between rows.
        wd = p["weed_density"]
        if wd > 0.0:
            gx = np.floor(wx / 0.12).astype(np.int64)
            gy = np.floor(wy / 0.12).astype(np.int64)
            occ = _hash01(gx, gy, seed + 5) < 0.45 * wd
            ox = (gx + 0.2 + 0.6 * _hash01(gx, gy, seed + 6)) * 0.12
            oy = (gy + 0.2 + 0.6 * _hash01(gx, gy, seed + 8)) * 0.12
            wr = 0.025 + 0.035 * _hash01(gx, gy, seed + 9)
            weed = occ & (np.hypot(wx - ox, wy - oy) < wr) & (np.abs(d) > 0.12)
            img = np.where(weed[..., None], _WEED[None, None, :] * shade[..., None], img)

        # Distance falloff toward the horizon, then the sky.
        fall = np.clip(self._dist / 25.0, 0.0, 1.0)[..., None]
        img = img * (1 - 0.35 * fall) + _SKY_HORIZON[None, None, :] * 0.35 * fall
        img = np.where(self._ground[..., None], img, self._sky)

        img = img * p["lighting_intensity"]
        img = self._apply_dust(img, p["dust_density"], seed)
        img = self._apply_glare(img, p["glare_intensity"])
        img = self._apply_motion_blur(img, p["motion_blur"])
        noise = (
            _hash01(self._uu.astype(np.int64) + 1000, self._vv.astype(np.int64), seed + self._t)
            - 0.5
        ) * 6.0
        img = img + noise[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_dust(
        self, img: NDArray[np.float64], density: float, seed: np.int64
    ) -> NDArray[np.float64]:
        if density <= 0.0:
            return img
        # Smooth low-frequency patches that drift past as the vehicle moves.
        patch = _value_noise(
            (self._uu + 40.0 * self._x) / (0.25 * self._w), self._vv / (0.3 * self._h), seed + 11
        )
        depth = np.clip(self._dist / 12.0, 0.0, 1.0)
        a = np.clip(density * (0.35 + 0.45 * depth + 0.3 * patch), 0.0, 0.95)[..., None]
        grey = img.mean(axis=2, keepdims=True)
        desat = img * (1 - 0.6 * density) + grey * 0.6 * density
        return np.asarray(desat * (1 - a) + _DUST[None, None, :] * a, dtype=np.float64)

    def _apply_glare(self, img: NDArray[np.float64], g: float) -> NDArray[np.float64]:
        if g <= 0.0:
            return img
        su = (self._sun_u - 0.5) * self._w
        sv = -0.42 * self._h
        r = np.hypot(self._uu - su, self._vv - sv) / self._w
        flare = np.exp(-np.square(r / 0.35)) * 260.0 + np.exp(-np.square(r / 0.9)) * 90.0
        streak = np.exp(-np.square((self._vv - sv) / (0.04 * self._h))) * 120.0
        veil = g * (flare + streak)[..., None]
        return np.asarray(img * (1 - 0.45 * g) + veil + 70.0 * g, dtype=np.float64)

    def _apply_motion_blur(self, img: NDArray[np.float64], m: float) -> NDArray[np.float64]:
        n = round(m * 15.0 * self._h / 96.0)
        if n <= 1:
            return img
        pad = np.concatenate([np.repeat(img[:1], n, axis=0), img], axis=0)
        cs = np.cumsum(pad, axis=0)
        out = (cs[n:] - cs[:-n]) / n
        return np.asarray(out, dtype=np.float64)
