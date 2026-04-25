"""Post-render HSV color-shift wrapper for ``color_shift_synthetic`` (B-43).

Backend-agnostic visual-bias perturbation. Wraps any
:class:`gauntlet.env.base.GauntletEnv` and applies a deterministic
HSV-space transform to ``obs["image"]`` (single-camera) and
``obs["images"][name]`` (multi-camera) **after** the inner env has
rendered them. Mirrors :mod:`gauntlet.env.image_attack` (B-31) — the two
wrappers compose: ``ColorShiftWrapper(ImageAttackWrapper(env))`` is well
defined.

Why a post-render shim rather than a per-backend renderer override? The
four backends (MuJoCo, PyBullet, Genesis, Isaac) each ship their own
material / shader path with incompatible APIs. A single post-render
shim lets one numpy-only implementation cover all of them.

Honest framing
--------------
HSV-shift on the rendered RGB does **NOT** faithfully simulate
real-world illumination changes. Real illumination changes alter
materials' specular response, shadow geometry, and inter-reflection;
post-render HSV cannot model any of those. The axis name carries the
``_synthetic`` suffix to be explicit about this — see the docstring on
:func:`gauntlet.env.perturbation.axes.color_shift_synthetic` for the
full anti-feature note. Reference: RoboView-Bias (arXiv 2509.22356)
shows VLAs are biased toward high-saturation hues over achromatic
scenes — this axis exposes that asymmetry within sim, with the caveat
that the underlying physics shortcut is intentional.

Encoding of the axis value
--------------------------
The axis is **categorical, integer-coded** (the suite YAML's ``values:``
shape carries floats, not strings; the float form survives YAML
round-trip and is rounded back to int by the wrapper). Each integer in
:data:`SHIFT_IDS` selects a named transform:

* ``0`` :data:`SHIFT_NONE` — pass-through (baseline).
* ``1`` :data:`SHIFT_HUE_PLUS_30` — rotate hue by +30 degrees.
* ``2`` :data:`SHIFT_HUE_MINUS_30` — rotate hue by -30 degrees.
* ``3`` :data:`SHIFT_SATURATION_0_5` — multiply saturation by 0.5.
* ``4`` :data:`SHIFT_SATURATION_1_5` — multiply saturation by 1.5
  (clipped to ``[0, 1]``).
* ``5`` :data:`SHIFT_ACHROMATIC` — saturation := 0 (grayscale-equivalent
  in HSV; ``R == G == B == V`` per pixel, where ``V`` is the original
  per-pixel max channel — NOT the luminance-weighted gray).

Determinism
-----------
The HSV transforms are pixel-wise pure functions; they consume no RNG.
The wrapper carries no random state (in contrast to
:class:`gauntlet.env.image_attack.ImageAttackWrapper`, which seeds noise
patterns). Same input frame, same axis value, bit-identical output.

Integration
-----------
Currently a *building block*. The runner does not yet auto-instantiate
the wrapper when a suite declares a ``color_shift_synthetic`` axis —
that wiring is left for a follow-up, mirroring B-31 (see
:mod:`gauntlet.env.image_attack`). Today, callers that want post-render
color shifts construct ``ColorShiftWrapper(inner_env)`` themselves and
hand the result to the runner via ``env_factory``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from gauntlet.env.base import Action, GauntletEnv, Observation

__all__ = [
    "SHIFT_ACHROMATIC",
    "SHIFT_HUE_MINUS_30",
    "SHIFT_HUE_PLUS_30",
    "SHIFT_IDS",
    "SHIFT_NAMES",
    "SHIFT_NONE",
    "SHIFT_SATURATION_0_5",
    "SHIFT_SATURATION_1_5",
    "ColorShiftWrapper",
    "apply_color_shift",
    "hsv_to_rgb",
    "rgb_to_hsv",
]


# ---------------------------------------------------------------------- IDs
# Integer-coded categorical axis values. The float form (e.g. ``0.0``)
# round-trips through the suite YAML ``values:`` shape unchanged.
SHIFT_NONE: Final[int] = 0
SHIFT_HUE_PLUS_30: Final[int] = 1
SHIFT_HUE_MINUS_30: Final[int] = 2
SHIFT_SATURATION_0_5: Final[int] = 3
SHIFT_SATURATION_1_5: Final[int] = 4
SHIFT_ACHROMATIC: Final[int] = 5

# Tuple of legal integer ids in declared order. ``axis.low`` / ``axis.high``
# in :func:`gauntlet.env.perturbation.axes.color_shift_synthetic` key off
# the extremes of this tuple.
SHIFT_IDS: Final[tuple[int, ...]] = (
    SHIFT_NONE,
    SHIFT_HUE_PLUS_30,
    SHIFT_HUE_MINUS_30,
    SHIFT_SATURATION_0_5,
    SHIFT_SATURATION_1_5,
    SHIFT_ACHROMATIC,
)

# Parallel name tuple — index i is the human label for ``SHIFT_IDS[i]``.
# Reports / logs may surface these.
SHIFT_NAMES: Final[tuple[str, ...]] = (
    "none",
    "hue_+30",
    "hue_-30",
    "saturation_0.5",
    "saturation_1.5",
    "achromatic",
)


# ----- per-shift constants ---------------------------------------------------
_HUE_DEGREES_PLUS: Final[float] = 30.0
_HUE_DEGREES_MINUS: Final[float] = -30.0
_SAT_FACTOR_LOW: Final[float] = 0.5
_SAT_FACTOR_HIGH: Final[float] = 1.5


# ---------------------------------------------------------------------- HSV core
def rgb_to_hsv(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a float32 ``(H, W, 3)`` RGB image in ``[0, 1]`` to HSV.

    Hue is encoded in ``[0, 1)`` (i.e. one full turn = 1.0, NOT
    degrees). Saturation and value are in ``[0, 1]``. Pure numpy — no
    Pillow, opencv, or colorsys dependency. The returned array is
    ``(H, W, 3)`` with channel order ``(H, S, V)``.

    Achromatic pixels (``cmax == cmin``) collapse to ``H = 0``,
    ``S = 0``; this matches the colorsys / opencv convention. The hue
    channel is always non-negative (the wrap to ``[0, 1)`` happens here)
    so downstream additive shifts can fold modulo 1 cleanly.
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    # Avoid division-by-zero on achromatic pixels; the hue lookup below
    # masks them out before the numerator can leak NaN through.
    safe_delta = np.where(delta == 0, 1.0, delta)
    # Per-channel hue contribution; only one branch is correct per pixel
    # but evaluating all three is cheaper than scatter+mask.
    hue_r = ((g - b) / safe_delta) % 6.0
    hue_g = (b - r) / safe_delta + 2.0
    hue_b = (r - g) / safe_delta + 4.0
    hue = np.where(cmax == r, hue_r, np.where(cmax == g, hue_g, hue_b))
    # Sextant -> [0, 1). Clamp the achromatic case to 0 explicitly so
    # downstream callers see a well-defined hue.
    hue = np.where(delta == 0, 0.0, hue / 6.0)
    hue = hue % 1.0
    sat = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    val = cmax
    return np.stack([hue, sat, val], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a float32 ``(H, W, 3)`` HSV image back to RGB in ``[0, 1]``.

    Inverse of :func:`rgb_to_hsv`. Hue in ``[0, 1)`` (callers fold their
    additive shifts modulo 1 first). Saturation and value in ``[0, 1]``.
    The output is the standard sextant decomposition; for ``S = 0`` the
    output collapses to ``R = G = B = V`` (the per-pixel original max
    channel — matching the colorsys / opencv convention; this is NOT the
    luminance-weighted gray). Tests for the ``achromatic`` shift assert
    the ``R == G == B == V`` invariant, not luminance.
    """
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    # Sextant dispatch via per-pixel index. Each branch reuses the
    # precomputed (p, q, t, v) channels — same op count as a switch but
    # vectorised over the whole image.
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------- shifts
def _shift_id_from_value(value: float) -> int:
    """Round a float axis value to the nearest legal shift id.

    Raises ``ValueError`` for out-of-range values. The axis value travels
    as a float through the categorical sampler; the wrapper converts it
    back to an int here.
    """
    idx = round(float(value))
    if idx not in SHIFT_IDS:
        legal = ", ".join(str(i) for i in SHIFT_IDS)
        raise ValueError(
            "color_shift_synthetic: value must be one of "
            f"{{{legal}}}; got {value!r} (rounded to {idx})"
        )
    return idx


def _hue_rotate(image: NDArray[np.uint8], degrees: float) -> NDArray[np.uint8]:
    """Rotate hue by ``degrees`` (positive = counter-clockwise on the wheel)."""
    floatimg = image.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(floatimg)
    hsv[..., 0] = (hsv[..., 0] + (float(degrees) / 360.0)) % 1.0
    rgb = hsv_to_rgb(hsv)
    out = np.clip(rgb, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _saturation_scale(image: NDArray[np.uint8], factor: float) -> NDArray[np.uint8]:
    """Scale saturation by ``factor`` (clipped to ``[0, 1]`` post-multiply)."""
    floatimg = image.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(floatimg)
    hsv[..., 1] = np.clip(hsv[..., 1] * float(factor), 0.0, 1.0)
    rgb = hsv_to_rgb(hsv)
    out = np.clip(rgb, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _achromatic(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Force saturation to zero (per-pixel ``R == G == B == V``).

    Note: this is the HSV ``S = 0`` collapse, NOT a luminance-weighted
    grayscale. Each pixel maps to its own ``V = max(R, G, B)`` so a
    pure-blue pixel and a pure-red pixel of equal V become the same gray.
    The honest framing is in this module's docstring.
    """
    floatimg = image.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(floatimg)
    hsv[..., 1] = 0.0
    rgb = hsv_to_rgb(hsv)
    out = np.clip(rgb, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def apply_color_shift(
    image: NDArray[np.uint8],
    shift_id: int,
) -> NDArray[np.uint8]:
    """Apply one named HSV shift to a single uint8 ``(H, W, 3)`` image.

    Pure function: same ``(image, shift_id)`` always yields the same
    output. No RNG — color shifts are deterministic.

    Raises:
        ValueError: ``shift_id`` is not one of :data:`SHIFT_IDS`.
    """
    if shift_id == SHIFT_NONE:
        return image
    if shift_id == SHIFT_HUE_PLUS_30:
        return _hue_rotate(image, _HUE_DEGREES_PLUS)
    if shift_id == SHIFT_HUE_MINUS_30:
        return _hue_rotate(image, _HUE_DEGREES_MINUS)
    if shift_id == SHIFT_SATURATION_0_5:
        return _saturation_scale(image, _SAT_FACTOR_LOW)
    if shift_id == SHIFT_SATURATION_1_5:
        return _saturation_scale(image, _SAT_FACTOR_HIGH)
    if shift_id == SHIFT_ACHROMATIC:
        return _achromatic(image)
    legal = ", ".join(str(i) for i in SHIFT_IDS)
    raise ValueError(f"color_shift_synthetic: unknown shift id {shift_id!r}; legal: {{{legal}}}")


# --------------------------------------------------------------------- wrapper
class ColorShiftWrapper:
    """Wrap a :class:`GauntletEnv` and apply post-render HSV color shifts.

    Structurally satisfies :class:`GauntletEnv`: forwards every Protocol
    method to the inner env, augmenting :meth:`AXIS_NAMES` with
    ``"color_shift_synthetic"`` and intercepting that axis in
    :meth:`set_perturbation`. Perturbations the inner env handles flow
    through unchanged.

    Attribute access for non-Protocol methods (e.g. backend-specific
    helpers like :meth:`TabletopEnv.set_initial_state_ood_prior`) is
    proxied via :meth:`__getattr__` so callers can keep using the inner
    surface without unwrapping.

    Composes with :class:`gauntlet.env.image_attack.ImageAttackWrapper`:
    ``ColorShiftWrapper(ImageAttackWrapper(env))`` applies image-attack
    noise first (inner) and then the color shift (outer), which is the
    natural sensor → color-pipeline order. Reverse stacking is also
    legal but flips the operator order.

    Determinism: the HSV transforms are pixel-wise pure functions, so
    the wrapper carries no RNG and same-input → same-output bit-for-bit.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()  # populated per-instance below

    def __init__(self, env: GauntletEnv) -> None:
        self._inner: GauntletEnv = env
        # Per-instance AXIS_NAMES override — append "color_shift_synthetic"
        # to the inner backend's set. We assign on the instance (shadowing
        # the ClassVar) so the GauntletEnv structural check on the wrapper
        # still passes. Mirrors the ImageAttackWrapper aliasing trick.
        inner_axes = type(env).AXIS_NAMES
        self.AXIS_NAMES = frozenset(inner_axes | {"color_shift_synthetic"})  # type: ignore[misc]
        self._pending_shift_id: int = SHIFT_NONE

    # ----- gym.Space surface --------------------------------------------------
    @property
    def observation_space(self) -> gym.spaces.Space[Any]:
        return self._inner.observation_space

    @property
    def action_space(self) -> gym.spaces.Space[Any]:
        return self._inner.action_space

    # ----- GauntletEnv.set_perturbation --------------------------------------
    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a perturbation; color_shift stays here, others delegate.

        ``color_shift_synthetic`` is intercepted and translated into an
        integer shift id; every other axis is forwarded to the inner env.
        Raises :class:`ValueError` for unknown names — same contract as
        the underlying :meth:`GauntletEnv.set_perturbation`.
        """
        if name == "color_shift_synthetic":
            self._pending_shift_id = _shift_id_from_value(value)
            return
        # Defensive: catch unknowns at the wrapper boundary even though
        # the inner env will also reject them, for a clearer error path.
        if name not in self.AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._inner.set_perturbation(name, value)

    def restore_baseline(self) -> None:
        """Reset shift queue + delegate to the inner env."""
        self._pending_shift_id = SHIFT_NONE
        self._inner.restore_baseline()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset inner env, then apply the queued shift to the first frame."""
        obs, info = self._inner.reset(seed=seed, options=options)
        return self._apply_to_obs(obs), info

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Step inner env, then apply the queued shift to the returned frame(s)."""
        obs, reward, terminated, truncated, info = self._inner.step(action)
        return self._apply_to_obs(obs), reward, terminated, truncated, info

    def close(self) -> None:
        """Release the inner env's resources. Idempotent."""
        self._inner.close()

    # ----- attribute proxy ---------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access for backend-specific methods.

        :meth:`__getattr__` is only invoked when normal attribute lookup
        fails, so the wrapper's own methods always win. Callers that
        rely on backend extras (e.g. ``set_initial_state_ood_prior``)
        keep working without unwrapping. The proxy intentionally does
        NOT proxy ``_inner`` itself — that would loop.
        """
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # ----- internals ---------------------------------------------------------
    def _apply_to_obs(self, obs: Observation) -> Observation:
        """Apply the queued shift to every image in ``obs``.

        Returns the same dict object with the relevant keys mutated to
        their shifted variants. ``obs["images"]`` is preserved intact;
        ``obs["image"]`` is re-aliased to the first camera so the legacy
        single-cam consumer stays in sync (matches the
        :class:`TabletopEnv` aliasing contract — see ``tabletop.py``
        around the ``obs["image"] = images[first].copy()`` site).
        """
        if self._pending_shift_id == SHIFT_NONE:
            return obs

        images_dict = obs.get("images")
        if isinstance(images_dict, dict) and len(images_dict) > 0:
            cam_names = list(images_dict.keys())
            for cam in cam_names:
                images_dict[cam] = apply_color_shift(images_dict[cam], self._pending_shift_id)
            # Re-alias the legacy ``obs["image"]`` to the (possibly
            # shifted) first camera, mirroring TabletopEnv's defensive
            # .copy().
            first = cam_names[0]
            obs["image"] = images_dict[first].copy()
        elif "image" in obs:
            obs["image"] = apply_color_shift(obs["image"], self._pending_shift_id)
        return obs
