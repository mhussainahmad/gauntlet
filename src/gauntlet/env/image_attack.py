"""Post-render image-attack wrapper for the ``image_attack`` axis (B-31).

Backend-agnostic adversarial-image / sensor-attack perturbation. Wraps any
:class:`gauntlet.env.base.GauntletEnv` and applies a deterministic per-step
perturbation to ``obs["image"]`` (single-camera) and ``obs["images"][name]``
(multi-camera) **after** the inner env has rendered them. The wrapper does
not enter the inner sim at all — it operates purely on the uint8 image
arrays the env returns.

Why a wrapper rather than a per-backend axis? The four backends
(MuJoCo, PyBullet, Genesis, Isaac) each ship their own renderer with
incompatible camera APIs. A single post-render shim lets one
implementation cover all of them and keeps the per-backend axis surface
focussed on physics-side perturbations. The trade-off is honest: the
attack sees the rendered uint8 frame, not the underlying scene, so it
cannot e.g. modify the lighting model — that is what
``lighting_intensity`` is for.

Encoding of the axis value
--------------------------
The axis is **categorical, integer-coded** (the suite YAML's ``values:``
shape only carries floats, not strings). Each integer in
:data:`ATTACK_IDS` selects a named attack:

* ``0`` :data:`ATTACK_NONE` — pass-through (baseline).
* ``1`` :data:`ATTACK_GAUSSIAN_LOW` — additive N(0, sigma=0.02 in [0,1]).
* ``2`` :data:`ATTACK_GAUSSIAN_HIGH` — additive N(0, sigma=0.10 in [0,1]).
* ``3`` :data:`ATTACK_JPEG_Q10` — encode-then-decode at JPEG quality 10
  (requires :mod:`PIL`; raises a clear ImportError otherwise).
* ``4`` :data:`ATTACK_RANDOM_PATCH_8X8` — zero out one random 8x8 patch.
* ``5`` :data:`ATTACK_DROPOUT_ONE_CAMERA` — zero one camera's frame
  (no-op when the env exposes a single camera).

Determinism
-----------
The wrapper seeds its own ``numpy.random.Generator`` from the env reset
seed (when available via the standard ``reset(seed=...)`` contract). The
RNG advances during ``step()`` so per-step attacks are reproducible from
``(seed, step_idx)``. The wrapper does NOT consume the inner env's RNG,
so adding the wrapper cannot perturb baselines.

Integration
-----------
Currently a *building block*. The runner does not yet auto-instantiate
the wrapper when a suite declares an ``image_attack`` axis — that wiring
is left for a follow-up. Today, callers that want post-render attacks
construct ``ImageAttackWrapper(inner_env)`` themselves and hand the
result to the runner via ``env_factory``.
"""

from __future__ import annotations

import io
from typing import Any, ClassVar, Final

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from gauntlet.env.base import Action, GauntletEnv, Observation

__all__ = [
    "ATTACK_DROPOUT_ONE_CAMERA",
    "ATTACK_GAUSSIAN_HIGH",
    "ATTACK_GAUSSIAN_LOW",
    "ATTACK_IDS",
    "ATTACK_JPEG_Q10",
    "ATTACK_NAMES",
    "ATTACK_NONE",
    "ATTACK_RANDOM_PATCH_8X8",
    "ImageAttackWrapper",
    "apply_image_attack",
]


# ---------------------------------------------------------------------- IDs
# Integer-coded categorical axis values. The float form (e.g. ``0.0``)
# round-trips through the suite YAML ``values:`` shape unchanged.
ATTACK_NONE: Final[int] = 0
ATTACK_GAUSSIAN_LOW: Final[int] = 1
ATTACK_GAUSSIAN_HIGH: Final[int] = 2
ATTACK_JPEG_Q10: Final[int] = 3
ATTACK_RANDOM_PATCH_8X8: Final[int] = 4
ATTACK_DROPOUT_ONE_CAMERA: Final[int] = 5

# Tuple of legal integer ids in declared order. ``axis.low`` / ``axis.high``
# in :func:`gauntlet.env.perturbation.axes.image_attack` key off the
# extremes of this tuple.
ATTACK_IDS: Final[tuple[int, ...]] = (
    ATTACK_NONE,
    ATTACK_GAUSSIAN_LOW,
    ATTACK_GAUSSIAN_HIGH,
    ATTACK_JPEG_Q10,
    ATTACK_RANDOM_PATCH_8X8,
    ATTACK_DROPOUT_ONE_CAMERA,
)

# Parallel name tuple — index i is the human label for ``ATTACK_IDS[i]``.
# Reports / logs may surface these.
ATTACK_NAMES: Final[tuple[str, ...]] = (
    "none",
    "gaussian_noise_low",
    "gaussian_noise_high",
    "jpeg_q10",
    "random_patch_8x8",
    "dropout_one_camera",
)


# ----- per-attack constants --------------------------------------------------
_GAUSSIAN_LOW_SIGMA: Final[float] = 0.02  # in [0, 1] image space
_GAUSSIAN_HIGH_SIGMA: Final[float] = 0.10
_JPEG_QUALITY: Final[int] = 10
_PATCH_SIZE: Final[int] = 8

_PIL_INSTALL_HINT: Final[str] = (
    "image_attack jpeg_q10 requires the optional Pillow dependency. "
    "Install via `pip install pillow` or any extra that pulls it in "
    "(e.g. the `[hf]` extra: `pip install gauntlet[hf]`)."
)


# ---------------------------------------------------------------------- core
def _attack_id_from_value(value: float) -> int:
    """Round a float axis value to the nearest legal attack id.

    Raises ``ValueError`` for out-of-range values. The axis value travels
    as a float through the categorical sampler; the wrapper converts it
    back to an int here.
    """
    idx = round(float(value))
    if idx not in ATTACK_IDS:
        legal = ", ".join(str(i) for i in ATTACK_IDS)
        raise ValueError(
            f"image_attack: value must be one of {{{legal}}}; got {value!r} (rounded to {idx})"
        )
    return idx


def _gaussian_noise(
    image: NDArray[np.uint8],
    sigma: float,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Apply additive Gaussian noise. ``sigma`` is in [0, 1] image space."""
    # Work in float to avoid uint8 wrap-around on negative noise samples.
    floatimg = image.astype(np.float32) / 255.0
    noise = rng.standard_normal(floatimg.shape).astype(np.float32) * float(sigma)
    out = np.clip(floatimg + noise, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _random_patch(
    image: NDArray[np.uint8], patch: int, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """Zero out one random ``patch x patch`` region of the image."""
    out = image.copy()
    h, w = out.shape[:2]
    if h < patch or w < patch:
        # Image too small to host the patch — leave unchanged. Honest
        # no-op rather than pretending; downstream sees identical bytes.
        return out
    y = int(rng.integers(0, h - patch + 1))
    x = int(rng.integers(0, w - patch + 1))
    out[y : y + patch, x : x + patch, ...] = 0
    return out


def _jpeg_compress(image: NDArray[np.uint8], quality: int) -> NDArray[np.uint8]:
    """Encode-then-decode the image at the given JPEG quality.

    Raises :class:`ImportError` with an install-hint message when Pillow
    is unavailable. The error is intentionally sharp — JPEG is the only
    attack that needs a non-stdlib codec, and silently skipping it would
    hide the configuration mismatch from the user.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised via test mock
        raise ImportError(_PIL_INSTALL_HINT) from exc
    pil = Image.fromarray(image, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    decoded = Image.open(buf)
    decoded.load()
    return np.asarray(decoded, dtype=np.uint8)


def apply_image_attack(
    image: NDArray[np.uint8],
    attack_id: int,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Apply one named attack to a single uint8 (H, W, 3) image.

    Pure function: same ``(image, attack_id, rng_state)`` always yields
    the same output. The RNG is advanced for noise / patch attacks.
    Per-camera dropout is not handled here — see
    :class:`ImageAttackWrapper` for the multi-camera dispatch.

    Raises:
        ValueError: ``attack_id`` is not one of :data:`ATTACK_IDS`.
        ImportError: ``attack_id == ATTACK_JPEG_Q10`` and Pillow is not
            installed.
    """
    if attack_id == ATTACK_NONE:
        return image
    if attack_id == ATTACK_GAUSSIAN_LOW:
        return _gaussian_noise(image, _GAUSSIAN_LOW_SIGMA, rng)
    if attack_id == ATTACK_GAUSSIAN_HIGH:
        return _gaussian_noise(image, _GAUSSIAN_HIGH_SIGMA, rng)
    if attack_id == ATTACK_JPEG_Q10:
        return _jpeg_compress(image, _JPEG_QUALITY)
    if attack_id == ATTACK_RANDOM_PATCH_8X8:
        return _random_patch(image, _PATCH_SIZE, rng)
    if attack_id == ATTACK_DROPOUT_ONE_CAMERA:
        # Single-image entry point — dropout only meaningful across
        # multiple cameras. The wrapper's per-camera path handles the
        # multi-cam case; here we no-op so single-cam consumers see
        # identical bytes.
        return image
    legal = ", ".join(str(i) for i in ATTACK_IDS)
    raise ValueError(f"image_attack: unknown attack id {attack_id!r}; legal: {{{legal}}}")


# --------------------------------------------------------------------- wrapper
class ImageAttackWrapper:
    """Wrap a :class:`GauntletEnv` and apply post-render image attacks.

    Structurally satisfies :class:`GauntletEnv`: forwards every Protocol
    method to the inner env, augmenting :meth:`AXIS_NAMES` with
    ``"image_attack"`` and intercepting that axis in
    :meth:`set_perturbation`. Perturbations the inner env handles flow
    through unchanged.

    Attribute access for non-Protocol methods (e.g. backend-specific
    helpers like :meth:`TabletopEnv.set_initial_state_ood_prior`) is
    proxied via :meth:`__getattr__` so callers can keep using the inner
    surface without unwrapping.

    Determinism: the wrapper holds its own ``np.random.Generator``,
    seeded from the env reset seed in :meth:`reset`. The RNG advances on
    every :meth:`step` so per-step attacks (Gaussian noise, random
    patches) differ across steps but are reproducible from
    ``(seed, step_idx)``. Same-seed reset+step sequences produce
    bit-identical attacked frames.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()  # populated per-instance below

    def __init__(self, env: GauntletEnv) -> None:
        self._inner: GauntletEnv = env
        # Per-instance AXIS_NAMES override — append "image_attack" to the
        # inner backend's set. We assign on the instance (shadowing the
        # ClassVar) so the GauntletEnv structural check on the wrapper
        # still passes.
        inner_axes = type(env).AXIS_NAMES
        self.AXIS_NAMES = frozenset(inner_axes | {"image_attack"})  # type: ignore[misc]
        self._pending_attack_id: int = ATTACK_NONE
        # Step counter is incremented inside step() before the per-step
        # RNG draw; reset() seeds the RNG from the env seed.
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._step_idx: int = 0

    # ----- gym.Space surface --------------------------------------------------
    @property
    def observation_space(self) -> gym.spaces.Space[Any]:
        return self._inner.observation_space

    @property
    def action_space(self) -> gym.spaces.Space[Any]:
        return self._inner.action_space

    # ----- GauntletEnv.set_perturbation --------------------------------------
    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a perturbation; image_attack stays here, others delegate.

        ``image_attack`` is intercepted and translated into an integer
        attack id; every other axis is forwarded to the inner env.
        Raises :class:`ValueError` for unknown names — same contract as
        the underlying :meth:`GauntletEnv.set_perturbation`.
        """
        if name == "image_attack":
            self._pending_attack_id = _attack_id_from_value(value)
            return
        # Defensive: catch unknowns at the wrapper boundary even though
        # the inner env will also reject them, for a clearer error path.
        if name not in self.AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._inner.set_perturbation(name, value)

    def restore_baseline(self) -> None:
        """Reset attack queue + delegate to the inner env."""
        self._pending_attack_id = ATTACK_NONE
        self._inner.restore_baseline()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset inner env, seed wrapper RNG, then attack the first frame.

        The wrapper RNG is seeded from ``seed`` if supplied. ``None``
        seeds yield a deterministic-but-unspecified default-seeded RNG;
        callers that care about reproducibility supply ``seed`` (the
        Runner does this).
        """
        obs, info = self._inner.reset(seed=seed, options=options)
        self._step_idx = 0
        # Use a wrapper-local RNG seeded from the env seed. ``None``
        # falls back to a fixed default so back-to-back reset(seed=None)
        # at least produces consistent attack draws within one process.
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        return self._apply_to_obs(obs), info

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Step inner env, then attack the returned frame(s)."""
        obs, reward, terminated, truncated, info = self._inner.step(action)
        self._step_idx += 1
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
        """Apply the queued attack to every image in ``obs``.

        Returns the same dict object with the relevant keys mutated to
        their attacked variants. ``obs["images"]`` is preserved intact;
        ``obs["image"]`` is re-aliased to the first camera so the
        legacy single-cam consumer stays in sync (matches the
        :class:`TabletopEnv` aliasing contract — see ``tabletop.py``
        around the ``obs["image"] = images[first].copy()`` site).
        """
        if self._pending_attack_id == ATTACK_NONE:
            return obs

        images_dict = obs.get("images")
        if isinstance(images_dict, dict) and len(images_dict) > 0:
            self._apply_to_multi_camera(obs, images_dict)
        elif "image" in obs:
            # Single-camera path. Dropout collapses to a no-op (per
            # :func:`apply_image_attack` contract).
            obs["image"] = apply_image_attack(obs["image"], self._pending_attack_id, self._rng)
        return obs

    def _apply_to_multi_camera(
        self,
        obs: Observation,
        images_dict: dict[str, NDArray[np.uint8]],
    ) -> None:
        """Apply the attack across the multi-camera dict in place.

        Dropout zeros exactly one camera (chosen via the wrapper RNG).
        Other attacks apply per-camera with independent noise samples.
        """
        cam_names = list(images_dict.keys())
        if self._pending_attack_id == ATTACK_DROPOUT_ONE_CAMERA:
            if len(cam_names) <= 1:
                # Single camera dressed as multi-cam — no-op so the
                # legacy single-cam contract holds.
                return
            victim = cam_names[int(self._rng.integers(0, len(cam_names)))]
            images_dict[victim] = np.zeros_like(images_dict[victim])
        else:
            for name in cam_names:
                images_dict[name] = apply_image_attack(
                    images_dict[name],
                    self._pending_attack_id,
                    self._rng,
                )
        # Re-alias the legacy ``obs["image"]`` to the (possibly attacked)
        # first camera, mirroring TabletopEnv's defensive .copy().
        first = cam_names[0]
        obs["image"] = images_dict[first].copy()
