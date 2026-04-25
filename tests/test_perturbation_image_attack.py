"""Tests for the ``image_attack`` post-render perturbation axis (B-31).

The axis lives in :mod:`gauntlet.env.image_attack` as a wrapper around
any :class:`gauntlet.env.base.GauntletEnv`. It operates on the rendered
``obs["image"]`` (single-camera) or ``obs["images"][name]`` (multi-cam)
arrays — backend-agnostic by construction.

Tests cover:

* registration in the canonical AXIS_NAMES tuple,
* per-attack semantics (gaussian noise, JPEG, random patch, dropout),
* determinism across same-seed reset+step pairs,
* multi-camera dropout vs. single-camera no-op,
* graceful ImportError when Pillow is missing for the JPEG attack.

These tests run headless: no real backend env is constructed; a small
fake env (``_FakeImageEnv``) provides the GauntletEnv surface needed by
:class:`ImageAttackWrapper`.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.image_attack import (
    ATTACK_DROPOUT_ONE_CAMERA,
    ATTACK_GAUSSIAN_HIGH,
    ATTACK_GAUSSIAN_LOW,
    ATTACK_IDS,
    ATTACK_JPEG_Q10,
    ATTACK_NAMES,
    ATTACK_NONE,
    ATTACK_RANDOM_PATCH_8X8,
    ImageAttackWrapper,
    apply_image_attack,
)
from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    image_attack,
)

# --------------------------------------------------------------- fake env
# A minimum GauntletEnv stand-in so the wrapper can exercise its
# step/reset/dispatch paths without touching MuJoCo. Mirrors the
# Protocol surface in :mod:`gauntlet.env.base`.


class _FakeImageEnv:
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"lighting_intensity"})

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (32, 32, 3),
        cameras: tuple[str, ...] = (),
    ) -> None:
        self._shape = image_shape
        self._cameras = cameras
        self.observation_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.last_perturbation: tuple[str, float] | None = None
        self._tick = 0

    def _make_image(self) -> NDArray[np.uint8]:
        # Mid-grey so additive noise has room above and below without
        # immediately saturating uint8.
        return np.full(self._shape, 128, dtype=np.uint8)

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        obs: dict[str, NDArray[Any]] = {}
        if self._cameras:
            images = {name: self._make_image() for name in self._cameras}
            obs["images"] = images  # type: ignore[assignment]
            obs["image"] = images[self._cameras[0]].copy()
        else:
            obs["image"] = self._make_image()
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        self._tick = 0
        return self._build_obs(), {}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        self._tick += 1
        return self._build_obs(), 0.0, False, False, {}

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self.last_perturbation = (name, float(value))

    def restore_baseline(self) -> None:
        self.last_perturbation = None

    def close(self) -> None:
        pass


# --------------------------------------------------------------- registry


class TestRegistry:
    def test_axis_in_canonical_registry(self) -> None:
        assert "image_attack" in AXIS_NAMES

    def test_axis_for_returns_categorical_axis(self) -> None:
        axis = axis_for("image_attack")
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "image_attack"
        assert axis.kind == AXIS_KIND_CATEGORICAL

    def test_axis_factory_samples_only_legal_ids(self) -> None:
        axis = image_attack()
        rng = np.random.default_rng(0)
        seen = {round(axis.sample(rng)) for _ in range(50)}
        assert seen.issubset(set(ATTACK_IDS))

    def test_attack_names_parallel_to_ids(self) -> None:
        # The two tuples must agree in length so reports can label
        # samples by name without an out-of-bounds risk.
        assert len(ATTACK_NAMES) == len(ATTACK_IDS)


# --------------------------------------------------------------- pure attacks


class TestPureAttackFunction:
    """Direct coverage of :func:`apply_image_attack` (no wrapper)."""

    def _baseline(self) -> NDArray[np.uint8]:
        return np.full((32, 32, 3), 128, dtype=np.uint8)

    def test_none_returns_input_unchanged(self) -> None:
        img = self._baseline()
        out = apply_image_attack(img, ATTACK_NONE, np.random.default_rng(0))
        np.testing.assert_array_equal(out, img)

    def test_gaussian_low_perturbs_within_uint8_bounds(self) -> None:
        img = self._baseline()
        out = apply_image_attack(img, ATTACK_GAUSSIAN_LOW, np.random.default_rng(0))
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        # Some pixels must have moved (noise is non-zero a.s.).
        assert not np.array_equal(out, img)
        # uint8 saturates at the boundaries — the clip path keeps values
        # inside [0, 255]. uint8 itself enforces this; we double-check
        # the clamp is exercised by ensuring no impossible casts.
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_gaussian_high_has_larger_spread_than_low(self) -> None:
        img = self._baseline()
        low = apply_image_attack(img, ATTACK_GAUSSIAN_LOW, np.random.default_rng(0))
        high = apply_image_attack(img, ATTACK_GAUSSIAN_HIGH, np.random.default_rng(0))
        std_low = float(np.std(low.astype(np.float32) - img.astype(np.float32)))
        std_high = float(np.std(high.astype(np.float32) - img.astype(np.float32)))
        # 0.10 sigma should swamp 0.02 sigma by a wide margin even on a
        # 32x32 image (~3000 samples).
        assert std_high > std_low * 2.5

    def test_random_patch_zeros_exactly_8x8_block(self) -> None:
        img = self._baseline()
        out = apply_image_attack(img, ATTACK_RANDOM_PATCH_8X8, np.random.default_rng(0))
        # The number of zero-pixels must be exactly 8 * 8 = 64 (one
        # patch on a uniform-128 image, no other pixel hits zero).
        zero_mask = np.all(out == 0, axis=-1)
        assert int(zero_mask.sum()) == 64

    def test_random_patch_deterministic_across_same_seed(self) -> None:
        img = self._baseline()
        out_a = apply_image_attack(img, ATTACK_RANDOM_PATCH_8X8, np.random.default_rng(2026))
        out_b = apply_image_attack(img, ATTACK_RANDOM_PATCH_8X8, np.random.default_rng(2026))
        np.testing.assert_array_equal(out_a, out_b)

    def test_unknown_attack_id_raises(self) -> None:
        img = self._baseline()
        with pytest.raises(ValueError, match="unknown attack id"):
            apply_image_attack(img, 999, np.random.default_rng(0))

    def test_dropout_single_image_is_noop(self) -> None:
        img = self._baseline()
        out = apply_image_attack(img, ATTACK_DROPOUT_ONE_CAMERA, np.random.default_rng(0))
        # Per the ``apply_image_attack`` contract — single-image
        # dropout no-ops; the wrapper handles multi-camera dispatch.
        np.testing.assert_array_equal(out, img)


# --------------------------------------------------------------- JPEG path


class TestJpegAttack:
    def test_jpeg_runs_when_pillow_present(self) -> None:
        # Pillow is in the [hf]/[lerobot]/[monitor] extras; if installed
        # in the dev env the call must succeed and return a uint8 image
        # with the same shape as the input.
        pytest.importorskip("PIL")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = apply_image_attack(img, ATTACK_JPEG_Q10, np.random.default_rng(0))
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_jpeg_raises_install_hint_when_pillow_missing(self) -> None:
        # Force PIL imports inside the JPEG branch to fail, mimicking
        # an env without the [hf] extra. The wrapper must turn this
        # into a clear ImportError naming the install path; the JPEG
        # value being categorical means the suite YAML can declare it
        # without inspecting the env's extras list, so the surface
        # error has to be the only signal.
        img = np.full((32, 32, 3), 128, dtype=np.uint8)

        # Patch ``builtins.__import__`` so a fresh ``import PIL``
        # inside the function raises. This is robust against PIL
        # being already imported at the test module top.
        real_import = __import__

        def _fake_import(
            name: str,
            import_globals: Any = None,
            import_locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError(f"mocked: {name} unavailable")
            return real_import(name, import_globals, import_locals, fromlist, level)

        # Also drop any cached PIL module so the in-function import
        # is forced to re-resolve.
        saved = {k: v for k, v in sys.modules.items() if k == "PIL" or k.startswith("PIL.")}
        for k in list(saved):
            del sys.modules[k]
        try:
            with (
                patch("builtins.__import__", side_effect=_fake_import),
                pytest.raises(ImportError, match="image_attack jpeg_q10 requires"),
            ):
                apply_image_attack(img, ATTACK_JPEG_Q10, np.random.default_rng(0))
        finally:
            sys.modules.update(saved)


# --------------------------------------------------------------- wrapper


class TestImageAttackWrapper:
    def test_wrapper_axis_names_unions_inner_with_image_attack(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            assert "image_attack" in wrapped.AXIS_NAMES
            assert "lighting_intensity" in wrapped.AXIS_NAMES
        finally:
            wrapped.close()

    def test_inner_axis_delegates_to_inner_env(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("lighting_intensity", 0.42)
            assert env.last_perturbation == ("lighting_intensity", 0.42)
        finally:
            wrapped.close()

    def test_image_attack_intercepted_not_forwarded(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("image_attack", float(ATTACK_GAUSSIAN_LOW))
            # Inner env never sees the image_attack value — it would
            # have raised since image_attack is not in its AXIS_NAMES.
            assert env.last_perturbation is None
        finally:
            wrapped.close()

    def test_image_attack_modifies_obs_image(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("image_attack", float(ATTACK_GAUSSIAN_HIGH))
            obs, _ = wrapped.reset(seed=0)
            baseline = np.full((32, 32, 3), 128, dtype=np.uint8)
            assert not np.array_equal(obs["image"], baseline)
        finally:
            wrapped.close()

    def test_attack_none_passes_obs_through(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("image_attack", float(ATTACK_NONE))
            obs, _ = wrapped.reset(seed=0)
            baseline = np.full((32, 32, 3), 128, dtype=np.uint8)
            np.testing.assert_array_equal(obs["image"], baseline)
        finally:
            wrapped.close()

    def test_same_seed_yields_identical_attacked_obs(self) -> None:
        env_a = _FakeImageEnv()
        env_b = _FakeImageEnv()
        wrapped_a = ImageAttackWrapper(env_a)
        wrapped_b = ImageAttackWrapper(env_b)
        try:
            wrapped_a.set_perturbation("image_attack", float(ATTACK_RANDOM_PATCH_8X8))
            wrapped_b.set_perturbation("image_attack", float(ATTACK_RANDOM_PATCH_8X8))
            obs_a, _ = wrapped_a.reset(seed=2026)
            obs_b, _ = wrapped_b.reset(seed=2026)
            np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
            # And the same is true for subsequent steps — the wrapper
            # RNG advances deterministically per step.
            for _ in range(3):
                obs_a, _, _, _, _ = wrapped_a.step(np.zeros(1))
                obs_b, _, _, _, _ = wrapped_b.step(np.zeros(1))
                np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
        finally:
            wrapped_a.close()
            wrapped_b.close()

    def test_dropout_single_camera_is_noop(self) -> None:
        env = _FakeImageEnv()  # single-camera path
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("image_attack", float(ATTACK_DROPOUT_ONE_CAMERA))
            obs, _ = wrapped.reset(seed=0)
            baseline = np.full((32, 32, 3), 128, dtype=np.uint8)
            np.testing.assert_array_equal(obs["image"], baseline)
        finally:
            wrapped.close()

    def test_dropout_multi_camera_zeros_exactly_one(self) -> None:
        env = _FakeImageEnv(cameras=("top", "wrist", "side"))
        wrapped = ImageAttackWrapper(env)
        try:
            wrapped.set_perturbation("image_attack", float(ATTACK_DROPOUT_ONE_CAMERA))
            obs, _ = wrapped.reset(seed=0)
            cams = obs["images"]
            assert isinstance(cams, dict)
            zero_count = sum(1 for img in cams.values() if int(img.sum()) == 0)
            assert zero_count == 1, f"expected exactly one zeroed cam; got {zero_count}"
            # Legacy alias must match the (possibly attacked) first cam.
            np.testing.assert_array_equal(obs["image"], cams["top"])
        finally:
            wrapped.close()

    def test_invalid_attack_value_rejected_at_set_perturbation(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            with pytest.raises(ValueError, match="image_attack: value must be one of"):
                wrapped.set_perturbation("image_attack", 99.0)
        finally:
            wrapped.close()

    def test_unknown_axis_rejected_at_wrapper(self) -> None:
        env = _FakeImageEnv()
        wrapped = ImageAttackWrapper(env)
        try:
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                wrapped.set_perturbation("not_a_real_axis", 0.0)
        finally:
            wrapped.close()
