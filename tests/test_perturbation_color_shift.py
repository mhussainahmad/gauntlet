"""Tests for the ``color_shift_synthetic`` post-render perturbation axis (B-43).

The axis lives in :mod:`gauntlet.env.color_attack` as a wrapper around
any :class:`gauntlet.env.base.GauntletEnv`. It operates on the rendered
``obs["image"]`` (single-camera) or ``obs["images"][name]`` (multi-cam)
arrays — backend-agnostic by construction. Mirrors the B-31
image_attack test layout.

Tests cover:

* registration in the canonical AXIS_NAMES tuple,
* HSV round-trip identity within float-tolerance,
* per-shift semantics (hue rotate, saturation scale, achromatic),
* observation shape / dtype preserved,
* axis YAML parsing through the suite schema,
* applying through wrapper -> inner ``set_perturbation`` delegation.

These tests run headless: no real backend env is constructed; a small
fake env (``_FakeImageEnv``) provides the GauntletEnv surface needed by
:class:`ColorShiftWrapper`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.color_attack import (
    SHIFT_ACHROMATIC,
    SHIFT_HUE_MINUS_30,
    SHIFT_HUE_PLUS_30,
    SHIFT_IDS,
    SHIFT_NAMES,
    SHIFT_NONE,
    SHIFT_SATURATION_0_5,
    SHIFT_SATURATION_1_5,
    ColorShiftWrapper,
    apply_color_shift,
    hsv_to_rgb,
    rgb_to_hsv,
)
from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    color_shift_synthetic,
)
from gauntlet.suite.schema import AxisSpec, Suite

# --------------------------------------------------------------- fake env
# A minimum GauntletEnv stand-in so the wrapper can exercise its
# step/reset/dispatch paths without touching MuJoCo. Mirrors the
# Protocol surface in :mod:`gauntlet.env.base`. The image content is
# parameterised so we can test HSV semantics on saturated colors that
# have well-defined hue (mid-grey would have undefined hue).


class _FakeImageEnv:
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"lighting_intensity"})

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (32, 32, 3),
        cameras: tuple[str, ...] = (),
        fill_rgb: tuple[int, int, int] = (200, 100, 50),
    ) -> None:
        self._shape = image_shape
        self._cameras = cameras
        self._fill = fill_rgb
        self.observation_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.last_perturbation: tuple[str, float] | None = None
        self._tick = 0

    def _make_image(self) -> NDArray[np.uint8]:
        # Saturated color so hue is well-defined; HSV transforms on a
        # uniform mid-grey patch are no-ops because S=0 there.
        img = np.empty(self._shape, dtype=np.uint8)
        img[..., 0] = self._fill[0]
        img[..., 1] = self._fill[1]
        img[..., 2] = self._fill[2]
        return img

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
        assert "color_shift_synthetic" in AXIS_NAMES

    def test_axis_for_returns_categorical_axis(self) -> None:
        axis = axis_for("color_shift_synthetic")
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "color_shift_synthetic"
        assert axis.kind == AXIS_KIND_CATEGORICAL

    def test_axis_factory_samples_only_legal_ids(self) -> None:
        axis = color_shift_synthetic()
        rng = np.random.default_rng(0)
        seen = {round(axis.sample(rng)) for _ in range(50)}
        assert seen.issubset(set(SHIFT_IDS))

    def test_shift_names_parallel_to_ids(self) -> None:
        # The two tuples must agree in length so reports can label
        # samples by name without an out-of-bounds risk.
        assert len(SHIFT_NAMES) == len(SHIFT_IDS)

    def test_shift_names_match_spec_labels(self) -> None:
        # Spec B-43 calls out exactly five labels (plus ``none``); we
        # guard the surface so a future rename in the wrapper doesn't
        # silently re-label the categorical axis.
        assert SHIFT_NAMES == (
            "none",
            "hue_+30",
            "hue_-30",
            "saturation_0.5",
            "saturation_1.5",
            "achromatic",
        )


# --------------------------------------------------------------- HSV round-trip


class TestHsvRoundTrip:
    """Direct coverage of :func:`rgb_to_hsv` / :func:`hsv_to_rgb`."""

    def test_round_trip_recovers_input_within_float_tolerance(self) -> None:
        # Random saturated RGB image; round-trip must recover it within
        # the float32 round-off (the conversion involves divisions and
        # the inverse sextant lookup).
        rng = np.random.default_rng(2026)
        rgb = rng.uniform(0.0, 1.0, size=(8, 8, 3)).astype(np.float32)
        hsv = rgb_to_hsv(rgb)
        back = hsv_to_rgb(hsv)
        np.testing.assert_allclose(back, rgb, atol=1e-5)

    def test_pure_red_has_hue_zero(self) -> None:
        red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        hsv = rgb_to_hsv(red)
        assert hsv[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert hsv[0, 0, 1] == pytest.approx(1.0, abs=1e-6)
        assert hsv[0, 0, 2] == pytest.approx(1.0, abs=1e-6)

    def test_pure_green_has_hue_one_third(self) -> None:
        green = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        hsv = rgb_to_hsv(green)
        # Green sits at H = 120deg = 1/3 of a turn.
        assert hsv[0, 0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_achromatic_input_has_zero_saturation(self) -> None:
        gray = np.full((4, 4, 3), 0.5, dtype=np.float32)
        hsv = rgb_to_hsv(gray)
        np.testing.assert_array_equal(hsv[..., 1], np.zeros((4, 4), dtype=np.float32))


# --------------------------------------------------------------- shifts


class TestPureShiftFunction:
    """Direct coverage of :func:`apply_color_shift` (no wrapper)."""

    def _saturated(self) -> NDArray[np.uint8]:
        # Pure red — H=0, S=1, V=1 — so hue rotations have a clean
        # geometric interpretation.
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[..., 0] = 255
        return img

    def _partially_saturated(self) -> NDArray[np.uint8]:
        # Custom (200, 100, 50) — H around 30deg (orange), S = 0.75,
        # V = 200/255. Useful for the saturation_1.5 test where pure
        # red would clip immediately at S=1.
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[..., 0] = 200
        img[..., 1] = 100
        img[..., 2] = 50
        return img

    def test_none_returns_input_unchanged(self) -> None:
        img = self._saturated()
        out = apply_color_shift(img, SHIFT_NONE)
        np.testing.assert_array_equal(out, img)
        # Identity branch returns the *same* object — the wrapper relies
        # on this to skip the obs-mutation path entirely.
        assert out is img

    def test_unknown_shift_id_raises(self) -> None:
        img = self._saturated()
        with pytest.raises(ValueError, match="unknown shift id"):
            apply_color_shift(img, 999)

    def test_hue_plus_30_shifts_hue_by_30_degrees(self) -> None:
        img = self._saturated()
        out = apply_color_shift(img, SHIFT_HUE_PLUS_30)
        # Convert both to HSV float and read the hue channel; difference
        # must be (30 / 360) within float-tolerance, modulo wrap.
        h_in = rgb_to_hsv(img.astype(np.float32) / 255.0)[0, 0, 0]
        h_out = rgb_to_hsv(out.astype(np.float32) / 255.0)[0, 0, 0]
        delta = (h_out - h_in) % 1.0
        assert delta == pytest.approx(30.0 / 360.0, abs=2e-3)

    def test_hue_minus_30_shifts_hue_by_negative_30_degrees(self) -> None:
        img = self._partially_saturated()
        out = apply_color_shift(img, SHIFT_HUE_MINUS_30)
        h_in = rgb_to_hsv(img.astype(np.float32) / 255.0)[0, 0, 0]
        h_out = rgb_to_hsv(out.astype(np.float32) / 255.0)[0, 0, 0]
        # (in - out) wraps to +30deg modulo 360; equivalent to
        # ``(in - out) mod 1 == 30/360``.
        delta = (h_in - h_out) % 1.0
        assert delta == pytest.approx(30.0 / 360.0, abs=2e-3)

    def test_saturation_0_5_halves_saturation_channel(self) -> None:
        img = self._partially_saturated()
        out = apply_color_shift(img, SHIFT_SATURATION_0_5)
        s_in = rgb_to_hsv(img.astype(np.float32) / 255.0)[0, 0, 1]
        s_out = rgb_to_hsv(out.astype(np.float32) / 255.0)[0, 0, 1]
        # uint8 round-trip introduces some quantisation; tolerance
        # ~1/255 over the saturation channel.
        assert s_out == pytest.approx(s_in * 0.5, abs=2e-3)

    def test_saturation_1_5_increases_saturation_channel(self) -> None:
        # Use a low-saturation source so 1.5x doesn't clip to 1.0.
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[..., 0] = 180
        img[..., 1] = 150
        img[..., 2] = 130
        out = apply_color_shift(img, SHIFT_SATURATION_1_5)
        s_in = rgb_to_hsv(img.astype(np.float32) / 255.0)[0, 0, 1]
        s_out = rgb_to_hsv(out.astype(np.float32) / 255.0)[0, 0, 1]
        assert s_out == pytest.approx(s_in * 1.5, abs=3e-3)
        assert s_out <= 1.0

    def test_saturation_1_5_clips_at_one(self) -> None:
        # Pure red has S=1 already — multiplying by 1.5 must clip back
        # to 1, NOT exceed it (uint8 saturation is enforced via the
        # ``np.clip`` in ``_saturation_scale``).
        img = self._saturated()
        out = apply_color_shift(img, SHIFT_SATURATION_1_5)
        s_out = rgb_to_hsv(out.astype(np.float32) / 255.0)[0, 0, 1]
        assert s_out == pytest.approx(1.0, abs=2e-3)
        # And the byte values must stay inside [0, 255] (uint8
        # enforces this; we double-check the clamp is exercised).
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_achromatic_produces_grayscale_per_pixel(self) -> None:
        # Heterogeneous image — different colors per pixel — ensures
        # the achromatic transform is applied pixel-wise rather than
        # collapsing the whole image to a single gray.
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = (200, 100, 50)
        img[0, 1] = (50, 200, 100)
        img[1, 0] = (100, 50, 200)
        img[1, 1] = (10, 240, 30)
        # Fill the rest with one solid color so the test array is well
        # defined; the assertion still walks every pixel.
        img[2:, :] = (128, 64, 192)
        img[:, 2:] = (200, 50, 80)
        out = apply_color_shift(img, SHIFT_ACHROMATIC)
        # R == G == B per pixel.
        np.testing.assert_array_equal(out[..., 0], out[..., 1])
        np.testing.assert_array_equal(out[..., 1], out[..., 2])
        # And the gray level matches the per-pixel max channel of the
        # input (HSV S=0 -> R=G=B=V; documented in the module
        # docstring as the colorsys/opencv convention).
        per_pixel_max = img.max(axis=-1)
        np.testing.assert_allclose(out[..., 0], per_pixel_max, atol=1)

    def test_shapes_and_dtype_preserved(self) -> None:
        img = self._partially_saturated()
        for shift_id in SHIFT_IDS:
            out = apply_color_shift(img, shift_id)
            assert out.shape == img.shape
            assert out.dtype == np.uint8


# --------------------------------------------------------------- wrapper


class TestColorShiftWrapper:
    def test_wrapper_axis_names_unions_inner_with_color_shift(self) -> None:
        env = _FakeImageEnv()
        wrapped = ColorShiftWrapper(env)
        try:
            assert "color_shift_synthetic" in wrapped.AXIS_NAMES
            assert "lighting_intensity" in wrapped.AXIS_NAMES
        finally:
            wrapped.close()

    def test_inner_axis_delegates_to_inner_env(self) -> None:
        env = _FakeImageEnv()
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("lighting_intensity", 0.42)
            assert env.last_perturbation == ("lighting_intensity", 0.42)
        finally:
            wrapped.close()

    def test_color_shift_intercepted_not_forwarded(self) -> None:
        env = _FakeImageEnv()
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_HUE_PLUS_30))
            # Inner env never sees the color_shift value — it would
            # have raised since color_shift_synthetic is not in its
            # AXIS_NAMES.
            assert env.last_perturbation is None
        finally:
            wrapped.close()

    def test_color_shift_modifies_obs_image(self) -> None:
        env = _FakeImageEnv(fill_rgb=(200, 100, 50))
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_ACHROMATIC))
            obs, _ = wrapped.reset(seed=0)
            # Achromatic forces R == G == B per pixel.
            np.testing.assert_array_equal(obs["image"][..., 0], obs["image"][..., 1])
            np.testing.assert_array_equal(obs["image"][..., 1], obs["image"][..., 2])
        finally:
            wrapped.close()

    def test_shift_none_passes_obs_through(self) -> None:
        env = _FakeImageEnv(fill_rgb=(200, 100, 50))
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_NONE))
            obs, _ = wrapped.reset(seed=0)
            baseline = np.zeros((32, 32, 3), dtype=np.uint8)
            baseline[..., 0] = 200
            baseline[..., 1] = 100
            baseline[..., 2] = 50
            np.testing.assert_array_equal(obs["image"], baseline)
        finally:
            wrapped.close()

    def test_observation_shape_preserved_through_wrapper(self) -> None:
        env = _FakeImageEnv(image_shape=(48, 64, 3))
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_HUE_PLUS_30))
            obs, _ = wrapped.reset(seed=0)
            assert obs["image"].shape == (48, 64, 3)
            assert obs["image"].dtype == np.uint8
            obs2, _, _, _, _ = wrapped.step(np.zeros(1))
            assert obs2["image"].shape == (48, 64, 3)
            assert obs2["image"].dtype == np.uint8
        finally:
            wrapped.close()

    def test_deterministic_same_input_same_output(self) -> None:
        # Two independently-constructed wrappers, same shift, same
        # input frame -> bit-identical output (the wrapper carries no
        # RNG, so determinism is structural rather than seeded).
        env_a = _FakeImageEnv()
        env_b = _FakeImageEnv()
        wrapped_a = ColorShiftWrapper(env_a)
        wrapped_b = ColorShiftWrapper(env_b)
        try:
            wrapped_a.set_perturbation("color_shift_synthetic", float(SHIFT_HUE_MINUS_30))
            wrapped_b.set_perturbation("color_shift_synthetic", float(SHIFT_HUE_MINUS_30))
            obs_a, _ = wrapped_a.reset(seed=0)
            obs_b, _ = wrapped_b.reset(seed=0)
            np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
            for _ in range(3):
                obs_a, _, _, _, _ = wrapped_a.step(np.zeros(1))
                obs_b, _, _, _, _ = wrapped_b.step(np.zeros(1))
                np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
        finally:
            wrapped_a.close()
            wrapped_b.close()

    def test_multi_camera_shifts_every_camera(self) -> None:
        env = _FakeImageEnv(cameras=("top", "wrist", "side"))
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_ACHROMATIC))
            obs, _ = wrapped.reset(seed=0)
            cams = obs["images"]
            assert isinstance(cams, dict)
            for name in ("top", "wrist", "side"):
                cam = cams[name]
                np.testing.assert_array_equal(cam[..., 0], cam[..., 1])
                np.testing.assert_array_equal(cam[..., 1], cam[..., 2])
            # Legacy alias must match the (shifted) first cam.
            np.testing.assert_array_equal(obs["image"], cams["top"])
        finally:
            wrapped.close()

    def test_invalid_shift_value_rejected_at_set_perturbation(self) -> None:
        env = _FakeImageEnv()
        wrapped = ColorShiftWrapper(env)
        try:
            with pytest.raises(ValueError, match="color_shift_synthetic: value must be one of"):
                wrapped.set_perturbation("color_shift_synthetic", 99.0)
        finally:
            wrapped.close()

    def test_unknown_axis_rejected_at_wrapper(self) -> None:
        env = _FakeImageEnv()
        wrapped = ColorShiftWrapper(env)
        try:
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                wrapped.set_perturbation("not_a_real_axis", 0.0)
        finally:
            wrapped.close()

    def test_restore_baseline_clears_pending_shift(self) -> None:
        env = _FakeImageEnv(fill_rgb=(200, 100, 50))
        wrapped = ColorShiftWrapper(env)
        try:
            wrapped.set_perturbation("color_shift_synthetic", float(SHIFT_ACHROMATIC))
            wrapped.restore_baseline()
            obs, _ = wrapped.reset(seed=0)
            # Baseline restored -> shift is SHIFT_NONE -> obs untouched.
            assert obs["image"][0, 0, 0] == 200
            assert obs["image"][0, 0, 1] == 100
            assert obs["image"][0, 0, 2] == 50
        finally:
            wrapped.close()

    def test_attribute_proxy_to_inner_env(self) -> None:
        # Inner env extras (e.g. backend-specific helpers) flow through
        # __getattr__ — this is the same proxy pattern as
        # ImageAttackWrapper / InstructionWrapper.
        env = _FakeImageEnv()

        # Tag the inner with an attribute the wrapper does not define.
        def _custom_helper() -> str:
            return "from_inner"

        env._custom_helper = _custom_helper  # type: ignore[attr-defined]
        wrapped = ColorShiftWrapper(env)
        try:
            # ``__getattr__`` proxy returns ``Any`` so mypy lets the
            # call site through without a per-call type ignore.
            assert wrapped._custom_helper() == "from_inner"
        finally:
            wrapped.close()


# --------------------------------------------------------------- YAML parsing


class TestSuiteSchemaParsing:
    """Suite-loader axis registration / YAML round-trip."""

    def test_axis_spec_accepts_color_shift_synthetic_categorical_values(self) -> None:
        # Direct AxisSpec construction (the YAML loader produces this
        # via pydantic). Categorical shape with the canonical 6-id list.
        spec = AxisSpec(values=[float(i) for i in SHIFT_IDS])
        # Enumerate must surface the same 6 ids back.
        enumerated = spec.enumerate()
        assert enumerated == tuple(float(i) for i in SHIFT_IDS)

    def test_suite_yaml_with_color_shift_synthetic_axis_loads(self) -> None:
        # End-to-end: build a Suite with the new axis and confirm the
        # schema validator (which checks ``axes`` keys against
        # ``AXIS_NAMES``) accepts the name.
        suite = Suite(
            name="color-shift-smoke",
            env="tabletop",
            episodes_per_cell=1,
            axes={
                "color_shift_synthetic": AxisSpec(values=[float(i) for i in SHIFT_IDS]),
            },
        )
        assert "color_shift_synthetic" in suite.axes
        # The grid expansion enumerates all 6 ids as separate cells.
        cells = list(suite.cells())
        assert len(cells) == len(SHIFT_IDS)
        observed = {round(cell.values["color_shift_synthetic"]) for cell in cells}
        assert observed == set(SHIFT_IDS)
