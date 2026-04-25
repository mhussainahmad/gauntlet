"""Tests for the ``instruction_paraphrase`` perturbation axis (B-05).

The axis lives in :mod:`gauntlet.env.instruction` as a wrapper around
any :class:`gauntlet.env.base.GauntletEnv`. It injects a single
``obs["instruction"]`` string per :meth:`step` / :meth:`reset` so VLA
policies (OpenVLA, SmolVLA, pi0 family) can observe the active task
phrasing — backend-agnostic by construction.

Tests cover:

* registration in the canonical AXIS_NAMES tuple,
* axis kind + factory shape (categorical, placeholder sampler),
* schema YAML accepts the string-list ``values:`` shape on the axis,
* schema YAML rejects string-list ``values:`` on every other axis,
* :meth:`AxisSpec.enumerate` emits indices for the string shape,
* :meth:`AxisSpec.paraphrases` round-trips the strings,
* the wrapper sets ``obs["instruction"]`` on reset and step,
* index toggles select the right paraphrase across cells,
* single-paraphrase suite collapses to baseline,
* invalid index raises a clear error,
* unknown axis raised at the wrapper boundary,
* inner-axis perturbations delegate to the inner env.

These tests run headless: no real backend env is constructed; a small
fake env (``_FakeInstructionEnv``) provides the GauntletEnv surface
needed by :class:`InstructionWrapper`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from gauntlet.env.instruction import (
    AXIS_NAME,
    DEFAULT_INSTRUCTION_KEY,
    InstructionWrapper,
)
from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    instruction_paraphrase,
)
from gauntlet.suite.loader import load_suite_from_string
from gauntlet.suite.schema import AxisSpec, Suite

# --------------------------------------------------------------- fake env
# Minimum GauntletEnv stand-in so the wrapper can exercise its
# step/reset/dispatch paths without touching MuJoCo. Mirrors the
# Protocol surface in :mod:`gauntlet.env.base`. Same shape as the
# ``_FakeImageEnv`` in tests/test_perturbation_image_attack.py — keep
# them parallel so fixes here can be ported there cleanly.


class _FakeInstructionEnv:
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"lighting_intensity"})

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8)}
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.last_perturbation: tuple[str, float] | None = None
        self.reset_calls: int = 0
        self.step_calls: int = 0

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        return {"image": np.full((4, 4, 3), 0, dtype=np.uint8)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        self.reset_calls += 1
        return self._build_obs(), {}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        self.step_calls += 1
        return self._build_obs(), 0.0, False, False, {}

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self.last_perturbation = (name, float(value))

    def restore_baseline(self) -> None:
        self.last_perturbation = None

    def close(self) -> None:
        pass


_PARAPHRASES = (
    "pick up the red cube",
    "grab the crimson block",
    "move the scarlet box",
)


# --------------------------------------------------------------- registry


class TestRegistry:
    def test_axis_in_canonical_registry(self) -> None:
        assert AXIS_NAME == "instruction_paraphrase"
        assert AXIS_NAME in AXIS_NAMES

    def test_axis_for_returns_categorical_axis(self) -> None:
        axis = axis_for(AXIS_NAME)
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == AXIS_NAME
        assert axis.kind == AXIS_KIND_CATEGORICAL

    def test_factory_default_sampler_emits_zero(self) -> None:
        # The default sampler is a placeholder over ``(0.0,)`` because
        # the real cardinality comes from the suite YAML's paraphrase
        # list. Sampling must always return index 0.
        axis = instruction_paraphrase()
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert axis.sample(rng) == 0.0

    def test_factory_default_bounds_cover_default_sample(self) -> None:
        axis = instruction_paraphrase()
        # Default sample (0.0) must lie inside the declared envelope.
        assert axis.low <= 0.0 <= axis.high


# --------------------------------------------------------------- schema


class TestSchemaStringValues:
    def test_axis_spec_accepts_string_values(self) -> None:
        spec = AxisSpec(values=list(_PARAPHRASES))
        # Strings preserved on the model.
        assert spec.values == list(_PARAPHRASES)
        # enumerate() emits indices, not strings — runner stays float-only.
        assert spec.enumerate() == (0.0, 1.0, 2.0)
        # paraphrases() round-trips the strings in declared order.
        assert spec.paraphrases() == _PARAPHRASES

    def test_paraphrases_returns_none_for_float_values(self) -> None:
        spec = AxisSpec(values=[0.0, 1.0])
        assert spec.paraphrases() is None

    def test_paraphrases_returns_none_for_continuous_axis(self) -> None:
        spec = AxisSpec(low=0.0, high=1.0, steps=2)
        assert spec.paraphrases() is None

    def test_string_values_only_legal_on_instruction_paraphrase_axis(self) -> None:
        # Building the AxisSpec alone is fine — the cross-axis check
        # lives on the parent ``Suite``. A confused user mapping
        # paraphrases to e.g. ``object_texture`` must hit the validator.
        with pytest.raises(ValidationError, match="instruction_paraphrase"):
            Suite(
                name="bad",
                env="tabletop",
                episodes_per_cell=1,
                axes={"object_texture": AxisSpec(values=["red", "green"])},
            )

    def test_yaml_loader_round_trips_paraphrase_axis(self) -> None:
        yaml_text = (
            "name: lang-suite\n"
            "env: tabletop\n"
            "episodes_per_cell: 1\n"
            "axes:\n"
            "  instruction_paraphrase:\n"
            "    values:\n"
            "      - pick up the red cube\n"
            "      - grab the crimson block\n"
            "      - move the scarlet box\n"
        )
        suite = load_suite_from_string(yaml_text)
        spec = suite.axes["instruction_paraphrase"]
        assert spec.paraphrases() == _PARAPHRASES
        # Cartesian enumeration: one cell per paraphrase, index-coded.
        cells = list(suite.cells())
        assert len(cells) == 3
        assert [c.values["instruction_paraphrase"] for c in cells] == [0.0, 1.0, 2.0]

    def test_single_paraphrase_collapses_to_baseline(self) -> None:
        # B-05 anti-feature — a one-element paraphrase list has no OOD
        # variation; the sweep degenerates to a single baseline cell so
        # users can wire a "no-paraphrase" suite without special-casing.
        spec = AxisSpec(values=["pick up the red cube"])
        assert spec.enumerate() == (0.0,)
        assert spec.paraphrases() == ("pick up the red cube",)


# --------------------------------------------------------------- wrapper


class TestInstructionWrapperBasics:
    def test_wrapper_axis_names_unions_inner_with_paraphrase(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            assert AXIS_NAME in wrapped.AXIS_NAMES
            assert "lighting_intensity" in wrapped.AXIS_NAMES
        finally:
            wrapped.close()

    def test_inner_axis_delegates_to_inner_env(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            wrapped.set_perturbation("lighting_intensity", 0.42)
            assert env.last_perturbation == ("lighting_intensity", 0.42)
        finally:
            wrapped.close()

    def test_paraphrase_axis_intercepted_not_forwarded(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            wrapped.set_perturbation(AXIS_NAME, 1.0)
            # Inner env never sees the paraphrase axis — it would have
            # raised since instruction_paraphrase is not in its
            # AXIS_NAMES. The wrapper picked up the index instead.
            assert env.last_perturbation is None
            assert wrapped.current_instruction == _PARAPHRASES[1]
        finally:
            wrapped.close()

    def test_default_index_is_zero(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            # No set_perturbation call — wrapper defaults to baseline.
            assert wrapped.current_instruction == _PARAPHRASES[0]
        finally:
            wrapped.close()

    def test_empty_paraphrases_rejected(self) -> None:
        env = _FakeInstructionEnv()
        with pytest.raises(ValueError, match="at least one paraphrase"):
            InstructionWrapper(env, ())


class TestInstructionInjection:
    def test_reset_injects_instruction_at_index_zero(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            obs, _ = wrapped.reset(seed=0)
            assert obs[DEFAULT_INSTRUCTION_KEY] == _PARAPHRASES[0]
        finally:
            wrapped.close()

    def test_step_injects_instruction_after_set_perturbation(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            wrapped.set_perturbation(AXIS_NAME, 2.0)
            obs, _ = wrapped.reset(seed=0)
            assert obs[DEFAULT_INSTRUCTION_KEY] == _PARAPHRASES[2]
            obs2, _, _, _, _ = wrapped.step(np.zeros(1))
            assert obs2[DEFAULT_INSTRUCTION_KEY] == _PARAPHRASES[2]
        finally:
            wrapped.close()

    def test_index_toggles_across_cells(self) -> None:
        # Mirrors the runner-side flow: between cells, the runner calls
        # ``set_perturbation`` to switch the active paraphrase and then
        # ``reset`` to start a new episode. The injected obs key must
        # follow the index.
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            seen: list[str] = []
            for idx in (0, 1, 2, 1, 0):
                wrapped.set_perturbation(AXIS_NAME, float(idx))
                obs, _ = wrapped.reset(seed=idx)
                seen.append(obs[DEFAULT_INSTRUCTION_KEY])
            assert seen == [
                _PARAPHRASES[0],
                _PARAPHRASES[1],
                _PARAPHRASES[2],
                _PARAPHRASES[1],
                _PARAPHRASES[0],
            ]
        finally:
            wrapped.close()

    def test_fractional_index_rounds_to_nearest(self) -> None:
        # Categorical samplers emit floats; the wrapper rounds to the
        # nearest legal index. ``round(0.6) == 1`` per Python's banker's
        # rounding behaviour.
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            wrapped.set_perturbation(AXIS_NAME, 0.6)
            obs, _ = wrapped.reset(seed=0)
            assert obs[DEFAULT_INSTRUCTION_KEY] == _PARAPHRASES[1]
        finally:
            wrapped.close()

    def test_single_paraphrase_collapses_to_baseline(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, ("pick up the red cube",))
        try:
            obs, _ = wrapped.reset(seed=0)
            assert obs[DEFAULT_INSTRUCTION_KEY] == "pick up the red cube"
            for _ in range(3):
                obs, _, _, _, _ = wrapped.step(np.zeros(1))
                assert obs[DEFAULT_INSTRUCTION_KEY] == "pick up the red cube"
        finally:
            wrapped.close()

    def test_restore_baseline_resets_to_index_zero(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            wrapped.set_perturbation(AXIS_NAME, 2.0)
            assert wrapped.current_instruction == _PARAPHRASES[2]
            wrapped.restore_baseline()
            assert wrapped.current_instruction == _PARAPHRASES[0]
            # And the inner env's baseline path was invoked too.
            assert env.last_perturbation is None
        finally:
            wrapped.close()


class TestErrorPaths:
    def test_invalid_index_rejected_at_set_perturbation(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            with pytest.raises(ValueError, match="out of range"):
                wrapped.set_perturbation(AXIS_NAME, 99.0)
            with pytest.raises(ValueError, match="out of range"):
                wrapped.set_perturbation(AXIS_NAME, -1.0)
        finally:
            wrapped.close()

    def test_unknown_axis_rejected_at_wrapper(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES)
        try:
            with pytest.raises(ValueError, match="unknown perturbation axis"):
                wrapped.set_perturbation("not_a_real_axis", 0.0)
        finally:
            wrapped.close()

    def test_custom_instruction_key_overrides_default(self) -> None:
        env = _FakeInstructionEnv()
        wrapped = InstructionWrapper(env, _PARAPHRASES, instruction_key="task")
        try:
            obs, _ = wrapped.reset(seed=0)
            assert obs["task"] == _PARAPHRASES[0]
            # The default key MUST not appear when overridden.
            assert DEFAULT_INSTRUCTION_KEY not in obs
        finally:
            wrapped.close()
