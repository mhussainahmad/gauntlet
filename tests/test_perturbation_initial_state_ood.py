"""Tests for the ``initial_state_ood`` perturbation axis (B-32).

The OOD axis extends ``ObjectInitialPose`` with the LIBERO-PRO /
LIBERO-Plus convention: an axis value ``V`` is a unitless sigma
multiplier and the env interprets it as a per-dim displacement of
``V * prior_std * sign`` from ``prior_mean``. The sign is drawn
deterministically from a seed-derived sub-stream so adding the axis
cannot perturb the env's main RNG (which other axes / target draws
consume).

These tests must run headless (no GL): no ``env.render()`` calls.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from gauntlet.env import TabletopEnv
from gauntlet.env.perturbation import (
    AXIS_KIND_CONTINUOUS,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    initial_state_ood,
)
from gauntlet.suite.schema import AxisSpec, Suite


@pytest.fixture
def env() -> Iterator[TabletopEnv]:
    """Fresh TabletopEnv per test, closed on teardown."""
    e = TabletopEnv()
    try:
        yield e
    finally:
        e.close()


# --------------------------------------------------------------- registry


class TestAxisRegistration:
    def test_axis_name_in_canonical_registry(self) -> None:
        assert "initial_state_ood" in AXIS_NAMES

    def test_axis_name_in_tabletop_axis_names(self) -> None:
        assert "initial_state_ood" in TabletopEnv.AXIS_NAMES

    def test_axis_for_returns_continuous_axis(self) -> None:
        axis = axis_for("initial_state_ood")
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "initial_state_ood"
        assert axis.kind == AXIS_KIND_CONTINUOUS

    def test_factory_default_bounds_are_nonneg(self) -> None:
        # Sigma multipliers are magnitudes; sign comes from the env seed.
        axis = initial_state_ood()
        assert axis.low >= 0.0
        assert axis.high > axis.low

    def test_set_perturbation_rejects_negative_sigma(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            env.set_perturbation("initial_state_ood", -0.5)


# --------------------------------------------------------------- YAML grammar


class TestYamlGrammar:
    def test_axis_spec_accepts_prior_fields(self) -> None:
        spec = AxisSpec(
            values=[1.0, 2.0, 3.0],
            prior_mean=[0.5, 0.0, 0.0],
            prior_std=[0.05, 0.05, 0.0],
        )
        assert spec.values == [1.0, 2.0, 3.0]
        assert spec.prior_mean == [0.5, 0.0, 0.0]
        assert spec.prior_std == [0.05, 0.05, 0.0]
        assert spec.enumerate() == (1.0, 2.0, 3.0)

    def test_suite_accepts_initial_state_ood_axis(self) -> None:
        suite = Suite(
            name="ood-demo",
            env="tabletop",
            episodes_per_cell=1,
            seed=0,
            axes={
                "initial_state_ood": AxisSpec(
                    values=[1.0, 2.0, 3.0],
                    prior_mean=[0.5, 0.0, 0.0],
                    prior_std=[0.05, 0.05, 0.0],
                ),
            },
        )
        cells = list(suite.cells())
        # Three sigma multipliers -> three cells.
        assert len(cells) == 3
        sampled = [cell.values["initial_state_ood"] for cell in cells]
        assert sampled == [1.0, 2.0, 3.0]

    def test_prior_mean_must_be_length_three(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            AxisSpec(values=[1.0], prior_mean=[0.5, 0.0])

    def test_prior_std_must_be_length_three(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            AxisSpec(values=[1.0], prior_std=[0.05])

    def test_prior_fields_rejected_on_other_axes(self) -> None:
        # The aux fields are valid on ``AxisSpec`` itself (alongside
        # ``values``) but the parent ``Suite`` restricts them to the
        # ``initial_state_ood`` axis.
        with pytest.raises(ValueError, match="initial_state_ood"):
            Suite(
                name="bad",
                env="tabletop",
                episodes_per_cell=1,
                axes={
                    "lighting_intensity": AxisSpec(
                        values=[0.5, 1.0],
                        prior_mean=[0.0, 0.0, 0.0],
                    ),
                },
            )

    def test_prior_fields_rejected_with_continuous_shape(self) -> None:
        with pytest.raises(ValueError, match="prior_mean / prior_std"):
            AxisSpec(low=0.0, high=3.0, steps=4, prior_mean=[0.0, 0.0, 0.0])


# ------------------------------------------------------------ apply path


class TestApplyPath:
    """The env-side apply path computes ``mean + V * std * sign_per_dim``."""

    def test_value_zero_collapses_to_prior_mean(self, env: TabletopEnv) -> None:
        env.set_initial_state_ood_prior(mean=(0.5, -0.1, 0.0), std=(0.05, 0.05, 0.0))
        env.set_perturbation("initial_state_ood", 0.0)
        obs, _ = env.reset(seed=0)
        assert float(obs["cube_pos"][0]) == pytest.approx(0.5, abs=1e-9)
        assert float(obs["cube_pos"][1]) == pytest.approx(-0.1, abs=1e-9)

    def test_value_nonzero_displaces_by_v_times_std(self, env: TabletopEnv) -> None:
        # Choose a prior whose displacement is large enough to dominate
        # any baseline jitter and easy to read off in test output.
        env.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=(0.10, 0.05, 0.0))
        env.set_perturbation("initial_state_ood", 2.0)
        obs, _ = env.reset(seed=0)
        # Magnitudes equal V * std exactly; sign is seed-dependent.
        assert abs(float(obs["cube_pos"][0])) == pytest.approx(0.20, abs=1e-9)
        assert abs(float(obs["cube_pos"][1])) == pytest.approx(0.10, abs=1e-9)

    def test_value_three_sigma_pushes_to_ood_tail(self, env: TabletopEnv) -> None:
        # Sanity: a 3-sigma multiplier overshoots the 1-sigma boundary.
        prior_std_x = 0.05
        env.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=(prior_std_x, 0.0, 0.0))
        env.set_perturbation("initial_state_ood", 3.0)
        obs, _ = env.reset(seed=42)
        assert abs(float(obs["cube_pos"][0])) > prior_std_x  # past 1-sigma
        assert abs(float(obs["cube_pos"][0])) == pytest.approx(3 * prior_std_x, abs=1e-9)

    def test_default_prior_uses_training_xy_distribution(self, env: TabletopEnv) -> None:
        # No explicit prior call -> defaults to the env's training-time
        # uniform-XY std (halfrange / sqrt(3)).
        expected_std = TabletopEnv._CUBE_INIT_HALFRANGE / np.sqrt(3.0)
        env.set_perturbation("initial_state_ood", 1.0)
        obs, _ = env.reset(seed=0)
        # 1-sigma displacement matches the training-distribution std.
        assert abs(float(obs["cube_pos"][0])) == pytest.approx(expected_std, abs=1e-9)
        assert abs(float(obs["cube_pos"][1])) == pytest.approx(expected_std, abs=1e-9)


# --------------------------------------------------------------- determinism


class TestDeterminism:
    def test_same_seed_same_pose(self) -> None:
        env_a = TabletopEnv()
        env_b = TabletopEnv()
        try:
            env_a.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=(0.05, 0.05, 0.0))
            env_b.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=(0.05, 0.05, 0.0))
            env_a.set_perturbation("initial_state_ood", 2.0)
            env_b.set_perturbation("initial_state_ood", 2.0)
            obs_a, _ = env_a.reset(seed=2026)
            obs_b, _ = env_b.reset(seed=2026)
            np.testing.assert_array_equal(obs_a["cube_pos"], obs_b["cube_pos"])
        finally:
            env_a.close()
            env_b.close()

    def test_different_seeds_can_yield_different_signs(self) -> None:
        # Sign is drawn from a seed-derived sub-stream. Across many
        # seeds the magnitude is constant but the sign varies.
        prior_std = (0.05, 0.05, 0.0)
        signs_seen: set[tuple[float, float]] = set()
        for seed in range(40):
            env = TabletopEnv()
            try:
                env.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=prior_std)
                env.set_perturbation("initial_state_ood", 1.0)
                obs, _ = env.reset(seed=seed)
                sx = float(np.sign(obs["cube_pos"][0]))
                sy = float(np.sign(obs["cube_pos"][1]))
                signs_seen.add((sx, sy))
            finally:
                env.close()
        # Across 40 seeds we should see at least two distinct sign
        # patterns (otherwise the sub-stream is broken).
        assert len(signs_seen) >= 2


# ---------------------------------------------------------- prior validation


class TestPriorValidation:
    def test_prior_setter_rejects_wrong_length(self, env: TabletopEnv) -> None:
        # The signature is typed as length-3 tuples; deliberately pass a
        # wrong-length tuple to exercise the runtime guard.
        with pytest.raises(ValueError, match="length 3"):
            env.set_initial_state_ood_prior(
                mean=(0.0, 0.0),  # type: ignore[arg-type]
                std=(0.05, 0.05, 0.0),
            )

    def test_prior_setter_rejects_negative_std(self, env: TabletopEnv) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            env.set_initial_state_ood_prior(mean=(0.0, 0.0, 0.0), std=(0.05, -0.01, 0.0))
