"""Worst-case continuous-perturbation sampler tests (B-44).

Properties pinned:

* The DE optimiser recovers the peak of a synthetic parabola failure
  surface within tight tolerance — the headline guarantee that the
  Eva-VLA-style search actually finds adversarial points.
* The ``max_evaluations`` budget cap is honoured: the optimiser never
  calls the objective more than
  ``baseline_evaluations + max_evaluations`` times.
* Categorical / structured (extrinsics) axes are filtered out of the
  search; only continuous axes participate in
  :attr:`WorstCaseResult.adversarial_axis_values`.
* An all-categorical suite raises a clear :class:`ValueError` at
  ``optimize()`` time (the schema rejects this at YAML load, but
  hand-constructed suites can still reach the optimiser).
* The result always carries :attr:`baseline_failure_rate` alongside
  :attr:`failure_rate_at_worst` — the anti-feature contract from the
  module docstring.
* The schema-side wiring round-trips: a YAML naming
  ``sampling: worst_case_continuous`` with a ``worst_case`` block
  parses cleanly, and validation rejects all-categorical-axis suites
  on this mode plus ``worst_case`` blocks paired with any other mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from gauntlet.suite import AxisSpec, Suite, WorstCaseConfig, load_suite_from_string
from gauntlet.suite.worst_case import (
    DEFAULT_BASELINE_EVALUATIONS,
    WorstCaseContinuousSampler,
    WorstCaseResult,
    continuous_axes,
)

# --------------------------------------------------------------------------- helpers


def _two_axis_continuous_suite(
    *,
    seed: int = 42,
    worst_case: WorstCaseConfig | None = None,
) -> Suite:
    """Suite with two continuous axes spanning ``[-0.1, 0.1]^2``.

    Used for the parabola-recovery test below: small box keeps DE
    convergence well-behaved within the default 40-evaluation budget.
    """
    return Suite.model_construct(
        name="worst-case-two-axis",
        env="tabletop",
        episodes_per_cell=1,
        seed=seed,
        axes={
            "camera_offset_x": AxisSpec(low=-0.1, high=0.1),
            "camera_offset_y": AxisSpec(low=-0.1, high=0.1),
        },
        sampling="worst_case_continuous",
        worst_case=worst_case,
    )


def _mixed_suite() -> Suite:
    """Mixed continuous + categorical suite — the categorical axis must
    be filtered out of the DE search.
    """
    return Suite.model_construct(
        name="worst-case-mixed",
        env="tabletop",
        episodes_per_cell=1,
        seed=42,
        axes={
            "camera_offset_x": AxisSpec(low=-0.1, high=0.1),
            "object_texture": AxisSpec(values=[0.0, 1.0]),
        },
        sampling="worst_case_continuous",
    )


def _all_categorical_suite() -> Suite:
    """Suite with no continuous axis — DE has nothing to optimise."""
    return Suite.model_construct(
        name="worst-case-all-cat",
        env="tabletop",
        episodes_per_cell=1,
        seed=42,
        axes={
            "object_texture": AxisSpec(values=[0.0, 1.0]),
        },
        sampling="worst_case_continuous",
    )


# --------------------------------------------------------------------------- tests


class TestSyntheticParabolaPeakRecovery:
    """Headline: DE recovers the known peak of a synthetic failure
    surface within tight tolerance."""

    def test_recovers_peak_of_2d_parabola(self) -> None:
        # Failure surface: 1.0 - ||(x, y) - peak||^2 / max_dist^2 so the
        # objective lies in [0, 1] with the maximum at peak.
        peak = np.array([0.05, -0.03], dtype=np.float64)
        # Max distance from peak to any corner of [-0.1, 0.1]^2 — use
        # this to normalise the parabola so the score stays in [0, 1].
        max_dist = float(np.linalg.norm(np.array([0.1, 0.1]) - peak))
        max_dist += float(np.linalg.norm(np.array([-0.1, -0.1]) - peak))
        # Pick a denominator that bounds the score in [0, 1] across the
        # whole box. ``2 * max_dist^2`` is conservative.
        denom = 2.0 * max_dist * max_dist

        def objective(axis_values: dict[str, float]) -> float:
            x = axis_values["camera_offset_x"]
            y = axis_values["camera_offset_y"]
            d2 = (x - peak[0]) ** 2 + (y - peak[1]) ** 2
            return float(max(0.0, 1.0 - d2 / denom))

        suite = _two_axis_continuous_suite()
        sampler = WorstCaseContinuousSampler(
            max_evaluations=80,
            episodes_per_eval=1,
            baseline_evaluations=5,
        )
        result = sampler.optimize(objective, suite, np.random.default_rng(0))

        # Recovered peak should land within 0.02 of the true peak in
        # both dims — DE on a 2-D parabola with 80 evals is comfortably
        # tighter than that. The tolerance is loose enough that random
        # entropy (different ``rng`` seed) still passes; the assertion
        # is "DE found the basin", not "DE converged to machine eps".
        x_hat = result.adversarial_axis_values["camera_offset_x"]
        y_hat = result.adversarial_axis_values["camera_offset_y"]
        assert abs(x_hat - peak[0]) < 0.02, (
            f"x recovery off: got {x_hat}, expected {peak[0]} +/- 0.02"
        )
        assert abs(y_hat - peak[1]) < 0.02, (
            f"y recovery off: got {y_hat}, expected {peak[1]} +/- 0.02"
        )
        # And the failure rate at the worst point should be much higher
        # than the baseline (random uniform draws don't land on the
        # peak).
        assert result.failure_rate_at_worst > result.baseline_failure_rate, (
            f"worst ({result.failure_rate_at_worst}) should exceed baseline "
            f"({result.baseline_failure_rate})"
        )
        # Worst-case must be near 1.0 (the parabola maximum).
        assert result.failure_rate_at_worst > 0.95, (
            f"DE failed to climb the parabola; got {result.failure_rate_at_worst}"
        )

    def test_optimizer_is_deterministic(self) -> None:
        """Same RNG seed -> identical WorstCaseResult (modulo objective
        determinism, which is trivially true here)."""

        def objective(axis_values: dict[str, float]) -> float:
            x = axis_values["camera_offset_x"]
            y = axis_values["camera_offset_y"]
            return float(0.5 + 0.5 * (x + y))

        suite = _two_axis_continuous_suite()
        sampler = WorstCaseContinuousSampler(
            max_evaluations=20,
            episodes_per_eval=1,
            baseline_evaluations=3,
        )
        a = sampler.optimize(objective, suite, np.random.default_rng(99))
        b = sampler.optimize(objective, suite, np.random.default_rng(99))
        assert a.adversarial_axis_values == b.adversarial_axis_values
        assert a.failure_rate_at_worst == b.failure_rate_at_worst
        assert a.baseline_failure_rate == b.baseline_failure_rate
        assert a.evaluations_used == b.evaluations_used


class TestBudgetCap:
    """The ``max_evaluations`` ceiling must be honoured."""

    def test_evaluations_used_capped_by_budget(self) -> None:
        calls = {"n": 0}

        def objective(axis_values: dict[str, float]) -> float:
            calls["n"] += 1
            return float(axis_values["camera_offset_x"])

        suite = _two_axis_continuous_suite()
        baseline = 4
        max_evals = 7
        sampler = WorstCaseContinuousSampler(
            max_evaluations=max_evals,
            episodes_per_eval=1,
            baseline_evaluations=baseline,
        )
        result = sampler.optimize(objective, suite, np.random.default_rng(0))
        # Total objective calls: baseline + DE evals; DE is capped at
        # ``max_evaluations``. The closure counts every call, so the
        # raw counter and the ``evaluations_used`` field must agree.
        assert calls["n"] == result.evaluations_used
        # And the total must lie within the budget envelope. Equality
        # holds when DE hits the cap mid-population (which it does for
        # any ``max_evaluations <= _DE_POPULATION``).
        assert result.evaluations_used <= baseline + max_evals
        # DE should at least burn the budget — otherwise the test would
        # be trivially satisfied by an early-exit bug. ``max_evals=7``
        # cannot be left over.
        assert result.evaluations_used == baseline + max_evals

    def test_default_budget_is_finite(self) -> None:
        """Sanity: even with default knobs, ``evaluations_used`` is
        bounded above by the public defaults."""

        def objective(axis_values: dict[str, float]) -> float:
            return 0.5

        suite = _two_axis_continuous_suite()
        sampler = WorstCaseContinuousSampler()
        result = sampler.optimize(objective, suite, np.random.default_rng(0))
        # Module-level defaults are 5 baseline + 40 DE = 45 max.
        assert result.evaluations_used <= DEFAULT_BASELINE_EVALUATIONS + 40


class TestCategoricalAxesFiltered:
    """The DE search must ignore categorical / structured axes — only
    continuous axes appear in ``adversarial_axis_values`` and the
    objective only sees continuous-axis names."""

    def test_continuous_axes_helper_filters_categorical(self) -> None:
        # ``continuous_axes`` is the helper the optimiser uses to pick
        # which axes participate. Pin its behaviour directly.
        suite = _mixed_suite()
        cont = continuous_axes(suite)
        names = tuple(name for name, _ in cont)
        assert names == ("camera_offset_x",), (
            f"only camera_offset_x is continuous in the mixed suite; got {names}"
        )

    def test_objective_only_receives_continuous_axes(self) -> None:
        seen_keys: set[frozenset[str]] = set()

        def objective(axis_values: dict[str, float]) -> float:
            seen_keys.add(frozenset(axis_values.keys()))
            return float(axis_values["camera_offset_x"])

        suite = _mixed_suite()
        sampler = WorstCaseContinuousSampler(
            max_evaluations=12,
            episodes_per_eval=1,
            baseline_evaluations=3,
        )
        result = sampler.optimize(objective, suite, np.random.default_rng(0))
        # The objective was called many times; every call's kwarg dict
        # must have exactly the continuous axis name. Categorical axes
        # never leak through.
        assert seen_keys == {frozenset({"camera_offset_x"})}, (
            f"categorical axis leaked into objective; got {seen_keys}"
        )
        # And the result mirrors that filter — categorical axes are not
        # included.
        assert set(result.adversarial_axis_values.keys()) == {"camera_offset_x"}


class TestEmptyContinuousSetRaises:
    """An all-categorical suite has no DE search space; the optimiser
    must raise a clean error."""

    def test_optimize_raises_on_all_categorical_suite(self) -> None:
        def objective(axis_values: dict[str, float]) -> float:
            return 0.0

        suite = _all_categorical_suite()
        sampler = WorstCaseContinuousSampler(
            max_evaluations=10,
            episodes_per_eval=1,
            baseline_evaluations=2,
        )
        with pytest.raises(ValueError, match="no continuous axes"):
            sampler.optimize(objective, suite, np.random.default_rng(0))

    def test_continuous_axes_helper_returns_empty_tuple(self) -> None:
        # Direct helper-level pin so the empty-set branch is testable
        # without going through ``optimize``.
        suite = _all_categorical_suite()
        assert continuous_axes(suite) == ()


class TestResultIncludesBaselineAlongsideWorstCase:
    """ANTI-FEATURE INVARIANT (B-44): the worst-case result must always
    carry an unbiased baseline alongside the worst observed rate.

    Surfaces the contract documented in
    ``src/gauntlet/suite/worst_case.py`` — a worst-case headline
    without its baseline is cherry-picking by construction.
    """

    def test_result_carries_both_rates(self) -> None:
        def objective(axis_values: dict[str, float]) -> float:
            return float(axis_values["camera_offset_x"])

        suite = _two_axis_continuous_suite()
        sampler = WorstCaseContinuousSampler(
            max_evaluations=20,
            episodes_per_eval=1,
            baseline_evaluations=4,
        )
        result = sampler.optimize(objective, suite, np.random.default_rng(0))
        # Both fields are required, both must be finite floats in [0, 1]
        # given the bounded objective above.
        assert isinstance(result, WorstCaseResult)
        assert 0.0 <= result.baseline_failure_rate <= 1.0
        assert 0.0 <= result.failure_rate_at_worst <= 1.0
        # Worst-case must be >= baseline by construction (DE can
        # always re-explore the baseline draws — the population is
        # initialised uniformly inside the same box). A strict
        # inequality is too tight for tiny budgets, but >= must hold.
        assert result.failure_rate_at_worst >= result.baseline_failure_rate, (
            f"worst ({result.failure_rate_at_worst}) must >= baseline "
            f"({result.baseline_failure_rate})"
        )

    def test_baseline_evaluations_below_one_rejected(self) -> None:
        """The constructor refuses to skip the baseline — the
        anti-feature note mandates a non-empty baseline."""
        with pytest.raises(ValueError, match="baseline_evaluations must be >= 1"):
            WorstCaseContinuousSampler(baseline_evaluations=0)


class TestSamplerProtocolRaises:
    """``Sampler.sample`` is a hard error for this mode — the protocol
    is offline but worst-case search needs online feedback."""

    def test_sample_raises_runtime_error(self) -> None:
        suite = _two_axis_continuous_suite()
        sampler = WorstCaseContinuousSampler()
        with pytest.raises(RuntimeError, match="requires an objective callable"):
            sampler.sample(suite, np.random.default_rng(0))


class TestSchemaWiring:
    """The ``sampling: worst_case_continuous`` mode round-trips through
    the YAML loader, and the schema rejects the documented invalid
    combinations.
    """

    def test_yaml_with_worst_case_block_loads(self) -> None:
        text = """
name: worst-case-yaml
env: tabletop
seed: 7
sampling: worst_case_continuous
episodes_per_cell: 1
worst_case:
  max_evaluations: 16
  episodes_per_eval: 3
  seed: 13
axes:
  camera_offset_x:
    low: -0.1
    high: 0.1
  camera_offset_y:
    low: -0.1
    high: 0.1
"""
        suite = load_suite_from_string(text)
        assert suite.sampling == "worst_case_continuous"
        assert suite.worst_case is not None
        assert suite.worst_case.max_evaluations == 16
        assert suite.worst_case.episodes_per_eval == 3
        assert suite.worst_case.seed == 13

    def test_yaml_without_worst_case_block_loads(self) -> None:
        # ``worst_case`` is optional — the sampler falls back to module
        # defaults when unset.
        text = """
name: worst-case-no-block
env: tabletop
seed: 7
sampling: worst_case_continuous
episodes_per_cell: 1
axes:
  camera_offset_x:
    low: -0.1
    high: 0.1
"""
        suite = load_suite_from_string(text)
        assert suite.sampling == "worst_case_continuous"
        assert suite.worst_case is None

    def test_all_categorical_suite_rejected(self) -> None:
        text = """
name: worst-case-all-cat
env: tabletop
seed: 7
sampling: worst_case_continuous
episodes_per_cell: 1
axes:
  object_texture:
    values: [0.0, 1.0]
"""
        with pytest.raises(Exception, match="at least one continuous axis"):
            load_suite_from_string(text)

    def test_n_samples_rejected_on_worst_case(self) -> None:
        text = """
name: worst-case-with-n-samples
env: tabletop
seed: 7
sampling: worst_case_continuous
n_samples: 16
episodes_per_cell: 1
axes:
  camera_offset_x:
    low: -0.1
    high: 0.1
"""
        with pytest.raises(Exception, match="n_samples must be omitted"):
            load_suite_from_string(text)

    def test_worst_case_block_on_other_mode_rejected(self) -> None:
        text = """
name: sobol-with-worst-case-block
env: tabletop
seed: 7
sampling: sobol
n_samples: 16
episodes_per_cell: 1
worst_case:
  max_evaluations: 5
axes:
  camera_offset_x:
    low: -0.1
    high: 0.1
"""
        with pytest.raises(Exception, match="worst_case sub-config is only valid"):
            load_suite_from_string(text)
