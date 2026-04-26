"""Worst-case continuous-perturbation search (B-44, Eva-VLA-inspired).

See ``docs/backlog.md`` B-44 for the design rationale: Eva-VLA
(arXiv 2509.18953) demonstrates that worst-case search over continuous
physical variations exposes failure modes invisible to grid / Sobol /
LHS sampling. OpenVLA fails >90% on LIBERO-Long under their found
worst-cases vs. ~20% on uniform sampling.

ANTI-FEATURE WARNING. Worst-case continuous search is the strongest
possible cherry-pick. A "this policy fails 100% on the adversarial
cell" headline reads as alarmist when the cell is measure-zero in the
operating distribution. The :class:`WorstCaseResult` always carries a
:attr:`baseline_failure_rate` derived from unbiased uniform draws over
the same continuous bounds; downstream callers MUST report the worst-
case rate alongside the baseline, never standalone. The unit tests in
``tests/test_worst_case_sampler.py`` pin this invariant
(``test_result_includes_baseline_alongside_worst_case``).

Distinction from B-07 (adversarial sampling). B-07 is a discrete
Thompson-sampling bandit over the existing axis grid (categorical bins
fitted from a pilot ``report.json``); B-44 is gradient-free continuous
optimisation over the underlying physical parameters. Same user-facing
pitch (find failure modes), different mechanism. The two modes are
disjoint by design — B-07 reuses :func:`gauntlet.suite.adversarial.AdversarialSampler`
machinery, B-44 lives entirely in this module.

Design notes
------------

1. **Optimiser**: differential evolution (Storn-Price 1997, ``rand/1/bin``)
   implemented from scratch in numpy. The Eva-VLA paper mentions CMA-ES,
   but ``cma`` is not a project dependency and ``scipy.optimize`` is
   not installed either (the codebase precedent — Sobol direction
   numbers, LHS, Wilson intervals, the adversarial Thompson sampler —
   is "implement small numerical kernels in-house, no heavy deps").
   DE is well-suited to the failure-surface objective: it makes no
   smoothness assumptions, handles bounded boxes natively, and converges
   well on the low-dimensional (5-axis) continuous hypercube the spec
   targets.

2. **Sampler-protocol entry**: :class:`WorstCaseContinuousSampler.sample`
   is intentionally a hard error. The :class:`gauntlet.suite.sampling.Sampler`
   protocol is offline (``(suite, rng) -> list[SuiteCell]``); worst-case
   continuous search is fundamentally online — it needs per-candidate
   feedback from running episodes against a real policy. Forcing it
   into a one-shot emitter would lie about what the mode does. Instead,
   the user invokes :meth:`WorstCaseContinuousSampler.optimize` directly
   with a callable objective.

3. **Categorical filter**: only axes with ``low`` / ``high`` set (and
   no ``values`` / ``extrinsics_values`` / ``extrinsics_range``) are
   passed to DE. The schema validator rejects suites where every axis
   is non-continuous (no continuous axes -> no search possible); a
   mixed suite is allowed but the optimiser silently ignores the
   categorical axes. Their values do not appear in
   :attr:`WorstCaseResult.adversarial_axis_values`.

4. **Baseline**: :attr:`WorstCaseResult.baseline_failure_rate` is the
   mean failure score over :attr:`WorstCaseContinuousSampler.baseline_evaluations`
   uniform-random draws inside the same continuous box. The default
   (``5``) is a small parameter chosen to keep the unit-test budget
   tight; real callers tune it up. Baseline draws count toward the
   total ``evaluations_used`` so the budget contract stays honest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gauntlet.suite.schema import AxisSpec, Suite, SuiteCell

__all__ = [
    "DEFAULT_BASELINE_EVALUATIONS",
    "DEFAULT_EPISODES_PER_EVAL",
    "DEFAULT_MAX_EVALUATIONS",
    "ObjectiveFn",
    "WorstCaseContinuousSampler",
    "WorstCaseResult",
    "continuous_axes",
]


# Sensible defaults — small enough that the unit tests run in <1 s,
# large enough that DE converges on a 5-axis continuous box.
DEFAULT_MAX_EVALUATIONS: Final[int] = 40
DEFAULT_EPISODES_PER_EVAL: Final[int] = 5
DEFAULT_BASELINE_EVALUATIONS: Final[int] = 5

# DE hyperparameters. ``rand/1/bin`` with these knobs is the textbook
# Storn-Price configuration — robust on smooth-ish failure surfaces.
_DE_POPULATION: Final[int] = 10
_DE_MUTATION_F: Final[float] = 0.5
_DE_CROSSOVER_CR: Final[float] = 0.7


# ``objective_fn(axis_values: dict[str, float]) -> float`` returning a
# **failure score** in ``[0, 1]`` (1.0 = certain failure). The spec
# convention is ``1.0 - per-cell success rate`` averaged over the per-
# evaluation episode budget; the optimiser maximises this.
ObjectiveFn = Callable[[dict[str, float]], float]


@dataclass(frozen=True)
class WorstCaseResult:
    """Result of a worst-case continuous search.

    Attributes:
        adversarial_axis_values: Mapping ``{axis_name: value}`` covering
            every continuous axis the search optimised over. Categorical
            axes (if present in the parent :class:`Suite`) are not
            included — they are filtered out before the search.
        failure_rate_at_worst: Failure score at the best (highest-
            failure) candidate the optimiser found. Always in ``[0, 1]``.
        baseline_failure_rate: Mean failure score over
            :attr:`WorstCaseContinuousSampler.baseline_evaluations`
            uniform-random draws inside the same continuous box. The
            anti-feature note above mandates reporting both numbers
            together — a worst-case rate without its baseline is
            cherry-picking by construction.
        evaluations_used: Total number of times :class:`ObjectiveFn`
            was called. Includes baseline draws AND DE candidate
            evaluations; bounded above by
            ``baseline_evaluations + max_evaluations``.
    """

    adversarial_axis_values: dict[str, float]
    failure_rate_at_worst: float
    baseline_failure_rate: float
    evaluations_used: int


def continuous_axes(suite: Suite) -> tuple[tuple[str, AxisSpec], ...]:
    """Return ``(name, spec)`` pairs for axes the worst-case search drives.

    A continuous axis has ``low`` / ``high`` set and *none* of the
    categorical / structured shapes (``values``,
    ``extrinsics_values``, ``extrinsics_range``). Order matches
    :attr:`Suite.axes` insertion order so DE column indices line up
    with the bounds tuple downstream callers can reason about.

    Returns:
        Empty tuple when the suite carries no continuous axis (the
        :class:`WorstCaseContinuousSampler` constructor rejects this
        case loudly; callers that just want to inspect a suite can
        consume an empty tuple safely).
    """
    out: list[tuple[str, AxisSpec]] = []
    for name, spec in suite.axes.items():
        if spec.values is not None:
            continue
        if spec.extrinsics_values is not None:
            continue
        if spec.extrinsics_range is not None:
            continue
        if spec.low is None or spec.high is None:
            continue
        out.append((name, spec))
    return tuple(out)


class WorstCaseContinuousSampler:
    """Differential-evolution worst-case search over continuous axes.

    Construction args:
        max_evaluations: Hard cap on :class:`ObjectiveFn` calls inside
            DE. Defaults to :data:`DEFAULT_MAX_EVALUATIONS` (= 40).
            The optimiser stops as soon as the count reaches this
            ceiling, even mid-generation; the best-so-far candidate is
            returned. Must be ``>= 1``.
        episodes_per_eval: Number of episodes the *caller*'s objective
            should average over per evaluation. Carried on the sampler
            so the YAML knob and the call site stay in sync; the
            sampler itself never runs episodes (the objective callable
            does).
        baseline_evaluations: Number of uniform-random draws used to
            estimate :attr:`WorstCaseResult.baseline_failure_rate`.
            Defaults to :data:`DEFAULT_BASELINE_EVALUATIONS` (= 5).
            Must be ``>= 1`` — a worst-case result without a baseline
            is exactly the cherry-pick the anti-feature warning calls
            out.

    The :meth:`sample` method is a deliberate hard error — the
    :class:`gauntlet.suite.sampling.Sampler` protocol is offline and
    worst-case search needs online feedback. Use :meth:`optimize`.
    """

    def __init__(
        self,
        *,
        max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
        episodes_per_eval: int = DEFAULT_EPISODES_PER_EVAL,
        baseline_evaluations: int = DEFAULT_BASELINE_EVALUATIONS,
    ) -> None:
        if max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1; got {max_evaluations}")
        if episodes_per_eval < 1:
            raise ValueError(f"episodes_per_eval must be >= 1; got {episodes_per_eval}")
        if baseline_evaluations < 1:
            raise ValueError(
                f"baseline_evaluations must be >= 1; got {baseline_evaluations} "
                "(a worst-case result without a baseline is cherry-picking by "
                "construction; see the anti-feature note in suite/worst_case.py)",
            )
        self.max_evaluations = max_evaluations
        self.episodes_per_eval = episodes_per_eval
        self.baseline_evaluations = baseline_evaluations

    # ------------------------------------------------------------------
    # Sampler protocol — intentional hard error.
    # ------------------------------------------------------------------

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[SuiteCell]:
        """Reject offline cell emission — worst-case needs online feedback.

        The :class:`gauntlet.suite.sampling.Sampler` contract is
        ``(suite, rng) -> list[SuiteCell]``; worst-case continuous
        search needs per-candidate failure-score feedback that only
        the runner / a live policy can produce. Forcing the search
        into a one-shot emitter would lie about what this mode does.

        Callers wiring this mode through the suite YAML instead use
        :meth:`optimize` with their own objective callable. The
        ``Suite.cells()`` path raises so a stale wiring fails loudly.
        """
        del suite, rng
        raise RuntimeError(
            "sampling=worst_case_continuous requires an objective callable; "
            "call WorstCaseContinuousSampler.optimize(objective_fn, suite, rng) "
            "directly instead of Suite.cells(). The Sampler protocol is offline "
            "but worst-case continuous search is online — see "
            "src/gauntlet/suite/worst_case.py for the design rationale.",
        )

    # ------------------------------------------------------------------
    # Optimiser entry point.
    # ------------------------------------------------------------------

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        suite: Suite,
        rng: np.random.Generator,
    ) -> WorstCaseResult:
        """Run DE worst-case search against ``objective_fn``.

        Args:
            objective_fn: Callable taking ``{axis_name: value}`` (covering
                every continuous axis) and returning a failure score in
                ``[0, 1]``. Convention: ``1.0 - per-cell success rate``
                averaged over :attr:`episodes_per_eval` episodes — the
                optimiser maximises this.
            suite: :class:`Suite` whose continuous axes drive the
                search. Must contain at least one continuous axis;
                an all-categorical suite raises :class:`ValueError`.
            rng: numpy ``Generator`` — the only source of entropy.
                Two ``optimize`` calls with two ``np.random.Generator``
                instances seeded with the same int produce identical
                :class:`WorstCaseResult` records (modulo whatever
                non-determinism the caller's ``objective_fn`` carries;
                a deterministic objective gives byte-identical results).

        Returns:
            :class:`WorstCaseResult` with the worst-observed continuous
            point, its failure rate, the unbiased baseline failure
            rate, and the total ``evaluations_used``.

        Raises:
            ValueError: when ``suite`` carries no continuous axis.
        """
        cont_pairs = continuous_axes(suite)
        if not cont_pairs:
            raise ValueError(
                "WorstCaseContinuousSampler.optimize: suite has no "
                "continuous axes; worst-case continuous search requires at "
                "least one axis with {low, high} bounds. Categorical "
                "(values), extrinsics, and string-valued axes are filtered "
                "out before the search runs.",
            )
        names = tuple(name for name, _ in cont_pairs)
        bounds = np.array(
            [[float(spec.low), float(spec.high)] for _, spec in cont_pairs],  # type: ignore[arg-type]
            dtype=np.float64,
        )

        # Closure that wraps the user objective: counts evaluations,
        # turns a 1-D numpy point into the dict the user asked for.
        # ``budget`` is a single-key dict so the closure can mutate the
        # call counter without needing ``nonlocal`` plumbing.
        budget = {"used": 0}

        def _eval_point(point: NDArray[np.float64]) -> float:
            kwargs = {name: float(point[i]) for i, name in enumerate(names)}
            score = float(objective_fn(kwargs))
            budget["used"] += 1
            return score

        # Step 1: unbiased baseline. ``baseline_evaluations`` uniform
        # draws inside the bounding box; mean of their failure scores.
        baseline_scores: list[float] = []
        for _ in range(self.baseline_evaluations):
            point = rng.uniform(bounds[:, 0], bounds[:, 1])
            baseline_scores.append(_eval_point(point))
        baseline_rate = float(np.mean(baseline_scores))

        # Step 2: DE. Initial population is uniform inside the box;
        # the per-member score array starts NaN so the budget cap can
        # cut DE off mid-population without polluting the argmax.
        pop = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(_DE_POPULATION, bounds.shape[0]),
        )
        scores = np.full(_DE_POPULATION, np.nan, dtype=np.float64)

        # Evaluate the initial DE population, stopping early if the
        # budget is exhausted. ``budget["used"]`` already reflects the
        # baseline evaluations; the DE budget is ``max_evaluations``
        # *additional* calls.
        max_de_evals = self.max_evaluations
        de_used = 0
        for i in range(_DE_POPULATION):
            if de_used >= max_de_evals:
                break
            scores[i] = _eval_point(pop[i])
            de_used += 1

        # If the DE budget didn't even cover the initial population,
        # return the best baseline candidate as the answer. This keeps
        # the budget cap test honest: ``max_evaluations=1`` evaluates
        # exactly one DE candidate and stops.
        # Otherwise, run rand/1/bin generations until the DE budget is
        # exhausted. Each generation is at most _DE_POPULATION new
        # evaluations; the inner loop checks the budget every step.
        while de_used < max_de_evals:
            # rand/1/bin: for each member i, pick three distinct other
            # members r1,r2,r3, build a mutant v = r1 + F * (r2 - r3),
            # crossover with i to get trial u, accept u if it scores
            # higher than the current member.
            for i in range(_DE_POPULATION):
                if de_used >= max_de_evals:
                    break
                # If this member never got an initial score (budget cut
                # off mid-population), skip it — it cannot be displaced
                # without a comparison.
                if np.isnan(scores[i]):
                    continue
                # Pick three distinct indices != i.
                idxs = [j for j in range(_DE_POPULATION) if j != i and not np.isnan(scores[j])]
                if len(idxs) < 3:
                    # Population too thin for rand/1; skip this member.
                    continue
                r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
                mutant = pop[r1] + _DE_MUTATION_F * (pop[r2] - pop[r3])
                # Reflect-clip the mutant back into the box. Standard
                # DE handles bound violations either by clipping or by
                # reflecting; clipping is simpler and works well on
                # this low-dim setting.
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                # Binomial crossover with rate ``_DE_CROSSOVER_CR``;
                # at least one dimension always swaps. ``cross`` is a
                # boolean mask the same length as the bounds vector; if
                # the random draws produced an all-False mask we force
                # exactly one dimension to swap so the trial is never a
                # bit-identical copy of the parent.
                draws = np.asarray(
                    rng.uniform(size=bounds.shape[0]),
                    dtype=np.float64,
                )
                cross: NDArray[np.bool_] = np.asarray(
                    draws < _DE_CROSSOVER_CR,
                    dtype=np.bool_,
                )
                if not bool(cross.any()):
                    forced = int(rng.integers(0, bounds.shape[0]))
                    cross[forced] = np.True_
                trial = np.where(cross, mutant, pop[i])
                trial_score = _eval_point(trial)
                de_used += 1
                if trial_score >= scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

        # Step 3: pick the best (highest-failure) member. ``np.nan``
        # entries mark untouched population slots; ``np.nanargmax``
        # ignores them. ``max_evaluations >= 1`` (constructor invariant)
        # guarantees the initial-population loop above evaluates at
        # least one slot, so at least one entry is non-NaN here.
        best_idx = int(np.nanargmax(scores))
        best_score = float(scores[best_idx])
        adv = {name: float(pop[best_idx, i]) for i, name in enumerate(names)}

        return WorstCaseResult(
            adversarial_axis_values=adv,
            failure_rate_at_worst=best_score,
            baseline_failure_rate=baseline_rate,
            evaluations_used=budget["used"],
        )
