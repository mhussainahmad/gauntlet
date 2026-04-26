"""Core bisect engine — binary-search a checkpoint list under paired CRN.

The engine is a pure function over the inputs: an ordered checkpoint
list ``[good, *intermediates, bad]``, a *resolver* from checkpoint id
to a zero-arg :class:`gauntlet.policy.base.Policy` factory, a
:class:`gauntlet.suite.schema.Suite`, and a target cell id. At each
midpoint of the binary-search interval we run the policy on the suite,
filter the resulting :class:`gauntlet.runner.Episode` list to the
target cell, and compare against the known-good baseline via
:func:`gauntlet.diff.paired.compute_paired_cells`. The decision rule
collapses the interval whenever the paired-CI upper bound on the
target-cell delta is strictly below zero.

Why a checkpoint *list* (not weight-interpolation):
    Weight-interpolation between two checkpoints only makes sense for
    same-architecture LoRA-style fine-tunes (and even there the
    behaviour is nonlinear). For the general case the user must
    supply intermediate checkpoint paths -- which means they need to
    have saved them. Most training runs don't, but for the runs that
    *did* the bisect is the natural CLI manifestation of "git bisect
    over policy training". We accept the discrete-list input
    explicitly per the B-39 backlog entry's anti-feature framing.

Why paired-CRN (not independent Wilson on each side):
    The same ``master_seed`` flows into both candidates' Suite, so the
    Runner's :meth:`numpy.random.SeedSequence.spawn` derivation gives
    identical per-episode env seeds across paired
    ``(cell_index, episode_index)`` keys. The McNemar / Newcombe-Tango
    paired-CI (:mod:`gauntlet.diff.paired`) folds in the positive
    outcome correlation that buys -- empirically -- a 2-4x reduction
    in episodes-per-step for the same delta-confidence. See
    "Beyond Binary Success" (arxiv 2603.13616) and SureSim
    (arxiv 2510.04354).

Decision rule (per midpoint):
    Let ``delta_ci_high`` be the upper bound of the paired CI on the
    target cell's ``b_success_rate - a_success_rate`` (where ``a`` is
    the known-good baseline and ``b`` is the midpoint). If
    ``delta_ci_high < 0`` the midpoint is *significantly* worse than
    good and we move ``hi = mid``; otherwise the midpoint is not yet
    distinguishable from good and we move ``lo = mid``. We stop when
    the interval has length 1 -- at that point ``ckpt_list[hi]`` is
    the first checkpoint that exhibits the regression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from gauntlet.diff.paired import (
    PairedCellDelta,
    PairingError,
    compute_paired_cells,
)
from gauntlet.policy.base import Policy
from gauntlet.runner import Episode, Runner
from gauntlet.suite.schema import Suite

__all__ = [
    "BisectError",
    "BisectResult",
    "BisectStep",
    "RunnerFactory",
    "bisect",
]


# ----------------------------------------------------------------------
# Errors.
# ----------------------------------------------------------------------


class BisectError(ValueError):
    """The bisect engine cannot proceed (bad inputs or degenerate state).

    Distinct from :class:`gauntlet.diff.paired.PairingError` (which is
    a runner-level CRN-violation signal) -- this one is for user-facing
    misconfiguration: an empty checkpoint list, a target cell missing
    from the suite, or a single-element list with no intermediate.
    """


# ----------------------------------------------------------------------
# Protocols.
# ----------------------------------------------------------------------


class RunnerFactory(Protocol):
    """Zero-arg callable returning a fresh :class:`Runner`.

    The bisect engine wants to inject the runner construction so
    tests can swap in a fake runner that returns synthetic episodes
    without spinning up MuJoCo. Production callers pass a partial
    over :class:`Runner` with the desired ``n_workers``,
    ``cache_dir``, etc. baked in.
    """

    def __call__(self) -> Runner:
        """Construct one :class:`Runner` instance."""
        ...


# ----------------------------------------------------------------------
# Result shapes.
# ----------------------------------------------------------------------


class BisectStep(BaseModel):
    """One step of the binary search.

    ``checkpoint`` is the midpoint candidate evaluated at this step.
    ``delta`` is the paired success-rate delta at the target cell
    (``b - a``, signed so a regression is negative; ``a`` is always
    the known-good baseline). ``delta_ci_low`` / ``delta_ci_high`` are
    the Newcombe / Tango paired-CI bounds; ``regressed`` is the
    decision the engine took at this step (``True`` -> upper bound
    strictly below zero -> the midpoint is significantly worse than
    good and the interval collapsed left).
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    index: int
    checkpoint: str
    n_paired: int
    a_success_rate: float
    b_success_rate: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    regressed: bool


class BisectResult(BaseModel):
    """Top-level bisect artefact.

    ``first_bad`` is the first checkpoint in ``ckpt_list`` whose target
    cell exhibits a paired-CRN-significant regression relative to the
    known-good baseline. ``steps`` is the ordered list of midpoints the
    engine evaluated -- a four-element checkpoint list bisects in at
    most two midpoint steps; the good and bad anchors are not counted
    as "steps" because they are evaluated once up front and re-used
    across the search.

    ``target_cell_delta`` is the paired success-rate delta at the
    target cell when comparing ``first_bad`` against the good anchor
    -- the headline number the operator reads to decide how bad the
    regression actually is.
    """

    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    suite_name: str
    target_cell_id: int
    good_checkpoint: str
    bad_checkpoint: str
    ckpt_list: list[str]
    first_bad: str
    target_cell_delta: float
    target_cell_delta_ci_low: float
    target_cell_delta_ci_high: float
    steps: list[BisectStep]


# ----------------------------------------------------------------------
# Engine.
# ----------------------------------------------------------------------


def _run_candidate(
    *,
    runner_factory: RunnerFactory,
    policy_factory: Callable[[], Policy],
    suite: Suite,
) -> list[Episode]:
    """Run one candidate end-to-end and return its Episode list.

    A fresh :class:`Runner` is constructed per call so each candidate
    starts from a clean cache / pruning-state -- the bisect signal
    must be self-contained, not influenced by leftover state from
    the previous candidate.
    """
    runner = runner_factory()
    return runner.run(policy_factory=policy_factory, suite=suite)


def _assert_target_cell_in_suite(suite: Suite, target_cell_id: int) -> None:
    """Reject a target cell id that is not produced by the suite.

    ``Suite.cells()`` is a stable iterator -- the cell index space
    is exactly ``range(suite.num_cells())`` for cartesian sampling
    and ``range(n_samples)`` for LHS / Sobol. A target cell outside
    that range is a user error at the boundary; we surface it loudly
    instead of silently producing a zero-paired-episode result.
    """
    n_cells = suite.num_cells()
    if not (0 <= target_cell_id < n_cells):
        raise BisectError(
            f"target cell id {target_cell_id} not in suite {suite.name!r} "
            f"(suite has {n_cells} cell(s); valid ids are 0..{n_cells - 1})"
        )


def _select_target_cell_delta(
    episodes_a: Sequence[Episode],
    episodes_b: Sequence[Episode],
    *,
    target_cell_id: int,
    suite_name: str,
) -> PairedCellDelta:
    """Compute paired CRN deltas across cells, return the target-cell row.

    The paired-CRN engine produces one :class:`PairedCellDelta` per
    shared cell; we pick out the target. A missing target -- either
    side has no episodes for the cell, which the runner can produce
    when caching skips work -- is a hard error: we cannot make a
    bisect decision without paired observations on the target cell.
    """
    paired = compute_paired_cells(episodes_a, episodes_b, suite_name=suite_name)
    for cell in paired.cells:
        if cell.cell_index == target_cell_id:
            return cell
    raise BisectError(
        f"target cell {target_cell_id} produced zero paired episodes "
        f"between the two checkpoints; cannot make a bisect decision. "
        f"Re-run with episodes_per_step >= 1 or pick a target cell "
        f"present in both runs."
    )


def _resolve_episodes_per_step(suite: Suite, episodes_per_step: int | None) -> Suite:
    """Apply an optional episodes-per-cell override to the suite.

    The bisect re-uses the suite's per-cell count by default because
    the user already calibrated it for a single-policy run; the
    override is for the case where the user wants more (or fewer)
    paired episodes per midpoint to trade off latency vs. signal.
    """
    if episodes_per_step is None:
        return suite
    if episodes_per_step < 1:
        raise BisectError(f"episodes_per_step must be >= 1; got {episodes_per_step}")
    return suite.model_copy(update={"episodes_per_cell": episodes_per_step})


def bisect(
    *,
    ckpt_list: Sequence[str],
    policy_factory_resolver: Callable[[str], Callable[[], Policy]],
    suite: Suite,
    target_cell_id: int,
    runner_factory: RunnerFactory,
    episodes_per_step: int | None = None,
) -> BisectResult:
    """Binary-search ``ckpt_list`` for the first regression at ``target_cell_id``.

    The list MUST satisfy the git-bisect invariant up front:
    ``ckpt_list[0]`` is the known-good checkpoint and
    ``ckpt_list[-1]`` is the known-bad one. The engine never re-runs
    the good anchor because every comparison is paired against it,
    so its episode list is computed once and re-used for every step.

    Args:
        ckpt_list: Ordered list ``[good, *intermediates, bad]``. Length
            ``>= 2`` (good and bad on the ends) is enforced; a list
            of length 1 means good == bad and the bisect is a no-op
            (returns the only element as ``first_bad``). Duplicate
            checkpoint ids are allowed -- the search treats them as
            distinct positions in the list and runs each one.
        policy_factory_resolver: Callable mapping a checkpoint id to a
            zero-arg :class:`gauntlet.policy.base.Policy` factory. The
            engine never inspects the checkpoint id; the resolver is
            the only place that knows how to load weights. Tests pass
            a stub that returns synthetic policies keyed off the id.
        suite: The :class:`Suite` to evaluate. ``target_cell_id`` MUST
            lie in ``range(suite.num_cells())``.
        target_cell_id: The :attr:`SuiteCell.index` to read the bisect
            decision off. Per the backlog spec the bisect tracks ONE
            failing cell; multi-cell bisection is out of scope.
        runner_factory: Zero-arg callable returning a fresh
            :class:`Runner`. Production callers wire this to a
            :class:`functools.partial` baking in ``n_workers``,
            ``cache_dir``, ``policy_id``, ``max_steps`` etc.; tests
            inject a fake runner that returns synthetic episodes.
        episodes_per_step: Optional per-cell episode count override.
            When ``None`` (the default) the suite's
            :attr:`Suite.episodes_per_cell` is used unchanged.

    Returns:
        :class:`BisectResult` with one :class:`BisectStep` per midpoint
        evaluated.

    Raises:
        BisectError: empty / length-zero ``ckpt_list``, ``target_cell_id``
            outside the suite's cell range, ``episodes_per_step < 1``,
            or a midpoint produced zero paired episodes for the target
            cell (the engine cannot decide in that case).
        PairingError: master-seed or env-seed mismatch between two
            paired runs -- a runner-level CRN-violation signal, not a
            user error. Surfaces verbatim so a wrapper script can
            distinguish "bisect did its job" from "the runner
            silently dropped CRN".
    """
    # ------------------------------------------------------------------
    # Input validation. Every error here is at the user/CLI boundary --
    # surface loudly with an actionable message rather than silently
    # producing a zero-step result.
    # ------------------------------------------------------------------
    if not ckpt_list:
        raise BisectError(
            "ckpt_list is empty; bisect requires at least the [good] anchor "
            "(and ideally a [good, ..., bad] sequence)."
        )
    _assert_target_cell_in_suite(suite, target_cell_id)
    suite_eff = _resolve_episodes_per_step(suite, episodes_per_step)

    good_ckpt = ckpt_list[0]
    bad_ckpt = ckpt_list[-1]

    # Degenerate case: a single-element list (or good == bad with no
    # intermediates). Nothing to bisect; return the only element as
    # ``first_bad`` with an empty step list. The caller (CLI) is
    # expected to surface this as a warning rather than a regression
    # signal -- the bisect is uninformative when there is no interval
    # to search.
    if len(ckpt_list) == 1:
        good_episodes = _run_candidate(
            runner_factory=runner_factory,
            policy_factory=policy_factory_resolver(good_ckpt),
            suite=suite_eff,
        )
        # Self-pair to give the operator a sanity-check delta_ci row;
        # a self-paired comparison has b == c == 0 by construction so
        # the CI collapses to ``[0, 0]`` and the McNemar p-value is
        # 1.0 -- exactly what we want to surface as "no regression".
        cell = _select_target_cell_delta(
            good_episodes,
            good_episodes,
            target_cell_id=target_cell_id,
            suite_name=suite_eff.name,
        )
        return BisectResult(
            suite_name=suite_eff.name,
            target_cell_id=target_cell_id,
            good_checkpoint=good_ckpt,
            bad_checkpoint=bad_ckpt,
            ckpt_list=list(ckpt_list),
            first_bad=good_ckpt,
            target_cell_delta=cell.delta,
            target_cell_delta_ci_low=cell.delta_ci_low,
            target_cell_delta_ci_high=cell.delta_ci_high,
            steps=[],
        )

    # ------------------------------------------------------------------
    # Run the good anchor once. Every step pairs against this list, so
    # we never re-evaluate it during the search loop.
    # ------------------------------------------------------------------
    good_episodes = _run_candidate(
        runner_factory=runner_factory,
        policy_factory=policy_factory_resolver(good_ckpt),
        suite=suite_eff,
    )

    # ------------------------------------------------------------------
    # Standard git-bisect invariant: lo points at a known-good index,
    # hi at a known-bad index. The search terminates when hi - lo == 1
    # -- at that point ckpt_list[hi] is the first checkpoint at which
    # the regression appeared.
    # ------------------------------------------------------------------
    lo = 0
    hi = len(ckpt_list) - 1
    steps: list[BisectStep] = []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        mid_ckpt = ckpt_list[mid]
        try:
            mid_episodes = _run_candidate(
                runner_factory=runner_factory,
                policy_factory=policy_factory_resolver(mid_ckpt),
                suite=suite_eff,
            )
        except PairingError:
            # The runner itself signalled a CRN violation -- propagate
            # so the wrapper script can distinguish "bisect ran" from
            # "the runner silently dropped pairing". Re-raised as-is.
            raise
        cell = _select_target_cell_delta(
            good_episodes,
            mid_episodes,
            target_cell_id=target_cell_id,
            suite_name=suite_eff.name,
        )
        # Decision rule: midpoint is "regressed vs good" only when the
        # paired-CI upper bound is strictly below zero. The strict
        # inequality is deliberate -- a CI that touches zero is not
        # yet evidence of a regression at the configured confidence
        # level (the CI collapses to ``[0, 0]`` when both sides agree
        # on every paired episode, and we should not call that a
        # regression).
        regressed = cell.delta_ci_high < 0.0
        steps.append(
            BisectStep(
                index=mid,
                checkpoint=mid_ckpt,
                n_paired=cell.n_paired,
                a_success_rate=cell.a_success_rate,
                b_success_rate=cell.b_success_rate,
                delta=cell.delta,
                delta_ci_low=cell.delta_ci_low,
                delta_ci_high=cell.delta_ci_high,
                regressed=regressed,
            )
        )
        if regressed:
            hi = mid
        else:
            lo = mid

    # ------------------------------------------------------------------
    # Resolve the final delta against the good anchor for the headline
    # number on the result. ``ckpt_list[hi]`` is the first-bad
    # checkpoint by the bisect invariant. If we never visited it
    # (the search bracket was already length-1, e.g. ckpt_list of
    # length 2), evaluate it now so target_cell_delta is always
    # populated against the good anchor.
    # ------------------------------------------------------------------
    first_bad_ckpt = ckpt_list[hi]
    final_step = next((s for s in steps if s.index == hi), None)
    if final_step is None:
        bad_episodes = _run_candidate(
            runner_factory=runner_factory,
            policy_factory=policy_factory_resolver(first_bad_ckpt),
            suite=suite_eff,
        )
        cell = _select_target_cell_delta(
            good_episodes,
            bad_episodes,
            target_cell_id=target_cell_id,
            suite_name=suite_eff.name,
        )
        target_delta = cell.delta
        target_ci_low = cell.delta_ci_low
        target_ci_high = cell.delta_ci_high
    else:
        target_delta = final_step.delta
        target_ci_low = final_step.delta_ci_low
        target_ci_high = final_step.delta_ci_high

    return BisectResult(
        suite_name=suite_eff.name,
        target_cell_id=target_cell_id,
        good_checkpoint=good_ckpt,
        bad_checkpoint=bad_ckpt,
        ckpt_list=list(ckpt_list),
        first_bad=first_bad_ckpt,
        target_cell_delta=target_delta,
        target_cell_delta_ci_low=target_ci_low,
        target_cell_delta_ci_high=target_ci_high,
        steps=steps,
    )
