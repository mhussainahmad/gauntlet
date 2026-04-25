"""Runner-level cross-process determinism audit — Phase 2.5 Task 17 case 5.

The Runner promises (``src/gauntlet/runner/runner.py`` "Output ordering"
docstring) that ``run(processes=1)`` and ``run(processes=N)`` produce
the SAME ``list[Episode]`` for the same inputs after the
``(cell_index, episode_index)`` sort. ``tests/test_runner.py``'s
``test_determinism_n_workers_one_vs_two`` already covers this for a
2x2 axis grid with two episodes per cell.

This file extends that coverage with two value-adds the advisor pass
called out:

1. **Multi-axis grid that surfaces sort bugs.** A 3-axis Suite
   (``lighting_intensity`` x ``camera_offset_x`` x ``distractor_count``)
   produces 8 cells whose iteration order is ``itertools.product``-y;
   if any worker-completion-order leak escapes the Runner's final
   ``episodes.sort(...)``, this test fails. Each cell runs 2 episodes
   to catch a per-cell episode-index leak too.

2. **``master_seed`` round-trip.** Episodes from
   ``processes=2`` carry the same ``metadata["master_seed"]`` as
   ``processes=1``; this is the integer a user re-feeds into
   ``Suite(seed=...)`` to reproduce a None-seeded run. If the master
   seed leaked across worker boundaries (e.g. each worker
   re-generated its own SeedSequence), reproduction would silently
   diverge.

All factories are module-level so they pickle under the ``spawn``
start method (lambdas defined inside test functions do not).

Worker count is capped at 2: the GitHub ``ubuntu-latest`` runner
exposes 4 logical CPUs but only ~2 usable cores under cgroup limits,
and spawning 4 fresh interpreters that each compile MuJoCo's MJCF
(numpy + mujoco import + scene compile) deadlocks the pool startup
on resource contention. ``processes=1`` vs ``processes=2`` is enough
to catch every worker-state-leak / sort-order / seed-leak bug the
``processes=1`` vs ``processes=N`` contract is designed to surface;
N=2 is the smallest N that exercises the multi-process path. See PR
hotfix/ci-pytest-hang for the diagnosis.

Default torch-free job; uses MuJoCo ``TabletopEnv`` with a small
``max_steps`` so the spawn overhead + per-cell rollout fits well
under the 30-s suite budget.
"""

from __future__ import annotations

from typing import Any

from gauntlet.policy.random import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite.schema import AxisSpec, Suite

# Action dim for the tabletop env; exposed so factories below stay tiny.
_ACTION_DIM = 7


def _make_random_policy() -> RandomPolicy:
    """Module-level factory so it pickles under spawn."""
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


def _make_fast_env() -> Any:
    """Tiny-budget env factory so the multi-cell test fits the time budget."""
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=10)


def _make_multi_axis_suite() -> Suite:
    """3-axis x 2 episodes-per-cell Suite, designed to surface a sort bug.

    ``itertools.product``'s natural order over three axes produces
    cell-index ordering that is NOT lexicographic on the axis values.
    A worker pool whose completion order leaks into the Runner's
    output would re-shuffle these cells; the test then fails on the
    ``model_dump`` comparison even before reaching the field-level
    asserts.
    """
    return Suite(
        name="determinism-runner-suite",
        env="tabletop",
        seed=2024,
        episodes_per_cell=2,
        axes={
            "lighting_intensity": AxisSpec(low=0.5, high=1.5, steps=2),
            "camera_offset_x": AxisSpec(values=[-0.02, 0.02]),
            "distractor_count": AxisSpec(values=[0.0, 2.0]),
        },
    )


# ---------------------------------------------------- case 5: serial vs parallel


def test_runner_serial_and_parallel_produce_identical_episode_lists() -> None:
    """``processes=1`` and ``processes=2`` must produce
    ``list[Episode]`` byte-equal under ``model_dump`` after the
    Runner's ``(cell_index, episode_index)`` sort.

    Suite is 2x2x2 = 8 cells x 2 eps = 16 episodes. Big enough that
    a sort bug or a worker-state leak would surface; small enough
    to fit the spawn-pool boot cost in the suite time budget.

    ``n_workers=2`` (not 4) because the GitHub runner deadlocks on
    pool startup with 4 spawn workers each importing MuJoCo — see
    module docstring.
    """
    suite = _make_multi_axis_suite()
    n_total = suite.num_cells() * suite.episodes_per_cell

    serial = Runner(n_workers=1, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )
    parallel = Runner(n_workers=2, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )

    assert len(serial) == len(parallel) == n_total
    for es, ep in zip(serial, parallel, strict=True):
        # ``model_dump()`` includes nested dicts (``perturbation_config``,
        # ``metadata``) whose key ordering Pydantic preserves. Direct
        # dict equality is the right granularity — every float in
        # ``total_reward`` must agree to the bit.
        assert es.model_dump() == ep.model_dump(), (
            f"cell={es.cell_index} ep={es.episode_index} diverged across "
            f"worker counts; serial={es.model_dump()} parallel={ep.model_dump()}"
        )


def test_runner_output_is_sorted_by_cell_then_episode() -> None:
    """The Runner contract is that the returned list is sorted by
    ``(cell_index, episode_index)`` regardless of worker completion
    order. Pin the contract directly so a future refactor that drops
    the sort is loud, not silently-divergent.

    ``n_workers=2`` (not 4) for the same CI-runner-deadlock reason
    documented in the module docstring; 2 workers is enough to make
    completion order non-deterministic relative to submission order
    on a 16-item job.
    """
    suite = _make_multi_axis_suite()
    parallel = Runner(n_workers=2, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )
    keys = [(ep.cell_index, ep.episode_index) for ep in parallel]
    assert keys == sorted(keys), (
        f"Runner output not sorted by (cell_index, episode_index) under processes=2 — got {keys}"
    )


# ---------------------------------------------------- master_seed round-trip


def test_master_seed_is_identical_across_worker_counts() -> None:
    """A literal-int master seed must propagate to every Episode's
    ``metadata["master_seed"]`` regardless of process count.

    If the multi-process path re-derives the SeedSequence in each
    worker (a class of bug we want to catch loud), the value would
    differ from the in-process run.
    """
    suite = _make_multi_axis_suite()

    serial = Runner(n_workers=1, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )
    # n_workers=2 (not 4) — see module docstring on CI runner deadlock.
    parallel = Runner(n_workers=2, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )

    serial_seeds = {ep.metadata["master_seed"] for ep in serial}
    parallel_seeds = {ep.metadata["master_seed"] for ep in parallel}

    # Every episode in one run shares one master_seed.
    assert len(serial_seeds) == 1, f"serial master_seed leaks: {serial_seeds}"
    assert len(parallel_seeds) == 1, f"parallel master_seed leaks: {parallel_seeds}"
    # The two runs share the SAME master_seed because the suite carries
    # an explicit literal-int ``seed``.
    assert serial_seeds == parallel_seeds, (
        f"master_seed differs across worker counts: serial={serial_seeds} parallel={parallel_seeds}"
    )


def test_episode_seeds_unique_across_worker_counts() -> None:
    """Each episode's ``Episode.seed`` (the per-episode env seed
    derived from the SeedSequence node) must be unique across the
    run AND identical between the serial and parallel runs.

    This is what makes per-episode reproduction possible: a user
    re-running with a single ``WorkItem``-equivalent seed must hit
    the same env initial state regardless of which worker handled
    it the first time.
    """
    suite = _make_multi_axis_suite()

    serial = Runner(n_workers=1, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )
    # n_workers=2 (not 4) — see module docstring on CI runner deadlock.
    parallel = Runner(n_workers=2, env_factory=_make_fast_env).run(
        policy_factory=_make_random_policy,
        suite=suite,
    )

    serial_seeds = [ep.seed for ep in serial]
    parallel_seeds = [ep.seed for ep in parallel]

    # All-unique within a run.
    assert len(set(serial_seeds)) == len(serial_seeds)
    assert len(set(parallel_seeds)) == len(parallel_seeds)
    # Same seed for the same (cell, episode) pair across worker counts.
    assert serial_seeds == parallel_seeds
