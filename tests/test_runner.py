"""Runner tests — see ``GAUNTLET_SPEC.md`` §5 task 6 and §6 (reproducibility).

The reproducibility tests are the load-bearing ones: they pin both
single-process and cross-process determinism, which together carry the
correctness contract of the entire harness.

All factory functions used in tests are defined at module scope so they
pickle cleanly under the ``spawn`` start method (lambdas do not). The
tests that exercise the multiprocessing path therefore look slightly
verbose, but every helper is intentional.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import register_env
from gauntlet.policy.base import Policy
from gauntlet.policy.random import RandomPolicy
from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.runner import Episode, Runner, WorkItem
from gauntlet.runner.worker import extract_env_seed
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------------
# Module-level factory helpers (must be picklable for spawn-based pools).
# ----------------------------------------------------------------------------


# Action dim for the tabletop env; exposed so factories below stay tiny.
_ACTION_DIM = 7

# Counter incremented by ``_counted_random_factory``. The Runner's
# in-process path runs in this very interpreter, so a module global is
# observable post-run. (For the multiprocess path each worker has its
# own copy; test #10 documents the behaviour difference.)
_FACTORY_CALL_COUNTER: list[int] = [0]


def make_random_policy() -> RandomPolicy:
    """Module-level factory so it pickles under spawn."""
    # seed=None — the runner re-seeds via ``policy.reset(rng)``.
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


def make_scripted_policy() -> ScriptedPolicy:
    """Default scripted trajectory factory."""
    return ScriptedPolicy()


def _counted_random_factory() -> RandomPolicy:
    """Random-policy factory that records call count in a module global."""
    _FACTORY_CALL_COUNTER[0] += 1
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


def make_fast_env() -> Any:
    """Env factory with a small ``max_steps`` to keep tests under a few seconds.

    The ``Any`` return is local to tests; the runner type is
    ``Callable[[], TabletopEnv]`` and our concrete return is exactly
    that — we just avoid importing TabletopEnv into the test module's
    annotations to keep the import surface tight.
    """
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=20)


# ----------------------------------------------------------------------------
# Suite builders.
# ----------------------------------------------------------------------------


def _make_suite(
    *,
    name: str = "test-suite",
    seed: int | None = 1234,
    episodes_per_cell: int = 2,
    axes: dict[str, AxisSpec] | None = None,
) -> Suite:
    if axes is None:
        # 2x2 grid -> 4 cells.
        axes = {
            "lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2),
            "camera_offset_x": AxisSpec(low=-0.02, high=0.02, steps=2),
        }
    return Suite(
        name=name,
        env="tabletop",
        seed=seed,
        episodes_per_cell=episodes_per_cell,
        axes=axes,
    )


# ----------------------------------------------------------------------------
# 1. Single-worker smoke.
# ----------------------------------------------------------------------------


def test_single_worker_returns_expected_episodes_in_order() -> None:
    """2x2 axes x 2 eps -> 8 episodes, sorted by (cell_index, episode_index)."""
    suite = _make_suite(seed=7, episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    assert len(episodes) == suite.num_cells() * suite.episodes_per_cell == 8
    # Sorted by (cell_index, episode_index).
    keys = [(ep.cell_index, ep.episode_index) for ep in episodes]
    assert keys == sorted(keys)
    # Identity bookkeeping.
    for ep in episodes:
        assert ep.suite_name == "test-suite"
        assert isinstance(ep, Episode)
        assert 0 <= ep.cell_index < suite.num_cells()
        assert 0 <= ep.episode_index < suite.episodes_per_cell


# ----------------------------------------------------------------------------
# 2. Per-cell perturbation_config matches.
# ----------------------------------------------------------------------------


def test_episode_perturbation_config_matches_originating_cell() -> None:
    suite = _make_suite(seed=11, episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    cell_values = {cell.index: dict(cell.values) for cell in suite.cells()}
    for ep in episodes:
        assert ep.perturbation_config == cell_values[ep.cell_index]


# ----------------------------------------------------------------------------
# 3. RandomPolicy integration.
# ----------------------------------------------------------------------------


def test_random_policy_integration_smoke() -> None:
    suite = _make_suite(
        seed=3,
        episodes_per_cell=1,
        axes={
            "lighting_intensity": AxisSpec(low=0.5, high=1.0, steps=2),
        },
    )
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_random_policy, suite=suite)

    assert len(episodes) == 2
    for ep in episodes:
        assert isinstance(ep.success, bool)
        assert isinstance(ep.terminated, bool)
        assert isinstance(ep.truncated, bool)
        # Either terminated, truncated, or both — at least one is true.
        assert ep.terminated or ep.truncated
        # max_steps in make_fast_env is 20; step_count must respect the budget.
        assert 0 <= ep.step_count <= 20


# ----------------------------------------------------------------------------
# 4. Determinism single-worker.
# ----------------------------------------------------------------------------


def test_determinism_single_worker_runs_match() -> None:
    suite = _make_suite(seed=999, episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    a = runner.run(policy_factory=make_random_policy, suite=suite)
    b = runner.run(policy_factory=make_random_policy, suite=suite)

    assert len(a) == len(b)
    for ea, eb in zip(a, b, strict=True):
        # Bit-for-bit equality across every Pydantic field.
        assert ea.model_dump() == eb.model_dump()


# ----------------------------------------------------------------------------
# 5. Determinism under parallelism (THE load-bearing test).
# ----------------------------------------------------------------------------


def test_determinism_n_workers_one_vs_two() -> None:
    """n_workers=1 and n_workers=2 must produce identical Episode lists.

    This catches seed-leak bugs that single-process tests miss: stale
    env state between episodes a worker handles, accidental dependence
    on wall-clock or PID, or non-deterministic completion ordering
    leaking into Episode payloads.

    Uses :class:`_FakeProtocolEnv` (not MuJoCo ``TabletopEnv``) for
    the env factory: the determinism contract under test is
    Runner-pool-internal (seed derivation, WorkItem dispatch, output
    sort) and is env-agnostic. The fake env constructs in microseconds
    and avoids the MuJoCo+spawn pool-startup deadlock on the GitHub
    ``ubuntu-latest`` runner (4 logical CPUs, ~2 usable under cgroup
    limits; two spawned interpreters each compiling MJCF wedge before
    any work item completes). The MuJoCo+Runner integration is
    exercised via the n=1 path in
    :func:`test_default_factory_dispatches_tabletop_via_registry` and
    by every other test in this file. See PR hotfix/ci-pytest-hang.

    ``n_workers=2`` (not 4) for the same reason: even with the fake
    env, 4 spawned interpreters each importing numpy + reimporting
    the test module is wasteful and adds CI-runner stress without
    strengthening the contract.
    """
    suite = _make_suite(seed=2024, episodes_per_cell=2)

    # Fake env (no MuJoCo). The Random policy still drives stochastic
    # actions, so the per-episode RNG-stream determinism contract is
    # exercised in full; the fake env just terminates on step 1 with
    # a constant observation, which is fine because the asserts target
    # Episode metadata (seed, master_seed, ordering), not env state.
    serial = Runner(n_workers=1, env_factory=_fake_env_factory).run(
        policy_factory=make_random_policy,
        suite=suite,
    )
    parallel = Runner(n_workers=2, env_factory=_fake_env_factory).run(
        policy_factory=make_random_policy,
        suite=suite,
    )

    assert len(serial) == len(parallel) == suite.num_cells() * suite.episodes_per_cell
    for es, ep in zip(serial, parallel, strict=True):
        assert es.model_dump() == ep.model_dump(), (
            f"cell={es.cell_index} ep={es.episode_index} diverged across worker counts; "
            f"serial={es.model_dump()} parallel={ep.model_dump()}"
        )


# ----------------------------------------------------------------------------
# 6. Master seed = None.
# ----------------------------------------------------------------------------


def test_master_seed_none_runs_and_records_master_seed() -> None:
    """A None seed must still produce a runnable suite and record the
    auto-generated master seed in every Episode for later reproduction."""
    suite = _make_suite(seed=None, episodes_per_cell=1)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    assert len(episodes) == suite.num_cells()
    master_seeds = {ep.metadata.get("master_seed") for ep in episodes}
    assert len(master_seeds) == 1, "all episodes in one run share one master_seed"
    only = master_seeds.pop()
    assert isinstance(only, int)
    assert only > 0  # OS entropy is overwhelmingly non-zero

    # Reproducibility: feed the recorded master_seed back into a Suite
    # and the resulting Episodes must match.
    suite2 = _make_suite(seed=only, episodes_per_cell=1)
    repro = runner.run(policy_factory=make_scripted_policy, suite=suite2)
    for a, b in zip(episodes, repro, strict=True):
        # ``suite_name`` and ``master_seed`` are identical too because we
        # rebuilt with the same name. Compare full payloads.
        assert a.model_dump() == b.model_dump()


# ----------------------------------------------------------------------------
# 7. Reject bad inputs.
# ----------------------------------------------------------------------------


def test_runner_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        Runner(n_workers=0)


def test_runner_rejects_negative_workers() -> None:
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        Runner(n_workers=-3)


def test_runner_rejects_fork_start_method() -> None:
    with pytest.raises(ValueError, match="start_method must be 'spawn'"):
        Runner(n_workers=2, start_method="fork")


def test_run_rejects_unsupported_env() -> None:
    """Defensive boundary check — Suite validates ``env`` at construction
    too, but the Runner is the config->execution boundary so it
    re-checks. We synthesise a Suite past validation by mutating the
    field in place, mirroring how a future loader bug might present."""
    suite = _make_suite(seed=1, episodes_per_cell=1)
    # ``model_config = ConfigDict(extra="forbid")`` does not freeze
    # existing fields; assignment validation is off by default in v2.
    object.__setattr__(suite, "env", "isaac")
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    with pytest.raises(ValueError, match="unsupported env"):
        runner.run(policy_factory=make_scripted_policy, suite=suite)


# ----------------------------------------------------------------------------
# 8. Episode.seed uniqueness across a run.
# ----------------------------------------------------------------------------


def test_episode_seeds_are_unique_within_a_run() -> None:
    suite = _make_suite(seed=5, episodes_per_cell=2)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    seeds = [ep.seed for ep in episodes]
    # SeedSequence.spawn produces statistically independent streams; the
    # uint32 we pull off each is overwhelmingly unique. Equality across
    # any two would point at a derivation bug (e.g. accidentally using
    # SeedSequence.entropy instead of generate_state).
    assert len(set(seeds)) == len(seeds)


# ----------------------------------------------------------------------------
# 9. ScriptedPolicy integration.
# ----------------------------------------------------------------------------


def test_scripted_policy_integration_smoke() -> None:
    """1 cell x 1 ep with the default scripted trajectory."""
    suite = _make_suite(
        seed=42,
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(low=0.7, high=0.7, steps=1)},
    )
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    assert len(episodes) == 1
    ep = episodes[0]
    # Scripted trajectory is open-loop; one of the two flags must fire
    # within the env's max_steps.
    assert ep.terminated or ep.truncated


# ----------------------------------------------------------------------------
# 10. Pickle / factory-boundary semantics.
# ----------------------------------------------------------------------------


def test_factory_called_once_per_episode_in_process() -> None:
    """Document the Runner contract: ``policy_factory()`` is invoked once
    per (cell, episode) pair, not once per worker.

    We can verify this directly in the in-process path because the
    counter lives in this very interpreter. For the multiprocess path
    the counter would be per-worker; the determinism test above is the
    end-to-end check that each worker's policies still drive the
    per-episode RNG stream the runner intends.
    """
    suite = _make_suite(seed=8, episodes_per_cell=2)
    _FACTORY_CALL_COUNTER[0] = 0

    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=_counted_random_factory, suite=suite)

    expected_calls = suite.num_cells() * suite.episodes_per_cell
    assert _FACTORY_CALL_COUNTER[0] == expected_calls
    assert len(episodes) == expected_calls


def test_lambdas_fail_loudly_under_spawn() -> None:
    """Documents the pickle contract for the multiworker path.

    Lambdas (and other unpicklable closures) cannot cross the spawn
    process boundary. Any caller that hands the Runner a lambda gets a
    pickling error from multiprocessing — that is the loud failure.
    We verify the pickle behaviour directly here so the contract is
    checked without spinning up an actual worker pool.
    """

    def local_factory() -> RandomPolicy:  # nested -> not picklable.
        return RandomPolicy(action_dim=_ACTION_DIM, seed=None)

    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(local_factory)

    # The module-level counterpart picks pickles fine.
    pickle.dumps(make_random_policy)


# ----------------------------------------------------------------------------
# Internal helpers — small, focused checks on the seed-derivation logic.
# ----------------------------------------------------------------------------


def test_episode_metadata_echoes_run_topology() -> None:
    """Every Episode records ``n_cells`` and ``episodes_per_cell`` so the
    replay tool can reconstruct the SeedSequence spawn tree even if the
    suite YAML is edited between run and replay.

    See ``docs/phase2-rfc-004-trajectory-replay.md`` §3.
    """
    suite = _make_suite(seed=321, episodes_per_cell=3)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    expected_n_cells = suite.num_cells()
    expected_eps_per_cell = suite.episodes_per_cell
    for ep in episodes:
        assert ep.metadata["n_cells"] == expected_n_cells
        assert ep.metadata["episodes_per_cell"] == expected_eps_per_cell


def test_extract_env_seed_is_deterministic_per_node() -> None:
    """Same SeedSequence node -> same env seed; siblings -> different seeds."""
    parent = np.random.SeedSequence(42)
    children = parent.spawn(3)
    seeds = [extract_env_seed(c) for c in children]
    assert len(set(seeds)) == 3
    # Determinism: re-derive from the same parent yields the same triple.
    children_again = np.random.SeedSequence(42).spawn(3)
    seeds_again = [extract_env_seed(c) for c in children_again]
    assert seeds == seeds_again


def test_work_item_round_trips_through_pickle() -> None:
    """WorkItem.episode_seq is a SeedSequence; verify it survives pickle
    so the multiprocessing path's contract holds."""
    seq = np.random.SeedSequence(123).spawn(1)[0]
    item = WorkItem(
        suite_name="x",
        cell_index=0,
        episode_index=0,
        perturbation_values={"lighting_intensity": 1.0},
        episode_seq=seq,
        master_seed=123,
        n_cells=1,
        episodes_per_cell=1,
    )
    restored = pickle.loads(pickle.dumps(item))
    assert restored.suite_name == "x"
    assert restored.episode_seq.spawn_key == seq.spawn_key
    assert extract_env_seed(restored.episode_seq) == extract_env_seed(seq)


def test_default_factory_dispatches_tabletop_via_registry() -> None:
    """When no ``env_factory`` is supplied, :meth:`Runner.run` dispatches
    through :func:`get_env_factory` on ``suite.env``. For the default
    MuJoCo ``tabletop`` backend the resolved factory is the
    :class:`TabletopEnv` class itself (registered by
    ``gauntlet.env.__init__``) — calling it with no args yields a
    stock instance, matching the pre-fix ``_default_env_factory`` shape.
    """
    from gauntlet.env.registry import get_env_factory
    from gauntlet.env.tabletop import TabletopEnv

    factory = get_env_factory("tabletop")
    env = factory()
    try:
        assert isinstance(env, TabletopEnv)
    finally:
        env.close()


# ----------------------------------------------------------------------------
# Helpers used by tests that build their own factories
# (reserved for future expansion; kept here to avoid local closures).
# ----------------------------------------------------------------------------


def _check_factory_pickles(factory: Callable[[], Policy]) -> None:
    """Helper: round-trip a factory through pickle to mirror what the
    spawn-context Pool will do at submit time."""
    pickle.dumps(factory)


# Confirm the spawn start method is at least available on this platform
# — a quick guard so the determinism test fails early with a useful
# message rather than deep inside multiprocessing if a CI image ever
# disables spawn.
def test_spawn_start_method_available() -> None:
    assert "spawn" in mp.get_all_start_methods()


# ----------------------------------------------------------------------------
# Phase 2 Task 5 step 6 — protect the GauntletEnv Protocol seam: a fake
# backend that does NOT subclass gym.Env or import MuJoCo must drive the
# Runner end-to-end. If the Runner's or worker's dispatch ever accidentally
# narrows to TabletopEnv, this test fails.
# ----------------------------------------------------------------------------


class _FakeProtocolEnv:
    """Minimal stand-in for a non-MuJoCo backend.

    Satisfies :class:`gauntlet.env.base.GauntletEnv` structurally without
    inheriting from ``gymnasium.Env``. Deterministic outputs (constant
    observation, constant reward, success-on-first-step) keep the test
    short and assertion-shaped.

    ``AXIS_NAMES`` enumerates every axis any test suite in this file or
    in ``test_determinism_runner_workers.py`` may declare. The fake
    accepts (and ignores) the value — it never affects rollout behaviour
    — so the multi-worker determinism tests can exercise the Runner's
    pool path without spinning up MuJoCo (whose MJCF compile under
    spawn deadlocks on the GitHub runner; see hotfix/ci-pytest-hang).
    """

    AXIS_NAMES = frozenset(
        {
            "distractor_count",
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "object_texture",
        }
    )
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {"cube_pos": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float64)
        self._pending: dict[str, float] = {}
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        self._step_count = 0
        self._pending.clear()
        return (
            {"cube_pos": np.zeros(3, dtype=np.float64)},
            {"seed_echo": seed},
        )

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        terminated = True  # Succeed on step 1; Runner records it as a success.
        return (
            {"cube_pos": np.zeros(3, dtype=np.float64)},
            1.0,
            terminated,
            False,
            {"success": True, "step_count": self._step_count},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()
        self._step_count = 0

    def close(self) -> None:
        return None


def _fake_env_factory() -> Any:
    """Module-level so it pickles under spawn (the Protocol seam still
    works under the pool path even though this test only exercises -w 1)."""
    return _FakeProtocolEnv()


def test_runner_accepts_non_tabletop_gauntlet_env_protocol_impl() -> None:
    """Step-6 regression test: a fake backend satisfying GauntletEnv
    structurally must drive the Runner end-to-end. If the Runner ever
    accidentally narrows to TabletopEnv again, this fails.
    """
    # Sanity — the fake actually satisfies the Protocol at runtime.
    fake = _FakeProtocolEnv()
    assert isinstance(fake, GauntletEnv)
    fake.close()

    suite = _make_suite(
        name="fake-protocol-suite",
        seed=13,
        episodes_per_cell=1,
        axes={"distractor_count": AxisSpec(values=[0.0, 1.0])},
    )

    runner = Runner(n_workers=1, env_factory=_fake_env_factory)
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    # 2 cells * 1 episode = 2 episodes, all successful per the fake's contract.
    assert len(episodes) == 2
    assert all(ep.success for ep in episodes)
    # The distractor_count axis values are preserved on the Episode.
    values = sorted(ep.perturbation_config["distractor_count"] for ep in episodes)
    assert values == [0.0, 1.0]


# ----------------------------------------------------------------------------
# RFC-006 §9 bullet 1 / RFC-005 §11 bug-fix regression: when the caller
# leaves ``env_factory`` unset, the Runner must dispatch through the env
# registry on ``suite.env``. The pre-fix Runner silently used MuJoCo
# ``TabletopEnv`` regardless of ``suite.env`` — so ``env: tabletop-pybullet``
# YAMLs ran on MuJoCo with the only clue being "results look wrong".
# ----------------------------------------------------------------------------


# Counter observable from the test-interpreter (the in-process path runs
# in this interpreter — same rationale as ``_FACTORY_CALL_COUNTER``).
_DISPATCH_ENV_CONSTRUCTIONS: list[int] = [0]


class _DispatchSentinelEnv(_FakeProtocolEnv):
    """Sub-Fake that records every construction in a module-level counter.

    Inheriting ``_FakeProtocolEnv`` preserves the Protocol-conformant
    reset/step/close surface; we only need to know *that* the class was
    instantiated — the exact rollout behaviour is immaterial to this
    test. Module-level so ``spawn`` would be able to pickle it even
    though the regression test exercises ``n_workers=1``.
    """

    def __init__(self) -> None:
        super().__init__()
        _DISPATCH_ENV_CONSTRUCTIONS[0] += 1


def test_runner_dispatches_via_registry_when_env_factory_is_none() -> None:
    """Register a sentinel backend, run Runner(env_factory=None) against
    a suite that names it, and assert the sentinel — not TabletopEnv —
    was constructed. Uses a unique registry key so the test is safe
    under pytest's session-scoped module-level registry.
    """
    env_name = "_dispatch_test_env_runner"
    register_env(env_name, cast("Callable[..., GauntletEnv]", _DispatchSentinelEnv))

    _DISPATCH_ENV_CONSTRUCTIONS[0] = 0

    suite = Suite(
        name="dispatch-test-suite",
        env=env_name,
        seed=3,
        episodes_per_cell=1,
        axes={"distractor_count": AxisSpec(values=[0.0])},
    )

    runner = Runner(n_workers=1)  # no env_factory -> registry dispatch
    episodes = runner.run(policy_factory=make_scripted_policy, suite=suite)

    assert len(episodes) == 1
    # One cell * one episode -> exactly one env construction via the
    # in-process fast path (which reuses a single env across items).
    assert _DISPATCH_ENV_CONSTRUCTIONS[0] == 1
