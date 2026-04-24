"""Property-based determinism check on :class:`Runner`.

Phase 2.5 Task 13 — for any seed, two ``Runner.run`` invocations with
the same ``Suite`` + ``policy_factory`` produce byte-identical Episode
lists. This is the headline reproducibility contract from
``GAUNTLET_SPEC.md`` §6.

Cost control: we use a tiny fake env (no MuJoCo load) so the property
explores 30 different seeds inside a few seconds. The fake satisfies
:class:`gauntlet.env.base.GauntletEnv` structurally; the Runner
dispatches through a caller-supplied ``env_factory`` and never touches
the registry, so the env name on the Suite (``"tabletop"`` — the only
default-registered env) is purely a schema-acceptable string.

Hypothesis budget: ``max_examples=30`` per test (each example runs the
Runner end-to-end twice, so cost compounds).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gauntlet.policy.random import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite.schema import AxisSpec, Suite

# ----- module-level helpers (must pickle for spawn — though we use
# ``n_workers=1`` here to skip pool overhead, the env_factory protocol
# still pickles anything via the in-process path's spawn-context check).


_ACTION_DIM = 3


class _TinyFakeEnv:
    """Minimal :class:`GauntletEnv` Protocol impl with no FFI cost."""

    AXIS_NAMES = frozenset({"lighting_intensity"})
    VISUAL_ONLY_AXES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {"x": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float64)
        self._pending: dict[str, float] = {}
        self._steps_taken = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        # Echo the seed in the obs so a downstream reward / success is
        # deterministic from the seed — this is the bit we're asserting
        # round-trips identically across runs.
        self._steps_taken = 0
        self._pending.clear()
        rng = np.random.default_rng(seed)
        return ({"x": rng.uniform(-1.0, 1.0, size=(1,))}, {})

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._steps_taken += 1
        # Reward derived from the action so step bookkeeping is observable.
        reward = float(np.sum(np.asarray(action, dtype=np.float64)))
        terminated = self._steps_taken >= 2
        return (
            {"x": np.asarray([reward], dtype=np.float64)},
            reward,
            terminated,
            False,
            {"success": True},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        self._pending[name] = value

    def restore_baseline(self) -> None:
        self._pending.clear()
        self._steps_taken = 0

    def close(self) -> None:
        return None


def _tiny_env_factory() -> Any:
    return _TinyFakeEnv()


def _random_policy_factory() -> RandomPolicy:
    # ``seed=None`` because the Runner re-seeds via ``policy.reset(rng)`` in
    # :func:`gauntlet.runner.worker.execute_one`. The two calls must
    # therefore re-seed identically, which is the determinism property.
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)


# ----- the property ---------------------------------------------------------


def _make_tiny_suite(seed: int) -> Suite:
    return Suite(
        name="prop-determinism",
        env="tabletop",  # any registered env name; we override the factory.
        seed=seed,
        episodes_per_cell=1,
        # 1 axis x 2 steps -> 2 cells x 1 ep = 2 Episodes per run.
        axes={"lighting_intensity": AxisSpec(low=0.4, high=0.8, steps=2)},
    )


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_runner_two_runs_same_seed_byte_identical_episodes(seed: int) -> None:
    """Spec §6 hard rule: two ``Runner.run`` invocations with the same
    inputs must produce byte-identical Episode lists."""
    suite = _make_tiny_suite(seed)

    runner_a = Runner(n_workers=1, env_factory=_tiny_env_factory)
    runner_b = Runner(n_workers=1, env_factory=_tiny_env_factory)

    eps_a = runner_a.run(policy_factory=_random_policy_factory, suite=suite)
    eps_b = runner_b.run(policy_factory=_random_policy_factory, suite=suite)

    assert len(eps_a) == len(eps_b) == suite.num_cells() * suite.episodes_per_cell
    # Episode equality is structural via pydantic; this asserts every
    # field including total_reward, seed, perturbation_config, metadata.
    assert eps_a == eps_b


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_runner_episode_count_matches_grid_size(seed: int) -> None:
    """Independent invariant: the Episode list length is exactly
    ``num_cells * episodes_per_cell``, regardless of seed. The Runner's
    output-sort step preserves this — a property test catches an
    accidental dedup / drop."""
    suite = _make_tiny_suite(seed)
    runner = Runner(n_workers=1, env_factory=_tiny_env_factory)
    eps = runner.run(policy_factory=_random_policy_factory, suite=suite)

    expected = suite.num_cells() * suite.episodes_per_cell
    assert len(eps) == expected
    # Output ordering is sorted by (cell_index, episode_index) — assert.
    sorted_keys = sorted((ep.cell_index, ep.episode_index) for ep in eps)
    assert sorted_keys == [(ep.cell_index, ep.episode_index) for ep in eps]
    # Every (cell, episode) tuple unique.
    assert len({(ep.cell_index, ep.episode_index) for ep in eps}) == expected


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=30,
    deadline=timedelta(seconds=2),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_runner_per_episode_seeds_pairwise_distinct(seed: int) -> None:
    """The seed-derivation tree (``master.spawn(n_cells)[i].spawn(eps)[j]``
    + ``generate_state``) must produce a unique env seed per episode.
    A regression that reused a parent's entropy across siblings
    (the bug the worker docstring warns about) would collide here."""
    suite = Suite(
        name="prop-distinct-seeds",
        env="tabletop",
        seed=seed,
        episodes_per_cell=2,
        axes={"lighting_intensity": AxisSpec(low=0.4, high=0.8, steps=2)},
    )
    runner = Runner(n_workers=1, env_factory=_tiny_env_factory)
    eps = runner.run(policy_factory=_random_policy_factory, suite=suite)
    assert len(eps) == 4
    seeds = [ep.seed for ep in eps]
    assert len(set(seeds)) == len(seeds), f"duplicate env seeds: {seeds}"
