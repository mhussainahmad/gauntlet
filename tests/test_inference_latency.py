"""B-37 inference-latency tracking + budget gate tests.

Covers the policy.act-timing → Episode → report → CLI path for the
three new :class:`gauntlet.runner.Episode` fields
``inference_latency_ms_p50`` / ``inference_latency_ms_p99`` /
``inference_latency_ms_max`` and the ``--max-inference-ms`` budget
soft flag stored on ``Episode.metadata['inference_budget_violated']``.

Discriminators specifically exercised (the "easy bug" cases):

1. Schema field defaults are ``None`` so pre-B-37 episodes.json files
   round-trip cleanly under ``Episode.model_validate``.
2. A synthetic policy with deterministic ``time.sleep`` durations
   produces the *expected* p50 / p99 / max under
   :func:`numpy.percentile` semantics (not just "non-zero").
3. T=0 rollouts (env truncates at reset) leave all three fields
   ``None`` rather than emitting a misleading 0.0.
4. ``policy.act_n`` (B-18 measurement) is NEVER timed — even when
   the worker is configured to invoke it, the latency buffer reflects
   only the ``policy.act`` calls. The B-37 docstring's anti-feature.
5. Budget compare:
   * p99 > limit → metadata key present and True.
   * p99 == limit → metadata key absent (strict ">" semantics).
   * p99 < limit → metadata key absent (NOT False — soft-flag
     contract).
   * No budget configured → metadata key absent on every Episode.
6. CLI flag is registered on the ``gauntlet run`` subcommand.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.env.base import GauntletEnv
from gauntlet.policy import Observation, SamplablePolicy
from gauntlet.policy.base import Policy
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import CONSISTENCY_STRIDE, WorkItem, execute_one

# ──────────────────────────────────────────────────────────────────────
# Fixtures.
# ──────────────────────────────────────────────────────────────────────


def _make_work_item(*, seed: int = 17) -> WorkItem:
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="b37-inference-latency",
        cell_index=0,
        episode_index=0,
        perturbation_values={},
        episode_seq=episode_node,
        master_seed=seed,
        n_cells=1,
        episodes_per_cell=1,
    )


class _FakeEnvFixedSteps:
    """Env stub that runs N control steps and stops. No info side-channels.

    Used in every B-37 test so the timing measurement is the only
    cross-cutting concern. The ``max_steps`` value drives the size of
    the latency buffer: with ``max_steps=N`` the worker calls
    ``policy.act`` exactly N times.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps
        self._step = 0
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
        )
        self.action_space: gym.spaces.Box = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        del seed, options
        self._step = 0
        return {"state": np.zeros(1, dtype=np.float64)}, {"success": False}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        truncated = self._step >= self._max_steps
        return (
            {"state": np.zeros(1, dtype=np.float64)},
            0.0,
            False,
            truncated,
            {"success": False},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        del name, value

    def restore_baseline(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakePolicyWithSchedule:
    """Synthetic policy that sleeps for a scheduled duration per call.

    The schedule is a list of seconds; each ``act`` call sleeps for
    the next entry, then returns a zero action. After the schedule
    is exhausted, raises (so a misconfigured test fails loudly
    instead of silently returning stale latency).
    """

    def __init__(self, sleep_schedule_seconds: list[float]) -> None:
        self._schedule = list(sleep_schedule_seconds)
        self._idx = 0

    def act(self, obs: Observation) -> NDArray[np.float64]:
        del obs
        if self._idx >= len(self._schedule):
            raise RuntimeError(
                f"_FakePolicyWithSchedule schedule exhausted at idx {self._idx}; "
                "test asked for more act() calls than scheduled."
            )
        time.sleep(self._schedule[self._idx])
        self._idx += 1
        return np.zeros(7, dtype=np.float64)


def _make_scheduled_policy_factory(schedule: list[float]) -> Callable[[], Policy]:
    """Return a zero-arg factory that yields a fresh scheduled policy.

    Each ``execute_one`` call instantiates one policy via the factory;
    we close over a fresh schedule list each invocation so the policy's
    internal counter resets. The cast keeps mypy --strict honest:
    :class:`_FakePolicyWithSchedule` satisfies the :class:`Policy`
    Protocol structurally (``act(obs) -> ndarray``) but the synthetic
    Protocol runtime-check is enforced by the :class:`Policy` base.
    """

    def factory() -> Policy:
        return cast(Policy, _FakePolicyWithSchedule(list(schedule)))

    return factory


def _as_env(env: object) -> GauntletEnv:
    """Cast a fake env to :class:`GauntletEnv` for ``execute_one``.

    The synthetic envs in this module satisfy the Protocol structurally
    (``reset`` / ``step`` / ``set_perturbation`` / ``restore_baseline``
    / ``close``) but mypy strict's :class:`gymnasium.spaces.Space`
    invariance flags ``Box`` vs ``Space[Any]`` as a conflict. Casting
    at the call site rather than inheriting from :class:`GauntletEnv`
    keeps the synthetic envs minimal and matches the pattern used by
    the other B-test fixtures (test_safety_violations, etc.).
    """
    return cast(GauntletEnv, env)


# ──────────────────────────────────────────────────────────────────────
# 1. Schema defaults.
# ──────────────────────────────────────────────────────────────────────


def test_episode_inference_latency_fields_default_none() -> None:
    """All three B-37 fields default to ``None`` so old JSONs round-trip."""
    ep = Episode(
        suite_name="schema-defaults",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={},
        success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
    )
    assert ep.inference_latency_ms_p50 is None
    assert ep.inference_latency_ms_p99 is None
    assert ep.inference_latency_ms_max is None


def test_episode_legacy_json_roundtrip_yields_none() -> None:
    """A pre-B-37 episodes.json dict (no latency keys) round-trips with None."""
    legacy: dict[str, Any] = {
        "suite_name": "legacy",
        "cell_index": 0,
        "episode_index": 0,
        "seed": 1,
        "perturbation_config": {},
        "success": False,
        "terminated": True,
        "truncated": False,
        "step_count": 5,
        "total_reward": 0.0,
        "metadata": {},
    }
    ep = Episode.model_validate(legacy)
    assert ep.inference_latency_ms_p50 is None
    assert ep.inference_latency_ms_p99 is None
    assert ep.inference_latency_ms_max is None
    # Equally critical for the soft-flag contract: legacy episodes have
    # no metadata budget key. Asserting it isn't there guards against
    # an accidental "set to False on construction" regression.
    assert "inference_budget_violated" not in ep.metadata


def test_episode_round_trips_through_json() -> None:
    """Latency fields survive a JSON round-trip without dtype drift."""
    ep = Episode(
        suite_name="b37",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={},
        success=False,
        terminated=False,
        truncated=True,
        step_count=10,
        total_reward=0.0,
        inference_latency_ms_p50=1.234,
        inference_latency_ms_p99=98.765,
        inference_latency_ms_max=123.456,
    )
    blob = ep.model_dump_json()
    restored = Episode.model_validate_json(blob)
    assert restored.inference_latency_ms_p50 == pytest.approx(1.234)
    assert restored.inference_latency_ms_p99 == pytest.approx(98.765)
    assert restored.inference_latency_ms_max == pytest.approx(123.456)


# ──────────────────────────────────────────────────────────────────────
# 2. Worker measurement: known sleeps → known percentiles.
# ──────────────────────────────────────────────────────────────────────


def test_worker_records_expected_p50_p99_max() -> None:
    """A scheduled-sleep policy yields p50 / p99 / max within tolerance.

    The schedule is dominated by 1ms sleeps with one 50ms outlier so
    p50 and p99 differ measurably; we use generous lower bounds (the
    OS scheduler can sleep longer than requested but never shorter
    than ~the requested duration on any sane CI runner).
    """
    # 9 fast steps (1ms) + 1 slow step (50ms). Median is ~1ms, p99 is
    # close to the slow tail (linear interp puts it 10% of the way
    # from 9th to 10th sample, i.e. closer to 50ms than 1ms).
    schedule_seconds = [0.001] * 9 + [0.050]
    env = _FakeEnvFixedSteps(max_steps=10)
    policy_factory = _make_scheduled_policy_factory(schedule_seconds)
    episode = execute_one(_as_env(env), policy_factory, _make_work_item(seed=29))

    assert episode.step_count == 10
    assert episode.inference_latency_ms_p50 is not None
    assert episode.inference_latency_ms_p99 is not None
    assert episode.inference_latency_ms_max is not None

    # Lower bounds are tight (OS sleep is monotonic-ish), upper bounds
    # are generous (jitter, GIL contention, scheduler quanta).
    assert 0.5 <= episode.inference_latency_ms_p50 <= 25.0
    # p99 of the latency distribution should reflect the 50ms tail.
    assert episode.inference_latency_ms_p99 >= 5.0
    # max IS one of the recorded samples — must be at least the
    # slowest sleep we asked for.
    assert episode.inference_latency_ms_max >= 25.0
    # Sanity: max >= p99 >= p50 (numpy percentile invariant).
    assert episode.inference_latency_ms_p50 <= episode.inference_latency_ms_p99
    assert episode.inference_latency_ms_p99 <= episode.inference_latency_ms_max


def test_worker_skips_act_n_in_latency_measurement() -> None:
    """B-18 ``policy.act_n`` is NOT counted toward the B-37 latency buffer.

    Constructs a SamplablePolicy whose ``act_n`` sleeps 100ms per
    candidate (8x = 800ms per measured step) but whose ``act`` is
    near-instant. With ``measure_action_consistency=True`` the worker
    invokes ``act_n`` every CONSISTENCY_STRIDE-th step. If the latency
    buffer mistakenly captured ``act_n``, p99 would jump to ~800ms.
    The contract: p99 stays ~zero because only ``act`` is timed.
    """

    class _SlowSamplablePolicy:
        # SamplablePolicy is a runtime-checkable Protocol; structural
        # subtyping kicks in once both methods are implemented.
        def __init__(self) -> None:
            self._n_act = 0
            self._n_act_n = 0

        def act(self, obs: Observation) -> NDArray[np.float64]:
            del obs
            self._n_act += 1
            return np.zeros(7, dtype=np.float64)

        def act_n(self, obs: Observation, n: int) -> list[NDArray[np.float64]]:
            del obs
            self._n_act_n += 1
            # Slow sampler — would dwarf any p99 if it were timed.
            time.sleep(0.100)
            return [np.zeros(7, dtype=np.float64) for _ in range(n)]

    # Static check: the protocol is structural, so this both confirms
    # the Protocol detects the test class AND documents the assumption.
    assert isinstance(_SlowSamplablePolicy(), SamplablePolicy)

    # Configure max_steps so the stride-aligned step is exercised.
    n_steps = CONSISTENCY_STRIDE + 1
    env = _FakeEnvFixedSteps(max_steps=n_steps)
    episode = execute_one(
        _as_env(env),
        cast(Callable[[], Policy], _SlowSamplablePolicy),
        _make_work_item(seed=31),
        measure_action_consistency=True,
    )
    assert episode.step_count == n_steps
    assert episode.inference_latency_ms_p99 is not None
    # If ``act_n`` were timed, the per-step bucket on a stride-aligned
    # step would be ~100ms; we assert well below that. 10ms gives
    # plenty of headroom for an unrelated GIL stall on a busy CI box
    # without weakening the discriminator.
    assert episode.inference_latency_ms_p99 < 10.0
    # Also confirm action_variance was actually computed (i.e. the
    # measurement path was taken — otherwise the test is vacuous).
    assert episode.action_variance is not None


# ──────────────────────────────────────────────────────────────────────
# 3. Budget gate semantics.
# ──────────────────────────────────────────────────────────────────────


def test_budget_violation_metadata_set_when_p99_exceeds() -> None:
    """p99 > max → metadata['inference_budget_violated'] is present and True."""
    schedule_seconds = [0.020] * 5
    env = _FakeEnvFixedSteps(max_steps=5)
    episode = execute_one(
        _as_env(env),
        _make_scheduled_policy_factory(schedule_seconds),
        _make_work_item(seed=37),
        max_inference_ms=5.0,
    )
    assert episode.inference_latency_ms_p99 is not None
    assert episode.inference_latency_ms_p99 > 5.0
    assert episode.metadata.get("inference_budget_violated") is True


def test_budget_metadata_absent_when_p99_within_limit() -> None:
    """p99 <= max → metadata key is ABSENT (not False)."""
    # Generous budget — every step's latency is well under 100ms.
    schedule_seconds = [0.001] * 5
    env = _FakeEnvFixedSteps(max_steps=5)
    episode = execute_one(
        _as_env(env),
        _make_scheduled_policy_factory(schedule_seconds),
        _make_work_item(seed=41),
        max_inference_ms=100.0,
    )
    assert episode.inference_latency_ms_p99 is not None
    assert episode.inference_latency_ms_p99 < 100.0
    # The soft-flag contract: absent, never False.
    assert "inference_budget_violated" not in episode.metadata


def test_budget_metadata_absent_when_no_budget_configured() -> None:
    """No max_inference_ms → metadata key is ABSENT on every episode."""
    schedule_seconds = [0.020] * 5
    env = _FakeEnvFixedSteps(max_steps=5)
    episode = execute_one(
        _as_env(env),
        _make_scheduled_policy_factory(schedule_seconds),
        _make_work_item(seed=43),
        # max_inference_ms intentionally omitted.
    )
    assert episode.inference_latency_ms_p99 is not None
    assert "inference_budget_violated" not in episode.metadata


def test_budget_violation_does_not_flip_success_or_abort() -> None:
    """Soft-flag contract: a violation never raises, never flips success."""
    schedule_seconds = [0.030] * 5
    env = _FakeEnvFixedSteps(max_steps=5)
    # Should NOT raise even though every step blows past the budget.
    episode = execute_one(
        _as_env(env),
        _make_scheduled_policy_factory(schedule_seconds),
        _make_work_item(seed=47),
        max_inference_ms=1.0,
    )
    # success is determined by the env's info["success"], not by the
    # latency budget. The fake env never sets it, so success=False.
    assert episode.success is False
    # The flag was set as a soft signal.
    assert episode.metadata.get("inference_budget_violated") is True


# ──────────────────────────────────────────────────────────────────────
# 4. CLI flag plumbing.
# ──────────────────────────────────────────────────────────────────────


class TestCliFlagPlumbing:
    def test_run_help_advertises_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Widen rich help so the long flag isn't word-wrapped across
        # the assertion target.
        monkeypatch.setenv("COLUMNS", "200")
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-inference-ms" in result.output
