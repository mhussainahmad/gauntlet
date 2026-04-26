"""Tests for the ``inference_delay_jitter`` runner-side perturbation axis (B-38).

The axis is *runner-owned*: unlike the env-direct axes
(``lighting_intensity``, ``object_swap``, etc.) and unlike the wrapper-
based axes (``image_attack``, ``instruction_paraphrase``), the staleness
FIFO lives entirely in :func:`gauntlet.runner.worker.execute_one`. The
backend env never sees the axis; the worker pops it out of
:attr:`WorkItem.perturbation_values` before the
:meth:`GauntletEnv.set_perturbation` loop.

Tests cover the contract surfaces:

* axis registration (canonical ``AXIS_NAMES`` membership; default
  constructor produces the (0, 50, 200, 500) ms grid from the backlog),
* :func:`resolve_inference_delay_steps` behaviour (env property,
  fallback warning, floor semantics, validation),
* FIFO behaviour (delay=2 -> step k delivers the action computed at
  k-2; warmup window sends ``np.zeros_like``; delay > episode length
  sends zeros throughout),
* zero-delay is byte-identical to the no-jitter path,
* B-37 latency metric is unaffected by the synthetic delay (only
  ``policy.act`` is timed; the FIFO bookkeeping is excluded),
* :attr:`Episode.perturbation_config` echoes the axis value for replay,
* the worker filters the axis out of the env loop (no
  ``ValueError: unknown perturbation axis`` from the backend).

Headless throughout: no real backend env is constructed.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.env.perturbation import (
    AXIS_KIND_CATEGORICAL,
    AXIS_NAMES,
    PerturbationAxis,
    axis_for,
    inference_delay_jitter,
)
from gauntlet.env.perturbation.axes import DEFAULT_BOUNDS
from gauntlet.policy import Observation
from gauntlet.policy.base import Policy
from gauntlet.runner.worker import (
    DEFAULT_CONTROL_DT_FALLBACK_S,
    INFERENCE_DELAY_JITTER_AXIS,
    WorkItem,
    execute_one,
    resolve_inference_delay_steps,
)

# ──────────────────────────────────────────────────────────────────────
# Fixtures.
# ──────────────────────────────────────────────────────────────────────


def _make_work_item(
    *,
    seed: int = 17,
    perturbation_values: dict[str, float] | None = None,
) -> WorkItem:
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="b38-inference-delay-jitter",
        cell_index=0,
        episode_index=0,
        perturbation_values=perturbation_values or {},
        episode_seq=episode_node,
        master_seed=seed,
        n_cells=1,
        episodes_per_cell=1,
    )


class _RecordingFakeEnv:
    """Env stub that records every action it receives via ``step``.

    The :attr:`delivered_actions` list is the test discriminator: it
    captures exactly what the worker handed to the env, post-FIFO. The
    fake env exposes a public ``control_dt`` (mirrors the
    :class:`gauntlet.env.tabletop.TabletopEnv` property) so the worker
    can resolve the FIFO depth without warning.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, max_steps: int, control_dt_s: float = 0.050) -> None:
        self._max_steps = max_steps
        self._step = 0
        self.control_dt: float = float(control_dt_s)
        self.delivered_actions: list[NDArray[np.float64]] = []
        self.set_perturbation_calls: list[tuple[str, float]] = []
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
        self.delivered_actions.clear()
        return {"state": np.zeros(1, dtype=np.float64)}, {"success": False}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        self.delivered_actions.append(np.asarray(action, dtype=np.float64).copy())
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
        self.set_perturbation_calls.append((name, value))

    def restore_baseline(self) -> None:
        pass

    def close(self) -> None:
        pass


class _NoControlDtEnv(_RecordingFakeEnv):
    """Identical to :class:`_RecordingFakeEnv` but without a ``control_dt`` attr.

    Used to exercise the worker's documented fallback behaviour when a
    backend lacks the property — :class:`UserWarning` plus the 50 ms /
    20 Hz default.
    """

    def __init__(self, *, max_steps: int) -> None:
        super().__init__(max_steps=max_steps)
        # Remove the control_dt attribute so getattr returns None.
        del self.control_dt


class _CountingPolicy:
    """Deterministic policy whose action[0] is the call index.

    Lets tests trace exactly which call's output ended up being
    delivered to the env: ``delivered_actions[k][0] == intended_index``
    pinpoints the FIFO offset.
    """

    def __init__(self) -> None:
        self._idx = 0

    def act(self, obs: Observation) -> NDArray[np.float64]:
        del obs
        out = np.zeros(7, dtype=np.float64)
        # Encode the call index in the first action component so test
        # assertions can read it back from the recording env.
        out[0] = float(self._idx)
        # Keep the rest distinguishable from a zero no-op: a positive
        # signature in component 1 means "fresh policy output". The
        # FIFO's warmup branch emits np.zeros_like which leaves
        # component 1 at zero — clean discriminator.
        out[1] = 1.0
        self._idx += 1
        return out


def _counting_policy_factory() -> Policy:
    return cast(Policy, _CountingPolicy())


def _as_env(env: object) -> GauntletEnv:
    """Cast a fake env to :class:`GauntletEnv` for ``execute_one``.

    Mirrors the cast helper used by ``test_inference_latency.py``:
    structural Protocol satisfaction with a small mypy escape hatch
    for :class:`gymnasium.spaces.Space` invariance.
    """
    return cast(GauntletEnv, env)


# ──────────────────────────────────────────────────────────────────────
# 1. Axis registration.
# ──────────────────────────────────────────────────────────────────────


class TestAxisRegistration:
    def test_axis_in_canonical_axis_names(self) -> None:
        """The axis name appears in ``AXIS_NAMES`` so suite YAML accepts it."""
        assert "inference_delay_jitter" in AXIS_NAMES

    def test_default_constructor_yields_categorical_axis(self) -> None:
        """Default constructor produces a categorical axis on the canonical grid."""
        axis = inference_delay_jitter()
        assert isinstance(axis, PerturbationAxis)
        assert axis.name == "inference_delay_jitter"
        assert axis.kind == AXIS_KIND_CATEGORICAL
        # Default sampler should draw from the (0, 50, 200, 500) ms
        # grid documented in the backlog.
        rng = np.random.default_rng(0)
        seen: set[float] = set()
        for _ in range(200):
            seen.add(axis.sample(rng))
        assert seen == {0.0, 50.0, 200.0, 500.0}

    def test_axis_for_returns_same_construct(self) -> None:
        """``axis_for`` lookup matches the direct constructor."""
        from_registry = axis_for("inference_delay_jitter")
        direct = inference_delay_jitter()
        assert from_registry.name == direct.name
        assert from_registry.kind == direct.kind
        assert from_registry.low == direct.low
        assert from_registry.high == direct.high

    def test_default_bounds_match_backlog_spec(self) -> None:
        """Default bounds cover the canonical (0..500) ms grid."""
        lo, hi = DEFAULT_BOUNDS["inference_delay_jitter"]
        assert lo == 0.0
        assert hi >= 500.0


# ──────────────────────────────────────────────────────────────────────
# 2. resolve_inference_delay_steps.
# ──────────────────────────────────────────────────────────────────────


class TestResolveInferenceDelaySteps:
    def test_zero_ms_yields_zero_steps(self) -> None:
        """0 ms always resolves to 0 steps regardless of control_dt."""
        env = _RecordingFakeEnv(max_steps=10, control_dt_s=0.050)
        assert resolve_inference_delay_steps(0.0, _as_env(env)) == 0

    def test_floor_semantics_below_one_step(self) -> None:
        """49 ms on a 50 ms loop floors to 0 (honest "no integer-step lag")."""
        env = _RecordingFakeEnv(max_steps=10, control_dt_s=0.050)
        assert resolve_inference_delay_steps(49.0, _as_env(env)) == 0

    def test_canonical_grid_resolution_at_50ms_dt(self) -> None:
        """The (0, 50, 200, 500) ms grid maps to (0, 1, 4, 10) steps at dt=50ms."""
        env = _RecordingFakeEnv(max_steps=20, control_dt_s=0.050)
        assert resolve_inference_delay_steps(0.0, _as_env(env)) == 0
        assert resolve_inference_delay_steps(50.0, _as_env(env)) == 1
        assert resolve_inference_delay_steps(200.0, _as_env(env)) == 4
        assert resolve_inference_delay_steps(500.0, _as_env(env)) == 10

    def test_resolution_changes_with_control_dt(self) -> None:
        """200 ms on a 100 ms loop resolves to 2 steps; on 25 ms loop, 8 steps."""
        env_slow = _RecordingFakeEnv(max_steps=10, control_dt_s=0.100)
        env_fast = _RecordingFakeEnv(max_steps=10, control_dt_s=0.025)
        assert resolve_inference_delay_steps(200.0, _as_env(env_slow)) == 2
        assert resolve_inference_delay_steps(200.0, _as_env(env_fast)) == 8

    def test_negative_delay_raises(self) -> None:
        """Negative ms is a programming error — fail loud."""
        env = _RecordingFakeEnv(max_steps=10)
        with pytest.raises(ValueError, match="must be >= 0"):
            resolve_inference_delay_steps(-50.0, _as_env(env))

    def test_missing_control_dt_falls_back_with_warning(self) -> None:
        """Backend without ``control_dt`` triggers UserWarning + 50ms default."""
        env = _NoControlDtEnv(max_steps=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            steps = resolve_inference_delay_steps(200.0, _as_env(env))
        assert steps == int(0.200 / DEFAULT_CONTROL_DT_FALLBACK_S)
        assert any("control_dt" in str(w.message) for w in caught)
        assert any(issubclass(w.category, UserWarning) for w in caught)


# ──────────────────────────────────────────────────────────────────────
# 3. FIFO behaviour in the rollout loop.
# ──────────────────────────────────────────────────────────────────────


class TestStaleActionFifo:
    def test_zero_delay_is_byte_identical_no_jitter_path(self) -> None:
        """delay=0 -> delivered_actions match the policy's fresh outputs.

        Discriminator: with the counting policy, ``action[0]`` for step
        k is exactly k (no FIFO offset). Anything else means the
        zero-delay codepath drifted from the pre-B-38 behaviour.
        """
        env = _RecordingFakeEnv(max_steps=5, control_dt_s=0.050)
        execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 0.0}),
        )
        assert len(env.delivered_actions) == 5
        for k, action in enumerate(env.delivered_actions):
            # action[0] == k means "this step received the freshly-computed action".
            assert action[0] == pytest.approx(float(k))
            # Component 1 == 1.0 means the action came from policy.act,
            # not from the np.zeros_like warmup branch.
            assert action[1] == pytest.approx(1.0)

    def test_axis_absent_is_byte_identical_no_jitter_path(self) -> None:
        """No axis in perturbation_values -> same as delay=0 (no FIFO)."""
        env = _RecordingFakeEnv(max_steps=5, control_dt_s=0.050)
        execute_one(_as_env(env), _counting_policy_factory, _make_work_item())
        assert len(env.delivered_actions) == 5
        for k, action in enumerate(env.delivered_actions):
            assert action[0] == pytest.approx(float(k))

    def test_delay_two_steps_delivers_action_from_two_steps_ago(self) -> None:
        """delay_steps=2 -> step k receives the action computed at k-2.

        FIFO semantics worked example (matches the docstring contract):

        * k=0: policy emits A0; warmup -> deliver zeros.
        * k=1: policy emits A1; warmup -> deliver zeros.
        * k=2: policy emits A2; warmup elapsed -> deliver A0.
        * k=3: policy emits A3; deliver A1.
        * k=4: policy emits A4; deliver A2.

        On a 50 ms control loop, 100 ms axis value floors to exactly 2
        steps (delay_steps=2). The counting policy makes the discriminator
        crisp: ``delivered_actions[k][0] == k - 2`` for k >= 2; warmup
        steps deliver an all-zero action.
        """
        env = _RecordingFakeEnv(max_steps=5, control_dt_s=0.050)
        execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 100.0}),
        )
        assert len(env.delivered_actions) == 5
        # First 2 steps are warmup -> all-zero no-op.
        for k in range(2):
            assert np.all(env.delivered_actions[k] == 0.0), f"step {k} not zero"
        # Remaining steps deliver the action computed delay_steps steps ago.
        for k in range(2, 5):
            expected_idx = float(k - 2)
            assert env.delivered_actions[k][0] == pytest.approx(expected_idx)
            # Fresh-policy signature must be present (i.e. it's a
            # buffered policy.act output, not the warmup zeros).
            assert env.delivered_actions[k][1] == pytest.approx(1.0)

    def test_delay_one_step(self) -> None:
        """delay_steps=1 -> step 0 zeros; from step 1 onward, lag-1 delivery."""
        env = _RecordingFakeEnv(max_steps=4, control_dt_s=0.050)
        execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 50.0}),
        )
        assert len(env.delivered_actions) == 4
        assert np.all(env.delivered_actions[0] == 0.0)
        for k in range(1, 4):
            assert env.delivered_actions[k][0] == pytest.approx(float(k - 1))

    def test_delay_greater_than_episode_length_sends_zeros_throughout(self) -> None:
        """delay_ms > episode_length * control_dt -> every step zero-action.

        With max_steps=3 and delay_steps=10, the FIFO never finishes
        warming up; every delivered action is np.zeros_like. The
        episode terminates normally (truncated when max_steps reached).
        """
        env = _RecordingFakeEnv(max_steps=3, control_dt_s=0.050)
        episode = execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 500.0}),
        )
        assert episode.step_count == 3
        assert episode.truncated is True
        assert len(env.delivered_actions) == 3
        for k, action in enumerate(env.delivered_actions):
            assert np.all(action == 0.0), f"step {k} got non-zero action {action}"

    def test_axis_filtered_out_of_env_set_perturbation_loop(self) -> None:
        """The axis is NOT routed through env.set_perturbation.

        No backend's ``AXIS_NAMES`` contains ``inference_delay_jitter``,
        so leaking the axis into the env loop would raise. The fake
        env records every set_perturbation call and asserts the axis
        is absent from that list.
        """
        env = _RecordingFakeEnv(max_steps=2, control_dt_s=0.050)
        execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(
                perturbation_values={
                    INFERENCE_DELAY_JITTER_AXIS: 100.0,
                },
            ),
        )
        names = [name for name, _ in env.set_perturbation_calls]
        assert "inference_delay_jitter" not in names

    def test_perturbation_config_records_axis_value_for_replay(self) -> None:
        """Episode.perturbation_config echoes the axis value verbatim.

        Replay machinery rebuilds the FIFO from this dict, so silently
        dropping the axis here would break determinism.
        """
        env = _RecordingFakeEnv(max_steps=2, control_dt_s=0.050)
        episode = execute_one(
            _as_env(env),
            _counting_policy_factory,
            _make_work_item(
                perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 200.0},
            ),
        )
        assert episode.perturbation_config.get("inference_delay_jitter") == pytest.approx(200.0)


# ──────────────────────────────────────────────────────────────────────
# 4. B-37 latency metric is unaffected by the synthetic delay.
# ──────────────────────────────────────────────────────────────────────


class _FastPolicy:
    """Near-instant policy. Useful for checking that the synthetic delay
    is NOT billed against the B-37 latency buffer.
    """

    def __init__(self) -> None:
        self._idx = 0

    def act(self, obs: Observation) -> NDArray[np.float64]:
        del obs
        # No sleep — the policy compute cost itself is sub-microsecond.
        out = np.zeros(7, dtype=np.float64)
        out[0] = float(self._idx)
        self._idx += 1
        return out


def _fast_policy_factory() -> Policy:
    return cast(Policy, _FastPolicy())


class _SlowPolicy:
    """Policy with a fixed per-call sleep. The sleep IS billed
    against B-37 (only ``policy.act`` is timed), independently of the
    synthetic FIFO delay.
    """

    def __init__(self, sleep_seconds: float) -> None:
        self._sleep = sleep_seconds
        self._idx = 0

    def act(self, obs: Observation) -> NDArray[np.float64]:
        del obs
        time.sleep(self._sleep)
        out = np.zeros(7, dtype=np.float64)
        out[0] = float(self._idx)
        self._idx += 1
        return out


def _slow_policy_factory(sleep_seconds: float) -> Callable[[], Policy]:
    def factory() -> Policy:
        return cast(Policy, _SlowPolicy(sleep_seconds))

    return factory


class TestLatencyMetricUnaffectedBySyntheticDelay:
    def test_p99_unaffected_when_only_synthetic_delay_is_present(self) -> None:
        """A 500 ms axis on a fast policy keeps p99 well below 500 ms.

        If the worker accidentally counted the FIFO bookkeeping (or
        np.zeros_like, or the deque append) as policy compute, p99
        would balloon. The contract: ``policy.act`` brackets only the
        policy call. The synthetic delay shows up downstream as failed
        rollouts via the success-rate metric, never via latency.
        """
        env = _RecordingFakeEnv(max_steps=20, control_dt_s=0.050)
        episode = execute_one(
            _as_env(env),
            _fast_policy_factory,
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 500.0}),
        )
        assert episode.inference_latency_ms_p99 is not None
        # Generous CI ceiling — a fast policy with no sleep should
        # latency well under 5 ms even on a busy runner. If the
        # synthetic delay leaked into the buffer, p99 would jump to
        # ~50 ms or beyond.
        assert episode.inference_latency_ms_p99 < 5.0
        assert episode.inference_latency_ms_max is not None
        assert episode.inference_latency_ms_max < 10.0

    def test_real_compute_cost_still_billed(self) -> None:
        """A slow ``policy.act`` IS billed even when synthetic delay is present.

        Discriminator vs. the prior test: this confirms the
        ``perf_counter`` brackets are still around ``policy.act``
        (i.e. we didn't accidentally remove the timing entirely).
        """
        env = _RecordingFakeEnv(max_steps=5, control_dt_s=0.050)
        episode = execute_one(
            _as_env(env),
            _slow_policy_factory(0.020),  # 20 ms per call.
            _make_work_item(perturbation_values={INFERENCE_DELAY_JITTER_AXIS: 100.0}),
        )
        assert episode.inference_latency_ms_p99 is not None
        # 20 ms sleep + scheduler jitter -> well above 10 ms.
        assert episode.inference_latency_ms_p99 >= 10.0
