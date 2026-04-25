"""B-02 behavioural-metrics tests (RoboEval RSS 2025).

Covers the env→worker→Episode→report path for the five flat
:class:`gauntlet.runner.Episode` fields ``time_to_success``,
``path_length_ratio``, ``jerk_rms``, ``near_collision_count``, and
``peak_force``:

1. Schema field defaults are ``None`` (backwards-compat with pre-B-02
   episodes.json).
2. The MuJoCo :class:`gauntlet.env.tabletop.TabletopEnv` publishes the
   four per-step ``info["behavior_*"]`` keys (``ee_pos``,
   ``control_dt``, ``near_collision_delta``, ``peak_contact_force``).
3. The worker derives ``time_to_success`` correctly from a synthetic
   env that signals ``info["success"]`` at a known step.
4. ``path_length_ratio`` is 1.0 for a straight-line trajectory and
   ``> 1.0`` for a U-shaped detour.
5. ``jerk_rms`` matches the closed-form value on a synthetic velocity
   trace whose third derivative is known.
6. A backend that publishes no behavioural telemetry leaves all five
   fields ``None`` (the cross-backend anti-feature gate).
7. Failure clusters surface ``mean_time_to_success`` /
   ``mean_path_length_ratio`` / ``mean_jerk_rms`` /
   ``mean_near_collision_count`` / ``mean_peak_force`` only when at
   least one cluster episode reported telemetry.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.report.analyze import build_report
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import WorkItem, execute_one


def _make_work_item(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    seed: int = 19,
) -> WorkItem:
    """One :class:`WorkItem` with a deterministic SeedSequence node."""
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="behavioral-metrics",
        cell_index=cell_index,
        episode_index=episode_index,
        perturbation_values={},
        episode_seq=episode_node,
        master_seed=seed,
        n_cells=1,
        episodes_per_cell=1,
    )


def _make_scripted() -> ScriptedPolicy:
    return ScriptedPolicy()


# ──────────────────────────────────────────────────────────────────────
# 1. Schema defaults — backwards-compat for pre-B-02 episodes.json.
# ──────────────────────────────────────────────────────────────────────


def test_episode_behavioral_fields_default_none() -> None:
    """All five B-02 fields default to ``None`` so old JSONs round-trip."""
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
    assert ep.time_to_success is None
    assert ep.path_length_ratio is None
    assert ep.jerk_rms is None
    assert ep.near_collision_count is None
    assert ep.peak_force is None


# ──────────────────────────────────────────────────────────────────────
# 2. MuJoCo tabletop publishes the four per-step ``behavior_*`` keys.
# ──────────────────────────────────────────────────────────────────────


def test_tabletop_step_info_publishes_behavior_keys() -> None:
    """MuJoCo tabletop emits the four ``behavior_*`` keys post-step."""
    env = TabletopEnv(max_steps=4)
    try:
        _, info_reset = env.reset(seed=5)
        # Reset must not surface the keys (worker treats absence as
        # "this is a reset, not a control step").
        assert "behavior_ee_pos" not in info_reset
        assert "behavior_control_dt" not in info_reset
        assert "behavior_near_collision_delta" not in info_reset
        assert "behavior_peak_contact_force" not in info_reset
        action = np.zeros(7, dtype=np.float64)
        _, _, _, _, info_step = env.step(action)
        assert "behavior_ee_pos" in info_step
        assert "behavior_control_dt" in info_step
        assert "behavior_near_collision_delta" in info_step
        assert "behavior_peak_contact_force" in info_step
        # Types as advertised on the contract.
        ee_pos = info_step["behavior_ee_pos"]
        assert isinstance(ee_pos, np.ndarray)
        assert ee_pos.shape == (3,)
        assert ee_pos.dtype == np.float64
        assert isinstance(info_step["behavior_control_dt"], float)
        assert info_step["behavior_control_dt"] > 0.0
        assert isinstance(info_step["behavior_near_collision_delta"], int)
        assert info_step["behavior_near_collision_delta"] >= 0
        assert isinstance(info_step["behavior_peak_contact_force"], float)
        assert info_step["behavior_peak_contact_force"] >= 0.0
    finally:
        env.close()


def test_tabletop_episode_carries_behavioral_metrics() -> None:
    """End-to-end: a MuJoCo rollout produces non-None behavioural fields.

    A failed-but-rolled-out episode populates ``path_length_ratio`` /
    ``jerk_rms`` / ``near_collision_count`` / ``peak_force`` (telemetry
    surfaced) but ``time_to_success`` stays ``None`` because the
    rollout did not succeed within the step cap.
    """
    env = TabletopEnv(max_steps=10)
    try:
        episode = execute_one(env, _make_scripted, _make_work_item())
    finally:
        env.close()
    # The cube doesn't reach the target in 10 steps with the scripted
    # policy at this seed → success is False, time_to_success is None.
    assert episode.success is False
    assert episode.time_to_success is None
    # Telemetry surfaced — these are NOT None.
    assert episode.near_collision_count is not None
    assert episode.near_collision_count >= 0
    assert episode.peak_force is not None
    assert episode.peak_force >= 0.0
    # path_length_ratio may be None if the EE didn't move (straight <
    # 1e-6); we just check it's None or >= 1.0 — never < 1.0 by
    # triangle inequality on the buffered EE samples.
    if episode.path_length_ratio is not None:
        assert episode.path_length_ratio >= 1.0 - 1e-9
    # jerk_rms requires T >= 4 EE samples; max_steps=10 ensures that.
    assert episode.jerk_rms is not None
    assert episode.jerk_rms >= 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. Synthetic env exercises the worker accumulator + derivation math.
# ──────────────────────────────────────────────────────────────────────


class _SyntheticBehaviorEnv:
    """Synthetic env that publishes scripted behavioural telemetry per step.

    Drives the worker's accumulator + episode-boundary derivations
    with deterministic values so the math can be checked against
    closed-form expected values, without depending on the MuJoCo
    physics integration.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        ee_schedule: list[NDArray[np.float64]],
        near_collision_schedule: list[int] | None = None,
        peak_force_schedule: list[float] | None = None,
        control_dt: float = 0.05,
        success_at_step: int | None = None,
    ) -> None:
        self._ee_schedule = list(ee_schedule)
        n = len(self._ee_schedule)
        self._near_collision_schedule = (
            list(near_collision_schedule) if near_collision_schedule is not None else [0] * n
        )
        self._peak_force_schedule = (
            list(peak_force_schedule) if peak_force_schedule is not None else [0.0] * n
        )
        if len(self._near_collision_schedule) != n or len(self._peak_force_schedule) != n:
            raise ValueError("schedules must match length")
        self._control_dt = control_dt
        self._success_at_step = success_at_step
        self._idx = 0
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
        self._idx = 0
        return {"state": np.zeros(1, dtype=np.float64)}, {"success": False}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        del action
        success = self._success_at_step is not None and self._idx + 1 >= self._success_at_step
        info: dict[str, Any] = {
            "success": success,
            "behavior_ee_pos": self._ee_schedule[self._idx].copy(),
            "behavior_control_dt": self._control_dt,
            "behavior_near_collision_delta": self._near_collision_schedule[self._idx],
            "behavior_peak_contact_force": self._peak_force_schedule[self._idx],
        }
        self._idx += 1
        truncated = (not success) and self._idx >= len(self._ee_schedule)
        terminated = success
        return (
            {"state": np.zeros(1, dtype=np.float64)},
            0.0,
            terminated,
            truncated,
            info,
        )

    def set_perturbation(self, name: str, value: float) -> None:
        del name, value

    def restore_baseline(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class _ZeroPolicy:
    def act(self, obs: dict[str, NDArray[Any]]) -> NDArray[Any]:
        del obs
        return np.zeros(7, dtype=np.float64)


def _make_zero_policy() -> _ZeroPolicy:
    return _ZeroPolicy()


def test_worker_time_to_success_from_step_count_and_dt() -> None:
    """``time_to_success = success_step * control_dt`` when success fires."""
    # Success on step 4 (1-indexed by ``step_count``), dt=0.1 → 0.4s.
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(6)]
    env = _SyntheticBehaviorEnv(
        ee_schedule=ee,
        control_dt=0.1,
        success_at_step=4,
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=21))
    assert episode.success is True
    assert episode.step_count == 4
    assert episode.time_to_success == pytest.approx(0.4)


def test_worker_time_to_success_none_on_failure() -> None:
    """Failed rollouts report ``time_to_success=None``."""
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(5)]
    env = _SyntheticBehaviorEnv(
        ee_schedule=ee,
        control_dt=0.1,
        success_at_step=None,  # never succeeds
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=23))
    assert episode.success is False
    assert episode.time_to_success is None


def test_path_length_ratio_straight_line_is_one() -> None:
    """A straight-line EE trajectory has path_length_ratio == 1.0."""
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(6)]  # 0..5 on X
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=0.1)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=25))
    assert episode.path_length_ratio is not None
    assert episode.path_length_ratio == pytest.approx(1.0)


def test_path_length_ratio_u_shape_above_one() -> None:
    """A U-shape detour reports path_length_ratio strictly > 1.0."""
    # Out-and-back over X then over Y: start (0,0,0), via (1,0,0), end
    # (1,1,0). Path length = 2.0 (1 along X + 1 along Y), straight =
    # sqrt(2). Ratio = 2/sqrt(2) = sqrt(2).
    ee = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=0.1)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=27))
    assert episode.path_length_ratio is not None
    assert episode.path_length_ratio == pytest.approx(np.sqrt(2.0))


def test_path_length_ratio_none_for_stationary_policy() -> None:
    """Stationary policy (start ≈ end) reports ``None``, not ``inf``."""
    # All samples within 1e-9 of origin → straight < 1e-6 m guard.
    ee = [np.array([1e-10 * i, 0.0, 0.0]) for i in range(5)]
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=0.1)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=29))
    assert episode.path_length_ratio is None


def test_jerk_rms_zero_for_constant_velocity() -> None:
    """Constant-velocity trajectory has zero third derivative."""
    # ee[t] = v * t (constant velocity v=1 along X). Third difference = 0.
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(6)]
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=0.1)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=31))
    assert episode.jerk_rms is not None
    assert episode.jerk_rms == pytest.approx(0.0, abs=1e-9)


def test_jerk_rms_matches_closed_form_on_cubic() -> None:
    """Cubic ee[t] = t^3 has third derivative = 6 / dt^3 (constant)."""
    # ee[t] = (t * dt)^3 along X (using continuous-time t = idx * dt).
    # Third forward difference: ee[t+3] - 3*ee[t+2] + 3*ee[t+1] - ee[t]
    # is the third forward difference of the discrete sequence and
    # asymptotically equals 6 * dt^3 for a continuous t^3 sampled at
    # integer indices, but the discrete formulation gives exactly 6
    # times h^3 where h is the index spacing — and dividing by dt^3
    # yields the third derivative ``d3/dt3 (t^3) = 6``. RMS of a
    # constant-magnitude jerk vector is its magnitude.
    dt = 0.1
    ee = [np.array([(i * dt) ** 3, 0.0, 0.0]) for i in range(6)]
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=dt)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=33))
    assert episode.jerk_rms is not None
    assert episode.jerk_rms == pytest.approx(6.0, rel=1e-6)


def test_jerk_rms_none_below_four_samples() -> None:
    """Rollouts with fewer than 4 EE samples report ``jerk_rms=None``."""
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(3)]  # only 3 samples
    env = _SyntheticBehaviorEnv(ee_schedule=ee, control_dt=0.1)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=35))
    assert episode.jerk_rms is None
    # path_length_ratio still works at T=3 → 1.0 along straight line.
    assert episode.path_length_ratio == pytest.approx(1.0)


def test_worker_accumulates_near_collision_and_peak_force() -> None:
    """Worker sums near-collision deltas and tracks peak force across steps."""
    ee = [np.array([float(i), 0.0, 0.0]) for i in range(5)]
    env = _SyntheticBehaviorEnv(
        ee_schedule=ee,
        near_collision_schedule=[0, 1, 2, 0, 3],
        peak_force_schedule=[0.0, 1.5, 0.5, 4.0, 2.0],
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=37))
    assert episode.near_collision_count == 6  # 0+1+2+0+3
    assert episode.peak_force == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────────────
# 4. Non-MuJoCo backend → all five fields stay None.
# ──────────────────────────────────────────────────────────────────────


class _FakeEnvNoBehavior:
    """Env stub that surfaces no behavioural telemetry (cross-backend gate)."""

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, max_steps: int = 4) -> None:
        self._max_steps = max_steps
        self._step = 0
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
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
        return {"state": np.zeros(7, dtype=np.float64)}, {"success": False}

    def step(
        self,
        action: NDArray[Any],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        truncated = self._step >= self._max_steps
        return (
            {"state": np.zeros(7, dtype=np.float64)},
            0.0,
            False,
            truncated,
            {"success": False},
        )

    def set_perturbation(self, name: str, value: float) -> None:
        del name, value

    def restore_baseline(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def test_non_mujoco_backend_yields_none_behavioral_fields() -> None:
    """A backend with no behaviour keys leaves all five Episode fields None."""
    env = _FakeEnvNoBehavior(max_steps=5)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=41))
    assert episode.time_to_success is None
    assert episode.path_length_ratio is None
    assert episode.jerk_rms is None
    assert episode.near_collision_count is None
    assert episode.peak_force is None


# ──────────────────────────────────────────────────────────────────────
# 5. Failure clusters surface the five mean_* aggregates.
# ──────────────────────────────────────────────────────────────────────


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    perturbation_config: dict[str, float],
    time_to_success: float | None = None,
    path_length_ratio: float | None = None,
    jerk_rms: float | None = None,
    near_collision_count: int | None = None,
    peak_force: float | None = None,
) -> Episode:
    return Episode(
        suite_name="behavioral-report",
        cell_index=cell_index,
        episode_index=episode_index,
        seed=cell_index * 100 + episode_index,
        perturbation_config=perturbation_config,
        success=success,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
        time_to_success=time_to_success,
        path_length_ratio=path_length_ratio,
        jerk_rms=jerk_rms,
        near_collision_count=near_collision_count,
        peak_force=peak_force,
    )


def test_failure_cluster_behavioral_means() -> None:
    """Cluster aggregates report the new behavioural means when telemetry exists."""
    eps: list[Episode] = []
    # 6 baseline successes pull the failure rate down.
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
                time_to_success=1.0,
                path_length_ratio=1.0,
                jerk_rms=0.5,
                near_collision_count=0,
                peak_force=0.1,
            )
        )
    # 4 failures at (a=1, b=1) with high near-collision / peak-force /
    # jerk; failed rollouts have time_to_success=None which gets dropped
    # from the cluster mean so mean_time_to_success stays None even
    # though path/jerk/near/peak are populated.
    for k in range(4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                time_to_success=None,
                path_length_ratio=2.5,
                jerk_rms=10.0,
                near_collision_count=8,
                peak_force=12.0,
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    # 4 failures, all with time_to_success=None → cluster mean is None.
    assert cluster.mean_time_to_success is None
    assert cluster.mean_path_length_ratio == pytest.approx(2.5)
    assert cluster.mean_jerk_rms == pytest.approx(10.0)
    assert cluster.mean_near_collision_count == pytest.approx(8.0)
    assert cluster.mean_peak_force == pytest.approx(12.0)


def test_failure_cluster_behavioral_means_none_when_no_telemetry() -> None:
    """All-None behaviour fields leave the cluster aggregates None."""
    eps: list[Episode] = []
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
            )
        )
    for k in range(4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    assert cluster.mean_time_to_success is None
    assert cluster.mean_path_length_ratio is None
    assert cluster.mean_jerk_rms is None
    assert cluster.mean_near_collision_count is None
    assert cluster.mean_peak_force is None


def test_failure_cluster_behavioral_partial_coverage_drops_unmeasured() -> None:
    """Episodes whose behaviour fields are None drop from numerator AND denominator."""
    eps: list[Episode] = []
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
            )
        )
    # 2 failures with telemetry, 2 failures without — the cluster mean
    # is over the 2 episodes that did report.
    for k in range(2):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                jerk_rms=8.0,
                near_collision_count=5,
                peak_force=10.0,
            )
        )
    for k in range(2, 4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    # Only 2 of 4 failures reported telemetry; cluster mean is over those 2.
    assert cluster.mean_jerk_rms == pytest.approx(8.0)
    assert cluster.mean_near_collision_count == pytest.approx(5.0)
    assert cluster.mean_peak_force == pytest.approx(10.0)
