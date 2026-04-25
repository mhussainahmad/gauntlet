"""B-30 safety-violation telemetry tests.

Covers the env→worker→Episode→report path for the four flat
:class:`gauntlet.runner.Episode` fields ``n_collisions``,
``n_joint_limit_excursions``, ``energy_over_budget``, and
``n_workspace_excursions``:

1. Schema field defaults are ``None`` (backwards-compat with pre-B-30
   episodes.json).
2. The MuJoCo :class:`gauntlet.env.tabletop.TabletopEnv` publishes the
   three per-step ``info["safety_*"]`` keys (``mj_data.ncon`` delta,
   ``model.jnt_range`` check, EE-vs-table-half-extents check).
3. The worker accumulates collisions / joint excursions / workspace
   excursions correctly from a synthetic env that injects scripted
   per-step values.
4. A backend that publishes no safety telemetry leaves all four
   fields ``None`` (the cross-backend anti-feature gate).
5. ``Report.success_safe_rate`` and the ``success_unsafe`` framing
   correctly identify successful-but-unsafe episodes; partial-coverage
   datasets (None on some episodes, measured on others) handle the
   numerator / denominator drop correctly.
6. ``energy_over_budget`` is derived in the worker from the
   accumulated ``actuator_energy`` versus the
   :class:`gauntlet.runner.worker.WorkerInitArgs.energy_budget` knob.
7. Failure clusters surface ``mean_collisions`` / ``mean_joint_excursions``
   only when at least one cluster episode reported telemetry.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.report.analyze import build_report, episode_has_safety_violation
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import WorkItem, execute_one


def _make_work_item(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    seed: int = 17,
) -> WorkItem:
    """One :class:`WorkItem` with a deterministic SeedSequence node."""
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="safety-violations",
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
# 1. Schema defaults — backwards-compat for pre-B-30 episodes.json.
# ──────────────────────────────────────────────────────────────────────


def test_episode_safety_fields_default_none() -> None:
    """All four B-30 fields default to ``None`` so old JSONs round-trip."""
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
    assert ep.n_collisions is None
    assert ep.n_joint_limit_excursions is None
    assert ep.energy_over_budget is None
    assert ep.n_workspace_excursions is None


# ──────────────────────────────────────────────────────────────────────
# 2. MuJoCo tabletop publishes the three per-step ``safety_*`` keys.
# ──────────────────────────────────────────────────────────────────────


def test_tabletop_step_info_publishes_safety_keys() -> None:
    """MuJoCo tabletop emits the three ``safety_*`` keys post-step."""
    env = TabletopEnv(max_steps=4)
    try:
        _, info_reset = env.reset(seed=5)
        # Reset must not surface the keys (worker treats absence as
        # "this is a reset, not a control step").
        assert "safety_n_collisions_delta" not in info_reset
        assert "safety_joint_limit_violation" not in info_reset
        assert "safety_workspace_excursion" not in info_reset
        action = np.zeros(7, dtype=np.float64)
        _, _, _, _, info_step = env.step(action)
        assert "safety_n_collisions_delta" in info_step
        assert "safety_joint_limit_violation" in info_step
        assert "safety_workspace_excursion" in info_step
        # Types as advertised on the contract.
        assert isinstance(info_step["safety_n_collisions_delta"], int)
        assert isinstance(info_step["safety_joint_limit_violation"], bool)
        assert isinstance(info_step["safety_workspace_excursion"], bool)
        # Non-negative collision delta (clamped at 0 by definition).
        assert info_step["safety_n_collisions_delta"] >= 0
    finally:
        env.close()


def test_tabletop_episode_carries_safety_counts() -> None:
    """End-to-end: a MuJoCo rollout produces non-None safety counts.

    The mocap-only tabletop has no bounded joints so
    ``n_joint_limit_excursions`` must be 0. Workspace excursions are
    bounded by EE motion against ``_TABLE_HALF_X / Y`` — ScriptedPolicy
    keeps the EE inside the table for canonical seeds, so the count
    is also 0 here. The contract is that the fields are NOT None
    (telemetry surfaced) — distinct from the non-MuJoCo backend test
    below where they STAY None.
    """
    env = TabletopEnv(max_steps=20)
    try:
        episode = execute_one(env, _make_scripted, _make_work_item())
    finally:
        env.close()
    assert episode.n_collisions is not None
    assert episode.n_collisions >= 0
    assert episode.n_joint_limit_excursions == 0
    assert episode.n_workspace_excursions == 0
    # ``energy_over_budget`` stays None because (a) no budget configured
    # AND (b) the mocap-only tabletop has nu==0 → actuator_energy is None.
    assert episode.energy_over_budget is None


# ──────────────────────────────────────────────────────────────────────
# 3. Synthetic env exercises the worker accumulator math.
# ──────────────────────────────────────────────────────────────────────


class _SyntheticSafetyEnv:
    """Synthetic env that publishes scripted safety telemetry per step.

    Drives the worker's accumulator with deterministic values so the
    sum / count math can be checked against a closed-form expected
    value, without depending on the MuJoCo physics integration.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        collision_schedule: list[int],
        joint_schedule: list[bool],
        workspace_schedule: list[bool],
        energy_schedule: list[float] | None = None,
        torque_schedule: list[float] | None = None,
    ) -> None:
        n = len(collision_schedule)
        if len(joint_schedule) != n or len(workspace_schedule) != n:
            raise ValueError("schedules must match length")
        self._collision_schedule = list(collision_schedule)
        self._joint_schedule = list(joint_schedule)
        self._workspace_schedule = list(workspace_schedule)
        self._energy_schedule = list(energy_schedule) if energy_schedule is not None else None
        self._torque_schedule = list(torque_schedule) if torque_schedule is not None else None
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
        info: dict[str, Any] = {
            "success": False,
            "safety_n_collisions_delta": self._collision_schedule[self._idx],
            "safety_joint_limit_violation": self._joint_schedule[self._idx],
            "safety_workspace_excursion": self._workspace_schedule[self._idx],
        }
        if self._energy_schedule is not None and self._torque_schedule is not None:
            info["actuator_energy_delta"] = self._energy_schedule[self._idx]
            info["actuator_torque_norm"] = self._torque_schedule[self._idx]
        self._idx += 1
        truncated = self._idx >= len(self._collision_schedule)
        return (
            {"state": np.zeros(1, dtype=np.float64)},
            0.0,
            False,
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


def test_worker_accumulates_synthetic_safety_telemetry() -> None:
    """Worker sums collisions and counts per-step joint/workspace flags."""
    env = _SyntheticSafetyEnv(
        collision_schedule=[0, 1, 2, 0, 3],
        joint_schedule=[False, True, False, True, True],
        workspace_schedule=[True, False, False, False, True],
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=29))
    assert episode.step_count == 5
    assert episode.n_collisions == 6  # 0+1+2+0+3
    assert episode.n_joint_limit_excursions == 3  # True at idx 1, 3, 4
    assert episode.n_workspace_excursions == 2  # True at idx 0, 4
    # No budget configured → field is None even though the env
    # surfaces no actuator telemetry. Distinct from False (= measured,
    # within budget).
    assert episode.energy_over_budget is None


def test_worker_energy_budget_derives_over_budget_bool() -> None:
    """``energy_budget`` + actuator telemetry → ``energy_over_budget`` bool."""
    # Total energy = 6.0 across 4 steps; budget=5.0 → over budget.
    env = _SyntheticSafetyEnv(
        collision_schedule=[0, 0, 0, 0],
        joint_schedule=[False] * 4,
        workspace_schedule=[False] * 4,
        energy_schedule=[1.0, 2.0, 1.5, 1.5],
        torque_schedule=[0.1, 0.2, 0.3, 0.4],
    )
    over = execute_one(
        env,
        _make_zero_policy,
        _make_work_item(seed=33),
        energy_budget=5.0,
    )
    assert over.actuator_energy == pytest.approx(6.0)
    assert over.energy_over_budget is True

    # Same rollout, budget=10.0 → within budget.
    env2 = _SyntheticSafetyEnv(
        collision_schedule=[0, 0, 0, 0],
        joint_schedule=[False] * 4,
        workspace_schedule=[False] * 4,
        energy_schedule=[1.0, 2.0, 1.5, 1.5],
        torque_schedule=[0.1, 0.2, 0.3, 0.4],
    )
    within = execute_one(
        env2,
        _make_zero_policy,
        _make_work_item(seed=33),
        energy_budget=10.0,
    )
    assert within.energy_over_budget is False


# ──────────────────────────────────────────────────────────────────────
# 4. Non-MuJoCo backend → all four fields stay None.
# ──────────────────────────────────────────────────────────────────────


class _FakeEnvNoSafety:
    """Env stub that surfaces no safety telemetry (cross-backend gate)."""

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


def test_non_mujoco_backend_yields_none_safety_fields() -> None:
    """A backend with no safety keys leaves all four Episode fields None."""
    env = _FakeEnvNoSafety(max_steps=5)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=41))
    assert episode.n_collisions is None
    assert episode.n_joint_limit_excursions is None
    assert episode.n_workspace_excursions is None
    assert episode.energy_over_budget is None


# ──────────────────────────────────────────────────────────────────────
# 5. ``success_safe_rate`` + ``success_unsafe`` framing in the report.
# ──────────────────────────────────────────────────────────────────────


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    perturbation_config: dict[str, float],
    n_collisions: int | None = None,
    n_joint_limit_excursions: int | None = None,
    energy_over_budget: bool | None = None,
    n_workspace_excursions: int | None = None,
) -> Episode:
    return Episode(
        suite_name="safety-report",
        cell_index=cell_index,
        episode_index=episode_index,
        seed=cell_index * 100 + episode_index,
        perturbation_config=perturbation_config,
        success=success,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
        n_collisions=n_collisions,
        n_joint_limit_excursions=n_joint_limit_excursions,
        energy_over_budget=energy_over_budget,
        n_workspace_excursions=n_workspace_excursions,
    )


def test_episode_has_safety_violation_predicate() -> None:
    """Predicate returns False for None fields (anti-feature contract)."""
    safe = _ep(
        cell_index=0,
        episode_index=0,
        success=True,
        perturbation_config={"a": 0.0},
        n_collisions=0,
        n_joint_limit_excursions=0,
        energy_over_budget=False,
        n_workspace_excursions=0,
    )
    assert episode_has_safety_violation(safe) is False
    # All-None: not measured, NOT a violation.
    unmeasured = _ep(
        cell_index=0,
        episode_index=0,
        success=True,
        perturbation_config={"a": 0.0},
    )
    assert episode_has_safety_violation(unmeasured) is False
    # Each field independently triggers the predicate.
    for kw in [
        {"n_collisions": 1},
        {"n_joint_limit_excursions": 1},
        {"energy_over_budget": True},
        {"n_workspace_excursions": 1},
    ]:
        ep = _ep(
            cell_index=0,
            episode_index=0,
            success=True,
            perturbation_config={"a": 0.0},
            **kw,  # type: ignore[arg-type]
        )
        assert episode_has_safety_violation(ep) is True, kw


def test_report_success_safe_rate_tags_unsafe_successes() -> None:
    """Successful episodes with violations drop ``success_safe_rate``."""
    eps: list[Episode] = []
    # 4 safe successes.
    for k in range(4):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
                n_collisions=0,
                n_joint_limit_excursions=0,
                energy_over_budget=False,
                n_workspace_excursions=0,
            )
        )
    # 2 unsafe successes (collisions > 0).
    for k in range(2):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=True,
                perturbation_config={"a": 1.0, "b": 0.0},
                n_collisions=3,
                n_joint_limit_excursions=0,
                energy_over_budget=False,
                n_workspace_excursions=0,
            )
        )
    # 4 failures (irrelevant for success_safe_rate denominator).
    for k in range(4):
        eps.append(
            _ep(
                cell_index=2,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                n_collisions=10,
                n_joint_limit_excursions=2,
                energy_over_budget=True,
                n_workspace_excursions=1,
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    # 4 of 6 successes were safe → success_safe_rate = 4/6.
    assert report.success_safe_rate == pytest.approx(4.0 / 6.0)


def test_report_success_safe_rate_none_when_no_telemetry() -> None:
    """All-None safety telemetry → success_safe_rate is ``None``."""
    eps = [
        _ep(
            cell_index=0,
            episode_index=k,
            success=True,
            perturbation_config={"a": 0.0},
        )
        for k in range(5)
    ]
    report = build_report(eps)
    assert report.success_safe_rate is None


def test_report_success_safe_rate_partial_coverage_drops_unmeasured() -> None:
    """Successes with all-None safety fields are dropped from the rate."""
    eps: list[Episode] = []
    # 2 successes with no telemetry at all (dropped from numerator AND denominator).
    for k in range(2):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
            )
        )
    # 2 successes with telemetry, all safe.
    for k in range(2):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=True,
                perturbation_config={"a": 1.0, "b": 0.0},
                n_collisions=0,
                n_joint_limit_excursions=0,
                energy_over_budget=False,
                n_workspace_excursions=0,
            )
        )
    # 2 successes with telemetry, unsafe.
    for k in range(2):
        eps.append(
            _ep(
                cell_index=2,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 1.0},
                n_collisions=2,
                n_joint_limit_excursions=0,
                energy_over_budget=False,
                n_workspace_excursions=0,
            )
        )
    report = build_report(eps)
    # 2 safe / 4 measured → 0.5; the 2 unmeasured successes were dropped.
    assert report.success_safe_rate == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────
# 6. Failure clusters surface mean_collisions / mean_joint_excursions.
# ──────────────────────────────────────────────────────────────────────


def test_failure_cluster_mean_collisions_and_joint_excursions() -> None:
    """Cluster aggregates report the new safety means when telemetry exists."""
    eps: list[Episode] = []
    # 6 baseline successes pull the failure rate down.
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
                n_collisions=0,
                n_joint_limit_excursions=0,
                energy_over_budget=False,
                n_workspace_excursions=0,
            )
        )
    # 4 failures at (a=1, b=1) with high collisions / joint excursions.
    for k in range(4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                n_collisions=8,
                n_joint_limit_excursions=2,
                energy_over_budget=True,
                n_workspace_excursions=1,
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    assert cluster.mean_collisions == pytest.approx(8.0)
    assert cluster.mean_joint_excursions == pytest.approx(2.0)


def test_failure_cluster_safety_means_none_when_no_telemetry() -> None:
    """All-None safety telemetry leaves the cluster aggregates None."""
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
    assert cluster.mean_collisions is None
    assert cluster.mean_joint_excursions is None
