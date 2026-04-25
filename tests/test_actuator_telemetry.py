"""B-21 actuator-energy / torque telemetry tests.

Covers the env→worker→Episode→report path:

1. Canonical :class:`TabletopEnv` is mocap-driven (``model.nu == 0``)
   so it intentionally leaves the three actuator-cost fields ``None``
   — emitting ``0.0`` would read as "policy did nothing" next to a
   high-failure cluster row, which is actively misleading. The B-21
   anti-feature note explicitly accepts this trade-off.
2. The worker accumulator math (sum / mean / peak) is verified
   against a synthetic env that injects deterministic per-step
   ``info["actuator_energy_delta"]`` / ``info["actuator_torque_norm"]``
   values; this tests "the wiring computes the right number" without
   depending on the env's MJCF actuator inventory.
3. A backend that publishes no telemetry leaves all three fields
   ``None`` — the cross-backend gate.
4. The failure-cluster aggregator carries per-cluster mean energy
   and mean peak-torque only when at least one cluster episode
   reported telemetry.
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
    seed: int = 11,
) -> WorkItem:
    """One :class:`WorkItem` with a deterministic SeedSequence node."""
    master = np.random.SeedSequence(seed)
    cell_node = master.spawn(1)[0]
    episode_node = cell_node.spawn(1)[0]
    return WorkItem(
        suite_name="actuator-telemetry",
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
# 1. Canonical TabletopEnv is mocap-driven → telemetry stays None.
# ──────────────────────────────────────────────────────────────────────


def test_tabletop_mocap_env_yields_none_telemetry() -> None:
    """Mocap-driven Tabletop has ``model.nu == 0`` → all three fields None.

    Documented anti-feature: emitting ``0.0`` here would be actively
    misleading next to a high-failure cluster row. A future joint-
    torque MJCF asset will surface real values without further wiring
    — see ``Episode.actuator_energy`` docstring.
    """
    env = TabletopEnv(max_steps=10)
    try:
        # Sanity: confirm the mocap-only assumption holds for this asset.
        assert env._model.nu == 0
        episode = execute_one(env, _make_scripted, _make_work_item())
    finally:
        env.close()
    assert episode.actuator_energy is None
    assert episode.mean_torque_norm is None
    assert episode.peak_torque_norm is None


def test_tabletop_step_info_omits_telemetry_keys_when_no_actuators() -> None:
    """Mocap-only Tabletop omits the per-step telemetry keys from info.

    Worker reads with ``info.get(...)``; key absence is the contract
    that signals "no telemetry, leave Episode field None".
    """
    env = TabletopEnv(max_steps=4)
    try:
        _, info_reset = env.reset(seed=3)
        assert "actuator_energy_delta" not in info_reset
        assert "actuator_torque_norm" not in info_reset
        action = np.zeros(7, dtype=np.float64)
        _, _, _, _, info_step = env.step(action)
        # nu==0 → step also omits the keys (not just reset).
        assert "actuator_energy_delta" not in info_step
        assert "actuator_torque_norm" not in info_step
    finally:
        env.close()


# ──────────────────────────────────────────────────────────────────────
# 2. Worker accumulator math, verified against a synthetic env that
#    injects deterministic per-step telemetry into ``info``.
# ──────────────────────────────────────────────────────────────────────


class _SyntheticTelemetryEnv:
    """Synthetic env that publishes scripted per-step telemetry values.

    Injects ``info["actuator_energy_delta"]`` and
    ``info["actuator_torque_norm"]`` from a fixed schedule so the
    worker's accumulator (sum / mean / peak) can be verified against a
    closed-form expected value. Stops after the schedule is exhausted
    (``truncated=True``) so the rollout terminates deterministically.
    """

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        energy_schedule: list[float],
        torque_schedule: list[float],
    ) -> None:
        if len(energy_schedule) != len(torque_schedule):
            raise ValueError("schedules must match length")
        self._energy_schedule = list(energy_schedule)
        self._torque_schedule = list(torque_schedule)
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
        energy = self._energy_schedule[self._idx]
        torque = self._torque_schedule[self._idx]
        self._idx += 1
        truncated = self._idx >= len(self._energy_schedule)
        info = {
            "success": False,
            "actuator_energy_delta": energy,
            "actuator_torque_norm": torque,
        }
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
    """Trivial policy — always returns the zero action."""

    def act(self, obs: dict[str, NDArray[Any]]) -> NDArray[Any]:
        del obs
        return np.zeros(7, dtype=np.float64)


def _make_zero_policy() -> _ZeroPolicy:
    return _ZeroPolicy()


def test_worker_accumulates_synthetic_telemetry_correctly() -> None:
    """Worker sums energy, averages torque, tracks peak from info dict."""
    energy = [0.5, 1.0, 1.5, 2.0]
    torques = [1.0, 3.0, 2.0, 4.0]
    env = _SyntheticTelemetryEnv(
        energy_schedule=energy,
        torque_schedule=torques,
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=7))
    assert episode.step_count == 4
    assert episode.actuator_energy == pytest.approx(sum(energy))
    assert episode.mean_torque_norm == pytest.approx(sum(torques) / len(torques))
    assert episode.peak_torque_norm == pytest.approx(max(torques))
    # Peak >= mean by construction.
    assert episode.peak_torque_norm >= episode.mean_torque_norm


def test_worker_zero_torque_synthetic_yields_zero_energy() -> None:
    """All-zero schedule still emits 0.0 (NOT None) — telemetry observed."""
    env = _SyntheticTelemetryEnv(
        energy_schedule=[0.0] * 3,
        torque_schedule=[0.0] * 3,
    )
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=13))
    # Distinct from the "no telemetry keys" path: here the env DID
    # publish values, they just happened to be zero. Episode must
    # carry float zeros, not None.
    assert episode.actuator_energy == 0.0
    assert episode.mean_torque_norm == 0.0
    assert episode.peak_torque_norm == 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. Non-MuJoCo backend → fields stay None (anti-feature contract).
# ──────────────────────────────────────────────────────────────────────


class _FakeEnvNoTelemetry:
    """Env stub that publishes no actuator info at all.

    Pins the cross-backend anti-feature: PyBullet, Genesis, Isaac, and
    any third-party env that does not surface
    ``info["actuator_energy_delta"]`` / ``info["actuator_torque_norm"]``
    must yield an Episode with all three actuator-cost fields ``None``
    — distinct from ``0.0``.
    """

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
        # info deliberately omits actuator telemetry keys.
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


def test_non_mujoco_backend_yields_none_telemetry() -> None:
    """A backend with no telemetry keys leaves all three fields None."""
    env = _FakeEnvNoTelemetry(max_steps=5)
    episode = execute_one(env, _make_zero_policy, _make_work_item(seed=53))
    assert episode.actuator_energy is None
    assert episode.mean_torque_norm is None
    assert episode.peak_torque_norm is None
    assert episode.step_count == 5


# ──────────────────────────────────────────────────────────────────────
# 4. Failure-cluster aggregation surfaces the new fields.
# ──────────────────────────────────────────────────────────────────────


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    perturbation_config: dict[str, float],
    actuator_energy: float | None,
    peak_torque_norm: float | None,
) -> Episode:
    return Episode(
        suite_name="cluster-agg",
        cell_index=cell_index,
        episode_index=episode_index,
        seed=cell_index * 100 + episode_index,
        perturbation_config=perturbation_config,
        success=success,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
        actuator_energy=actuator_energy,
        mean_torque_norm=(peak_torque_norm * 0.5 if peak_torque_norm is not None else None),
        peak_torque_norm=peak_torque_norm,
    )


def test_failure_cluster_carries_mean_energy_and_peak_torque() -> None:
    """Cluster aggregates report per-cluster mean energy / peak torque.

    Episodes whose backend left telemetry None are dropped from both
    numerator and denominator so a partial-coverage cluster doesn't
    bias toward zero.
    """
    eps: list[Episode] = []
    # Baseline: 6 successes at (a=0, b=0) — pulls the overall failure
    # rate down so the (a=1, b=1) failure cluster fires above the 2x
    # baseline threshold.
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
                actuator_energy=0.5,
                peak_torque_norm=1.0,
            )
        )
    # Failure cluster: 4 failures at (a=1, b=1) with high energy.
    for k in range(4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                actuator_energy=10.0,
                peak_torque_norm=5.0,
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    assert len(report.failure_clusters) >= 1
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    assert cluster.mean_actuator_energy == pytest.approx(10.0)
    assert cluster.mean_peak_torque_norm == pytest.approx(5.0)


def test_failure_cluster_none_when_no_telemetry() -> None:
    """A cluster whose episodes carry no telemetry surfaces None aggregates."""
    eps: list[Episode] = []
    for k in range(6):
        eps.append(
            _ep(
                cell_index=0,
                episode_index=k,
                success=True,
                perturbation_config={"a": 0.0, "b": 0.0},
                actuator_energy=None,
                peak_torque_norm=None,
            )
        )
    for k in range(4):
        eps.append(
            _ep(
                cell_index=1,
                episode_index=k,
                success=False,
                perturbation_config={"a": 1.0, "b": 1.0},
                actuator_energy=None,
                peak_torque_norm=None,
            )
        )
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    assert len(report.failure_clusters) >= 1
    cluster = next(c for c in report.failure_clusters if c.axes == {"a": 1.0, "b": 1.0})
    assert cluster.mean_actuator_energy is None
    assert cluster.mean_peak_torque_norm is None
