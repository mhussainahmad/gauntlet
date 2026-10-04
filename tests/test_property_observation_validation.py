"""Property tests for non-finite observation / action handling.

A NaN that leaks out of the solver (or through a buggy env wrapper),
or a NaN action from a diverged checkpoint, must not flow silently into
the next ``policy.act`` / ``env.step`` and corrupt the rollout's
outcome. Two surfaces cover it:

1. :func:`gauntlet.runner.worker.validate_observation` raises
   :class:`ValueError` on any NaN / +-Inf in a floating-point obs entry.
2. :func:`gauntlet.runner.worker.execute_one` ends the rollout on the
   first non-finite observation or action, records it as a failure and
   sets :attr:`Episode.observation_invalid` / :attr:`Episode.action_invalid`
   instead of raising, so one bad episode never crashes a sweep.

Hypothesis usage: the per-test ``@settings(max_examples=...)`` knob
overrides the conftest profile; 50 examples is plenty to cover the
``(nan, +inf, -inf)`` cross-product with random positions.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.policy import Observation
from gauntlet.runner.determinism import episode_hash
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import WorkItem, execute_one, validate_observation

pytestmark = pytest.mark.hypothesis_property

_NON_FINITE = [np.nan, np.inf, -np.inf]


def _obs(rng: np.random.Generator) -> dict[str, NDArray[Any]]:
    return {
        "cube_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "ee_pos": rng.uniform(-1, 1, size=3).astype(np.float64),
        "gripper": np.array([0.0], dtype=np.float64),
        "image": rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8),
    }


# ----- validate_observation --------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    non_finite=st.sampled_from(_NON_FINITE),
    key=st.sampled_from(["cube_pos", "ee_pos", "gripper"]),
)
@settings(max_examples=50)
def test_validate_observation_rejects_any_non_finite(
    seed: int, non_finite: float, key: str
) -> None:
    rng = np.random.default_rng(seed)
    obs = _obs(rng)
    obs[key][int(rng.integers(0, obs[key].size))] = non_finite
    with pytest.raises(ValueError, match=rf"{key}.*non-finite"):
        validate_observation(obs)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_validate_observation_accepts_finite(seed: int) -> None:
    """No false positives on finite state, uint8 images or string entries."""
    obs: dict[str, Any] = dict(_obs(np.random.default_rng(seed)))
    obs["instruction"] = "put the red cube on the target"
    validate_observation(obs)


def test_episode_flags_default_false_for_old_records() -> None:
    """Episodes written before the flags existed still load, unflagged."""
    ep = Episode(
        suite_name="s",
        cell_index=0,
        episode_index=0,
        seed=1,
        perturbation_config={},
        success=True,
        terminated=True,
        truncated=False,
        step_count=3,
        total_reward=1.0,
    )
    assert ep.observation_invalid is False
    assert ep.action_invalid is False


# ----- execute_one ------------------------------------------------------------


class _FakeEnv:
    """Minimal env that can emit a NaN observation at a chosen step."""

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset()
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, nan_at_step: int | None = None, max_steps: int = 10) -> None:
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {"x": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)}
        )
        self.action_space: gym.spaces.Box = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float64)
        self._nan_at_step = nan_at_step
        self._max_steps = max_steps
        self._t = 0
        self.steps_taken = 0

    def _ob(self) -> dict[str, NDArray[np.float64]]:
        x = np.zeros(2, dtype=np.float64)
        if self._nan_at_step is not None and self._t >= self._nan_at_step:
            x[0] = np.nan
        return {"x": x}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        del seed, options
        self._t = 0
        return self._ob(), {}

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        del action
        self._t += 1
        self.steps_taken += 1
        # Reports success on every step, so a flagged episode proves the
        # flag overrides the env's own success signal.
        return self._ob(), 1.0, False, self._t >= self._max_steps, {"success": True}

    def set_perturbation(self, name: str, value: float) -> None:
        raise ValueError(name)

    def restore_baseline(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def control_dt(self) -> float:
        return 0.05


class _ConstPolicy:
    def __init__(self, value: float = 0.0, *, nan_after: int | None = None) -> None:
        self._value = value
        self._nan_after = nan_after
        self._calls = 0

    def act(self, obs: Observation) -> NDArray[np.float64]:
        self._calls += 1
        a = np.full(7, self._value, dtype=np.float64)
        if self._nan_after is not None and self._calls > self._nan_after:
            a[2] = np.nan
        return a


def _item() -> WorkItem:
    node = np.random.SeedSequence(5).spawn(1)[0].spawn(1)[0]
    return WorkItem(
        suite_name="non-finite",
        cell_index=0,
        episode_index=0,
        perturbation_values={},
        episode_seq=node,
        master_seed=5,
        n_cells=1,
        episodes_per_cell=1,
    )


def test_clean_rollout_is_not_flagged() -> None:
    env = _FakeEnv()
    ep = execute_one(cast(GauntletEnv, env), _ConstPolicy, _item())
    assert ep.success is True
    assert ep.step_count == 10
    assert not ep.observation_invalid and not ep.action_invalid


@pytest.mark.parametrize("nan_at_step", [0, 1, 4])
def test_non_finite_observation_ends_rollout_as_flagged_failure(nan_at_step: int) -> None:
    env = _FakeEnv(nan_at_step=nan_at_step)
    ep = execute_one(cast(GauntletEnv, env), _ConstPolicy, _item())
    assert ep.observation_invalid is True
    assert ep.action_invalid is False
    assert ep.success is False
    # The policy never sees the bad observation: no step after it.
    assert env.steps_taken == nan_at_step
    assert ep.step_count == nan_at_step


@pytest.mark.parametrize("nan_after", [0, 3])
def test_non_finite_action_ends_rollout_before_env_step(nan_after: int) -> None:
    env = _FakeEnv()
    ep = execute_one(cast(GauntletEnv, env), lambda: _ConstPolicy(nan_after=nan_after), _item())
    assert ep.action_invalid is True
    assert ep.observation_invalid is False
    assert ep.success is False
    # The bad action never reaches the env.
    assert env.steps_taken == nan_after


def test_flags_do_not_change_episode_hash_of_clean_rollouts() -> None:
    """The flags are outside the hashed field set, so hashes of existing
    clean episodes stay comparable across versions."""
    ep = execute_one(cast(GauntletEnv, _FakeEnv()), _ConstPolicy, _item())
    flipped = ep.model_copy(update={"observation_invalid": True, "action_invalid": True})
    assert episode_hash(ep) == episode_hash(flipped)
