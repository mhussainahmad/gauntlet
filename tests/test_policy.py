"""Tests for the policy protocol and reference implementations."""

from __future__ import annotations

import numpy as np
import pytest

from gauntlet.policy import (
    DEFAULT_PICK_AND_PLACE_TRAJECTORY,
    Observation,
    Policy,
    RandomPolicy,
    ResettablePolicy,
    ScriptedPolicy,
)

_EMPTY_OBS: Observation = {"state": np.zeros(3, dtype=np.float64)}


class TestRandomPolicy:
    def test_instantiates(self) -> None:
        p = RandomPolicy(action_dim=7)
        assert p.action_dim == 7
        assert p.action_low == -1.0
        assert p.action_high == 1.0

    def test_rejects_nonpositive_dim(self) -> None:
        with pytest.raises(ValueError, match="action_dim must be positive"):
            RandomPolicy(action_dim=0)
        with pytest.raises(ValueError, match="action_dim must be positive"):
            RandomPolicy(action_dim=-3)

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            RandomPolicy(action_dim=3, action_low=1.0, action_high=-1.0)
        with pytest.raises(ValueError, match="strictly less than"):
            RandomPolicy(action_dim=3, action_low=0.5, action_high=0.5)

    def test_action_has_correct_shape_and_dtype(self) -> None:
        p = RandomPolicy(action_dim=5, seed=0)
        action = p.act(_EMPTY_OBS)
        assert action.shape == (5,)
        assert action.dtype == np.float64

    def test_action_within_bounds(self) -> None:
        p = RandomPolicy(action_dim=4, action_low=-0.3, action_high=0.7, seed=1)
        for _ in range(200):
            a = p.act(_EMPTY_OBS)
            assert np.all(a >= -0.3)
            assert np.all(a < 0.7)

    def test_reproducible_with_same_seed(self) -> None:
        p1 = RandomPolicy(action_dim=3, seed=42)
        p2 = RandomPolicy(action_dim=3, seed=42)
        for _ in range(10):
            np.testing.assert_array_equal(p1.act(_EMPTY_OBS), p2.act(_EMPTY_OBS))

    def test_different_seeds_yield_different_actions(self) -> None:
        p1 = RandomPolicy(action_dim=3, seed=1)
        p2 = RandomPolicy(action_dim=3, seed=2)
        assert not np.array_equal(p1.act(_EMPTY_OBS), p2.act(_EMPTY_OBS))

    def test_reset_reseeds_from_provided_generator(self) -> None:
        p = RandomPolicy(action_dim=3, seed=0)
        p.reset(np.random.default_rng(123))
        first = p.act(_EMPTY_OBS)
        p.reset(np.random.default_rng(123))
        second = p.act(_EMPTY_OBS)
        np.testing.assert_array_equal(first, second)

    def test_satisfies_policy_protocols(self) -> None:
        p = RandomPolicy(action_dim=3)
        assert isinstance(p, Policy)
        assert isinstance(p, ResettablePolicy)


class TestScriptedPolicy:
    def test_instantiates_with_default_trajectory(self) -> None:
        p = ScriptedPolicy()
        assert p.length == DEFAULT_PICK_AND_PLACE_TRAJECTORY.shape[0]
        assert p.action_dim == DEFAULT_PICK_AND_PLACE_TRAJECTORY.shape[1]

    def test_does_not_mutate_module_level_default(self) -> None:
        snapshot = DEFAULT_PICK_AND_PLACE_TRAJECTORY.copy()
        p = ScriptedPolicy()
        for _ in range(p.length):
            p.act(_EMPTY_OBS)
        np.testing.assert_array_equal(DEFAULT_PICK_AND_PLACE_TRAJECTORY, snapshot)

    def test_action_shape_matches_trajectory_dim(self) -> None:
        traj = np.zeros((5, 4), dtype=np.float64)
        p = ScriptedPolicy(trajectory=traj)
        assert p.act(_EMPTY_OBS).shape == (4,)

    def test_advances_through_steps_in_order(self) -> None:
        traj = np.arange(12, dtype=np.float64).reshape(4, 3)
        p = ScriptedPolicy(trajectory=traj)
        for i in range(4):
            np.testing.assert_array_equal(p.act(_EMPTY_OBS), traj[i])

    def test_holds_last_action_when_not_looping(self) -> None:
        traj = np.array([[1.0], [2.0]], dtype=np.float64)
        p = ScriptedPolicy(trajectory=traj, loop=False)
        p.act(_EMPTY_OBS)  # -> [1.0]
        p.act(_EMPTY_OBS)  # -> [2.0]
        for _ in range(3):
            np.testing.assert_array_equal(p.act(_EMPTY_OBS), [2.0])

    def test_loops_back_to_start_when_looping(self) -> None:
        traj = np.array([[1.0], [2.0]], dtype=np.float64)
        p = ScriptedPolicy(trajectory=traj, loop=True)
        actions = [float(p.act(_EMPTY_OBS)[0]) for _ in range(5)]
        assert actions == [1.0, 2.0, 1.0, 2.0, 1.0]

    def test_reset_returns_to_step_zero(self) -> None:
        traj = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
        p = ScriptedPolicy(trajectory=traj)
        p.act(_EMPTY_OBS)
        p.act(_EMPTY_OBS)
        p.reset(np.random.default_rng(0))
        np.testing.assert_array_equal(p.act(_EMPTY_OBS), [10.0])

    def test_rejects_empty_trajectory(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            ScriptedPolicy(trajectory=np.empty((0, 3), dtype=np.float64))

    def test_rejects_one_dimensional_trajectory(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            ScriptedPolicy(trajectory=np.zeros(5, dtype=np.float64))

    def test_act_returns_a_copy_not_a_view(self) -> None:
        traj = np.array([[1.0, 2.0]], dtype=np.float64)
        p = ScriptedPolicy(trajectory=traj)
        out = p.act(_EMPTY_OBS)
        out[0] = 99.0
        # Mutating the returned action must not affect the stored trajectory.
        np.testing.assert_array_equal(traj, [[1.0, 2.0]])

    def test_satisfies_policy_protocols(self) -> None:
        p = ScriptedPolicy()
        assert isinstance(p, Policy)
        assert isinstance(p, ResettablePolicy)


def test_policy_protocol_rejects_non_policy() -> None:
    class NotAPolicy:
        pass

    assert not isinstance(NotAPolicy(), Policy)
    assert not isinstance(NotAPolicy(), ResettablePolicy)
