"""Tests pinning the public surface of :class:`SubtaskMilestone` (B-09).

These tests assert the Protocol's shape — the attribute names, the
method name, and the parameter / return-type signatures — so a future
refactor that silently renames or drops the seam is loud at test
time. They also confirm the only env that currently satisfies the
Protocol (``TabletopStackEnv``) round-trips ``isinstance``.

The runtime-checkable Protocol's ``isinstance`` only verifies attribute
*presence*, not signatures (PEP 544 by design). The signature-stability
assertions therefore go through :mod:`inspect` directly — that's what
gives the test its reverse-compat teeth.
"""

from __future__ import annotations

import inspect
import typing

import numpy as np

from gauntlet.env import SubtaskMilestone, TabletopEnv, TabletopStackEnv
from gauntlet.env.base import Observation


class TestProtocolSurface:
    def test_n_subtasks_is_a_protocol_member(self) -> None:
        # The Protocol declares ``n_subtasks: int`` at class scope; a
        # backend MUST surface the attribute (instance OR class) for
        # the runtime check to pass. ``get_type_hints`` resolves the
        # string-form annotation produced by ``from __future__ import
        # annotations`` against the module's namespace, returning the
        # live ``int`` class object.
        hints = typing.get_type_hints(SubtaskMilestone)
        assert "n_subtasks" in hints
        assert hints["n_subtasks"] is int

    def test_is_subtask_done_signature_is_stable(self) -> None:
        # Pin the public signature so a refactor that swaps argument
        # order or renames a parameter breaks loudly. ``inspect.signature``
        # with ``eval_str=True`` resolves the stringified annotations
        # (``from __future__ import annotations``) against the module's
        # namespace so the comparison runs against live class objects,
        # not strings.
        sig = inspect.signature(SubtaskMilestone.is_subtask_done, eval_str=True)
        params = list(sig.parameters.values())
        # ``self`` + ``idx`` + ``obs``.
        assert [p.name for p in params] == ["self", "idx", "obs"]
        assert params[1].annotation is int
        assert params[2].annotation is Observation
        assert sig.return_annotation is bool


class TestTabletopStackSatisfiesProtocol:
    def test_isinstance_passes(self) -> None:
        env = TabletopStackEnv()
        try:
            assert isinstance(env, SubtaskMilestone)
        finally:
            env.close()

    def test_n_subtasks_value(self) -> None:
        env = TabletopStackEnv()
        try:
            # Both the class and the instance must agree (ClassVar
            # contract in TabletopStackEnv).
            assert TabletopStackEnv.n_subtasks == 3
            assert env.n_subtasks == 3
        finally:
            env.close()

    def test_is_subtask_done_returns_false_at_reset(self) -> None:
        env = TabletopStackEnv()
        try:
            obs, _ = env.reset(seed=7)
            for idx in range(env.n_subtasks):
                assert env.is_subtask_done(idx, obs) is False
        finally:
            env.close()

    def test_is_subtask_done_rejects_out_of_range_index(self) -> None:
        env = TabletopStackEnv()
        try:
            obs, _ = env.reset(seed=0)
            import pytest

            with pytest.raises(IndexError):
                env.is_subtask_done(-1, obs)
            with pytest.raises(IndexError):
                env.is_subtask_done(env.n_subtasks, obs)
        finally:
            env.close()


class TestNonMilestoneEnvDoesNotSatisfyProtocol:
    """The single-step Tabletop env must NOT advertise the Protocol.

    The Protocol's whole point is to mark the long-horizon envs as a
    distinct opt-in capability. If TabletopEnv accidentally grew an
    ``n_subtasks`` attribute the runtime check would silently flip
    True; this test guards that drift.
    """

    def test_tabletop_does_not_satisfy_subtask_milestone(self) -> None:
        env = TabletopEnv()
        try:
            assert not isinstance(env, SubtaskMilestone)
        finally:
            env.close()


class TestProtocolPredicatesMatchInfo:
    """``is_subtask_done`` (un-latched) must agree with the live info
    payload (latched) at the moment a subtask FIRST flips."""

    def test_first_flip_agrees_with_obs_predicate(self) -> None:
        # Drive the env with zero actions — no subtask should fire,
        # so both sources agree on ``False`` for every step. This is
        # the cheapest cross-check between the two readout paths.
        env = TabletopStackEnv()
        try:
            obs, info = env.reset(seed=11)
            for _ in range(10):
                obs, _, _, _, info = env.step(np.zeros(7, dtype=np.float64))
                latched: list[bool] = info["subtask_completion"]
                for idx in range(env.n_subtasks):
                    # Subtask 1 latches under "C grasped" (un-latched
                    # predicate is "grasped right now"); both are
                    # False under no-op so they agree trivially.
                    assert latched[idx] is False
                    assert env.is_subtask_done(idx, obs) is False
        finally:
            env.close()
