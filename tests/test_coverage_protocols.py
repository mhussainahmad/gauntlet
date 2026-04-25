"""Branch-coverage backfill for Protocol-only modules.

Phase 2.5 Task 11. The :class:`gauntlet.policy.base.Policy` /
:class:`ResettablePolicy` and :class:`gauntlet.env.base.GauntletEnv`
Protocols carry method bodies of the form ``...`` (docstring +
ellipsis). Concrete implementations satisfy the Protocol structurally,
but normal call sites dispatch to those concrete bodies — the Protocol
body bytecode itself never runs and stays "uncovered" under
``branch=True`` coverage.

We force the Protocol body bytecode to execute via direct unbound
calls (``Protocol.method(None, ...)``). The bodies are pure ``...``
expressions so they have no observable side-effect — the call is
purely about pulling the bytecode through the coverage tracer.

Same pattern applied to :class:`gauntlet.env.perturbation.base.AxisSampler`
which carries a single ``__call__(rng)`` Protocol body.

This file also contains structural ``isinstance`` checks against
:func:`runtime_checkable`-decorated Protocols to confirm the existing
concrete classes still satisfy the documented surface — a regression
test on the Protocol contract itself.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np

from gauntlet.env.base import CameraSpec, GauntletEnv
from gauntlet.env.perturbation.base import AxisSampler, make_continuous_sampler
from gauntlet.policy.base import Policy, ResettablePolicy

# ----------------------------------------------------------------------------
# Concrete satisfiers used both for ``isinstance`` checks AND as the
# ``self`` argument to direct unbound Protocol-method calls below.
# ----------------------------------------------------------------------------


class _StatelessPolicy:
    def act(self, obs: Any) -> Any:
        return np.zeros(7, dtype=np.float64)


class _ResettablePolicyImpl:
    def __init__(self) -> None:
        self.reset_count = 0

    def act(self, obs: Any) -> Any:
        return np.zeros(7, dtype=np.float64)

    def reset(self, rng: np.random.Generator) -> None:
        self.reset_count += 1


class _MinimalEnv:
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset({"distractor_count"})
    observation_space: Any = None
    action_space: Any = None

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        return ({"x": np.zeros(1)}, {})

    def step(self, action: Any) -> Any:
        return ({"x": np.zeros(1)}, 0.0, True, False, {"success": True})

    def set_perturbation(self, name: str, value: float) -> None:
        return None

    def restore_baseline(self) -> None:
        return None

    def close(self) -> None:
        return None


# ----------------------------------------------------------------------------
# isinstance checks against runtime_checkable Protocols — the structural
# contract regression test.
# ----------------------------------------------------------------------------


def test_concrete_policy_satisfies_policy_protocol() -> None:
    p = _StatelessPolicy()
    assert isinstance(p, Policy)
    # Stateless policies do NOT satisfy ResettablePolicy by design
    # (no ``reset`` method).
    assert not isinstance(p, ResettablePolicy)


def test_resettable_policy_satisfies_both_protocols() -> None:
    p = _ResettablePolicyImpl()
    assert isinstance(p, Policy)
    assert isinstance(p, ResettablePolicy)


def test_minimal_env_satisfies_gauntlet_env_protocol() -> None:
    e = _MinimalEnv()
    assert isinstance(e, GauntletEnv)


def test_camera_spec_named_tuple_layout() -> None:
    """CameraSpec is a NamedTuple — round-trip the documented layout."""
    spec = CameraSpec(name="wrist", pose=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0), size=(64, 64))
    assert spec.name == "wrist"
    assert spec.pose == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert spec.size == (64, 64)


# ----------------------------------------------------------------------------
# Protocol-body bytecode coverage — direct unbound calls.
#
# The Protocol method bodies are ``...`` (LOAD_CONST None / RETURN_VALUE)
# and have no side effects. Invoking the unbound method on a concrete
# instance executes that bytecode while the concrete class's own
# implementation handles all real call sites in the rest of the suite.
# This is purely a coverage-tracing exercise; the assertion checks are
# trivially "did not raise".
# ----------------------------------------------------------------------------


def test_policy_protocol_act_body_runs() -> None:
    """policy.base L50 — Protocol ``Policy.act`` body bytecode executes."""
    p = _StatelessPolicy()
    # Direct unbound dispatch to the Protocol's body. The body is
    # ``...`` so the return is None (we don't assert on it).
    Policy.act(cast(Policy, p), {"x": np.zeros(1)})


def test_resettable_policy_protocol_method_bodies_run() -> None:
    """policy.base L64 + L68 — Protocol ``ResettablePolicy.act`` and
    ``ResettablePolicy.reset`` Protocol bodies execute."""
    p = _ResettablePolicyImpl()
    ResettablePolicy.act(cast(ResettablePolicy, p), {"x": np.zeros(1)})
    ResettablePolicy.reset(cast(ResettablePolicy, p), np.random.default_rng(0))


def test_gauntlet_env_protocol_method_bodies_run() -> None:
    """env.base L141, L153, L164, L174, L178 — every Protocol method
    body on :class:`GauntletEnv` executes."""
    e = _MinimalEnv()
    GauntletEnv.reset(cast(GauntletEnv, e), seed=0, options=None)
    GauntletEnv.step(cast(GauntletEnv, e), np.zeros(7))
    GauntletEnv.set_perturbation(cast(GauntletEnv, e), "distractor_count", 1.0)
    GauntletEnv.restore_baseline(cast(GauntletEnv, e))
    GauntletEnv.close(cast(GauntletEnv, e))


def test_axis_sampler_protocol_call_body_runs() -> None:
    """perturbation.base L63 — Protocol ``AxisSampler.__call__`` body executes."""
    sampler = make_continuous_sampler(0.0, 1.0)
    AxisSampler.__call__(sampler, np.random.default_rng(0))
