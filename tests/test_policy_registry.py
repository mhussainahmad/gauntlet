"""Direct unit tests for ``gauntlet.policy.registry.resolve_policy_factory``.

Phase 2.5 Task 11. The CLI tests in ``tests/test_cli.py`` already cover
the happy paths (``"random"``, ``"module:attr"``) and one error path
(unknown spec). This file fills in the under-covered ``_resolve_module_attr``
branches: empty / multi-colon spec, missing attribute, non-callable
attribute, unimportable module, and the empty-string guard at the
top-level resolver.

All tests are pure unit checks — no env, no Runner — and run in well
under 50 ms each.
"""

from __future__ import annotations

import pytest

from gauntlet.policy.random import RandomPolicy
from gauntlet.policy.registry import PolicySpecError, resolve_policy_factory
from gauntlet.policy.scripted import ScriptedPolicy

# Module-level non-callable + zero-arg-callable used by the
# ``module:attr`` error-path tests below. They must live at module scope
# so ``importlib.import_module`` finds them.

_NOT_A_CALLABLE: int = 42


def _ok_policy_factory() -> RandomPolicy:
    return RandomPolicy(action_dim=7)


# ──────────────────────────────────────────────────────────────────────
# Happy paths.
# ──────────────────────────────────────────────────────────────────────


def test_resolve_random_returns_random_policy_factory() -> None:
    factory = resolve_policy_factory("random")
    policy = factory()
    assert isinstance(policy, RandomPolicy)


def test_resolve_scripted_returns_scripted_policy_class() -> None:
    factory = resolve_policy_factory("scripted")
    policy = factory()
    assert isinstance(policy, ScriptedPolicy)


def test_resolve_module_attr_happy_path() -> None:
    factory = resolve_policy_factory("tests.test_policy_registry:_ok_policy_factory")
    assert isinstance(factory(), RandomPolicy)


def test_resolve_strips_surrounding_whitespace() -> None:
    factory = resolve_policy_factory("  random  ")
    assert isinstance(factory(), RandomPolicy)


# ──────────────────────────────────────────────────────────────────────
# Error paths — top-level resolve_policy_factory.
# ──────────────────────────────────────────────────────────────────────


def test_empty_string_spec_rejected() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("")


def test_whitespace_only_spec_rejected() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("   ")


def test_unknown_shortcut_rejected() -> None:
    """A bare word that is neither ``random`` / ``scripted`` nor
    contains ``:`` is rejected with the ``unknown policy spec`` hint."""
    with pytest.raises(PolicySpecError, match="unknown policy spec"):
        resolve_policy_factory("policy-name-without-colon")


# ──────────────────────────────────────────────────────────────────────
# Error paths — _resolve_module_attr.
# ──────────────────────────────────────────────────────────────────────


def test_module_spec_without_colon_rejected() -> None:
    """A spec whose left/right halves cannot be parsed because there
    are zero colons hits the top-level ``unknown`` branch (covered
    above). A spec with TWO colons hits the
    ``_resolve_module_attr`` count check."""
    with pytest.raises(PolicySpecError, match="exactly one ':'"):
        resolve_policy_factory("a:b:c")


def test_module_spec_empty_module_path() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory(":attr")


def test_module_spec_empty_attribute_name() -> None:
    with pytest.raises(PolicySpecError, match="non-empty"):
        resolve_policy_factory("module:")


def test_module_spec_unimportable_module() -> None:
    with pytest.raises(PolicySpecError, match="could not import module"):
        resolve_policy_factory("definitely_not_a_real_module_42:attr")


def test_module_spec_missing_attribute() -> None:
    with pytest.raises(PolicySpecError, match="has no attribute"):
        resolve_policy_factory("tests.test_policy_registry:does_not_exist")


def test_module_spec_attribute_is_not_callable() -> None:
    with pytest.raises(PolicySpecError, match="not callable"):
        resolve_policy_factory("tests.test_policy_registry:_NOT_A_CALLABLE")


# ──────────────────────────────────────────────────────────────────────
# PolicySpecError is a ValueError subclass — catchable both ways.
# ──────────────────────────────────────────────────────────────────────


def test_policy_spec_error_is_value_error() -> None:
    """Existing ``except ValueError`` handlers must keep working."""
    with pytest.raises(ValueError):
        resolve_policy_factory("")
