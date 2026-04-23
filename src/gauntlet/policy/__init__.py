"""Policy adapters — see ``base.py`` for the core Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gauntlet.policy.base import Action as Action
from gauntlet.policy.base import Observation as Observation
from gauntlet.policy.base import Policy as Policy
from gauntlet.policy.base import ResettablePolicy as ResettablePolicy
from gauntlet.policy.random import RandomPolicy as RandomPolicy
from gauntlet.policy.registry import PolicySpecError as PolicySpecError
from gauntlet.policy.registry import resolve_policy_factory as resolve_policy_factory
from gauntlet.policy.scripted import (
    DEFAULT_PICK_AND_PLACE_TRAJECTORY as DEFAULT_PICK_AND_PLACE_TRAJECTORY,
)
from gauntlet.policy.scripted import ScriptedPolicy as ScriptedPolicy

if TYPE_CHECKING:  # pragma: no cover — re-export is dynamic, see __getattr__ below.
    from gauntlet.policy.huggingface import HuggingFacePolicy as HuggingFacePolicy

__all__ = [
    "DEFAULT_PICK_AND_PLACE_TRAJECTORY",
    "Action",
    "HuggingFacePolicy",
    "Observation",
    "Policy",
    "PolicySpecError",
    "RandomPolicy",
    "ResettablePolicy",
    "ScriptedPolicy",
    "resolve_policy_factory",
]


def __getattr__(name: str) -> Any:
    """Lazily expose ``HuggingFacePolicy`` without importing torch on package load.

    ``from gauntlet.policy import RandomPolicy`` must keep working on a
    torch-free install (spec §6). Accessing ``HuggingFacePolicy`` — whether
    via ``from gauntlet.policy import HuggingFacePolicy`` or
    ``gauntlet.policy.HuggingFacePolicy`` — triggers the import-time check
    here: if torch is unavailable we raise ``ImportError`` with the
    install hint *at attribute-access time*, per
    docs/phase2-rfc-001-huggingface-policy.md §3.
    """
    if name == "HuggingFacePolicy":
        from gauntlet.policy.huggingface import _HF_INSTALL_HINT

        try:
            import torch  # noqa: F401 — presence check for the [hf] extra.
        except ImportError as exc:
            raise ImportError(_HF_INSTALL_HINT) from exc
        from gauntlet.policy.huggingface import HuggingFacePolicy

        return HuggingFacePolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
