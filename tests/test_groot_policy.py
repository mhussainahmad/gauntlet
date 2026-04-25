"""Tests for :class:`gauntlet.policy.groot.GrootN1Policy` (backlog B-15).

All tests are gated behind ``@pytest.mark.groot`` (registered in
pyproject.toml) so the default torch-/lerobot-free pytest run skips
them. Mocks attach at the lerobot groot factory seams — same pattern
as ``tests/test_pi0_policy.py`` — so no real GR00T-N1 weights are
ever fetched.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.policy.base import Observation, Policy, ResettablePolicy

pytestmark = pytest.mark.groot


def _zeros_image() -> np.ndarray[Any, Any]:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _default_obs() -> Observation:
    return {
        "image": _zeros_image(),
        "ee_pos": np.zeros(3, dtype=np.float64),
        "gripper": np.zeros(1, dtype=np.float64),
    }


def _install_groot_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action_dim: int = 7,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Install mocks at the lerobot GR00T + factory seams.

    Returns ``(policy_loader, factory, fake_policy, fake_preprocessor)``.
    """
    import torch

    action_tensor = torch.zeros(action_dim, dtype=torch.float32)

    fake_policy = MagicMock(name="fake_groot_policy")
    fake_policy.config = MagicMock(name="fake_groot_config")
    fake_policy.select_action.return_value = action_tensor
    fake_policy.to.return_value = fake_policy

    fake_preprocessor = MagicMock(name="fake_preprocessor")
    fake_preprocessor.return_value = {"batched": "frame"}

    fake_postprocessor = MagicMock(name="fake_postprocessor")
    fake_postprocessor.side_effect = lambda t: t

    policy_loader = MagicMock(return_value=fake_policy)
    factory = MagicMock(return_value=(fake_preprocessor, fake_postprocessor))

    monkeypatch.setattr(
        "lerobot.policies.groot.modeling_groot.GrootPolicy.from_pretrained",
        policy_loader,
    )
    monkeypatch.setattr(
        "lerobot.policies.groot.processor_groot.make_groot_pre_post_processors",
        factory,
    )
    return policy_loader, factory, fake_policy, fake_preprocessor


# ──────────────────────────────────────────────────────────────────────
# Constructor + Protocol conformance
# ──────────────────────────────────────────────────────────────────────


def test_constructs_with_mocked_lerobot_and_satisfies_policy_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, factory, fake_policy, _ = _install_groot_mocks(monkeypatch)
    from gauntlet.policy.groot import GrootN1Policy

    policy = GrootN1Policy(model_id="nvidia/groot-n1", device="cpu", action_horizon=8)

    assert isinstance(policy, Policy)
    assert isinstance(policy, ResettablePolicy)
    assert loader.call_args.args[0] == "nvidia/groot-n1"
    # B-15 spec mirrors the lerobot/π0 adapters: do NOT forward trust_remote_code.
    assert "trust_remote_code" not in loader.call_args.kwargs
    assert fake_policy.to.call_args.kwargs["device"] == "cpu"
    assert factory.call_args.args[0] is fake_policy.config
    assert factory.call_args.args[1] == "nvidia/groot-n1"


def test_default_model_id_is_nvidia_groot_n1(monkeypatch: pytest.MonkeyPatch) -> None:
    loader, _, _, _ = _install_groot_mocks(monkeypatch)
    from gauntlet.policy.groot import GrootN1Policy

    policy = GrootN1Policy()
    assert policy.model_id == "nvidia/groot-n1"
    assert loader.call_args.args[0] == "nvidia/groot-n1"


def test_action_horizon_parameter_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_groot_mocks(monkeypatch)
    from gauntlet.policy.groot import GrootN1Policy

    policy = GrootN1Policy(action_horizon=16)
    assert policy.action_horizon == 16

    other = GrootN1Policy(action_horizon=4)
    assert other.action_horizon == 4


# ──────────────────────────────────────────────────────────────────────
# act() pipeline
# ──────────────────────────────────────────────────────────────────────


def test_act_returns_float64_vector_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_groot_mocks(monkeypatch, action_dim=7)
    from gauntlet.policy.groot import GrootN1Policy

    policy = GrootN1Policy()
    action = policy.act(_default_obs())

    assert action.shape == (7,)
    assert action.dtype == np.float64


# ──────────────────────────────────────────────────────────────────────
# Import-guard: extra missing → clean ImportError with install hint
# ──────────────────────────────────────────────────────────────────────


def test_import_guard_raises_install_hint_when_lerobot_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing without the [groot] extra yields a clean ImportError.

    Simulated by setting ``sys.modules['lerobot'] = None`` then
    reloading the adapter module so its ``try: import lerobot.*`` block
    re-runs against the patched cache.
    """
    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "lerobot", None)

    import gauntlet.policy.groot as groot_mod

    try:
        importlib.reload(groot_mod)
        with pytest.raises(ImportError, match="uv sync --extra groot"):
            groot_mod.GrootN1Policy(model_id="nvidia/groot-n1")
    finally:
        monkeypatch.delitem(sys.modules, "lerobot", raising=False)
        importlib.reload(groot_mod)
