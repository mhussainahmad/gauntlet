"""Tests for :class:`gauntlet.policy.rdt.RdtPolicy` (backlog B-15).

All tests are gated behind ``@pytest.mark.rdt`` (registered in
pyproject.toml) so the default torch-/transformers-free pytest run
skips them. Mocks attach at the
``transformers.AutoModel.from_pretrained`` seam — same shape as
``tests/test_pi0_policy.py`` — so no real RDT weights are ever
fetched.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.policy.base import Observation, Policy, ResettablePolicy

pytestmark = pytest.mark.rdt


def _zeros_image() -> np.ndarray[Any, Any]:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _default_obs() -> Observation:
    return {"image": _zeros_image()}


def _install_rdt_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action_dim: int = 7,
) -> tuple[MagicMock, MagicMock]:
    """Install mocks at the ``transformers.AutoModel.from_pretrained`` seam.

    Returns ``(model_loader, fake_model)``.
    """
    import torch

    action_tensor = torch.zeros(action_dim, dtype=torch.float32)

    fake_model = MagicMock(name="fake_rdt_model")
    fake_model.predict_action.return_value = action_tensor
    fake_model.to.return_value = fake_model

    model_loader = MagicMock(return_value=fake_model)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", model_loader)
    return model_loader, fake_model


# ──────────────────────────────────────────────────────────────────────
# Constructor + Protocol conformance
# ──────────────────────────────────────────────────────────────────────


def test_constructs_with_mocked_transformers_and_satisfies_policy_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, fake_model = _install_rdt_mocks(monkeypatch)
    from gauntlet.policy.rdt import RdtPolicy

    policy = RdtPolicy(
        model_id="robotics-diffusion-transformer/rdt-1b",
        device="cpu",
        action_horizon=64,
    )

    assert isinstance(policy, Policy)
    assert isinstance(policy, ResettablePolicy)
    assert loader.call_args.args[0] == "robotics-diffusion-transformer/rdt-1b"
    # RDT lives outside the transformers tree — trust_remote_code is required.
    assert loader.call_args.kwargs["trust_remote_code"] is True
    assert fake_model.to.call_args.args[0] == "cpu"


def test_default_model_id_is_rdt_1b(monkeypatch: pytest.MonkeyPatch) -> None:
    loader, _ = _install_rdt_mocks(monkeypatch)
    from gauntlet.policy.rdt import RdtPolicy

    policy = RdtPolicy()
    assert policy.model_id == "robotics-diffusion-transformer/rdt-1b"
    assert loader.call_args.args[0] == "robotics-diffusion-transformer/rdt-1b"


def test_action_horizon_parameter_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_rdt_mocks(monkeypatch)
    from gauntlet.policy.rdt import RdtPolicy

    policy = RdtPolicy(action_horizon=128)
    assert policy.action_horizon == 128

    other = RdtPolicy(action_horizon=32)
    assert other.action_horizon == 32


# ──────────────────────────────────────────────────────────────────────
# act() pipeline
# ──────────────────────────────────────────────────────────────────────


def test_act_returns_float64_vector_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_rdt_mocks(monkeypatch, action_dim=7)
    from gauntlet.policy.rdt import RdtPolicy

    policy = RdtPolicy()
    action = policy.act(_default_obs())

    assert action.shape == (7,)
    assert action.dtype == np.float64


# ──────────────────────────────────────────────────────────────────────
# Import-guard: extra missing → clean ImportError with install hint
# ──────────────────────────────────────────────────────────────────────


def test_import_guard_raises_install_hint_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing without the [rdt] extra yields a clean ImportError.

    Simulated by setting ``sys.modules['transformers'] = None`` then
    reloading the adapter module so its ``try: import transformers``
    block re-runs against the patched cache.
    """
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)

    import gauntlet.policy.rdt as rdt_mod

    try:
        importlib.reload(rdt_mod)
        with pytest.raises(ImportError, match="uv sync --extra rdt"):
            rdt_mod.RdtPolicy(model_id="robotics-diffusion-transformer/rdt-1b")
    finally:
        monkeypatch.delitem(sys.modules, "transformers", raising=False)
        importlib.reload(rdt_mod)
