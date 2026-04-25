"""Tests for :class:`gauntlet.policy.dt.DecisionTransformerPolicy` (B-16).

All tests are gated behind ``@pytest.mark.dt`` (registered in
pyproject.toml) so the default torch-/transformers-free pytest run
skips them. Mocks attach at the
``transformers.DecisionTransformerModel.from_pretrained`` seam — same
shape as ``tests/test_rdt_policy.py`` — so no real DT weights are
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

pytestmark = pytest.mark.dt


_STATE_DIM = 11
_ACT_DIM = 3


def _default_obs() -> Observation:
    return {"state": np.zeros(_STATE_DIM, dtype=np.float64)}


def _install_dt_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dim: int = _STATE_DIM,
    act_dim: int = _ACT_DIM,
) -> tuple[MagicMock, MagicMock]:
    """Install mocks at the ``DecisionTransformerModel.from_pretrained`` seam.

    Returns ``(model_loader, fake_model)``.
    """
    import torch

    fake_model = MagicMock(name="fake_dt_model")
    fake_config = MagicMock(name="fake_dt_config")
    fake_config.state_dim = state_dim
    fake_config.act_dim = act_dim
    fake_model.config = fake_config
    fake_model.to.return_value = fake_model

    def _call(**kwargs: Any) -> Any:
        # DT returns an object with ``.action_preds`` shape (B, T, act_dim);
        # mirror the seq length of the supplied state tensor.
        states_t = kwargs["states"]
        seq_len = int(states_t.shape[1])
        out = MagicMock(name="fake_dt_output")
        out.action_preds = torch.zeros(1, seq_len, act_dim, dtype=torch.float32)
        return out

    fake_model.side_effect = _call

    model_loader = MagicMock(return_value=fake_model)
    monkeypatch.setattr(
        "transformers.DecisionTransformerModel.from_pretrained",
        model_loader,
    )
    return model_loader, fake_model


# ──────────────────────────────────────────────────────────────────────
# Constructor + Protocol conformance
# ──────────────────────────────────────────────────────────────────────


def test_constructs_with_mocked_transformers_and_satisfies_policy_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, fake_model = _install_dt_mocks(monkeypatch)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(
        model_id="edbeeching/decision-transformer-gym-hopper-medium",
        device="cpu",
        target_return=200.0,
        context_length=10,
    )

    assert isinstance(policy, Policy)
    assert isinstance(policy, ResettablePolicy)
    assert loader.call_args.args[0] == "edbeeching/decision-transformer-gym-hopper-medium"
    # DT ships in the transformers tree — trust_remote_code MUST NOT be forwarded.
    assert "trust_remote_code" not in loader.call_args.kwargs
    assert fake_model.to.call_args.args[0] == "cpu"
    assert policy.state_dim == _STATE_DIM
    assert policy.act_dim == _ACT_DIM


def test_target_return_parameter_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dt_mocks(monkeypatch)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(model_id="x", target_return=42.0)
    assert policy.target_return == 42.0

    other = DecisionTransformerPolicy(model_id="x", target_return=999.5)
    assert other.target_return == 999.5


# ──────────────────────────────────────────────────────────────────────
# act() pipeline
# ──────────────────────────────────────────────────────────────────────


def test_act_returns_float64_vector_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dt_mocks(monkeypatch)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(model_id="x")
    action = policy.act(_default_obs())

    assert action.shape == (_ACT_DIM,)
    assert action.dtype == np.float64


def test_context_buffer_rolls_after_filling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffer length is capped at ``context_length`` once exceeded."""
    _install_dt_mocks(monkeypatch)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(model_id="x", context_length=3)
    for _ in range(5):
        policy.act(_default_obs())

    # All four buffers stay aligned at exactly context_length.
    assert len(policy._states) == 3
    assert len(policy._actions) == 3
    assert len(policy._returns_to_go) == 3
    assert len(policy._timesteps) == 3
    # Timesteps record the absolute step index — the rolling window
    # should hold the most recent three (steps 2, 3, 4).
    assert policy._timesteps == [2, 3, 4]


def test_reset_flushes_context_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dt_mocks(monkeypatch)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(model_id="x")
    for _ in range(4):
        policy.act(_default_obs())
    assert len(policy._states) == 4

    policy.reset(np.random.default_rng(0))

    assert policy._states == []
    assert policy._actions == []
    assert policy._returns_to_go == []
    assert policy._timesteps == []


# ──────────────────────────────────────────────────────────────────────
# Validation errors
# ──────────────────────────────────────────────────────────────────────


def test_act_rejects_state_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dt_mocks(monkeypatch, state_dim=8)
    from gauntlet.policy.dt import DecisionTransformerPolicy

    policy = DecisionTransformerPolicy(model_id="x")
    bad_obs: Observation = {"state": np.zeros(5, dtype=np.float64)}
    with pytest.raises(ValueError, match="state dim mismatch"):
        policy.act(bad_obs)


# ──────────────────────────────────────────────────────────────────────
# Import-guard: extra missing → clean ImportError with install hint
# ──────────────────────────────────────────────────────────────────────


def test_import_guard_raises_install_hint_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing without the [dt] extra yields a clean ImportError.

    Simulated by setting ``sys.modules['transformers'] = None`` then
    reloading the adapter module so its ``try: import transformers``
    block re-runs against the patched cache.
    """
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)

    import gauntlet.policy.dt as dt_mod

    try:
        importlib.reload(dt_mod)
        with pytest.raises(ImportError, match="uv sync --extra dt"):
            dt_mod.DecisionTransformerPolicy(model_id="x")
    finally:
        monkeypatch.delitem(sys.modules, "transformers", raising=False)
        importlib.reload(dt_mod)
