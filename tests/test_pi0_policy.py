"""Tests for :class:`gauntlet.policy.pi0.Pi0Policy` (backlog B-14).

All tests are gated behind ``@pytest.mark.pi0`` (registered in
pyproject.toml) so the default torch-/lerobot-free pytest run skips
them. Mocks attach at the lerobot factory seams — same pattern as
``tests/lerobot/test_lerobot_policy.py`` — so no real π0 weights are
ever fetched.

The ImportError-when-extra-missing case is covered inline here rather
than in ``tests/test_import_guards.py``: B-14's disjoint scope rule
forbids touching the import-guard module.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.policy.base import Observation, Policy, ResettablePolicy

pytestmark = pytest.mark.pi0


def _zeros_image() -> np.ndarray[Any, Any]:
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _default_obs() -> Observation:
    return {
        "image": _zeros_image(),
        "ee_pos": np.zeros(3, dtype=np.float64),
        "gripper": np.zeros(1, dtype=np.float64),
    }


def _install_pi0_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action_dim: int = 7,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Install mocks at the lerobot π0 + factory seams.

    Returns ``(policy_loader, factory, fake_policy, fake_preprocessor)``.
    """
    import torch

    action_tensor = torch.zeros(action_dim, dtype=torch.float32)

    fake_policy = MagicMock(name="fake_pi0_policy")
    fake_policy.config = MagicMock(name="fake_pi0_config")
    fake_policy.select_action.return_value = action_tensor
    fake_policy.to.return_value = fake_policy

    fake_preprocessor = MagicMock(name="fake_preprocessor")
    fake_preprocessor.return_value = {"batched": "frame"}

    fake_postprocessor = MagicMock(name="fake_postprocessor")
    fake_postprocessor.side_effect = lambda t: t

    policy_loader = MagicMock(return_value=fake_policy)
    factory = MagicMock(return_value=(fake_preprocessor, fake_postprocessor))

    monkeypatch.setattr(
        "lerobot.policies.pi0.modeling_pi0.PI0Policy.from_pretrained",
        policy_loader,
    )
    monkeypatch.setattr(
        "lerobot.policies.factory.make_pre_post_processors",
        factory,
    )
    return policy_loader, factory, fake_policy, fake_preprocessor


# ──────────────────────────────────────────────────────────────────────
# Constructor + Protocol conformance
# ──────────────────────────────────────────────────────────────────────


def test_constructs_with_mocked_lerobot_and_satisfies_policy_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, factory, fake_policy, _ = _install_pi0_mocks(monkeypatch)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy(model_id="lerobot/pi0", device="cpu", action_horizon=8)

    assert isinstance(policy, Policy)
    assert isinstance(policy, ResettablePolicy)
    assert loader.call_args.args[0] == "lerobot/pi0"
    # B-14 spec mirrors the lerobot adapter: do NOT forward trust_remote_code.
    assert "trust_remote_code" not in loader.call_args.kwargs
    # ``.to(device=...)`` reaches the policy with the configured device.
    assert fake_policy.to.call_args.kwargs["device"] == "cpu"
    # Factory gets the policy config + model_id positional args.
    assert factory.call_args.args[0] is fake_policy.config
    assert factory.call_args.args[1] == "lerobot/pi0"


def test_action_horizon_parameter_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pi0_mocks(monkeypatch)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy(action_horizon=16)
    assert policy.action_horizon == 16

    other = Pi0Policy(action_horizon=4)
    assert other.action_horizon == 4


# ──────────────────────────────────────────────────────────────────────
# act() pipeline
# ──────────────────────────────────────────────────────────────────────


def test_act_returns_float64_vector_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pi0_mocks(monkeypatch, action_dim=7)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy()
    action = policy.act(_default_obs())

    assert action.shape == (7,)
    assert action.dtype == np.float64


def test_act_builds_frame_with_three_cameras_state_and_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, fake_pre = _install_pi0_mocks(monkeypatch)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy()
    policy.act(_default_obs())

    frame = fake_pre.call_args.args[0]
    for cam in (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ):
        assert cam in frame
    assert frame["observation.state"].dtype == np.float32
    assert frame["observation.state"].shape == (4,)
    assert isinstance(frame["task"], str)


# ──────────────────────────────────────────────────────────────────────
# reset() flushes the chunk queue (the correctness delta π0 inherits)
# ──────────────────────────────────────────────────────────────────────


def test_reset_flushes_action_chunk_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, fake_policy, _ = _install_pi0_mocks(monkeypatch)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy()
    fake_policy.reset.reset_mock()

    policy.reset(np.random.default_rng(0))

    assert fake_policy.reset.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# Validation errors
# ──────────────────────────────────────────────────────────────────────


def test_act_rejects_non_uint8_image(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pi0_mocks(monkeypatch)
    from gauntlet.policy.pi0 import Pi0Policy

    policy = Pi0Policy()
    obs = {
        "image": np.zeros((224, 224, 3), dtype=np.float32),
        "ee_pos": np.zeros(3, dtype=np.float64),
        "gripper": np.zeros(1, dtype=np.float64),
    }
    with pytest.raises(ValueError, match="uint8"):
        policy.act(obs)


# ──────────────────────────────────────────────────────────────────────
# Import-guard: extra missing → clean ImportError with install hint
# ──────────────────────────────────────────────────────────────────────


def test_import_guard_raises_install_hint_when_lerobot_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing without the [pi0] extra yields a clean ImportError.

    Simulated by setting ``sys.modules['lerobot'] = None`` then
    reloading the adapter module so its ``try: import lerobot.*`` block
    re-runs against the patched cache.
    """
    # Flush every cached lerobot.* submodule so a subsequent ``from
    # lerobot.policies.pi0...`` is forced to re-traverse ``sys.modules``
    # and hit the ``None`` sentinel (Python returns cached child
    # modules even when the parent is None, so flushing children is
    # required for this to actually simulate the missing extra).
    for name in list(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "lerobot", None)

    import gauntlet.policy.pi0 as pi0_mod

    try:
        importlib.reload(pi0_mod)
        with pytest.raises(ImportError, match="uv sync --extra pi0"):
            pi0_mod.Pi0Policy(model_id="lerobot/pi0")
    finally:
        monkeypatch.delitem(sys.modules, "lerobot", raising=False)
        importlib.reload(pi0_mod)
