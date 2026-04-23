"""Tests for :class:`gauntlet.policy.lerobot.LeRobotPolicy`.

Every test in this file is marked ``@pytest.mark.lerobot`` — they require
the ``[lerobot]`` extra (lerobot, torch, PIL) and are gated out of the
default pytest run. The default CI job runs
``pytest -m 'not hf and not lerobot'``; the dedicated ``lerobot-tests``
CI job runs ``pytest -m lerobot``.

Import-guard tests (§6 cases 1 & 2) live in ``tests/test_import_guards.py``
so they run in the torch-/lerobot-free default job — that's where the
contract actually needs enforcing.

Mocks attach at the ``from_pretrained`` / ``make_pre_post_processors``
seams — that's the ``modeling_smolvla`` / ``factory`` module where the
adapter's ``import`` statement resolves the name, NOT an instance-level
attribute (RFC §6 case 4).

See docs/phase2-rfc-002-lerobot-smolvla.md §6 for case numbering.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.policy.base import Observation, Policy, ResettablePolicy

pytestmark = pytest.mark.lerobot


# --------------------------------------------------------------------- shared fixtures


def _zeros_image(h: int = 224, w: int = 224) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _zeros_ee_pos() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


def _zeros_gripper() -> np.ndarray:
    return np.zeros(1, dtype=np.float64)


def _default_obs() -> Observation:
    return {
        "image": _zeros_image(),
        "ee_pos": _zeros_ee_pos(),
        "gripper": _zeros_gripper(),
    }


def _install_lerobot_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action_dim: int = 6,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Install mocks at the lerobot factory seams.

    Returns ``(policy_loader, factory, fake_policy, fake_preprocessor)``
    so tests can assert on call args.

    ``fake_policy.select_action`` returns a ``MagicMock`` tensor with the
    configured dim; ``fake_postprocessor`` passes it through unchanged.
    The adapter converts the returned object to numpy via
    ``.detach().to(...).numpy()`` — we stub those hops so no real torch
    tensor math runs.
    """
    import torch

    # Tensor returned by select_action: a real float32 torch tensor — the
    # adapter's numpy path (detach → cpu → numpy) is a real code path, not
    # a mock surface, so let it exercise for realism.
    action_tensor = torch.zeros(action_dim, dtype=torch.float32)

    fake_policy = MagicMock(name="fake_policy")
    fake_policy.config = MagicMock(name="fake_config")
    fake_policy.select_action.return_value = action_tensor
    # ``.to(device=..., dtype=...)`` returns the same mock so the adapter
    # can bind ``self._policy = policy.to(...)`` without surprise.
    fake_policy.to.return_value = fake_policy

    fake_preprocessor = MagicMock(name="fake_preprocessor")
    # Preprocessor is called with the raw frame; its return value is
    # whatever batch select_action will see. We don't assert on the batch
    # contents — we assert on the *call args* to the preprocessor.
    fake_preprocessor.return_value = {"batched": "frame"}

    fake_postprocessor = MagicMock(name="fake_postprocessor")
    # Postprocessor returns the same tensor it got, unmodified.
    fake_postprocessor.side_effect = lambda t: t

    policy_loader = MagicMock(return_value=fake_policy)
    factory = MagicMock(return_value=(fake_preprocessor, fake_postprocessor))

    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained",
        policy_loader,
    )
    monkeypatch.setattr(
        "lerobot.policies.factory.make_pre_post_processors",
        factory,
    )
    return policy_loader, factory, fake_policy, fake_preprocessor


# --------------------------------------------------------------------- case 3 (signature-level)


class TestProtocolConformance:
    """RFC §6 case 3 — duck-typed Policy / ResettablePolicy conformance."""

    def test_satisfies_policy_protocols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        assert isinstance(policy, Policy)
        assert isinstance(policy, ResettablePolicy)


# --------------------------------------------------------------------- case 4


class TestConstructor:
    """RFC §6 case 4 — constructor contract with mocked weights."""

    def test_forwards_repo_id_and_no_trust_remote_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader, factory, fake_policy, _ = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        LeRobotPolicy(
            repo_id="lerobot/fake-smolvla",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )

        # repo_id reaches the loader as the positional first arg.
        assert loader.call_args.args[0] == "lerobot/fake-smolvla"
        # RFC §7: the adapter MUST NOT forward trust_remote_code to lerobot.
        assert "trust_remote_code" not in loader.call_args.kwargs
        # Factory gets the policy config + repo_id.
        assert factory.call_args.args[0] is fake_policy.config
        assert factory.call_args.args[1] == "lerobot/fake-smolvla"

    def test_device_and_dtype_reach_policy_to(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, fake_policy, _ = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        # ``policy.to(device=..., dtype=...)`` with the configured values.
        to_call = fake_policy.to.call_args
        assert to_call.kwargs["device"] == "cpu"
        # dtype is the actual torch dtype object — just check it's float32.
        import torch

        assert to_call.kwargs["dtype"] is torch.float32

    def test_preprocessor_and_postprocessor_overrides_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, factory, _, _ = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        pre = {"resize": (256, 256)}
        post = {"unnormalize": False}
        LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
            preprocessor_overrides=pre,
            postprocessor_overrides=post,
        )
        fkw = factory.call_args.kwargs
        # dict(...) copies are fine — contents must match.
        assert fkw["preprocessor_overrides"] == pre
        assert fkw["postprocessor_overrides"] == post


# --------------------------------------------------------------------- case 5


class TestActShapeDtype:
    """RFC §6 case 5 — act() shape/dtype + preprocessor receives frame."""

    def test_returns_float64_seven_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        # Silence the once-per-instance pad warning — case 10 tests it
        # explicitly.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            action = policy.act(_default_obs())

        assert action.shape == (7,)
        assert action.dtype == np.float64

    def test_preprocessor_receives_frame_with_expected_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, _, fake_pre = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            policy.act(_default_obs())

        frame = fake_pre.call_args.args[0]
        # All three default camera slots populated.
        for cam in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            assert cam in frame
        assert "observation.state" in frame
        assert "task" in frame


# --------------------------------------------------------------------- case 6


class TestImageDuplication:
    """RFC §6 case 6 — one image → all configured camera slots."""

    def test_three_cameras_all_get_same_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, _, fake_pre = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
            camera_keys=("a", "b", "c"),
        )
        obs: Observation = {
            "image": _zeros_image(),
            "ee_pos": _zeros_ee_pos(),
            "gripper": _zeros_gripper(),
        }
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            policy.act(obs)
        frame = fake_pre.call_args.args[0]
        assert frame["a"] is frame["b"] is frame["c"]  # identity-shared.
        assert frame["a"].shape == (224, 224, 3)

    def test_single_camera_key_only_that_key_populated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, _, fake_pre = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
            camera_keys=("solo",),
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            policy.act(_default_obs())
        frame = fake_pre.call_args.args[0]
        assert "solo" in frame
        assert "observation.images.camera1" not in frame


# --------------------------------------------------------------------- case 7


class TestStateConcatenation:
    """RFC §6 case 7 — state_obs_keys are hstacked as float32."""

    def test_default_state_keys_concat_to_four_dim_float32(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, _, fake_pre = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
        )
        obs: Observation = {
            "image": _zeros_image(),
            "ee_pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "gripper": np.array([0.5], dtype=np.float64),
        }
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            policy.act(obs)
        frame = fake_pre.call_args.args[0]
        state = frame["observation.state"]
        assert state.dtype == np.float32
        assert state.shape == (4,)
        assert np.allclose(state, [0.1, 0.2, 0.3, 0.5])


# --------------------------------------------------------------------- case 8


class TestInstructionSlot:
    """RFC §6 case 8 — instruction reaches the frame's ``task`` slot."""

    def test_instruction_becomes_task_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, _, fake_pre = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            policy.act(_default_obs())
        frame = fake_pre.call_args.args[0]
        assert frame["task"] == "grasp the cube"


# --------------------------------------------------------------------- case 9 (the critical one)


class TestResetFlushesActionChunkQueue:
    """RFC §6 case 9 — reset() must call self._policy.reset().

    This is THE correctness delta from HuggingFacePolicy. Without it,
    episode N opens by executing the tail of episode N-1's 50-step
    chunk queue.
    """

    def test_reset_calls_underlying_policy_reset_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, fake_policy, _ = _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
        )
        # Reset the call counter — construction doesn't call reset, but
        # be explicit so a future change that does can't accidentally
        # pass this test.
        fake_policy.reset.reset_mock()

        policy.reset(np.random.default_rng(0))

        assert fake_policy.reset.call_count == 1


# --------------------------------------------------------------------- case 10


class TestDefaultActionRemap:
    """RFC §6 case 10 — default 6→7 pad with once-per-instance warning."""

    def test_pads_six_to_seven_with_zero_gripper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_lerobot_mocks(monkeypatch, action_dim=6)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            action = policy.act(_default_obs())
        assert action.shape == (7,)
        assert action[6] == 0.0

    def test_pad_warning_fires_exactly_once_across_many_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_lerobot_mocks(monkeypatch, action_dim=6)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
        )

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            for _ in range(5):
                policy.act(_default_obs())

        pad_warnings = [w for w in caught if "default action_remap is padding" in str(w.message)]
        assert len(pad_warnings) == 1


# --------------------------------------------------------------------- case 11


class TestUserSuppliedRemap:
    """RFC §6 case 11 — user action_remap bypasses the pad warning."""

    def test_custom_remap_skips_pad_warning_and_sets_gripper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_lerobot_mocks(monkeypatch, action_dim=6)
        from gauntlet.policy.lerobot import LeRobotPolicy

        def _remap(action: np.ndarray) -> np.ndarray:
            return np.concatenate([action, [1.0]])

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
            action_remap=_remap,
        )

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            action = policy.act(_default_obs())

        pad_warnings = [w for w in caught if "default action_remap is padding" in str(w.message)]
        assert pad_warnings == []
        assert action[6] == 1.0


# --------------------------------------------------------------------- case 12


class TestOOBTwistWarning:
    """RFC §6 case 12 — |action[i]| > 1 on twist coords emits RuntimeWarning."""

    def test_warns_when_twist_exceeds_unit_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, fake_policy, _ = _install_lerobot_mocks(monkeypatch, action_dim=7)
        import torch

        fake_policy.select_action.return_value = torch.tensor(
            [2.0, 0, 0, 0, 0, 0, 0.5], dtype=torch.float32
        )
        from gauntlet.policy.lerobot import LeRobotPolicy

        # Use a remap that preserves the 7-D output so no pad-warn fires.
        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
            action_remap=lambda a: a,
        )

        with pytest.warns(RuntimeWarning, match="twist command exceeds"):
            action = policy.act(_default_obs())
        # Pass-through — no silent clip inside the adapter.
        assert action[0] == pytest.approx(2.0)
        assert action[6] == pytest.approx(0.5)


# --------------------------------------------------------------------- case 13


class TestImageValidation:
    """RFC §6 case 13 — image dtype/shape + missing-key errors."""

    def _make_policy(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        return LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
        )

    def test_rejects_float_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        obs = {
            "image": np.zeros((224, 224, 3), dtype=np.float32),
            "ee_pos": _zeros_ee_pos(),
            "gripper": _zeros_gripper(),
        }
        with pytest.raises(ValueError, match="uint8"):
            policy.act(obs)

    def test_rejects_missing_channel_dim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        obs = {
            "image": np.zeros((224, 224), dtype=np.uint8),
            "ee_pos": _zeros_ee_pos(),
            "gripper": _zeros_gripper(),
        }
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            policy.act(obs)

    def test_missing_image_key_raises_keyerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_lerobot_mocks(monkeypatch)
        from gauntlet.policy.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            repo_id="dummy/repo",
            instruction="t",
            device="cpu",
            dtype="float32",
            image_obs_key="camera_front",
        )
        obs = {
            "image": _zeros_image(),  # present but NOT camera_front
            "ee_pos": _zeros_ee_pos(),
            "gripper": _zeros_gripper(),
        }
        with pytest.raises(KeyError, match="camera_front"):
            policy.act(obs)

    def test_missing_state_key_raises_keyerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        obs: Observation = {
            "image": _zeros_image(),
            "ee_pos": _zeros_ee_pos(),
            # gripper missing
        }
        with pytest.raises(KeyError, match="gripper"):
            policy.act(obs)
