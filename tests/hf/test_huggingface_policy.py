"""Tests for :class:`gauntlet.policy.huggingface.HuggingFacePolicy`.

Every test in this file is marked ``@pytest.mark.hf`` — they require the
``[hf]`` extra (torch, transformers, PIL) and are gated out of the
default pytest run. The default CI job runs ``pytest -m 'not hf'``; the
dedicated ``hf-tests`` CI job runs ``pytest -m hf``.

See docs/phase2-rfc-001-huggingface-policy.md §6 for case numbering.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gauntlet.policy.base import Observation, Policy, ResettablePolicy

pytestmark = pytest.mark.hf


def _zeros_image(h: int = 224, w: int = 224) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _install_hf_mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install mocks at the HF ``from_pretrained`` seam.

    RFC §6 case 3 — we mock the class method rather than instance
    attributes, so callers that hit ``trust_remote_code=True`` get a
    MagicMock back without the real loader code running.
    """
    # Mock processor: calling ``processor(prompt, image, return_tensors="pt")``
    # returns an object with ``.to(device, dtype=...)`` that yields a dict.
    proc_call_return = MagicMock(name="processor_call_return")
    pixel_values = MagicMock(name="pixel_values")
    input_ids = MagicMock(name="input_ids")
    proc_call_return.to.return_value = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }
    processor_instance = MagicMock(name="processor_instance")
    processor_instance.return_value = proc_call_return

    # Mock model: ``.to(device)`` returns a model whose ``predict_action``
    # returns a (7,) float32 numpy array.
    model_instance = MagicMock(name="model_instance")
    model_instance.predict_action.return_value = np.zeros(7, dtype=np.float32)
    model_factory = MagicMock(name="raw_model")
    model_factory.to.return_value = model_instance

    processor_loader = MagicMock(return_value=processor_instance)
    model_loader = MagicMock(return_value=model_factory)

    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        processor_loader,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForVision2Seq.from_pretrained",
        model_loader,
    )
    return processor_loader, model_loader


# --------------------------------------------------------------------- case 1 & 2


class TestImportGuards:
    """Cases 1 & 2 — these tests also run in the default job because the
    import-guard contract must hold whether or not ``[hf]`` is installed.
    """

    def test_import_guard_raises_install_hint_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force ``import torch`` inside ``HuggingFacePolicy.__init__`` to fail
        # even though torch IS installed in the hf-tests job. Putting ``None``
        # in ``sys.modules`` is the documented sentinel that turns the next
        # ``import x`` into ImportError.
        monkeypatch.setitem(sys.modules, "torch", None)
        # Reload the module so the ``try/except ImportError`` inside __init__
        # sees the patched sys.modules rather than a cached-in reference.
        import gauntlet.policy.huggingface as hf_mod

        importlib.reload(hf_mod)

        with pytest.raises(ImportError, match="uv sync --extra hf"):
            hf_mod.HuggingFacePolicy(repo_id="dummy/repo", instruction="pick up the red cube")

        # Restore for later tests in this file.
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        importlib.reload(hf_mod)

    def test_reexport_guard_lazy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing ``gauntlet.policy`` must not import torch.

        The re-export in ``gauntlet.policy.__init__`` uses a module-level
        ``__getattr__`` so ``from gauntlet.policy import RandomPolicy``
        still works on torch-free installs. Accessing
        ``HuggingFacePolicy`` is what triggers the torch presence check
        and raises ``ImportError(_HF_INSTALL_HINT)`` at attribute-access
        time (RFC §6 case 2).
        """
        import gauntlet.policy as pkg

        # RandomPolicy must be reachable even if torch is mocked out.
        assert pkg.RandomPolicy is not None
        # Unknown attr raises AttributeError, not ImportError.
        with pytest.raises(AttributeError, match="HuggingNothingPolicy"):
            pkg.__getattr__("HuggingNothingPolicy")

    def test_reexport_raises_install_hint_at_attribute_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``from gauntlet.policy import HuggingFacePolicy`` must fail loudly
        at attribute-access time when the ``[hf]`` extra is missing — NOT
        at ``import gauntlet.policy`` time, which would break the
        torch-free install promise for every other policy user.
        """
        # Sentinel-None in sys.modules makes the next ``import torch`` raise.
        monkeypatch.setitem(sys.modules, "torch", None)
        import gauntlet.policy as pkg

        # ``from gauntlet.policy import RandomPolicy`` must still work.
        assert pkg.RandomPolicy is not None

        # Attribute access (which is what ``from pkg import HuggingFacePolicy``
        # performs) is the point of failure.
        with pytest.raises(ImportError, match="uv sync --extra hf"):
            pkg.__getattr__("HuggingFacePolicy")

        monkeypatch.delitem(sys.modules, "torch", raising=False)


# --------------------------------------------------------------------- case 3


class TestConstructor:
    def test_forwards_repo_id_and_trust_remote_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        proc_loader, model_loader = _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        HuggingFacePolicy(
            repo_id="openvla/fake-checkpoint",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
            model_kwargs={"attn_implementation": "eager"},
        )

        # Both loaders called with the repo id.
        assert proc_loader.call_args.args[0] == "openvla/fake-checkpoint"
        assert model_loader.call_args.args[0] == "openvla/fake-checkpoint"

        # ``trust_remote_code=True`` forced on the model loader regardless
        # of what the caller passed in ``model_kwargs``.
        assert model_loader.call_args.kwargs["trust_remote_code"] is True
        # Extra model kwargs forwarded verbatim.
        assert model_loader.call_args.kwargs["attn_implementation"] == "eager"


# --------------------------------------------------------------------- case 4


class TestActShapeDtype:
    def test_returns_float64_seven_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
            unnorm_key="bridge_orig",
        )
        obs: Observation = {"image": _zeros_image()}
        action = policy.act(obs)

        assert action.shape == (7,)
        assert action.dtype == np.float64

    def test_passes_unnorm_key_and_do_sample_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
            unnorm_key="bridge_orig",
        )
        policy.act({"image": _zeros_image()})

        # The mocked .to(...) returns the dict fixture; predict_action is
        # called with those keys plus our policy-level kwargs.
        predict_args = policy._model.predict_action.call_args
        assert predict_args.kwargs["do_sample"] is False
        assert predict_args.kwargs["unnorm_key"] == "bridge_orig"

    def test_omits_unnorm_key_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        policy.act({"image": _zeros_image()})
        predict_args = policy._model.predict_action.call_args
        assert "unnorm_key" not in predict_args.kwargs


# --------------------------------------------------------------------- case 5


class TestPromptTemplating:
    def test_prompt_matches_openvla_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        policy.act({"image": _zeros_image()})

        # processor(prompt, image, return_tensors="pt")
        call = policy._processor.call_args
        prompt_arg = call.args[0]
        assert prompt_arg == "In: What action should the robot take to grasp the cube?\nOut:"
        assert call.kwargs["return_tensors"] == "pt"


# --------------------------------------------------------------------- case 6


class TestImageValidation:
    def _make_policy(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        return HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )

    def test_rejects_float_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        with pytest.raises(ValueError, match="uint8"):
            policy.act({"image": np.zeros((224, 224, 3), dtype=np.float32)})

    def test_rejects_missing_channel_dim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            policy.act({"image": np.zeros((224, 224), dtype=np.uint8)})

    def test_rejects_non_rgb_channel_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._make_policy(monkeypatch)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            policy.act({"image": np.zeros((224, 224, 4), dtype=np.uint8)})

    def test_missing_image_key_raises_keyerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
            image_obs_key="camera_front",
        )
        with pytest.raises(KeyError, match="camera_front"):
            policy.act({"image": _zeros_image()})


# --------------------------------------------------------------------- case 7


class TestProtocolConformance:
    def test_satisfies_policy_protocols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        assert isinstance(policy, Policy)
        assert isinstance(policy, ResettablePolicy)

    def test_reset_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        # Reset must not raise and must not mutate anything we can observe.
        policy.reset(np.random.default_rng(0))
        # Still satisfies protocol after reset.
        assert isinstance(policy, Policy)


# --------------------------------------------------------------------- RFC §7 — action adaptation


class TestActionAdaptation:
    """RFC §7 defaults: gripper convention flip + OOB twist warning."""

    def test_gripper_convention_is_flipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )

        # Case A: 0.7 (OpenVLA leaning closed) → 1.0 - 1.4 = -0.4.
        policy._model.predict_action.return_value = np.array(
            [0, 0, 0, 0, 0, 0, 0.7], dtype=np.float32
        )
        action = policy.act({"image": _zeros_image()})
        assert action[6] == pytest.approx(-0.4)

        # Case B: 0.0 (OpenVLA open) → +1.0 (TabletopEnv open).
        policy._model.predict_action.return_value = np.array(
            [0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32
        )
        action = policy.act({"image": _zeros_image()})
        assert action[6] == pytest.approx(1.0)

        # Case C: 1.0 (OpenVLA fully closed) → -1.0 (TabletopEnv closed).
        policy._model.predict_action.return_value = np.array(
            [0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32
        )
        action = policy.act({"image": _zeros_image()})
        assert action[6] == pytest.approx(-1.0)

    def test_warns_when_twist_exceeds_unit_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_hf_mocks(monkeypatch)
        from gauntlet.policy.huggingface import HuggingFacePolicy

        policy = HuggingFacePolicy(
            repo_id="dummy/repo",
            instruction="grasp the cube",
            device="cpu",
            dtype="float32",
        )
        policy._model.predict_action.return_value = np.array(
            [2.0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32
        )

        with pytest.warns(RuntimeWarning, match="twist command exceeds"):
            action = policy.act({"image": _zeros_image()})

        # Adapter does NOT rescale twist — passes through unchanged.
        assert action[0] == pytest.approx(2.0)
        assert np.all(action[1:6] == 0.0)
        # Gripper flip still applied: 0.5 → 1.0 - 1.0 = 0.0.
        assert action[6] == pytest.approx(0.0)
