"""GR00T-N1 (NVIDIA humanoid foundation) :class:`Policy` adapter via lerobot.

Backlog item B-15 — first half. GR00T-N1 is NVIDIA's humanoid foundation
model; lerobot 0.4.4 ships it as
:class:`lerobot.policies.groot.modeling_groot.GrootPolicy` (re-exported
as ``lerobot.policies.groot.GrootPolicy``). It is a ``PreTrainedPolicy``
subclass with the same ``from_pretrained`` / ``select_action`` /
``reset`` contract as :class:`gauntlet.policy.pi0.Pi0Policy`. We wrap it
here behind the ``[groot]`` extra (which pulls ``lerobot[groot]``).

Everything torch / lerobot / PIL is imported **lazily** inside
:meth:`GrootN1Policy.__init__`. Importing this module with the
``[groot]`` extra uninstalled is fine — instantiating the class is what
raises ``ImportError(_GROOT_INSTALL_HINT)``.

Honesty caveat (the B-15 anti-feature). GR00T-N1 inference needs serious
GPU: the Eagle-2.5 VL backbone alone is ~7B params. The adapter ships
anyway because presence is the value — running ``gauntlet`` against
GR00T becomes possible the moment a user has the hardware. Embodiment
mismatch with TabletopEnv is also expected: GR00T-N1 was trained on
humanoid bimanual data, not a 7-D EE-twist single-arm. See
:class:`gauntlet.policy.lerobot.LeRobotPolicy` for the same caveat
applied to SmolVLA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:  # pragma: no cover — strings at runtime per `from __future__`.
    import torch  # noqa: F401

__all__ = ["GrootN1Policy"]


_GROOT_INSTALL_HINT = (
    "GrootN1Policy requires the 'groot' extra. Install with:\n"
    "    uv sync --extra groot\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[groot]'"
)

# Default lerobot frame layout for GR00T-N1 — three camera slots + a
# state vector, mirroring the SmolVLA / π0 conventions. The actual
# expected slot names depend on the GR00T fine-tune; these defaults are
# fine for shelfware: a real deployment overrides them by patching
# ``GrootN1Policy._build_frame``.
_DEFAULT_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
_DEFAULT_STATE_KEYS: tuple[str, ...] = ("ee_pos", "gripper")
_DEFAULT_INSTRUCTION: str = "perform the demonstrated task"


class GrootN1Policy:
    """Policy adapter for GR00T-N1 checkpoints loaded through lerobot.

    Wraps :class:`lerobot.policies.groot.modeling_groot.GrootPolicy`
    (re-exported as ``lerobot.policies.groot.GrootPolicy``). The
    contract mirrors :class:`gauntlet.policy.pi0.Pi0Policy` — both share
    lerobot's ``PreTrainedPolicy`` base — but the constructor surface is
    intentionally narrow per the B-15 spec: only ``model_id`` /
    ``device`` / ``action_horizon``. Camera-slot, state-key, and
    instruction defaults live as module-level constants.

    Parameters
    ----------
    model_id:
        HF Hub repo ID or local checkpoint path. Defaults to
        ``"nvidia/groot-n1"``. Per the B-15 anti-feature note: GR00T-N1
        weights are gated; users may need to authenticate or otherwise
        prove access before this resolves.
    device:
        Torch device string. Defaults to ``"cpu"``.
    action_horizon:
        Number of actions to dequeue from the underlying chunk queue per
        ``select_action`` round-trip. Stored on the instance and exposed
        for inspection; GR00T-N1's flow-matching head produces an action
        chunk whose length is governed by the model config. Default 8.

    Raises
    ------
    ImportError: if the ``[groot]`` extra is not installed.
    KeyError: on ``act`` if the configured image / state keys are
        missing from the observation.
    ValueError: on ``act`` if the image is not ``uint8 (H, W, 3)``.
    """

    def __init__(
        self,
        model_id: str = "nvidia/groot-n1",
        device: str = "cpu",
        action_horizon: int = 8,
    ) -> None:
        try:
            import torch
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import (
                make_groot_pre_post_processors,
            )
            from PIL import Image  # noqa: F401 — presence check for [groot] extra.
        except ImportError as exc:
            raise ImportError(_GROOT_INSTALL_HINT) from exc

        self._torch: Any = torch
        self.model_id = model_id
        self.device = device
        self.action_horizon = int(action_horizon)

        # FFI seam: lerobot's public ``from_pretrained`` / factory surface
        # is untyped — bind via ``Any`` (same pattern as Pi0Policy /
        # LeRobotPolicy — spec §6 carve-out for FFI boundaries). We do
        # NOT forward ``trust_remote_code=True``: lerobot first-party
        # policies live in the installed package.
        loader: Any = GrootPolicy.from_pretrained
        self._policy: Any = loader(model_id)
        self._policy = self._policy.to(device=device)
        self._policy.eval()

        factory: Any = make_groot_pre_post_processors
        preprocessor, postprocessor = factory(
            self._policy.config,
            model_id,
            preprocessor_overrides=None,
            postprocessor_overrides=None,
        )
        self._preprocess: Any = preprocessor
        self._postprocess: Any = postprocessor

    # ---- public API ------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to an action vector."""
        frame = self._build_frame(obs)
        batch = self._preprocess(frame)
        raw = self._policy.select_action(batch)
        post = self._postprocess(raw)

        tensor: Any = post
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "to"):
            tensor = tensor.to(device="cpu", dtype=self._torch.float32)
        if hasattr(tensor, "numpy"):
            arr = np.asarray(tensor.numpy(), dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(tensor, dtype=np.float64).reshape(-1)
        return arr

    def reset(self, rng: np.random.Generator) -> None:
        """Flush the underlying GR00T-N1 action-chunk queue.

        **Critical for correctness** — same delta as
        :meth:`gauntlet.policy.pi0.Pi0Policy.reset`. GR00T-N1's
        flow-matching head caches an action chunk per inference call
        and dequeues one per step; without this flush, episode N would
        start by executing the tail of episode N-1's chunk.
        """
        del rng
        self._policy.reset()

    # ---- private helpers -------------------------------------------------

    def _build_frame(self, obs: Observation) -> dict[str, Any]:
        """Assemble the lerobot-style frame dict before preprocessing."""
        if "image" not in obs:
            raise KeyError(
                "GrootN1Policy.act: observation missing image key 'image'; "
                f"got keys {sorted(obs.keys())}"
            )
        image = self._validate_image(np.asarray(obs["image"]))

        missing = [k for k in _DEFAULT_STATE_KEYS if k not in obs]
        if missing:
            raise KeyError(
                f"GrootN1Policy.act: observation missing state key(s) {missing}; "
                f"got keys {sorted(obs.keys())}"
            )
        state_parts = [np.asarray(obs[k]).reshape(-1) for k in _DEFAULT_STATE_KEYS]
        state_vec = np.concatenate(state_parts).astype(np.float32, copy=False)

        frame: dict[str, Any] = dict.fromkeys(_DEFAULT_CAMERA_KEYS, image)
        frame["observation.state"] = state_vec
        frame["task"] = _DEFAULT_INSTRUCTION
        return frame

    @staticmethod
    def _validate_image(image_arr: NDArray[Any]) -> NDArray[np.uint8]:
        if image_arr.dtype != np.uint8:
            raise ValueError(f"GrootN1Policy expects uint8 images, got dtype {image_arr.dtype!r}")
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                "GrootN1Policy expects RGB images with shape (H, W, 3); "
                f"got shape {image_arr.shape}"
            )
        return cast("NDArray[np.uint8]", image_arr)
