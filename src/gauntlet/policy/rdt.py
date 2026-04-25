"""RDT (Robotics Diffusion Transformer) :class:`Policy` adapter.

Backlog item B-15 — second half. RDT is the Robotics Diffusion
Transformer (``robotics-diffusion-transformer/rdt-1b``), a
1B-parameter bimanual VLA distributed on the Hugging Face Hub. Unlike
GR00T-N1 / SmolVLA / π0, RDT does NOT ship through the lerobot policy
factory (no ``lerobot[rdt]`` extra exists as of lerobot 0.4.4 — checked
via ``uv pip install --dry-run`` at adapter build time). We therefore
load it via ``transformers.AutoModel.from_pretrained(..., trust_remote_code=True)``
and call its model-card-documented ``predict_action`` surface, mirroring
the OpenVLA path in :class:`gauntlet.policy.huggingface.HuggingFacePolicy`.

Everything torch / transformers / PIL is imported **lazily** inside
:meth:`RdtPolicy.__init__`. Importing this module with the ``[rdt]``
extra uninstalled is fine — instantiating the class is what raises
``ImportError(_RDT_INSTALL_HINT)``.

Honesty caveat (the B-15 anti-feature). RDT-1B inference needs a CUDA
GPU; the adapter ships anyway because presence is the value. The
``predict_action`` surface assumed here is what RDT's published model
card documents, but the upstream ``trust_remote_code`` script is
expected to evolve — if the symbol shifts, the adapter's lazy-import
guard surfaces the failure cleanly. Embodiment mismatch with TabletopEnv
(7-D EE-twist + gripper) is also expected: RDT was trained on bimanual
data, so single-arm zero-shot success is ~0% by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:  # pragma: no cover — strings at runtime per `from __future__`.
    import torch  # noqa: F401

__all__ = ["RdtPolicy"]


_RDT_INSTALL_HINT = (
    "RdtPolicy requires the 'rdt' extra. Install with:\n"
    "    uv sync --extra rdt\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[rdt]'"
)

# RDT's published prompt template — fixed task description slotted into
# a single string per the model card. We keep this as a module constant
# so the adapter's surface stays minimal (``model_id`` / ``device`` /
# ``action_horizon``); a real deployment overrides ``_DEFAULT_INSTRUCTION``
# or replaces ``_build_inputs``.
_DEFAULT_INSTRUCTION: str = "perform the demonstrated task"


class RdtPolicy:
    """Policy adapter for RDT (Robotics Diffusion Transformer) checkpoints.

    Loads via ``transformers.AutoModel.from_pretrained(model_id,
    trust_remote_code=True)`` (RDT lives outside the ``transformers``
    tree — same shape as OpenVLA's loader path) and invokes the
    documented ``predict_action`` surface per inference step.

    Parameters
    ----------
    model_id:
        HF Hub repo ID or local checkpoint path. Defaults to
        ``"robotics-diffusion-transformer/rdt-1b"``.
    device:
        Torch device string. Defaults to ``"cpu"``.
    action_horizon:
        Length of the diffusion-policy action chunk dequeued per
        ``select_action`` round-trip. Stored on the instance and
        exposed for inspection; downstream callers (notably the
        runner's CRN paired-compare paths) read ``policy.action_horizon``
        to size buffers. Default 64 (RDT's published horizon).

    Raises
    ------
    ImportError: if the ``[rdt]`` extra is not installed.
    KeyError: on ``act`` if ``"image"`` is missing from the observation.
    ValueError: on ``act`` if the image is not ``uint8 (H, W, 3)``.
    """

    def __init__(
        self,
        model_id: str = "robotics-diffusion-transformer/rdt-1b",
        device: str = "cpu",
        action_horizon: int = 64,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(_RDT_INSTALL_HINT) from exc

        self._torch: Any = torch
        self._Image: Any = Image
        self.model_id = model_id
        self.device = device
        self.action_horizon = int(action_horizon)
        self._instruction = _DEFAULT_INSTRUCTION

        # FFI seam: transformers' public ``from_pretrained`` is untyped —
        # bind via ``Any`` (same pattern as HuggingFacePolicy — spec §6
        # carve-out for FFI boundaries). RDT's modeling code lives
        # outside the transformers tree, so ``trust_remote_code=True`` is
        # required.
        loader: Any = AutoModel.from_pretrained
        model: Any = loader(model_id, trust_remote_code=True)
        self._model: Any = model.to(device)
        self._model.eval()

    # ---- public API ------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to an action vector."""
        inputs = self._build_inputs(obs)
        raw = self._model.predict_action(**inputs)

        tensor: Any = raw
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
        """Flush any cached diffusion-policy chunk on the underlying model.

        RDT's diffusion head caches an action chunk per inference call
        and dequeues one per step (same shape as π0 / GR00T-N1). The
        underlying model exposes a ``reset`` hook when the cached chunk
        is per-episode state — best-effort: if absent, the diffusion
        sampler simply re-rolls on the next ``act``.
        """
        del rng
        reset_fn = getattr(self._model, "reset", None)
        if callable(reset_fn):
            reset_fn()

    # ---- private helpers -------------------------------------------------

    def _build_inputs(self, obs: Observation) -> dict[str, Any]:
        """Assemble the RDT input dict before inference.

        Single image + a fixed instruction string. The exact key names
        match what RDT's published ``predict_action`` signature expects;
        a real deployment with a fine-tuned checkpoint overrides this.
        """
        if "image" not in obs:
            raise KeyError(
                f"RdtPolicy.act: observation missing image key 'image'; "
                f"got keys {sorted(obs.keys())}"
            )
        image = self._validate_image(np.asarray(obs["image"]))
        # ``mode=`` was deprecated in Pillow 11 (removal in 13); the array
        # is already (H, W, 3) uint8 so PIL infers RGB unambiguously.
        pil_image: Any = self._Image.fromarray(image)
        return {"image": pil_image, "instruction": self._instruction}

    @staticmethod
    def _validate_image(image_arr: NDArray[Any]) -> NDArray[np.uint8]:
        if image_arr.dtype != np.uint8:
            raise ValueError(f"RdtPolicy expects uint8 images, got dtype {image_arr.dtype!r}")
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"RdtPolicy expects RGB images with shape (H, W, 3); got shape {image_arr.shape}"
            )
        return cast("NDArray[np.uint8]", image_arr)
