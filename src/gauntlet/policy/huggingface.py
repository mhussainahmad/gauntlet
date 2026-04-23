"""HuggingFace-backed :class:`Policy` adapter for VLA checkpoints.

See ``docs/phase2-rfc-001-huggingface-policy.md`` §4 for the design.

Everything torch / transformers / PIL is imported **lazily** inside
:meth:`HuggingFacePolicy.__init__`. Importing this module with the
``[hf]`` extra uninstalled is fine — instantiating the class is what
raises ``ImportError(_HF_INSTALL_HINT)``. This keeps ``gauntlet.core``
torch-free per spec §6.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:
    # Static-typing only — never executed. ``from __future__ import
    # annotations`` turns every annotation below into a string at runtime,
    # so referencing these names in signatures costs nothing when torch /
    # transformers / PIL are not installed. The ``ignore_missing_imports``
    # override in pyproject.toml covers the missing-dep case for mypy.
    import torch
    from PIL.Image import Image as PILImage

__all__ = ["HuggingFacePolicy"]


Dtype = Literal["float32", "float16", "bfloat16"]


_HF_INSTALL_HINT = (
    "HuggingFacePolicy requires the 'hf' extra. Install with:\n"
    "    uv sync --extra hf\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[hf]'"
)

# OpenVLA's prompt template — per the published model card and RFC §4.
_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


class HuggingFacePolicy:
    """Policy adapter for HF checkpoints loadable via ``AutoModelForVision2Seq``.

    Compatible with OpenVLA-7B and derivatives that expose a
    ``predict_action(**inputs, unnorm_key=...) -> np.ndarray`` method via
    ``trust_remote_code=True``. Not compatible with SmolVLA (see RFC-002).

    The adapter holds a fixed task ``instruction`` string for its lifetime;
    multi-task instruction-per-episode is out of scope for this RFC.

    Parameters
    ----------
    repo_id:
        HF Hub repo ID or local checkpoint path, forwarded to
        ``AutoModelForVision2Seq.from_pretrained`` / ``AutoProcessor.from_pretrained``.
    instruction:
        Natural-language task description. Inserted into the OpenVLA prompt
        template ``"In: What action should the robot take to {instruction}?\\nOut:"``.
    unnorm_key:
        Dataset key passed through to ``model.predict_action`` for action
        de-normalisation (e.g. ``"bridge_orig"`` for BridgeData V2). If
        ``None`` the model picks its sole dataset or raises.
    device:
        Torch device string (``"cuda:0"``, ``"cpu"``). Defaults to ``"cuda"``
        when available, else ``"cpu"``.
    dtype:
        Parameter dtype. ``"bfloat16"`` is the OpenVLA default.
    image_obs_key:
        Key in the observation dict that holds the ``uint8`` ``(H, W, 3)``
        RGB frame. Defaults to ``"image"``. Requires the env to surface a
        rendered frame — see :class:`gauntlet.env.TabletopEnv`'s
        ``render_in_obs`` kwarg.
    processor_kwargs / model_kwargs:
        Extra keyword args forwarded verbatim to the two ``from_pretrained``
        calls (e.g. ``{"attn_implementation": "flash_attention_2"}``).
        ``trust_remote_code=True`` is forced on the model loader regardless.

    Raises
    ------
    ImportError: if the ``[hf]`` extra is not installed (clear install hint).
    KeyError: on ``act`` if ``image_obs_key`` is missing from ``obs``.
    ValueError: on ``act`` if the image shape/dtype is not ``(H, W, 3), uint8``.

    Notes
    -----
    ``act()`` applies the gripper convention flip
    (OpenVLA ``[0, 1]`` with ``0 = open, 1 = close`` →
    TabletopEnv ``[+1 open, -1 close]``) unconditionally, per RFC §7. It
    does NOT rescale the twist magnitudes on ``action[:6]``: BridgeData V2's
    ``unnorm_key`` emits world-frame metre deltas that can exceed
    TabletopEnv's ``[-1, 1]`` per-step bounds. We pass them through and
    emit a ``RuntimeWarning`` when any twist component is OOB, so an
    adapter/unnorm-key mismatch surfaces loudly instead of being silently
    clipped by ``TabletopEnv.step``.
    """

    def __init__(
        self,
        repo_id: str,
        instruction: str,
        *,
        unnorm_key: str | None = None,
        device: str | None = None,
        dtype: Dtype = "bfloat16",
        image_obs_key: str = "image",
        processor_kwargs: Mapping[str, object] | None = None,
        model_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError(_HF_INSTALL_HINT) from exc

        # Stash the modules / classes we'll need later. Typed ``Any`` — this is
        # an FFI boundary (spec §6 carve-out for MuJoCo / torch).
        self._torch: Any = torch
        self._Image: Any = Image

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map: dict[str, Any] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}; expected one of {list(dtype_map)}")

        self.repo_id = repo_id
        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.device = device
        self.dtype_name: Dtype = dtype
        self.image_obs_key = image_obs_key
        self._torch_dtype: Any = dtype_map[dtype]
        self._prompt = _PROMPT_TEMPLATE.format(instruction=instruction)

        proc_kwargs: dict[str, Any] = dict(processor_kwargs or {})
        proc_kwargs.setdefault("trust_remote_code", True)
        mdl_kwargs: dict[str, Any] = dict(model_kwargs or {})
        # trust_remote_code is required by OpenVLA's custom modeling code.
        mdl_kwargs["trust_remote_code"] = True
        mdl_kwargs.setdefault("torch_dtype", self._torch_dtype)

        # Loader call sites go through ``Any``-typed bindings: transformers'
        # public stubs mark ``from_pretrained`` as untyped, which collides
        # with our ``disallow_untyped_calls`` strict-mypy rule. Per spec §6
        # ``Any`` is permitted at FFI boundaries, and the HF model surface
        # is exactly that.
        proc_from_pretrained: Any = AutoProcessor.from_pretrained
        model_from_pretrained: Any = AutoModelForVision2Seq.from_pretrained
        self._processor: Any = proc_from_pretrained(repo_id, **proc_kwargs)
        model: Any = model_from_pretrained(repo_id, **mdl_kwargs)
        self._model: Any = model.to(device)

    # ---- public API ------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to a 7-DoF ``float64`` action vector.

        Extracts ``obs[image_obs_key]`` as a ``uint8 (H, W, 3)`` array,
        wraps it in ``PIL.Image``, runs the processor with the cached prompt,
        and calls ``predict_action`` on the loaded model. The returned
        ``np.float64`` array has the gripper convention flipped from OpenVLA's
        ``[0, 1]`` (``0 = open, 1 = close``) to TabletopEnv's
        ``[+1 open, -1 close]``; twist magnitudes on ``action[:6]`` are NOT
        rescaled, and a ``RuntimeWarning`` is emitted if any twist coordinate
        exceeds ``[-1, 1]`` (see RFC §7).
        """
        if self.image_obs_key not in obs:
            raise KeyError(
                f"HuggingFacePolicy.act: observation missing image key "
                f"{self.image_obs_key!r}; got keys {sorted(obs.keys())}"
            )
        image_arr = np.asarray(obs[self.image_obs_key])
        pil_image = self._to_pil(image_arr)
        inputs = self._prep_inputs(pil_image)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key is not None:
            kwargs["unnorm_key"] = self.unnorm_key
        raw_action = self._model.predict_action(**inputs, **kwargs)

        action = np.asarray(raw_action, dtype=np.float64).reshape(-1)

        # OpenVLA emits gripper in [0, 1] (0 = open, 1 = close).
        # TabletopEnv's action_space expects [+1 = open, -1 = close] with a
        # binary snap in TabletopEnv.step. Flip once so downstream sees the
        # env's convention.
        action[6] = 1.0 - 2.0 * action[6]

        # RFC §7: the BridgeData V2 unnorm_key emits world-frame metre deltas
        # that can exceed TabletopEnv's [-1, 1] per-step bounds. Don't silently
        # clip in TabletopEnv.step — surface the mismatch so "my adapter is
        # broken" doesn't look like "my policy is dumb". Spec §6: never
        # aggregate away failures.
        twist = action[:6]
        max_twist = float(np.max(np.abs(twist)))
        if max_twist > 1.0:
            warnings.warn(
                f"HuggingFacePolicy: twist command exceeds TabletopEnv's [-1, 1] "
                f"bounds (max |twist| = {max_twist:.3f}); TabletopEnv.step will clip. "
                f"Check your unnorm_key (BridgeData V2 uses metre deltas) and "
                f"consider rescaling.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cast("Action", action)

    def reset(self, rng: np.random.Generator) -> None:
        """No-op — the HF model is stateless between episodes.

        Implemented so the adapter satisfies :class:`ResettablePolicy` and
        the Runner can hand it an RNG like every other policy.
        """
        del rng

    # ---- private helpers -------------------------------------------------

    def _build_prompt(self) -> str:
        """Return the OpenVLA prompt template with ``self.instruction`` slotted in."""
        return self._prompt

    def _prep_inputs(self, image: PILImage) -> Mapping[str, torch.Tensor]:
        """Run the processor and move tensors to device+dtype.

        Kept as a single choke-point so that if we later need multi-view or
        proprioceptive inputs, only this method changes.
        """
        inputs = self._processor(self._prompt, image, return_tensors="pt")
        moved = inputs.to(self.device, dtype=self._torch_dtype)
        # ``BatchFeature`` supports ``**`` unpacking via dict semantics, but
        # some processors return plain dicts. Normalise to dict so downstream
        # ``**kwargs`` works regardless of the upstream class.
        return cast("Mapping[str, torch.Tensor]", dict(moved))

    def _to_pil(self, image_arr: NDArray[Any]) -> PILImage:
        """Validate ``(H, W, 3) uint8`` and wrap in ``PIL.Image``.

        Rejects float arrays and non-RGB shapes with a clear error — the
        adapter's image contract is specific; we surface violations rather
        than silently coercing.
        """
        if image_arr.dtype != np.uint8:
            raise ValueError(
                f"HuggingFacePolicy expects uint8 images, got dtype {image_arr.dtype!r}"
            )
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                "HuggingFacePolicy expects RGB images with shape (H, W, 3); "
                f"got shape {image_arr.shape}"
            )
        return cast("PILImage", self._Image.fromarray(image_arr, mode="RGB"))
