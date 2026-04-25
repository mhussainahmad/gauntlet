"""π0 (Physical Intelligence) :class:`Policy` adapter via lerobot.

Backlog item B-14. π0 is the highest-profile open VLA after OpenVLA /
SmolVLA. It is distributed through the same lerobot policy factory as
SmolVLA — :class:`lerobot.policies.pi0.modeling_pi0.PI0Policy` is a
``PreTrainedPolicy`` subclass with the same ``from_pretrained`` /
``select_action`` / ``reset`` contract. We wrap it here behind the
``[pi0]`` extra (which pulls ``lerobot[pi]``).

Everything torch / lerobot / PIL is imported **lazily** inside
:meth:`Pi0Policy.__init__`. Importing this module with the ``[pi0]``
extra uninstalled is fine — instantiating the class is what raises
``ImportError(_PI0_INSTALL_HINT)``.

Honesty caveat — same shape as the lerobot/SmolVLA adapter. PI's
licensing on weights is restrictive; users may not be able to download
what the adapter expects. The adapter ships anyway: presence is the
value (eval-on-π0 becomes possible the moment licensing clears or the
user has access). Embodiment / action-space mismatch with TabletopEnv
is also expected — see :class:`gauntlet.policy.lerobot.LeRobotPolicy`
for the same caveat applied to SmolVLA.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:  # pragma: no cover — strings at runtime per `from __future__`.
    import torch  # noqa: F401

__all__ = ["Pi0Policy"]


_PI0_INSTALL_HINT = (
    "Pi0Policy requires the 'pi0' extra. Install with:\n"
    "    uv sync --extra pi0\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[pi0]'"
)

# Default lerobot frame layout for π0 — three camera slots + a state
# vector, mirroring the SmolVLA convention. PI0 was trained against the
# DROID + Bridge mixture; the actual frame keys depend on the
# fine-tune. These defaults are fine for shelfware: a real deployment
# overrides them by patching ``Pi0Policy._build_frame``.
_DEFAULT_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
_DEFAULT_STATE_KEYS: tuple[str, ...] = ("ee_pos", "gripper")
_DEFAULT_INSTRUCTION: str = "perform the demonstrated task"


class Pi0Policy:
    """Policy adapter for π0 / π0-fast checkpoints loaded through lerobot.

    Wraps :class:`lerobot.policies.pi0.modeling_pi0.PI0Policy` (re-exported
    as ``lerobot.policies.pi0.PI0Policy``). The contract mirrors
    :class:`gauntlet.policy.lerobot.LeRobotPolicy` exactly — π0 and
    SmolVLA share lerobot's ``PreTrainedPolicy`` base class — but the
    constructor surface is intentionally narrower (B-14 spec): only
    ``model_id`` / ``device`` / ``action_horizon``. Camera-slot, state-
    key, and instruction defaults live as module-level constants.

    Parameters
    ----------
    model_id:
        HF Hub repo ID or local checkpoint path. Defaults to
        ``"lerobot/pi0"``. Per the B-14 anti-feature note: PI's licensing
        on weights is restrictive; users may need to authenticate or
        otherwise prove access before this resolves.
    device:
        Torch device string. Defaults to ``"cpu"``.
    action_horizon:
        Number of actions to dequeue from the underlying chunk queue per
        ``select_action`` round-trip. Stored on the instance and exposed
        for inspection; π0's flow-matching head produces an action chunk
        whose length is governed by the model config, but downstream
        callers (notably the runner's CRN paired-compare paths) read
        ``policy.action_horizon`` to size buffers. Default 8.

    Raises
    ------
    ImportError: if the ``[pi0]`` extra is not installed.
    KeyError: on ``act`` if the configured image / state keys are
        missing from the observation.
    ValueError: on ``act`` if the image is not ``uint8 (H, W, 3)``.
    """

    def __init__(
        self,
        model_id: str = "lerobot/pi0",
        device: str = "cpu",
        action_horizon: int = 8,
    ) -> None:
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            from PIL import Image  # noqa: F401 — presence check for [pi0] extra.
        except ImportError as exc:
            raise ImportError(_PI0_INSTALL_HINT) from exc

        self._torch: Any = torch
        self.model_id = model_id
        self.device = device
        self.action_horizon = int(action_horizon)

        # FFI seam: lerobot's public ``from_pretrained`` / factory surface
        # is untyped, so bind via ``Any`` (same pattern as
        # gauntlet.policy.lerobot — spec §6 carve-out for FFI boundaries).
        # We intentionally do NOT forward ``trust_remote_code=True``:
        # lerobot first-party policies live in the installed package.
        loader: Any = PI0Policy.from_pretrained
        self._policy: Any = loader(model_id)
        self._policy = self._policy.to(device=device)
        self._policy.eval()

        factory: Any = make_pre_post_processors
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
        """Flush the underlying π0 action-chunk queue.

        **Critical for correctness** — same delta as
        :meth:`gauntlet.policy.lerobot.LeRobotPolicy.reset`. π0's
        flow-matching head caches an action chunk per inference call
        and dequeues one per step; without this flush, episode N would
        start by executing the tail of episode N-1's chunk.
        """
        del rng
        self._policy.reset()

    def act_n(self, obs: Observation, n: int = 8) -> Sequence[Action]:
        """Sample ``n`` independent flow-matching actions for ``obs`` (B-18).

        State-preserving (B-18 :class:`gauntlet.policy.SamplablePolicy`
        contract). Snapshots the underlying chunk queue via
        :func:`copy.deepcopy`, calls :meth:`_policy.reset` +
        :meth:`_policy.select_action` ``n`` times to draw N independent
        flow-matching samples, and restores the queue afterwards so the
        rollout's next ``act`` continues from the cached chunk
        position. See :meth:`gauntlet.policy.lerobot.LeRobotPolicy.act_n`
        for the matching pattern (π0 and SmolVLA share lerobot's
        ``PreTrainedPolicy`` chunk-queue API).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        frame = self._build_frame(obs)
        batch = self._preprocess(frame)
        snapshot: Any = None
        queue_attr = getattr(self._policy, "_action_queue", None)
        if queue_attr is not None:
            try:
                snapshot = copy.deepcopy(queue_attr)
            except Exception:
                snapshot = None
        samples: list[Action] = []
        try:
            for _ in range(n):
                self._policy.reset()
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
                samples.append(arr)
        finally:
            if snapshot is not None and queue_attr is not None:
                try:
                    self._policy._action_queue = snapshot
                except Exception:
                    self._policy.reset()
        return samples

    # ---- private helpers -------------------------------------------------

    def _build_frame(self, obs: Observation) -> dict[str, Any]:
        """Assemble the lerobot-style frame dict before preprocessing."""
        if "image" not in obs:
            raise KeyError(
                "Pi0Policy.act: observation missing image key 'image'; "
                f"got keys {sorted(obs.keys())}"
            )
        image = self._validate_image(np.asarray(obs["image"]))

        missing = [k for k in _DEFAULT_STATE_KEYS if k not in obs]
        if missing:
            raise KeyError(
                f"Pi0Policy.act: observation missing state key(s) {missing}; "
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
            raise ValueError(f"Pi0Policy expects uint8 images, got dtype {image_arr.dtype!r}")
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                f"Pi0Policy expects RGB images with shape (H, W, 3); got shape {image_arr.shape}"
            )
        return cast("NDArray[np.uint8]", image_arr)
