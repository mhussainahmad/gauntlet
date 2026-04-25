"""LeRobot-factory :class:`Policy` adapter for SmolVLA checkpoints.

See ``docs/phase2-rfc-002-lerobot-smolvla.md`` §4 for the design.

Everything torch / lerobot / PIL is imported **lazily** inside
:meth:`LeRobotPolicy.__init__`. Importing this module with the
``[lerobot]`` extra uninstalled is fine — instantiating the class is what
raises ``ImportError(_LEROBOT_INSTALL_HINT)``. This keeps ``gauntlet.core``
torch-free per spec §6.

``LeRobotPolicy`` is a *sibling* to :class:`~gauntlet.policy.huggingface.HuggingFacePolicy`
— NOT a subclass. Per RFC-002 §7 we reject a shared base class for now;
the ≤20-line rule (spec §6) is about user-facing wrap code, not adapter
bodies. If a third VLA adapter lands, revisit via a helper module.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:
    # Static-typing only — never executed. ``from __future__ import
    # annotations`` turns every annotation below into a string at runtime,
    # so referencing these names in signatures costs nothing when torch /
    # lerobot / PIL are not installed. The ``ignore_missing_imports``
    # overrides in pyproject.toml cover the missing-dep case for mypy.
    import torch  # noqa: F401

__all__ = ["LeRobotPolicy"]


Dtype = Literal["float32", "float16", "bfloat16"]


_LEROBOT_INSTALL_HINT = (
    "LeRobotPolicy requires the 'lerobot' extra. Install with:\n"
    "    uv sync --extra lerobot\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[lerobot]'"
)


class LeRobotPolicy:
    """Policy adapter for lerobot-factory checkpoints (SmolVLA and derivatives).

    Wraps :class:`lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy`
    (re-exported as ``lerobot.policies.smolvla.SmolVLAPolicy``) or any
    ``PreTrainedPolicy`` subclass with a compatible ``select_action`` /
    ``reset`` contract. The adapter:

    * Lazy-imports ``torch`` / ``lerobot`` / ``PIL`` in ``__init__``; raises
      ``ImportError(_LEROBOT_INSTALL_HINT)`` when the extra is missing.
    * Constructs the pre/post processors via
      ``lerobot.policies.factory.make_pre_post_processors`` with optional
      user overrides. Both the processors and the underlying
      ``SmolVLAPolicy`` are cached on the instance.
    * Concatenates the proprioceptive state from ``state_obs_keys`` (default
      ``("ee_pos", "gripper")``) into a single ``float32`` vector placed at
      ``"observation.state"``. SmolVLA-base pads this internally to 32-D.
    * Duplicates the single ``obs[image_obs_key]`` frame into every entry of
      ``camera_keys`` (default three SmolVLA-base cameras). Override
      ``camera_keys`` for a fine-tune with a different camera set.
    * Calls ``self._preprocess(frame)`` → ``self._policy.select_action(batch)``
      → ``self._postprocess(action)`` each step; returns a single
      per-step action (the internal chunk queue dequeues once per call).
    * Flushes the action-chunk queue on :meth:`reset` — **critical**, the
      queue is per-episode state (see RFC §4 "Action chunk queue").
    * Applies an ``action_remap`` function (default: 6→7 pad with warning)
      to bridge SmolVLA-base's SO-100 action layout to TabletopEnv's
      ``[dx, dy, dz, drx, dry, drz, gripper]``.

    Compatible with:
        * ``lerobot/smolvla_base`` (loads, but prediction quality is ~0% on
          TabletopEnv — embodiment mismatch, see RFC §2 honesty caveat).
        * User fine-tunes of ``smolvla_base`` on TabletopEnv-compatible data.

    Not compatible with:
        * OpenVLA / any ``AutoModelForVision2Seq`` checkpoint. Use
          :class:`gauntlet.policy.HuggingFacePolicy` (RFC-001).
        * π0 / diffusion-policy. Future RFC.

    Parameters
    ----------
    repo_id:
        HF Hub repo ID or local checkpoint path, passed to
        ``SmolVLAPolicy.from_pretrained`` and ``make_pre_post_processors``.
    instruction:
        Natural-language task description. Placed into the observation dict
        at ``"task"`` before preprocessing. A fixed-for-the-adapter-lifetime
        string (same as RFC-001); multi-task instruction is out of scope.
    device:
        Torch device string (``"cuda:0"``, ``"cpu"``). Defaults to ``"cuda"``
        if available, else ``"cpu"``.
    dtype:
        Torch dtype for model weights. SmolVLA-base is trained bf16;
        inference also runs at bf16 by default. CPU-only users can pass
        ``"float32"``; bf16 on CPU is slow but functional.
    image_obs_key:
        Key in Gauntlet's observation dict that holds the ``uint8 (H, W, 3)``
        RGB frame. Default ``"image"`` matches
        ``TabletopEnv(render_in_obs=True)``.
    camera_keys:
        Names of the camera slots to populate in the lerobot observation
        frame. Default is the three SmolVLA-base slots; all three receive
        the same frame unless the env supplies more keys. Set to
        ``("observation.images.camera1",)`` for a single-camera fine-tune.
    state_obs_keys:
        Gauntlet-obs keys to concatenate into ``observation.state``. Default
        ``("ee_pos", "gripper")`` is 4-D; SmolVLA-base pads internally to
        its 32-D ``max_state_dim``. The adapter does not try to match the
        pretrained 6-D SO-100 layout — fine-tuning is required for that.
    action_remap:
        Optional callable ``NDArray[np.float64] -> NDArray[np.float64]`` that
        maps the postprocessed lerobot action (shape ``(action_dim,)``, 6
        for SmolVLA-base) to TabletopEnv's 7-D action. Default: pad 6→7
        with a zero gripper **and** emit a ``RuntimeWarning`` once per
        adapter instance saying "the pretrained SO-100 action layout does
        not match TabletopEnv; pass an explicit action_remap or fine-tune".
    preprocessor_overrides, postprocessor_overrides:
        Passthroughs to ``make_pre_post_processors``. Use these to bypass
        lerobot default stats (e.g., ``dataset_stats=...``) or to tweak
        image resizing without upstreaming a custom processor.

    Raises
    ------
    ImportError: if the ``[lerobot]`` extra is not installed.
    KeyError: on ``act`` if ``image_obs_key`` or any configured
        ``state_obs_keys`` is missing from ``obs``.
    ValueError: on ``act`` if the image is not ``uint8 (H, W, 3)``.

    Notes
    -----
    Per RFC §7, the adapter intentionally does NOT forward
    ``trust_remote_code=True`` to ``SmolVLAPolicy.from_pretrained``: lerobot
    first-party policy classes live in the installed package, so the HF
    Hub remote-code path is neither needed nor safe to enable by default.

    The default action remap is a "don't fail silently" bridge — it warns
    once that SmolVLA-base's SO-100 layout is the wrong embodiment for
    TabletopEnv. OOB twist magnitudes (any ``|action[i]| > 1.0`` for
    ``i < 6``) also trigger a per-step ``RuntimeWarning``, matching
    :class:`~gauntlet.policy.huggingface.HuggingFacePolicy`'s §7 OOB
    behaviour. Both warnings exist so the harness never hides a model/env
    mismatch behind a clipped action.
    """

    def __init__(
        self,
        repo_id: str,
        instruction: str,
        *,
        device: str | None = None,
        dtype: Dtype = "bfloat16",
        image_obs_key: str = "image",
        camera_keys: Sequence[str] = (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ),
        state_obs_keys: Sequence[str] = ("ee_pos", "gripper"),
        action_remap: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        preprocessor_overrides: Mapping[str, object] | None = None,
        postprocessor_overrides: Mapping[str, object] | None = None,
    ) -> None:
        """Load SmolVLA from *repo_id* and build pre/post processors.

        ``device`` defaults to ``"cuda"`` when available else ``"cpu"``.
        ``camera_keys`` and ``state_obs_keys`` map TabletopEnv obs into
        the lerobot frame layout. ``action_remap=None`` selects the
        default SO-100 → TabletopEnv padding remap (warns once, RFC §4).

        Does NOT forward ``trust_remote_code=True``: lerobot first-party
        policies live in the installed package, not on the HF Hub.

        Raises:
            ImportError: when the ``[lerobot]`` extra is not installed.
            ValueError: on an unsupported ``dtype``.
        """
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors

            # RFC §4 cites ``lerobot.policies.smolvla.SmolVLAPolicy`` but
            # lerobot 0.4.4 does not re-export the class at the subpackage
            # root — we import from ``modeling_smolvla`` directly. (RFC §7
            # open-question default: "if lerobot's import paths have shifted,
            # use what lerobot actually exposes".)
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from PIL import Image  # noqa: F401 — presence check for [lerobot] extra.
        except ImportError as exc:
            raise ImportError(_LEROBOT_INSTALL_HINT) from exc

        # Stash torch typed as ``Any`` — this is an FFI boundary
        # (spec §6 carve-out for MuJoCo / torch / lerobot surfaces).
        self._torch: Any = torch

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
        self.device = device
        self.dtype_name: Dtype = dtype
        self.image_obs_key = image_obs_key
        self.camera_keys: tuple[str, ...] = tuple(camera_keys)
        self.state_obs_keys: tuple[str, ...] = tuple(state_obs_keys)
        self._torch_dtype: Any = dtype_map[dtype]

        # Resolve the default remap lazily — ``action_remap=None`` cannot
        # default to a bound method at class-definition time.
        self._pad_warned: bool = False
        if action_remap is None:
            self._action_remap: Callable[[NDArray[np.float64]], NDArray[np.float64]] = (
                self._default_action_remap
            )
        else:
            self._action_remap = action_remap

        # Load policy + processors via ``Any``-typed bindings: lerobot's
        # public stubs mark ``from_pretrained`` and ``make_pre_post_processors``
        # as untyped, which collides with our ``disallow_untyped_calls``
        # strict-mypy rule. Per spec §6, Any is permitted at FFI boundaries,
        # and the lerobot policy/factory surface is exactly that. Note that
        # we intentionally do NOT forward ``trust_remote_code=True`` (RFC §7):
        # lerobot first-party policies don't use the HF Hub remote-code path.
        policy_from_pretrained: Any = SmolVLAPolicy.from_pretrained
        self._policy: Any = policy_from_pretrained(repo_id)
        self._policy = self._policy.to(device=device, dtype=self._torch_dtype)
        self._policy.eval()

        factory: Any = make_pre_post_processors
        preprocessor, postprocessor = factory(
            self._policy.config,
            repo_id,
            preprocessor_overrides=(
                dict(preprocessor_overrides) if preprocessor_overrides is not None else None
            ),
            postprocessor_overrides=(
                dict(postprocessor_overrides) if postprocessor_overrides is not None else None
            ),
        )
        self._preprocess: Any = preprocessor
        self._postprocess: Any = postprocessor

    # ---- public API ------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to a 7-DoF ``float64`` action vector.

        Pipeline: validate image + concat state from ``state_obs_keys`` +
        slot ``task`` → build a lerobot "frame" dict with ``camera_keys``
        populated + an ``observation.state`` vector + the ``task`` string →
        ``self._preprocess(frame)`` → ``self._policy.select_action(batch)``
        (dequeues one step from the internal chunk queue; populates the
        queue on an empty dequeue) → ``self._postprocess(action)`` → cast
        to ``np.float64`` → ``self._action_remap(action)``.
        """
        frame = self._build_frame(obs)
        batch = self._preprocess(frame)
        raw = self._policy.select_action(batch)
        post = self._postprocess(raw)

        # ``post`` is a torch tensor (possibly on GPU, possibly bf16). Bring
        # it to CPU float32 before casting — torch.Tensor.numpy() does not
        # support bf16 directly.
        tensor: Any = post
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "to"):
            tensor = tensor.to(device="cpu", dtype=self._torch.float32)
        if hasattr(tensor, "numpy"):
            action_arr = np.asarray(tensor.numpy(), dtype=np.float64).reshape(-1)
        else:
            action_arr = np.asarray(tensor, dtype=np.float64).reshape(-1)

        action = self._action_remap(action_arr)

        # RFC §4 OOB-twist guard — mirrors HuggingFacePolicy §7. This fires
        # per-step (different cardinality from the pad-warn below), so that
        # a silently-clipped twist never looks like a dumb policy.
        twist = action[:6]
        max_twist = float(np.max(np.abs(twist)))
        if max_twist > 1.0:
            warnings.warn(
                f"LeRobotPolicy: twist command exceeds TabletopEnv's [-1, 1] "
                f"bounds (max |twist| = {max_twist:.3f}); TabletopEnv.step will clip. "
                f"The pretrained SmolVLA-base output is SO-100 joint-space, "
                f"not TabletopEnv EE-twist — fine-tune or pass an explicit "
                f"action_remap.",
                RuntimeWarning,
                stacklevel=2,
            )

        return action

    def reset(self, rng: np.random.Generator) -> None:
        """Flush the underlying SmolVLA action-chunk queue.

        **Critical for correctness.** ``SmolVLAPolicy.select_action`` caches
        a chunk of ``chunk_size`` (=50 for SmolVLA-base) actions per
        inference call and dequeues one per step. If the adapter does NOT
        call ``self._policy.reset()`` between episodes, episode N starts
        by executing the tail of episode N-1's cached chunk — subtle and
        silent. See RFC §4 "Action chunk queue".
        """
        del rng
        self._policy.reset()  # RFC §4 "Action chunk queue" — critical.

    def act_n(self, obs: Observation, n: int = 8) -> Sequence[Action]:
        """Sample ``n`` independent diffusion-policy actions for ``obs`` (B-18).

        State-preserving (B-18 :class:`gauntlet.policy.SamplablePolicy`
        contract): the underlying SmolVLA action-chunk queue is
        snapshotted via :func:`copy.deepcopy` before sampling and
        restored after, so a B-18 mode-collapse measurement does not
        consume rollout queue entries (which would silently shift the
        rollout's actions by one chunk position).

        Each draw calls :meth:`_policy.reset` to flush the queue, then
        :meth:`_policy.select_action` to populate a fresh chunk and
        dequeue its first action — diffusion's stochastic noise gives
        independent samples per call. Best-effort: if the lerobot
        internal queue attribute name shifts between releases the
        snapshot becomes a no-op and the rollout's queue is rebuilt at
        the next ``act``; we accept that over crashing the run.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        frame = self._build_frame(obs)
        batch = self._preprocess(frame)
        # Snapshot the chunk-queue attribute. lerobot exposes it as
        # ``_action_queue`` (a ``collections.deque``). deepcopy is safe
        # for plain Python deques and returns a fully detached clone.
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
                samples.append(self._action_remap(arr))
        finally:
            if snapshot is not None and queue_attr is not None:
                # Restore the original queue so the rollout's next
                # ``act`` continues from the cached chunk position.
                try:
                    self._policy._action_queue = snapshot
                except Exception:
                    self._policy.reset()
        return samples

    # ---- private helpers -------------------------------------------------

    def _build_frame(self, obs: Observation) -> dict[str, Any]:
        """Assemble the lerobot-style frame dict before preprocessing.

        Produces ``{camera_key_i: uint8_hwc_array, ...,
        "observation.state": float32_1d_array, "task": self.instruction}``.
        One place to change if we later want multi-view-from-distinct-keys.
        """
        if self.image_obs_key not in obs:
            raise KeyError(
                f"LeRobotPolicy.act: observation missing image key "
                f"{self.image_obs_key!r}; got keys {sorted(obs.keys())}"
            )
        image = self._validate_image(np.asarray(obs[self.image_obs_key]))

        missing = [k for k in self.state_obs_keys if k not in obs]
        if missing:
            raise KeyError(
                f"LeRobotPolicy.act: observation missing state key(s) {missing}; "
                f"got keys {sorted(obs.keys())}"
            )
        state_parts: list[NDArray[Any]] = [
            np.asarray(obs[k]).reshape(-1) for k in self.state_obs_keys
        ]
        state_vec = np.concatenate(state_parts).astype(np.float32, copy=False)

        frame: dict[str, Any] = dict.fromkeys(self.camera_keys, image)
        frame["observation.state"] = state_vec
        frame["task"] = self.instruction
        return frame

    def _default_action_remap(
        self,
        action: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pad a 6-D action to 7-D with a zero gripper; warn once per instance.

        Default bridge for pretrained ``smolvla_base`` against TabletopEnv.
        The warning fires exactly once per adapter instance (per RFC §4)
        to avoid log-spam at 50 Hz; the OOB-twist warning in :meth:`act`
        remains per-step.
        """
        if not self._pad_warned:
            warnings.warn(
                "LeRobotPolicy: default action_remap is padding the pretrained "
                "SO-100 6-D output to TabletopEnv's 7-D [twist, gripper] with a "
                "zero gripper. The embodiment mismatch means zero-shot success "
                "on TabletopEnv is ~0% by construction — fine-tune SmolVLA on "
                "TabletopEnv-compatible data or pass an explicit action_remap. "
                "This warning fires once per adapter instance.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._pad_warned = True

        flat = np.asarray(action, dtype=np.float64).reshape(-1)
        if flat.shape[0] == 7:
            return flat
        if flat.shape[0] == 6:
            return np.concatenate([flat, np.zeros(1, dtype=np.float64)])
        raise ValueError(
            f"LeRobotPolicy default action_remap expects 6-D or 7-D action; "
            f"got shape {flat.shape}. Pass an explicit action_remap for other "
            f"action-space widths."
        )

    @staticmethod
    def _validate_image(image_arr: NDArray[Any]) -> NDArray[np.uint8]:
        """Validate ``(H, W, 3) uint8`` and return as ``np.uint8``.

        Identical semantics to
        :meth:`~gauntlet.policy.huggingface.HuggingFacePolicy._to_pil` up to
        the PIL wrap — lerobot's preprocessor consumes ``uint8 (H, W, 3)``
        arrays directly via cv2/numpy paths, so no PIL intermediate here.
        """
        if image_arr.dtype != np.uint8:
            raise ValueError(f"LeRobotPolicy expects uint8 images, got dtype {image_arr.dtype!r}")
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError(
                "LeRobotPolicy expects RGB images with shape (H, W, 3); "
                f"got shape {image_arr.shape}"
            )
        return cast("NDArray[np.uint8]", image_arr)
