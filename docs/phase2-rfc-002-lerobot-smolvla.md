# Phase 2 RFC 002 — `LeRobotPolicy` (SmolVLA adapter)

- **Status**: Draft
- **Phase**: 2, Task 2 (follow-up to RFC-001 `HuggingFacePolicy`)
- **Author**: innovator / architect agent
- **Date**: 2026-04-22
- **Supersedes**: n/a
- **References**: `docs/phase2-rfc-001-huggingface-policy.md` (OpenVLA-shape adapter; defers SmolVLA explicitly in its §2 non-goals).

---

## 1. Summary

RFC-001 shipped `HuggingFacePolicy` for OpenVLA-shape checkpoints loadable via `AutoModelForVision2Seq` + `predict_action(unnorm_key=...)` and explicitly scoped SmolVLA out because its loading path is different: `lerobot.policies.smolvla.SmolVLAPolicy.from_pretrained(...)`, plus a `make_pre_post_processors` pipeline, plus an internal 50-step action-chunk queue that must be flushed on `reset`. This RFC adds a sibling adapter `gauntlet.policy.lerobot.LeRobotPolicy` behind a **new `[lerobot]` extra** (not folded into `[hf]`), reusing RFC-001's lazy-import / extras-guard / image-validation patterns but not sharing a base class — the ≤20-line rule is about user-facing wrap code (unchanged at ~15 lines), not the adapter body. The core stays torch-free per spec §6; the env grows **no new kwargs** (the existing `render_in_obs=True` from RFC-001 suffices; state comes from the obs dict via a `state_obs_keys` adapter knob); and the `Policy` protocol does **not** change. One honesty caveat is foregrounded: `lerobot/smolvla_base`'s pretraining embodiment is SO-100/SO-101 (6-DoF joint-position control, gripper as joint-6) whereas TabletopEnv is a mocap EE with a 7-D `[twist, gripper]` action, so zero-shot success on TabletopEnv will be ~0% by construction — users must fine-tune. The adapter works correctly; the harness must not read 0% as a bug.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.policy.lerobot.LeRobotPolicy` — a reference adapter that wraps any `lerobot.policies.smolvla.SmolVLAPolicy`-compatible checkpoint (SmolVLA-base and user fine-tunes thereof).
- Keep `gauntlet.core` (everything under `src/gauntlet/` except `policy/huggingface.py` and `policy/lerobot.py`) torch-free: importing `gauntlet`, `gauntlet.policy`, `gauntlet.env`, `gauntlet.runner` must not transitively import `torch`, `lerobot`, `transformers`, or `PIL`.
- Keep the public `Policy` / `ResettablePolicy` protocols unchanged (§6 hard rule).
- Keep `mypy --strict` passing regardless of whether the `[lerobot]` extra is installed.
- `uv sync --extra lerobot` — one command to turn the adapter on. Composes with `--extra hf` for users who want both adapters on one machine.
- Correctly call `policy.reset()` on the underlying `SmolVLAPolicy` between episodes to flush the 50-step action-chunk queue (see §4).

### Non-goals

- π0, OpenPI, or any non-lerobot-factory VLA. A third RFC if/when needed.
- Multi-task / per-episode language instructions. Same as RFC-001: the adapter holds one fixed string for its lifetime. An `EpisodeContext`-via-`reset` extension is Phase-2-next-RFC material.
- Fine-tuning support. The adapter loads and infers; training is out of scope for the evaluation harness per §2.
- An end-to-end success test against real SmolVLA weights in CI. Like RFC-001, CI mocks at the `SmolVLAPolicy.from_pretrained` / `make_pre_post_processors` seams; real-weight smoke tests are a `workflow_dispatch` GPU job.
- Any change to RFC-001's `[hf]` extras set (keep blast radius minimal — RFC-001 just landed).

## 3. Dependency placement decision

### Choice: **(B) new `[lerobot]` extra** — separate from `[hf]`.

### Options considered

- **(A) Extend `[hf]` to include `lerobot[smolvla]`.** One extras install, two adapters. Rejected.
- **(B) New `[lerobot]` extra.** Users opt into `--extra hf`, `--extra lerobot`, or both. **Chosen.**
- **(C) Rename to a single mega-extra `[vla]` = OpenVLA + lerobot deps.** Rejected.
- **(D) Separate `gauntlet-lerobot` distribution package.** Rejected (premature — RFC-001 already rejected this for `[hf]` on the same grounds).

### Why (B)

Four discriminating facts:

1. **`lerobot[smolvla]` is a heavy and orthogonal dep set.** Confirmed against April 2026 lerobot (`pip install "lerobot[smolvla]"` / `pip install -e ".[smolvla]"`): it pulls the full `lerobot` package itself (torch, torchvision, opencv-python-headless, einops, safetensors, accelerate, huggingface_hub, and friends), the SmolVLM2 VLM backbone via transformers (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`), and ffmpeg-linked video deps via torchcodec. An OpenVLA-only user installing `[hf]` should not pay for opencv + ffmpeg + lerobot's dataset stack.
2. **Version-pin conflict risk is non-trivial.** RFC-001's `[hf]` floors `transformers>=4.40,<5` and `torch>=2.2,<3`. `lerobot` does its own transformers / torch / tokenizers pinning, and has in practice been more aggressive about floors (SmolVLA PR #1175 bumped several). Putting both in one extras set ties Gauntlet's release cadence to the intersection of both upstreams. Two extras let each evolve independently.
3. **Composition is already a first-class uv primitive.** `uv sync --extra hf --extra lerobot` is the native way to get both — exactly the composition RFC-001 §3 used to justify (A) over a separate package. Extras *compose* cleanly; they're *not* mutually exclusive.
4. **Adapter-per-extra is the cleanest blame boundary.** "Install the matching extra for the checkpoint you're running" is a one-line rule. If a third VLA adapter lands (π0), it gets its own extra too; precedent is set.

### `pyproject.toml` diff (fragments only — inline, not applied in this RFC)

```toml
# EXISTING from RFC-001 — unchanged.
[project.optional-dependencies]
hf = [
    "torch>=2.2,<3",
    "transformers>=4.40,<5",
    "timm>=0.9.10,<2",
    "tokenizers>=0.19,<1",
    "pillow>=10.0,<12",
]

# NEW — independent extra for the lerobot / SmolVLA adapter.
# lerobot[smolvla] pulls torch, torchvision, transformers, safetensors,
# accelerate, huggingface_hub, opencv-python-headless, einops, and the
# SmolVLM2 backbone via transformers. Version floors follow lerobot's own
# floors in April 2026; revisit when lerobot pins shift. No ceiling on
# lerobot itself because its public API is not yet stable — see §7.
lerobot = [
    "lerobot[smolvla]>=0.4,<1",
    # transformers pin kept in sync with lerobot's own floor so a joint
    # install with --extra hf resolves without pip backtracking.
    "transformers>=4.40,<5",
    "pillow>=10.0,<12",
]

# NEW — dev group analogue of hf-dev, for the lerobot pytest job.
[dependency-groups]
# (existing dev + hf-dev groups unchanged)
lerobot-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]

# NEW mypy override — let mypy import-check ``policy/lerobot.py`` even when
# lerobot isn't installed (default CI job stays torch-/lerobot-free).
[[tool.mypy.overrides]]
module = ["lerobot", "lerobot.*"]
ignore_missing_imports = true

# NEW pytest marker.
[tool.pytest.ini_options]
markers = [
    "hf: tests that require the [hf] extra (torch, transformers, PIL)",
    "lerobot: tests that require the [lerobot] extra (lerobot, torch, PIL)",
]
```

### CI structure

One new job, mirroring RFC-001's `hf-tests`:

- **Default job (unchanged):** `uv sync` (no extras) → `ruff` → `mypy` → `pytest -m 'not hf and not lerobot'`. Still the continuous enforcement of §6 "no torch in the core".
- **`hf-tests` (from RFC-001, unchanged):** `uv sync --extra hf --group hf-dev` → `pytest -m hf`.
- **NEW `lerobot-tests`:** `uv sync --extra lerobot --group lerobot-dev` → `pytest -m lerobot`. Standard `ubuntu-latest` (no GPU) — tests mock at the `SmolVLAPolicy.from_pretrained` / `make_pre_post_processors` seams (see §6).

All three jobs block merges.

### File placement

- `src/gauntlet/policy/lerobot.py` — sibling to `huggingface.py`. Matches the naming pattern of `random.py` / `scripted.py` / `huggingface.py`.
- Re-exported from `policy/__init__.py` via the same `__getattr__`-based lazy-import guard already established for `HuggingFacePolicy`, so `from gauntlet.policy import RandomPolicy` keeps working on a lerobot-free install.
- Tests: `tests/lerobot/test_lerobot_policy.py`, all marked `@pytest.mark.lerobot`.

## 4. Adapter API

### User-facing wrap (≤20-line rule, §6)

```python
# examples/evaluate_smolvla.py  (illustrative — not code committed in this task)
from gauntlet.policy import LeRobotPolicy

policy = LeRobotPolicy(
    repo_id="lerobot/smolvla_base",
    instruction="pick up the red cube and place it on the target",
    device="cuda:0",
    # TabletopEnv renders one camera; SmolVLA expects three. Duplicate by
    # default; override to map 1:1 onto a fine-tune that was trained on a
    # single-camera dataset.
    image_obs_key="image",
    camera_keys=("observation.images.camera1",
                 "observation.images.camera2",
                 "observation.images.camera3"),
    # SmolVLA-base is 6-D joint-position output for SO-100. TabletopEnv
    # expects 7-D [dx, dy, dz, drx, dry, drz, gripper]. The default remap
    # pads 6→7 with a zero gripper AND emits a loud RuntimeWarning; pass
    # an explicit callable once you've fine-tuned on TabletopEnv.
    state_obs_keys=("ee_pos", "gripper"),
    action_remap=None,  # default: pad+warn; see §4 "Action adaptation"
)
```

### Class sketch (signatures + docstrings + 1-line bodies only)

```python
# src/gauntlet/policy/lerobot.py
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:
    import torch  # noqa: F401

Dtype = Literal["float32", "float16", "bfloat16"]

_LEROBOT_INSTALL_HINT = (
    "LeRobotPolicy requires the 'lerobot' extra. Install with:\n"
    "    uv sync --extra lerobot\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[lerobot]'"
)


class LeRobotPolicy:
    """Policy adapter for lerobot-factory checkpoints (SmolVLA and derivatives).

    Wraps ``lerobot.policies.smolvla.SmolVLAPolicy`` (or any
    ``PreTrainedPolicy`` subclass with a compatible ``select_action`` /
    ``reset`` contract). The adapter:

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
        * ``lerobot/smolvla_base`` (cold, but prediction quality is ~0% on
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
        inference also runs at bf16 by default.
    image_obs_key:
        Key in Gauntlet's observation dict that holds the ``uint8 (H, W, 3)``
        RGB frame. Default ``"image"`` matches ``TabletopEnv(render_in_obs=True)``.
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
        for SmolVLA-base) to TabletopEnv's 7-D action. Default: pad 6→7 with
        a zero gripper **and** emit a ``RuntimeWarning`` once per adapter
        instance saying "the pretrained SO-100 action layout does not match
        TabletopEnv; pass an explicit action_remap or fine-tune".
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
        """Lazy-import lerobot/torch/PIL; load SmolVLAPolicy + processors."""
        ...  # raise ImportError(_LEROBOT_INSTALL_HINT) on import failure.

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to a 7-DoF ``float64`` action vector.

        Pipeline: extract image + concat state + slot ``task`` → build a
        lerobot "frame" dict with ``camera_keys`` populated + an
        ``observation.state`` vector + the ``task`` string →
        ``self._preprocess(frame)`` → ``self._policy.select_action(batch)``
        (dequeues one step from the internal chunk queue; populates the
        queue on an empty dequeue) → ``self._postprocess(action)`` → cast to
        ``np.float64`` → ``self._action_remap(action)``.
        """
        ...  # see pipeline above; return cast("Action", action_7d)

    def reset(self, rng: np.random.Generator) -> None:
        """Flush the underlying SmolVLA action-chunk queue.

        **Critical for correctness.** ``SmolVLAPolicy.select_action`` caches
        a chunk of ``chunk_size`` (=50 for SmolVLA-base) actions per
        inference call and dequeues one per step. If the adapter does NOT
        call ``self._policy.reset()`` between episodes, episode N starts by
        executing the tail of episode N-1's cached chunk — subtle and
        silent. The unit tests assert queue emptiness after ``reset``.
        """
        ...  # del rng; self._policy.reset()

    # ---- private helpers -------------------------------------------------

    def _build_frame(self, obs: Observation) -> "dict[str, Any]":
        """Assemble the lerobot-style frame dict before preprocessing.

        Produces: ``{camera_key_i: uint8_hwc_array, ...,
        "observation.state": float32_1d_array, "task": self._instruction}``.
        One place to change if we later want multi-view-from-distinct-keys.
        """
        ...

    def _default_action_remap(
        self,
        action: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pad a 6-D action to 7-D with a zero gripper; warn once per instance.

        Default bridge for pretrained ``smolvla_base`` against TabletopEnv.
        The warning fires exactly once per adapter instance to avoid
        log-spam at 50 Hz; subsequent OOB twist magnitudes are still
        warned per-step the same way RFC-001 §4 handles them.
        """
        ...

    @staticmethod
    def _validate_image(image_arr: NDArray[Any]) -> NDArray[np.uint8]:
        """Validate ``(H, W, 3) uint8`` and return as ``np.uint8``.

        Identical semantics to ``HuggingFacePolicy._to_pil`` up to the
        PIL wrap — lerobot's preprocessor consumes ``uint8 (H, W, 3)``
        arrays directly via cv2 paths, so no PIL intermediate here.
        Lives as a private static so a future ``_vla_utils`` helper
        module can promote it without breaking this file's contract
        (see §10 refactor hook).
        """
        ...
```

### Obs → lerobot-frame mapping (concrete)

| Source (TabletopEnv obs)             | Destination (lerobot frame)                 | Dtype / shape                |
|--------------------------------------|---------------------------------------------|------------------------------|
| `obs["image"]` (`render_in_obs=True`) | Each key in `camera_keys` (duplicated)      | `uint8 (H, W, 3)`            |
| `np.concat([obs[k] for k in state_obs_keys]).astype(float32)` | `"observation.state"` | `float32 (sum_dims,)`        |
| `self._instruction` (constructor arg) | `"task"`                                    | Python `str`                 |
| n/a                                   | `"robot_type"` (optional, `None`)          | Python `str \| None`         |

After preprocessing, `build_inference_frame` / the pipeline yields a batched tensor dict the policy consumes; the adapter does not touch intermediate tensors.

### Action-chunk queue (the bit RFC-001 did not need)

`SmolVLAPolicy.select_action` returns one action per call, but internally:

```python
# From lerobot/policies/smolvla/modeling_smolvla.py (main, April 2026).
if self._check_get_actions_condition():
    actions = self._get_action_chunk(batch, noise)
    self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
return self._queues[ACTION].popleft()
```

With `chunk_size = n_action_steps = 50`, one forward pass every 50 steps, dequeued one per step. `SmolVLAPolicy.reset()` clears `self._queues`. **The adapter's `reset` MUST call this**, otherwise episode N opens with the tail of episode N-1's chunk. This is the single most important correctness difference from RFC-001 (where `HuggingFacePolicy.reset` was a documented no-op).

### Action adaptation

SmolVLA-base emits a 6-D action per step for SO-100: joint positions `[q0, q1, q2, q3, q4, q_gripper]`. TabletopEnv's action is `[-1, +1]`-bounded 7-D `[dx, dy, dz, drx, dry, drz, gripper]` per RFC-001 §4. These are:

- **Different dimensionality** (6 vs 7) — requires an explicit remap, not a flip.
- **Different frame** (joint-position vs EE-twist) — no simple conversion exists; fine-tuning is the real answer.
- **Different gripper convention** (joint-6 angle vs `[-1, +1]` snap) — also remap-dependent.

Default `action_remap`: zero-pad 6→7 and append a zero gripper; emit a **single** `RuntimeWarning` on the first `act` call explaining the embodiment mismatch. This mirrors RFC-001's "don't hide failures" stance — we surface the mismatch loudly and let the OOB-twist check (inherited helper; see §10) fire per-step if any component exceeds `[-1, 1]`. Users with a TabletopEnv-fine-tuned checkpoint pass a real `action_remap`.

## 5. Required core changes

### Policy protocol: **unchanged.**

Same reasoning as RFC-001 §5. `Policy.act(obs) -> Action` accepts a `Mapping[str, NDArray[Any]]`; the adapter holds its own instruction. `ResettablePolicy.reset(rng)` covers the action-queue flush.

### `TabletopEnv`: **no new kwargs.**

Three options:

1. **No env change.** Use RFC-001's `render_in_obs=True` for the image; concat `ee_pos` + `gripper` (the existing state keys) into `observation.state` inside the adapter via `state_obs_keys`. **Chosen.**
2. Add a `state_in_obs: bool` / `state_layout: Sequence[str]` kwarg to `TabletopEnv.__init__` that materialises a concatenated `state` key.
3. Add a `render_state_in_obs: bool` kwarg that emits a single named `state` vector in a VLA-friendly layout.

Rejected (2) and (3): every VLA family wants a different state vector (SmolVLA 6-D joint positions for SO-100; future π0 wants whatever π0 wants; OpenVLA doesn't want state at all). Pushing the concat into each adapter keeps `TabletopEnv` honest about what it actually produces (`ee_pos`, `gripper`, `cube_pos`, ...) and puts the embodiment-mismatch burden on the adapter, where it belongs. If the third VLA adapter lands and re-duplicates the concat logic, promote to a helper module (see §10) — but do not grow an env kwarg for it.

One matching-default note: RFC-001 set `render_size=(224, 224)` to avoid a double-resize against OpenVLA's 224 processor. SmolVLA's `resize_imgs_with_padding` default is `(512, 512)`; the adapter's docstring notes that users who care about preserving every pixel pass `TabletopEnv(render_in_obs=True, render_size=(512, 512))`. 224→512 upsample is lossy but not broken — default behaviour is still correct.

### `gauntlet.policy.__init__`: one small patch.

Extend the existing `__getattr__` (landed in RFC-001) to also handle `"LeRobotPolicy"` under a lerobot-presence check. Same guard pattern; same `ImportError` surface.

```python
# src/gauntlet/policy/__init__.py  (extension, not rewrite)
def __getattr__(name: str) -> Any:
    if name == "HuggingFacePolicy":
        ...  # existing RFC-001 branch.
    if name == "LeRobotPolicy":
        from gauntlet.policy.lerobot import _LEROBOT_INSTALL_HINT
        try:
            import lerobot  # noqa: F401 — presence check.
        except ImportError as exc:
            raise ImportError(_LEROBOT_INSTALL_HINT) from exc
        from gauntlet.policy.lerobot import LeRobotPolicy
        return LeRobotPolicy
    raise AttributeError(...)
```

That's the full blast radius outside `policy/lerobot.py`.

## 6. Test plan

All lerobot tests live under `tests/lerobot/` and are marked `@pytest.mark.lerobot`. Default `pytest` de-selects them; the `lerobot-tests` CI job runs `pytest -m lerobot`.

### Unit tests (run in the default, torch-free job — `@pytest.mark.lerobot` *not* required for these three)

1. **Import guard.** Monkeypatch `sys.modules["lerobot"]` to `None`; assert `LeRobotPolicy(...)` raises `ImportError` whose message contains `uv sync --extra lerobot`.
2. **Re-export guard.** `from gauntlet.policy import LeRobotPolicy` with lerobot missing raises a clean `ImportError` at attribute-access time, not module-import time.
3. **Protocol conformance markers.** `LeRobotPolicy` has `act` and `reset` methods with the right signatures so `isinstance(p, Policy)` and `isinstance(p, ResettablePolicy)` are `True` (duck-typed, like `RandomPolicy`). These are signature-level tests that do not need the extra installed.

### Lerobot-extra tests (run in the new `lerobot-tests` job — `@pytest.mark.lerobot`)

4. **Constructor contract (mocked weights).** `monkeypatch.setattr("lerobot.policies.smolvla.SmolVLAPolicy.from_pretrained", fake_policy_loader)` and `monkeypatch.setattr("lerobot.policies.factory.make_pre_post_processors", fake_factory)`. `fake_policy_loader` returns a `MagicMock` with `.config.image_features = {...}`, `.config.n_action_steps = 50`, and a `.select_action` returning a fake tensor. Assert `repo_id` reaches the loader, `device` and `dtype` reach `.to()`, and both processor overrides pass through.
5. **`act()` shape/dtype.** Hand an obs with `"image"` (224×224×3 uint8), `"ee_pos"`, and `"gripper"`. Assert the returned action is shape `(7,)`, `dtype float64`, and the preprocessor was called with a dict whose keys include all three `camera_keys`, `"observation.state"`, and `"task"`.
6. **Image duplication.** With `camera_keys=("a", "b", "c")`, assert all three receive `obs["image"]` (identity). With `camera_keys=("a",)`, only `"a"` is populated.
7. **State concatenation.** With `state_obs_keys=("ee_pos", "gripper")`, assert `"observation.state"` is the hstack of those two as `float32` with total shape `(4,)`.
8. **Instruction slot.** `instruction="grasp the cube"` → frame dict passed to preprocess has `"task": "grasp the cube"`.
9. **Action-chunk queue flush on reset (the critical one).** Spy on `fake_policy.reset`. Construct adapter → call `reset(rng)` → assert `fake_policy.reset` was called exactly once. This is the RFC-001 → RFC-002 delta.
10. **Default action-remap (6→7 pad + warn).** Have `fake_policy.select_action` return shape `(6,)`. Assert returned action shape is `(7,)`, `action[6] == 0.0`, and a `RuntimeWarning` was raised exactly once across N>1 calls (warn-once semantics; use `pytest.warns` + `warnings.catch_warnings`).
11. **User-supplied `action_remap` bypasses the warning.** Pass `action_remap=lambda a: np.concatenate([a, [1.0]])`; assert no warning fires and `action[6] == 1.0`.
12. **OOB twist magnitude warns.** When `action_remap` produces `|action[i]| > 1.0` for `i < 6`, a `RuntimeWarning` fires (mirrors RFC-001 §4). Reuses the same helper if promoted to `_vla_utils`; otherwise duplicated here.
13. **Image validation.** Float image → `ValueError`. Shape `(H, W)` (no channels) → `ValueError`. Missing `image_obs_key` → `KeyError`.

### Explicitly out-of-scope for this task

- **Real-weight smoke test against `lerobot/smolvla_base`.** The weights are ~3 GB; no CI download. `examples/evaluate_smolvla.py` (checklist item, not a test) lets maintainers verify end-to-end on a GPU box. Because of the embodiment mismatch, that smoke test will show ~0% success — document this in the example's top-level comment so no one misreads it as a regression.
- **Pinning numerical outputs from a mocked model.**

## 7. Open questions

Each has a reasonable default in parentheses so the implementation agent is not blocked.

- **How much of the `build_inference_frame` / `make_pre_post_processors` / dequeue pipeline is hidden inside the adapter?** Hiding everything is ergonomic but couples us to lerobot's current factory signatures, which are *not* API-stable (April 2026: `build_inference_frame` lives in `policy/utils.py`; earlier releases had it elsewhere). (**Default: hide behind sensible defaults, but expose both `preprocessor_overrides` and `postprocessor_overrides` as explicit kwargs so users can patch around breaking lerobot changes without forking Gauntlet.**)
- **Does `preprocessor_overrides={"empty_cameras": 2}` actually replace the camera-duplication path?** The `empty_cameras` config field exists on `SmolVLAConfig`, but the preprocessor's handling needs verification against lerobot main at implementation time. (**Default: do NOT rely on `empty_cameras`; keep the explicit camera-duplication in `_build_frame`. Revisit if users complain.**)
- **`state_obs_keys` default.** `("ee_pos", "gripper")` is 4-D; SmolVLA-base was trained on a 6-D SO-100 joint vector. The padding is internal (max 32-D), so shape-wise it's fine — but semantically the adapter is handing the model an off-distribution state. (**Default: `("ee_pos", "gripper")` with the honesty caveat in the docstring; users who fine-tune override `state_obs_keys` and their fine-tune's dataset stats match.**)
- **`dtype` default.** `bfloat16` is the SmolVLA default and matches the model card. On CPU-only hosts, bf16 is slow; should we auto-fall-back to `float32` when `device="cpu"`? (**Default: bf16 everywhere, document the CPU perf penalty; users who care pass `dtype="float32"`.**)
- **`trust_remote_code`.** Not required for lerobot-first-party policies (unlike OpenVLA). Adapter should explicitly **not** forward a `trust_remote_code=True` to `from_pretrained`. (**Default: do not pass `trust_remote_code` at all — lerobot weights go through its own loading path.**)
- **HF Hub token.** `lerobot/smolvla_base` is public; no token needed for baseline. Private fine-tunes need `HF_TOKEN` / `huggingface-cli login` — document this in the docstring. (**Default: no explicit token kwarg; let `huggingface_hub` read from env/config as it always does.**)
- **Registry shortcut in `gauntlet run --policy smolvla`.** Same answer as RFC-001: no shortcut. Users write a three-line factory (`repo_id` + `instruction` are user-specific). (**Default: no registry entry in this RFC.**)
- **Multi-task / per-episode instructions.** Still not in scope. If/when both VLA adapters need it, design an `EpisodeContext` threaded via `env.reset(...) -> (obs, info)` with `info["instruction"]`. (**Default: defer; flag in a future RFC.**)
- **Shared `_vla_utils` helper module.** If the `action_remap` + OOB-twist-warn + image validation helpers start to diverge between `huggingface.py` and `lerobot.py`, extract to `src/gauntlet/policy/_vla_utils.py` (a helper module, **not** a shared base class — inheritance here obscures the ≤20-line user-facing wrap). (**Default: no extraction in this RFC; revisit on RFC-003.**)

## 8. Rough implementation checklist

Sized as one §9-style PR, ~8 commits.

1. **`pyproject.toml`**: add `[project.optional-dependencies] lerobot = [...]`, the `lerobot-dev` dependency group, the new mypy override for `lerobot.*`, and the `lerobot` pytest marker. Regenerate `uv.lock`.
2. **`src/gauntlet/policy/lerobot.py`**: write the class per §4 — lazy-import guard, `_build_frame`, `act`, `reset` (with `self._policy.reset()`), `_default_action_remap`, `_validate_image`. Keep string annotations on all torch/lerobot/PIL types. Reuse the RFC-001 OOB-twist-warn pattern inline for now (do NOT premature-extract a base class or helper module — see §7).
3. **`src/gauntlet/policy/__init__.py`**: extend the existing `__getattr__` to handle `"LeRobotPolicy"`; update `TYPE_CHECKING` re-export; add to `__all__`.
4. **`tests/lerobot/test_lerobot_policy.py`**: cases 1-13 from §6. Mock at `SmolVLAPolicy.from_pretrained` and `make_pre_post_processors`. Mark all with `@pytest.mark.lerobot` except the three torch-free import-guard tests.
5. **CI**: add the `lerobot-tests` job to `.github/workflows/ci.yml` mirroring the `hf-tests` job. Confirm the default job now runs `pytest -m 'not hf and not lerobot'`.
6. **`examples/evaluate_smolvla.py`**: docstring with the embodiment-mismatch warning, `if __name__ == "__main__":`, ≤20-line user-facing wrap. Mention fine-tuning is required for non-zero success and link to `huggingface.co/docs/lerobot/smolvla`.
7. **`README.md`**: one bullet under the existing "Using a real VLA" section (added in RFC-001): `uv sync --extra lerobot` + a one-line pointer to the example, including the "fine-tune required" honesty caveat.
8. **Local gate**: `uv run ruff check && uv run mypy && uv run pytest -m 'not hf and not lerobot'` AND `uv sync --extra lerobot --group lerobot-dev && uv run pytest -m lerobot`. Both must be green. PR description explicitly links this RFC and RFC-001, and calls out the zero-env-change claim as a design wins.

---

## Appendix A — External facts anchoring this RFC (as of April 2026)

- `lerobot[smolvla]` is a single pip extra (`pip install "lerobot[smolvla]"`, or `pip install -e ".[smolvla]"` for editable). Confirmed against `huggingface.co/docs/lerobot/smolvla`.
- `SmolVLAPolicy.from_pretrained(model_id)` + `make_pre_post_processors(policy.config, model_id, preprocessor_overrides=...)` is the canonical inference entrypoint. Confirmed against `github.com/huggingface/lerobot/blob/main/examples/tutorial/smolvla/using_smolvla_example.py`.
- `SmolVLAPolicy.select_action` returns one action per call by popping from `self._queues[ACTION]`; populates the deque from a 50-step chunk when empty. `SmolVLAPolicy.reset` clears `self._queues`. Confirmed against `src/lerobot/policies/smolvla/modeling_smolvla.py` on `main` (April 2026).
- `lerobot/smolvla_base` config: `input_features` has three cameras (`observation.images.camera1/2/3`, shape `[3, 256, 256]`) + `observation.state` shape `[6]`; `output_features.action` shape `[6]`; `max_state_dim = max_action_dim = 32`; `chunk_size = n_action_steps = 50`; `resize_imgs_with_padding = [512, 512]`; VLM backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Confirmed against `huggingface.co/lerobot/smolvla_base/raw/main/config.json`.
- Pretraining embodiment is SO-100/SO-101 (per the model card's `so101_follower` inference example and `docs/source/smolvla.mdx`). There is **no OpenVLA-style `unnorm_key`** — normalisation stats are baked into the checkpoint and applied via `make_pre_post_processors`.
- `trust_remote_code` is **not** required by lerobot's first-party policy classes. Do not forward it from the RFC-001 pattern.
- HF Hub token is not required for the public `smolvla_base` checkpoint; private fine-tunes follow the usual `huggingface_hub` env/config flow.
