# Phase 2 RFC 001 — `HuggingFacePolicy` adapter

- **Status**: Draft
- **Phase**: 2, Task 1 (first item in `GAUNTLET_SPEC.md` §7)
- **Author**: innovator / architect agent
- **Date**: 2026-04-22
- **Supersedes**: n/a

---

## 1. Summary

Phase 1 ships a `Policy` protocol plus `Random`/`Scripted` reference adapters; §4 of the spec calls for a `HuggingFace`-backed reference wrapper so external teams can drop in real VLA checkpoints. This RFC narrows that wrapper to the OpenVLA-shaped loading path (`AutoModelForVision2Seq` + `AutoProcessor` + `predict_action(**inputs, unnorm_key=...)`) and defers the lerobot-shaped loading path (SmolVLA) to a follow-up RFC. To keep §6's "no `torch` in the core" rule intact we place all heavy deps behind a `[project.optional-dependencies] hf = [...]` extras group, lazy-import everything inside `HuggingFacePolicy.__init__`, and add an opt-in `render_in_obs: bool` constructor flag on `TabletopEnv` so the adapter receives a camera frame without changing the `Policy` protocol or touching the `Runner`. The `Policy` protocol does **not** change.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.policy.huggingface.HuggingFacePolicy` — a reference adapter that wraps OpenVLA-7B and any other checkpoint loadable via `AutoModelForVision2Seq` + `trust_remote_code=True` + a `predict_action(unnorm_key=...)` method.
- Keep `gauntlet.core` (everything under `src/gauntlet/` except `policy/huggingface.py`) torch-free: importing `gauntlet`, `gauntlet.policy`, `gauntlet.env`, `gauntlet.runner`, etc., must not transitively import `torch`, `transformers`, or `PIL`.
- Keep the public `Policy` protocol unchanged (§6: "Minimal policy adapter … more than 20 lines, the `Policy` protocol is wrong").
- Keep `mypy --strict` passing whether or not torch is installed.
- `uv sync --extra hf` must be the single command that turns the adapter on.

### Non-goals

- **SmolVLA support** in this task. SmolVLA does not load via `AutoModelForVision2Seq`; it requires `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained(...)` + `lerobot[smolvla]` + `make_pre_post_processors`. Cramming both loading APIs into one class turns the wrapper into a strategy-pattern mess and breaks the ≤20-line rule. Tracked separately as Phase 2 RFC-002 `LeRobotPolicy` (sibling adapter in the same `[hf]` extra or a new `[lerobot]` extra — decided in that RFC).
- π0, diffusion-policy, or any non-HF checkpoint format.
- Per-episode language instructions. TabletopEnv is single-task. The adapter holds one fixed instruction string for its lifetime.
- Action-space adaptation beyond a 7-DoF pass-through (OpenVLA already emits 7-DoF `[dx, dy, dz, rx, ry, rz, gripper]` which matches `TabletopEnv`'s action space; scaling/unnormalisation strategy is an open question, see §7).
- Downloading model weights in CI. Unit tests mock at the HF `from_pretrained` boundary; real-weight smoke tests run only as opt-in jobs.

## 3. Dependency placement decision

### Choice: **(A) Optional extras group**

### Why not (B), (C), (D)

- **(B) Separate `gauntlet-hf` package.** Overkill for Phase 2 Task 1. A second package adds release plumbing (versioning, CI publish jobs, matched-version matrix) before the first external user. Revisit when a third adapter (e.g. lerobot) lands and the extras set actually fragments.
- **(C) Dev-only `[dependency-groups] hf`.** Makes `HuggingFacePolicy` uninstallable by end users. The entire *point* of this adapter is `gauntlet run --policy module.path:my_openvla_factory` from someone else's checkout; a dev-only group defeats that.
- **(D) Something else.** Namespace-package plugin (`gauntlet_hf/`) was considered but adds import-time discovery complexity that buys nothing over (A).

### Why (A)

- Matches §6 exactly: core stays torch-free, `gauntlet.core` still imports cleanly without torch.
- One command (`uv sync --extra hf`) to enable; one module (`gauntlet.policy.huggingface`) to blame.
- Extras compose cleanly with future adapters (`uv sync --extra hf --extra lerobot`).
- `uv`'s extras resolution is first-class, same ergonomics as `pip install gauntlet[hf]`.

### `pyproject.toml` diff (fragments)

```toml
# NEW: optional extras — end-user-facing.
[project.optional-dependencies]
hf = [
    # Version floors mirror openvla/requirements-min.txt (Apr 2026) with
    # ceilings one major above to catch surprise breakage early.
    "torch>=2.2,<3",
    "transformers>=4.40,<5",
    "timm>=0.9.10,<2",
    "tokenizers>=0.19,<1",
    "pillow>=10.0,<12",
    # flash-attn is intentionally NOT listed: it requires CUDA at pip-install
    # time and breaks CPU-only installs. Users opt in separately per the
    # HuggingFacePolicy docstring.
]

# NEW: CI-only group so the HF pytest job can pull dev + hf together.
[dependency-groups]
dev = [ ... ]  # unchanged
hf-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]

# NEW mypy override: let mypy import-check our `huggingface.py` even when
# torch/transformers/PIL aren't installed.
[[tool.mypy.overrides]]
module = ["torch", "torch.*", "transformers", "transformers.*", "PIL", "PIL.*"]
ignore_missing_imports = true

# NEW pytest marker for the hf-only test subset.
[tool.pytest.ini_options]
markers = [
    "hf: tests that require the [hf] extra (torch, transformers, PIL)",
]
```

### File placement

- `src/gauntlet/policy/huggingface.py` — **confirmed**. Matches the naming pattern of `random.py` / `scripted.py`. Re-exported from `policy/__init__.py` under an `ImportError` guard so `from gauntlet.policy import HuggingFacePolicy` raises a clear error when the extra isn't installed, but `from gauntlet.policy import RandomPolicy` keeps working.
- Tests: `tests/hf/test_huggingface_policy.py` — new subdirectory, all tests marked `@pytest.mark.hf`. Default `uv run pytest` de-selects this marker; the HF CI job runs `pytest -m hf`.

### Import-guard pattern

```python
# src/gauntlet/policy/huggingface.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

# Keep static type-checking alive even on machines without torch installed.
# These names are only used inside string annotations below.
if TYPE_CHECKING:
    import torch  # noqa: F401

_HF_INSTALL_HINT = (
    "HuggingFacePolicy requires the 'hf' extra. Install with:\n"
    "    uv sync --extra hf\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[hf]'"
)


class HuggingFacePolicy:
    def __init__(self, repo_id: str, instruction: str, ...) -> None:
        try:
            import torch
            from PIL import Image  # noqa: F401 — used later
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError(_HF_INSTALL_HINT) from exc
        ...
```

- Static-typing annotations on method signatures use string form: `def _to_pil(self, img: NDArray[np.uint8]) -> "Image.Image": ...` and `image_tensor: "torch.Tensor"`.
- `if TYPE_CHECKING` re-imports satisfy mypy when the extras ARE installed and silently skip when they aren't; the `[[tool.mypy.overrides]] ignore_missing_imports = true` block handles the missing-dep case.

### CI structure

GitHub Actions grows one new job. Current layout (`.github/workflows/ci.yml`, created in Phase 1 Task 1) is assumed to be a single matrix running lint + type-check + test.

- **Default job (unchanged)**: `uv sync` (no extras) → `uv run ruff check` → `uv run mypy` → `uv run pytest -m 'not hf'`. This job must pass without torch, transformers, or PIL installed — it is the continuous enforcement of §6's "no torch in the core".
- **New `hf-tests` job**: `uv sync --extra hf --group hf-dev` → `uv run pytest -m hf`. Runs on the same Python version matrix. No GPU runner; tests mock the model boundary (see §6), so a standard `ubuntu-latest` image is enough.

Both jobs block merges. Real-weight integration is out of CI — a separately triggered `workflow_dispatch` job exists to pull `openvla/openvla-7b` on a GPU runner, but that's not in scope for this RFC.

## 4. Adapter API

Wrapping OpenVLA is ≤15 user lines (§6's ≤20-line rule holds):

```python
# examples/evaluate_openvla.py  (illustrative — not code to commit in this task)
from gauntlet.policy import HuggingFacePolicy

policy = HuggingFacePolicy(
    repo_id="openvla/openvla-7b",
    instruction="pick up the red cube and place it on the target",
    unnorm_key="bridge_orig",
    device="cuda:0",
    dtype="bfloat16",
    image_obs_key="image",
)
# Hand `policy` to `gauntlet run` the same way as RandomPolicy/ScriptedPolicy.
```

Class sketch (signatures + docstrings + one-line bodies only — not a full implementation):

```python
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as PILImage
    from transformers import PreTrainedModel  # noqa: F401

Dtype = Literal["float32", "float16", "bfloat16"]


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
        rendered frame — see §5 for the ``TabletopEnv.render_in_obs`` flag.
    processor_kwargs / model_kwargs:
        Extra keyword args forwarded verbatim to the two ``from_pretrained``
        calls (e.g. ``{"attn_implementation": "flash_attention_2"}``).

    Raises
    ------
    ImportError: if the ``[hf]`` extra is not installed (clear install hint).
    KeyError: on ``act`` if ``image_obs_key`` is missing from ``obs``.
    ValueError: on ``act`` if the image shape/dtype is not ``(H, W, 3), uint8``.
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
        """Lazy-import torch/transformers/PIL, then load processor + model."""
        ...  # raise ImportError(_HF_INSTALL_HINT) if imports fail, else load.

    def act(self, obs: Observation) -> Action:
        """Map a single-frame observation to a 7-DoF ``float64`` action vector.

        Extracts ``obs[image_obs_key]`` as a ``uint8 (H, W, 3)`` array,
        wraps it in ``PIL.Image``, runs the processor with the cached prompt,
        calls ``predict_action`` on the loaded model, and returns a
        ``np.float64`` array matching ``TabletopEnv.action_space``.
        """
        ...  # return np.asarray(self._model.predict_action(**self._prep(obs)), dtype=np.float64)

    def reset(self, rng: "np.random.Generator") -> None:
        """No-op — the HF model is stateless between episodes. Kept for
        ``ResettablePolicy`` conformance so the Runner treats this adapter
        like every other policy."""
        ...  # del rng

    # ---- private helpers -------------------------------------------------

    def _build_prompt(self) -> str:
        """Return the OpenVLA prompt template with ``self._instruction`` slotted in."""
        ...

    def _prep_inputs(self, image: "PILImage") -> Mapping[str, "torch.Tensor"]:
        """Run the processor and move tensors to device+dtype. One place to
        change if we later support multi-view images."""
        ...

    @staticmethod
    def _to_pil(image_arr: NDArray[np.uint8]) -> "PILImage":
        """Validate ``(H, W, 3) uint8`` and wrap in ``PIL.Image``. Rejects
        float arrays with a clear error — keeps the adapter honest about
        its image contract."""
        ...
```

### How obs → model inputs, model output → env action

- **Image**: pulled from `obs[image_obs_key]` (default `"image"`). Adapter validates `(H, W, 3) uint8`, converts to `PIL.Image`. No explicit resize — the HF `AutoProcessor` handles target resolution (OpenVLA processor standardises to 224×224).
- **Proprioceptive state**: ignored. OpenVLA is an image-only VLA; no state head exists on the model. (SmolVLA *does* consume state; that's one reason it needs a different adapter.)
- **Instruction**: single fixed string held by the adapter, slotted into `"In: What action should the robot take to {instruction}?\nOut:"`. Not read from obs. Not reset between episodes.
- **Action out**: `model.predict_action(**inputs, unnorm_key=..., do_sample=False)` returns a `np.ndarray` of shape `(7,)`. The adapter casts to `float64` to match `Action`. No coordinate-frame or scaling conversion in this RFC — see §7 open questions.

## 5. Required core changes

### Policy protocol: **unchanged.**

`Policy.act(obs) -> Action` already accepts a `Mapping[str, NDArray[Any]]`, which can carry an image channel. Text-conditioned policies slot in by holding their instruction internally and reading the image from a well-known obs key. No API break, no caller migration.

### `TabletopEnv`: one additive change (opt-in).

The env currently returns `{cube_pos, cube_quat, ee_pos, gripper, target_pos}` — no pixels. `HuggingFacePolicy` needs an image. Three options were considered:

1. **`TabletopEnv.__init__(..., render_in_obs: bool = False)`** — when `True`, `_build_obs` additionally emits `"image"` as a `uint8 (H, W, 3)` array via a cached offscreen renderer, and `observation_space` grows an `"image"` Box. Default remains `False`, so every existing test and caller is byte-identical.
2. **Runner merges `env.render()` into obs before `policy.act`.** Touches `Runner`, changes the contract for all policies, not just image-conditioned ones.
3. **Policy gets a reference to the env.** Breaks the protocol and re-litigates §6.

**Chosen: option 1.** Smallest blast radius, opt-in, one file changed outside `policy/huggingface.py`, no existing test touched. Implementation details — renderer is constructed lazily on first `reset` to keep headless CI workers happy, size is a second kwarg (`render_size: tuple[int, int] = (224, 224)`), camera is the existing `"main"` camera. The runner is untouched; it simply passes through whatever the env returns.

That's the only core change. Everything else lives under `src/gauntlet/policy/huggingface.py`.

## 6. Test plan

### Unit tests (run in default `pytest` job — `@pytest.mark.hf` skipped)

1. **Import guard**: simulate `torch` absence (monkeypatch `sys.modules`) and assert `HuggingFacePolicy(...)` raises `ImportError` whose message contains `uv sync --extra hf`.
2. **Re-export guard**: `from gauntlet.policy import HuggingFacePolicy` with torch missing raises a clean `ImportError` at attribute-access time, not module-import time (so `from gauntlet.policy import RandomPolicy` still works).

### HF tests (run in `hf-tests` CI job — `@pytest.mark.hf`, under `tests/hf/`)

3. **Constructor contract (mocked weights)**: `monkeypatch.setattr("transformers.AutoModelForVision2Seq.from_pretrained", fake_loader)` + same for `AutoProcessor.from_pretrained`. `fake_loader` returns a `MagicMock` whose `.predict_action(**kw)` returns `np.zeros(7, dtype=np.float32)`. Assert the adapter forwards `repo_id`, `trust_remote_code=True`, and any extra `model_kwargs` into the loader. **Mock at the `from_pretrained` seam — do not try to mock `predict_action` on a real instance; you'll fight `trust_remote_code`.**
4. **`act()` shape/dtype**: feed a synthetic `{"image": np.zeros((224, 224, 3), dtype=np.uint8)}` obs; assert returned action has shape `(7,)` and dtype `np.float64`.
5. **Prompt templating**: with a spy processor mock, assert the processor was called with the literal string `"In: What action should the robot take to grasp the cube?\nOut:"` for `instruction="grasp the cube"`.
6. **Image validation**: pass a float array → `ValueError`. Pass shape `(224, 224)` → `ValueError`. Pass missing key → `KeyError` naming the configured `image_obs_key`.
7. **Protocol conformance**: `assert isinstance(HuggingFacePolicy(...), Policy)` and `ResettablePolicy` (mirrors the `tests/test_policy.py` style).
8. **`TabletopEnv(render_in_obs=True)`**: `reset()` returns an obs that contains `"image"` with dtype `uint8` and shape `(render_size_h, render_size_w, 3)`. `render_in_obs=False` (default) leaves the obs dict byte-identical to Phase 1 — enforced with a key-set equality check.

### Explicitly out-of-scope for this task

- **Real-weight smoke test**: `openvla/openvla-7b` is ~15 GB; no CI download. A manual `examples/evaluate_openvla.py` (RFC checklist item, not a test) lets maintainers verify end-to-end on a GPU box.
- **Action-value regression tests**: we don't pin numerical outputs from a mocked model. Real-weight determinism is a separate concern.

## 7. Open questions

Things the implementation agent must resolve at code time; each has a reasonable default in parentheses so the implementer is not blocked.

- **Action unnormalisation → TabletopEnv scaling.** OpenVLA's `unnorm_key="bridge_orig"` returns actions in BridgeData V2's world-frame-delta scale (metres). `TabletopEnv` expects `[-1, 1]`-bounded per-step twist commands scaled by `MAX_LINEAR_STEP = 0.05 m`. Does the adapter (a) clip, (b) rescale by dividing by `MAX_LINEAR_STEP`, or (c) pass through and let the env clip? (**Default: (c) pass-through with a warning when any coordinate exceeds `[-1, 1]` — matches "don't hide failures" from §6.**)
- **Gripper convention.** OpenVLA emits `gripper ∈ [0, 1]` (0 = open, 1 = close). `TabletopEnv` takes `+1 = open / -1 = close` with a snap. Map via `action[6] = 1.0 - 2.0 * action[6]`? (**Default: yes, documented in the docstring; flagged for real-weight validation.**)
- **`flash_attn` story.** Users on CPU-only machines (and CI) must not hit a `flash_attn` import error. Leave it out of the extras set (as this RFC does) and let advanced users pass `model_kwargs={"attn_implementation": "flash_attention_2"}` when they've pip-installed `flash-attn` themselves. Is that friction acceptable or do we add an `[hf-gpu]` extra? (**Default: friction is fine for MVP; revisit when a real user complains.**)
- **`render_size` default.** 224×224 matches the OpenVLA processor's internal resize and avoids a double-resize. Do we hardcode that or make it a top-level env kwarg? (**Default: kwarg with `(224, 224)` default on `TabletopEnv`.**)
- **Registry entry for `gauntlet run --policy openvla`.** The CLI resolver in `policy/registry.py` accepts `"random"`, `"scripted"`, or `"module.path:attr"`. Do we add an `"openvla"` shortcut, or force users through `module.path:attr`? (**Default: no shortcut in this RFC — the shortcut requires baking in a `repo_id` and `instruction`, which are user-specific. Users write a three-line factory in their own module.**)
- **Multi-task / per-episode instructions.** Not in scope here, but the natural extension point is an `EpisodeContext` object threaded through `env.reset(...) -> (obs, info)` with `info["instruction"]`. Flag for a future RFC, do not design for it now.

## 8. Rough implementation checklist

Sized to land as one §9-style PR. Each bullet maps to one commit.

1. Add `[project.optional-dependencies] hf = [...]`, `[dependency-groups] hf-dev`, the mypy override, and the `hf` pytest marker to `pyproject.toml`. Regenerate `uv.lock` with `uv lock`.
2. Add `TabletopEnv.__init__(..., render_in_obs: bool = False, render_size: tuple[int, int] = (224, 224))`; extend `observation_space` and `_build_obs` conditionally. Ensure the offscreen `mujoco.Renderer` is lazy (first `reset` call).
3. Write `src/gauntlet/policy/huggingface.py` with the class sketched in §4: lazy imports guarded by the `_HF_INSTALL_HINT` ImportError, string annotations for torch/PIL, `act` + `reset`, `_to_pil`, `_build_prompt`, `_prep_inputs`.
4. Wire `HuggingFacePolicy` into `src/gauntlet/policy/__init__.py` behind a try/except ImportError re-export that preserves `from gauntlet.policy import RandomPolicy` on torch-free installs.
5. Write `tests/hf/test_huggingface_policy.py` covering the eight cases in §6; mark them all `@pytest.mark.hf`. Add `tests/test_env.py` cases for `render_in_obs=True/False` (unmarked — these don't need torch).
6. Add a `hf-tests` job to `.github/workflows/ci.yml`: `uv sync --extra hf --group hf-dev` → `uv run pytest -m hf`. Confirm the default job still runs `pytest -m 'not hf'`.
7. Add `examples/evaluate_openvla.py` (docstring + `if __name__ == "__main__":`) showing the ≤20-line wiring. Reference it from the README under a new "Using a real VLA" section.
8. Update `README.md` Quickstart: one extra bullet on `uv sync --extra hf` and a pointer to the example.
9. Run the full gate locally: `uv run ruff check && uv run mypy && uv run pytest -m 'not hf'` AND `uv sync --extra hf --group hf-dev && uv run pytest -m hf`. Both must be green before opening the PR.
10. PR description explicitly links this RFC and calls out the `TabletopEnv(render_in_obs=...)` kwarg as the one core-side change.
