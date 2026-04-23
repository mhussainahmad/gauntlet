# Phase 2 Task 2 Reuse Audit: LeRobotPolicy vs. HuggingFacePolicy

This audit examines what Task 1 (HuggingFacePolicy for OpenVLA) built that Task 2 (LeRobotPolicy for SmolVLA) can directly reuse, and what must be forked or added.

---

## 1. Infrastructure Task 1 Built That Task 2 Gets for Free

### 1.1 `pyproject.toml` — Optional Dependencies

**Location:** `/home/hussain/dev/gauntlet/pyproject.toml:35–47` and `99–126`

Task 1 added the `[project.optional-dependencies] hf` group:

```toml
[project.optional-dependencies]
hf = [
    "torch>=2.2,<3",
    "transformers>=4.40,<5",
    "timm>=0.9.10,<2",
    "tokenizers>=0.19,<1",
    "pillow>=10.0,<12",
]
```

And the dependency group for CI:

```toml
[dependency-groups]
hf-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]
```

**Mypy overrides** (lines 124–126) to allow torch/transformers/PIL imports without breaking torch-free builds:

```toml
[[tool.mypy.overrides]]
module = ["torch", "torch.*", "transformers", "transformers.*", "PIL", "PIL.*"]
ignore_missing_imports = true
```

**Pytest marker** (lines 138–140):

```toml
markers = [
    "hf: tests that require the [hf] extra (torch, transformers, PIL)",
]
```

**Reuse for Task 2:** The `hf` extras group can absorb `lerobot` dependencies (which also depend on torch). No new extras group needed. Mypy overrides already cover the imports. The `hf` pytest marker can gate both OpenVLA and SmolVLA tests. **✓ Directly reusable.**

---

### 1.2 `.github/workflows/ci.yml` — CI Job Structure

**Location:** `/home/hussain/dev/gauntlet/.github/workflows/ci.yml:13–79`

Task 1 established a two-job split:

- **`lint-typecheck-test` job (lines 14–50):** Torch-free default. Runs `pytest -m "not hf"` to enforce no torch in the core.
- **`hf-tests` job (lines 52–79):** With `--extra hf --group hf-dev`. Runs `pytest -m hf`.

**Reuse for Task 2:** The structure trivially scales. A third job (e.g., `lerobot-tests`) would follow the same pattern:

```yaml
lerobot-tests:
  name: lerobot tests (py${{ matrix.python-version }})
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with:
        enable-cache: true
        cache-dependency-glob: "uv.lock"
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    - name: Sync dependencies (with lerobot extra + lerobot-dev group)
      run: uv sync --extra lerobot --group lerobot-dev --python ${{ matrix.python-version }}
    - name: Run lerobot tests
      run: uv run pytest -m lerobot
```

**✓ No restructuring needed.** Existing job definitions are a blueprint for new loaders.

---

### 1.3 `src/gauntlet/policy/__init__.py` — Lazy Re-export Pattern

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/policy/__init__.py:36–57`

The `__getattr__` pattern for `HuggingFacePolicy`:

```python
def __getattr__(name: str) -> Any:
    """Lazily expose ``HuggingFacePolicy`` without importing torch on package load."""
    if name == "HuggingFacePolicy":
        from gauntlet.policy.huggingface import _HF_INSTALL_HINT
        try:
            import torch
        except ImportError as exc:
            raise ImportError(_HF_INSTALL_HINT) from exc
        from gauntlet.policy.huggingface import HuggingFacePolicy
        return HuggingFacePolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Reuse for Task 2:** Adding a second deferred symbol scales trivially:

```python
def __getattr__(name: str) -> Any:
    if name == "HuggingFacePolicy":
        # ... existing HF logic ...
    elif name == "LeRobotPolicy":
        from gauntlet.policy.lerobot import _LEROBOT_INSTALL_HINT
        try:
            import torch
        except ImportError as exc:
            raise ImportError(_LEROBOT_INSTALL_HINT) from exc
        from gauntlet.policy.lerobot import LeRobotPolicy
        return LeRobotPolicy
    raise AttributeError(...)
```

Add `"LeRobotPolicy"` to `__all__` and a conditional TYPE_CHECKING import. **✓ Pattern directly reusable with minimal boilerplate.**

---

### 1.4 `src/gauntlet/env/tabletop.py` — `render_in_obs` Kwarg

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/env/tabletop.py:152–159` and `633–650`

Task 1 added a `render_in_obs: bool` constructor parameter (default `False`) and a `render_size: tuple[int, int] = (224, 224)` parameter. When `render_in_obs=True`, the observation dict includes an `"image"` key:

```python
if self._render_in_obs:
    h, w = self._render_size
    obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
```

Rendering happens in `_render_obs_image()` (lines 637–650), which returns `NDArray[np.uint8]` of shape `(h, w, 3)`.

**Reuse for Task 2:** SmolVLA uses the LeRobot processor pipeline. The exact image resolution requirement depends on SmolVLA's `VisionTransformer` backbone. OpenVLA's OpenAI CLIP-based processor expects 224×224; SmolVLA may differ. **The kwarg already supports arbitrary resolutions via `render_size` parameter. No env change needed.** However, Task 2 should validate that its SmolVLA loader works with TabletopEnv's uint8 RGB output without resampling if possible.

**Note:** No "robot state" (joint positions, velocities) is exposed separately in observations. Only geometric pose + gripper state + rendered image. **Flag for Task 2:** If SmolVLA requires proprioceptive input (e.g., joint positions), a new env kwarg will be needed.

**✓ Directly reusable; no changes required.**

---

### 1.5 `tests/test_import_guards.py` — Torch-Free Guard Tests

**Location:** `/home/hussain/dev/gauntlet/tests/test_import_guards.py:25–70`

Task 1 added unmarked tests (run in both torch-free default job and hf-tests job) that:

1. Verify importing `gauntlet.policy` does not import torch (lines 26–47).
2. Verify accessing `HuggingFacePolicy` via `__getattr__` raises `ImportError` with the install hint when torch is absent (lines 49–70).

These are **unmarked** — they run in both CI jobs. The torch-absence check is enforced by `monkeypatch.setitem(sys.modules, "torch", None)` so the same code path executes even on machines with torch installed.

**Reuse for Task 2:** The pattern is fully reusable. A new test method can follow the same structure:

```python
def test_lerobot_policy_guard_raises_install_hint_when_torch_missing(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    import gauntlet.policy.lerobot as lr_mod
    try:
        importlib.reload(lr_mod)
        with pytest.raises(ImportError, match="uv sync --extra lerobot"):
            lr_mod.LeRobotPolicy(repo_id="dummy/repo", instruction="...")
    finally:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        importlib.reload(lr_mod)
```

**Not OpenVLA-specific.** The test pattern is protocol-level. **✓ Directly reusable.**

---

## 2. What Task 1 Did That Task 2 Probably CAN'T Reuse

### 2.1 `HuggingFacePolicy` Constructor Design

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/policy/huggingface.py:105–165`

Task 1's loader hardcodes:

- `AutoModelForVision2Seq.from_pretrained(repo_id, **model_kwargs)` (line 164)
- `AutoProcessor.from_pretrained(repo_id, **proc_kwargs)` (line 163)
- `trust_remote_code=True` forced on both loaders (lines 150, 153)

SmolVLA's loading API is fundamentally different. Per the spec, SmolVLA requires:

```python
from lerobot.policies.smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained(repo_id)
```

**Can task 2 subclass HuggingFacePolicy?** Unlikely to be cleaner. The loader logic is hardwired into `__init__`, and SmolVLA returns a different object shape (likely with different processor / model attribute names and methods). A sibling class (`LeRobotPolicy`) is cleaner than a `HuggingFace` base class.

**✓ Task 2 should fork: create a new sibling `LeRobotPolicy` class in `src/gauntlet/policy/lerobot.py`.**

---

### 2.2 Prompt Template Constant

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/policy/huggingface.py:45–46`

```python
_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
```

This is OpenVLA-specific. SmolVLA's model card and processing pipeline will define its own prompt format (if any). **Task 2 will need its own constant or factory.**

**✓ Not reusable.**

---

### 2.3 Action-Adaptation Defaults

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/policy/huggingface.py:195–218`

Two conversions are hardcoded:

1. **Gripper convention flip** (line 201): `action[6] = 1.0 - 2.0 * action[6]`
   - OpenVLA emits `[0, 1]` (0 = open, 1 = close).
   - TabletopEnv expects `[+1 open, -1 close]`.

2. **OOB twist warning** (lines 208–218):
   - OpenVLA's BridgeData V2 unnorm_key emits world-frame metre deltas that can exceed `[-1, 1]`.
   - Pass through with a `RuntimeWarning` rather than silently clipping.

**Are these general or loader-specific?**

- **Gripper flip:** Unknown until SmolVLA is trained. SmolVLA's gripper convention may differ. This is likely a per-loader choice.
- **Twist warning:** This is general good practice for any VLA — expose OOB actions rather than hiding them. But SmolVLA may emit differently scaled actions (e.g., already in `[-1, 1]`), so the warning threshold may differ.

**Recommendation for Task 2's RFC author:** Check SmolVLA's model card for its gripper and action conventions. If they differ from OpenVLA, define new adaptation defaults in `LeRobotPolicy.act()`. If they match (unlikely), consider extraction.

**✓ Not reusable as-is; task 2 must validate and potentially fork the logic.**

---

## 3. Shared Utilities — Candidates for Extraction

Task 1 defined several helper functions in `huggingface.py` that future loaders might share. **Do NOT propose extraction as done deals — the RFC-002 author decides.** Below are candidates:

### Genuinely Protocol-Level

1. **`_to_pil()` helper** (lines 249–265)
   - Validates `uint8 (H, W, 3)` and wraps in `PIL.Image`.
   - **Protocol-level:** Any vision-based loader that accepts PIL images could reuse this.
   - **Line range:** 249–265

2. **`_prep_inputs()` helper** (lines 236–247)
   - Runs processor and moves tensors to device+dtype.
   - **Generic:** Applies to any torch-based model loader, not OpenVLA-specific.
   - **Line range:** 236–247

3. **Install hint text** (lines 38–43)
   - `_HF_INSTALL_HINT = "HuggingFacePolicy requires..."`.
   - **Protocol-level:** Every optional-dependency loader needs an install hint.
   - **Line range:** 38–43

### Loader-Specific

1. **`_build_prompt()` method** (lines 232–234)
   - Applies the OpenVLA template. SmolVLA will have a different prompt or none at all.
   - **Not reusable.**

2. **Dtype mapping** (lines 132–146)
   - Maps `"float32"` → `torch.float32`, etc.
   - **Could be shared,** but it's simple enough to duplicate.

### Summary

**Candidates for extraction (if RFC-002 decides):**
- Image validation (`_to_pil`)
- Tensor preparation (`_prep_inputs`)
- Install hint string pattern

**Cannot extract:**
- Prompt template (loader-specific)
- Action adaptation logic (needs per-loader validation)

---

## 4. Observation Contract for VLAs on TabletopEnv

### 4.1 Default Observation Keys (render_in_obs=False)

**Location:** `/home/hussain/dev/gauntlet/tests/test_env.py` (implicit) and `src/gauntlet/env/tabletop.py:621–635`

When `render_in_obs=False` (Phase 1 default, unchanged):

```python
obs = {
    "cube_pos": NDArray[np.float64] shape (3,),
    "cube_quat": NDArray[np.float64] shape (4,),  # wxyz order
    "ee_pos": NDArray[np.float64] shape (3,),
    "gripper": NDArray[np.float64] shape (1,),
    "target_pos": NDArray[np.float64] shape (3,),
}
```

### 4.2 Observation Keys with Rendering (render_in_obs=True)

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/env/tabletop.py:633–634` and test verification in `tests/test_env.py`

When `render_in_obs=True` (Task 1's opt-in flag):

```python
obs = {
    "cube_pos": NDArray[np.float64] shape (3,),
    "cube_quat": NDArray[np.float64] shape (4,),
    "ee_pos": NDArray[np.float64] shape (3,),
    "gripper": NDArray[np.float64] shape (1,),
    "target_pos": NDArray[np.float64] shape (3,),
    "image": NDArray[np.uint8] shape (render_size[0], render_size[1], 3),  # default (224, 224, 3)
}
```

**Image dtype and shape:** uint8 RGB, shape `(H, W, 3)` where H, W default to 224. Task 1 specifically chose 224×224 to match OpenVLA's processor internal resize (see RFC §7).

### 4.3 Robot State in Observations

**Current observation structure includes:**
- `cube_pos` — target object position (3D Cartesian)
- `cube_quat` — target object orientation (quaternion)
- `ee_pos` — end-effector position (3D Cartesian, mocap body)
- `gripper` — gripper state (scalar, snap value)
- `target_pos` — goal position (3D Cartesian)
- `image` — rendered view (opt-in)

**What is NOT included:**
- **Joint positions / angles** — There are no actuated joints. The EE is a mocap body (kinematically free).
- **Joint velocities** — No joint DOFs to report.
- **Generalized velocities** — Not surfaced.

**Implication for Task 2:** SmolVLA may or may not expect proprioceptive (joint state) input alongside the image. If SmolVLA's processor expects a flat joint-position vector or proprioceptive context, **Task 2 must add a new `include_proprioceptive: bool` or similar env kwarg** to optionally pack joint state into observations. Current env does not expose this.

**✓ Flag for Task 2:** Validate SmolVLA's processor signature. If it expects `(image, proprioceptive_state)`, design a kwarg to surface it.

---

## 5. Runner / Picklability Contract

**No changes since Phase 1.** Task 1 (HuggingFacePolicy) was implemented without modifying `src/gauntlet/runner/`.

**Location check:** `/home/hussain/dev/gauntlet/src/gauntlet/runner/runner.py` and `worker.py` — no git changes related to Task 1.

**Policy instantiation contract:** The runner calls `policy_factory()` — a zero-arg callable stashed in worker globals — to build a fresh policy per episode. It never pickles the policy itself; the factory is pickled. Both `HuggingFacePolicy(repo_id=..., instruction=...)` and (by analogy) `LeRobotPolicy(repo_id=..., instruction=...)` constructors are zero-arg via a user-provided factory function.

**Example factory for HuggingFacePolicy:**

```python
def hf_factory():
    from gauntlet.policy import HuggingFacePolicy
    return HuggingFacePolicy(repo_id="openvla/openvla-7b", instruction="pick up the red cube")
```

**LeRobotPolicy will follow the same pattern.** No runner changes needed.

**✓ Picklability contract unchanged. Task 2 inherits the pattern.**

---

## Summary

### Task 2 Inherits (No Work)

- **`pyproject.toml` extras/dev-groups/mypy/pytest structure:** The `hf` extras group absorbs lerobot deps. No new extras group needed.
- **CI job template:** Existing two-job split scales to three (torch-free default, `hf-tests`, and optionally `lerobot-tests`).
- **Lazy re-export pattern in `policy/__init__.py`:** `__getattr__` scales to two or more deferred imports with minimal boilerplate.
- **`TabletopEnv.render_in_obs` kwarg:** Produces uint8 (H, W, 3) image suitable for any vision-based loader. No env changes needed for SmolVLA unless it requires proprioceptive input.
- **Import guard test pattern:** Unmarked tests for torch-free contract are reusable verbatim.
- **Runner picklability contract:** Factory-based policy instantiation. No runner changes needed.

### Task 2 Must Add (Net New)

- **`src/gauntlet/policy/lerobot.py`:** New sibling loader class (`LeRobotPolicy`) with SmolVLA-specific `from_pretrained` logic.
- **SmolVLA prompt template constant:** If SmolVLA requires a prompt, define it in `lerobot.py`.
- **SmolVLA action-adaptation logic:** Validate gripper convention and action scale; implement per SmolVLA's model card.
- **`[dependency-groups] lerobot-dev`:** If SmolVLA has lerobot-specific test deps (unlikely; torch/PIL are already in `hf`).
- **`pytest.ini` marker for SmolVLA tests:** If tests are lerobot-specific (e.g., `@pytest.mark.lerobot`), add to `pyproject.toml`. Otherwise, reuse `@pytest.mark.hf`.
- **Observation validation for proprioceptive input:** If SmolVLA requires joint state, add a new `TabletopEnv` kwarg (e.g., `include_proprioceptive`) and surface it in observations.
- **Task 2 RFC-002 section on action conventions:** Document SmolVLA's gripper and twist conventions vs. TabletopEnv.

### Task 2 Should Consider Refactoring (Optional)

- **Extract `_to_pil()` image validation helper** to `src/gauntlet/policy/image_utils.py` if both HuggingFacePolicy and LeRobotPolicy share uint8 RGB validation.
- **Extract install-hint pattern** to a factory function (e.g., `make_install_hint(extra_name, packages_list)`) so both loaders avoid duplicating the message.
- **Extract `_prep_inputs()` / tensor-to-device logic** if SmolVLA's processor also requires moving tensors to device+dtype.

**Rationale:** These extractions are optional — sibling code duplication is acceptable for MVP. Only extract if a third loader (e.g., Diffusion Policy) lands and the pattern becomes clear.

---

## Three-Sentence Summary

**What Task 2 gets for free:** Task 1's infrastructure (extras group, CI split, lazy re-export pattern, `render_in_obs` kwarg, import guards) is directly reusable; Task 2 adds a sibling `LeRobotPolicy` class without modifying core systems. **What must be net new:** SmolVLA's proprietary loader API (`lerobot.policies.smolvla.SmolVLAPolicy.from_pretrained`), prompt template, and action-adaptation defaults differ from OpenVLA; Task 2 must implement these from scratch and validate against SmolVLA's model card. **Shared base class?** Premature. Both classes inherit the `Policy` protocol and factory pattern, but the loaders are distinct enough (different `from_pretrained` APIs, prompt formats, action conventions) that a shared base would force either over-generalization or conditional branches. Sibling classes with common test infrastructure (import guards, pytest markers) is cleaner for MVP; revisit if a third loader lands.

