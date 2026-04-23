# Phase 2 Exploration: Policy Interface Touch-Point Map for HuggingFacePolicy

**Date:** 2026-04-22  
**Task:** Exhaustive touch-point map for adding a new Policy implementation (HuggingFacePolicy) that wraps VLA models (OpenVLA, SmolVLA).  
**Scope:** Read-only analysis. No code written.

---

## 1. The Policy Protocol Itself

### Location: `src/gauntlet/policy/base.py`

**Core Protocol Definition:**
```python
# File: src/gauntlet/policy/base.py:39–50
@runtime_checkable
class Policy(Protocol):
    """Minimal policy adapter.

    A policy maps a (possibly multi-modal) observation to an action vector.
    Policies may be stateful internally; stateful policies should also
    satisfy :class:`ResettablePolicy`.
    """

    def act(self, obs: Observation) -> Action:
        """Return an action for the current observation."""
        ...
```

**ResettablePolicy (Optional, for Stateful Policies):**
```python
# File: src/gauntlet/policy/base.py:54–68
@runtime_checkable
class ResettablePolicy(Protocol):
    """Policy with per-episode reset.

    The runner calls :meth:`reset` at the start of each episode, passing the
    episode's deterministic RNG so stochastic policies can re-seed and
    scripted policies can rewind their step counter.
    """

    def act(self, obs: Observation) -> Action:
        """Return an action for the current observation."""
        ...

    def reset(self, rng: np.random.Generator) -> None:
        """Reset per-episode state using the supplied RNG."""
        ...
```

### Type Aliases

**Observation:**
```python
# File: src/gauntlet/policy/base.py:32
Observation: TypeAlias = Mapping[str, NDArray[Any]]
```
- A dict-like mapping from string keys to numpy arrays of any dtype.
- Bridges the MuJoCo/gymnasium FFI boundary and carries mixed dtypes (uint8 images, float32 proprio).
- Per spec §6, `Any` inside `NDArray` is **permitted at FFI boundaries only**.

**Action:**
```python
# File: src/gauntlet/policy/base.py:36
Action: TypeAlias = NDArray[np.float64]
```
- Continuous control vector, shape `(action_dim,)`.
- Standardized on `float64` for direct MuJoCo control buffer compatibility (`mjData.ctrl`).

### Existing Implementations

#### RandomPolicy
**File:** `src/gauntlet/policy/random.py`

**Constructor:**
```python
# File: src/gauntlet/policy/random.py:20–37
def __init__(
    self,
    action_dim: int,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
    seed: int | None = None,
) -> None:
```

**Side Effects in `act()`:**
```python
# File: src/gauntlet/policy/random.py:39–46
def act(self, obs: Observation) -> Action:
    del obs  # random policy ignores observations by design
    sample = self._rng.uniform(
        low=self.action_low,
        high=self.action_high,
        size=self.action_dim,
    )
    return np.asarray(sample, dtype=np.float64)
```
- **Advances internal RNG state** on each call.
- **Reset behavior:** `reset(rng)` at line 48–50 adopts the runner's episode RNG.

#### ScriptedPolicy
**File:** `src/gauntlet/policy/scripted.py`

**Constructor:**
```python
# File: src/gauntlet/policy/scripted.py:46–63
def __init__(
    self,
    trajectory: NDArray[np.float64] | None = None,
    *,
    loop: bool = False,
) -> None:
```

**Side Effects in `act()`:**
```python
# File: src/gauntlet/policy/scripted.py:73–79
def act(self, obs: Observation) -> Action:
    del obs  # open-loop: scripted policy does not read observations
    idx = self._step % self.length if self.loop else min(self._step, self.length - 1)
    self._step += 1
    return cast("Action", self._trajectory[idx].copy())
```
- **Advances internal step counter** on each call.
- **Reset behavior:** `reset()` at line 81–84 rewinds `_step` to 0.

### Policy Registry & Factory Pattern

**File:** `src/gauntlet/policy/registry.py`

**Main Entry Point:**
```python
# File: src/gauntlet/policy/registry.py:90–118
def resolve_policy_factory(spec: str) -> Callable[[], Policy]:
    """Turn a ``--policy`` CLI string into a zero-arg policy factory.

    Args:
        spec: One of ``"random"``, ``"scripted"``, or ``"module.path:attr"``.

    Returns:
        Zero-arg callable that returns a fresh :class:`Policy` on each
        call. Picklable under :mod:`multiprocessing` ``spawn`` so the
        same factory works for ``n_workers == 1`` and ``n_workers >= 2``.

    Raises:
        PolicySpecError: If ``spec`` is empty, has the wrong shape, or
            references an unimportable module / missing attribute.
    """
    if not spec or not spec.strip():
        raise PolicySpecError("policy spec must be a non-empty string")
    spec = spec.strip()
    if spec == "random":
        # ``partial`` over a class pickles cleanly; a lambda does not.
        return partial(RandomPolicy, action_dim=_DEFAULT_ACTION_DIM)
    if spec == "scripted":
        # The class is itself a zero-arg callable (all kwargs default).
        return ScriptedPolicy
    if ":" in spec:
        return _resolve_module_attr(spec)
    raise PolicySpecError(
        f"unknown policy spec {spec!r}: expected 'random', 'scripted', or 'module.path:attr'"
    )
```

**Registry Acceptance Criteria:**
- Built-in shortcuts: `"random"` → `partial(RandomPolicy, action_dim=7)`; `"scripted"` → `ScriptedPolicy`.
- Custom module form: `"module.path:attr"` resolves via `importlib.import_module()` + `getattr()`.
- **No explicit registry dict.** Dispatch is by string pattern matching, then dynamic import.
- **Custom policy must be a zero-arg callable** that returns a `Policy` instance.

**Picklability Requirement (CRITICAL):**
```python
# File: src/gauntlet/policy/registry.py:5–18
"""
The resolved factory is intentionally a top-level callable
(:class:`functools.partial` over a class, or the class itself, or the
imported attribute). Lambdas / closures would not pickle under the
``spawn`` start method that :class:`gauntlet.runner.Runner` requires for
``n_workers >= 2``; using ``partial`` keeps the parallel path alive even
though the unit tests only exercise ``-w 1``.
"""
```
- **No lambdas, no closures.** Factories must be picklable module-level functions or classes.
- `functools.partial` is acceptable.
- This is enforced at the `spawn` multiprocessing boundary.

---

## 2. How the Runner Consumes a Policy

### File: `src/gauntlet/runner/runner.py`

**Runner Construction:**
```python
# File: src/gauntlet/runner/runner.py:95–134
def __init__(
    self,
    *,
    n_workers: int = 1,
    env_factory: Callable[[], TabletopEnv] | None = None,
    start_method: str = "spawn",
) -> None:
```
- Policy is **not** constructed in the Runner; it's passed in as a factory.

**Runner.run() Entry Point:**
```python
# File: src/gauntlet/runner/runner.py:140–182
def run(
    self,
    *,
    policy_factory: Callable[[], Policy],
    suite: Suite,
) -> list[Episode]:
    """Execute every (cell x episode) rollout.

    Args:
        policy_factory: Zero-arg callable returning a fresh
            :class:`Policy`. Each worker calls it exactly once per
            episode it handles.
```
- Policy factory is called **once per episode** (not once per cell, not once per worker).
- Each worker instantiates a fresh policy via `policy_factory()` for every episode it handles.

### File: `src/gauntlet/runner/worker.py`

**Core Execution Pipeline (_execute_one):**
```python
# File: src/gauntlet/runner/worker.py:168–225
def _execute_one(env: TabletopEnv, policy_factory: Callable[[], Policy], item: WorkItem) -> Episode:
    """Drive one (cell, episode) rollout to completion.
    
    Pipeline (mirrors Pin 3):

    1. ``env.restore_baseline()`` wipes any model mutation from the
       previous episode handled by this worker.
    2. Apply every queued ``perturbation_value`` via
       :meth:`TabletopEnv.set_perturbation`.
    3. Build a fresh ``policy`` via ``policy_factory()``.
    4. Build the policy RNG from ``item.episode_seq`` (decorrelated from
       the env stream but reproducible from the same SeedSequence node).
    5. ``env.reset(seed=...)`` with the derived uint32 env seed.
    6. If the policy is :class:`ResettablePolicy`, hand it the RNG.
    7. Step until terminated / truncated, accumulating reward.
    """
    env.restore_baseline()
    for name, value in item.perturbation_values.items():
        env.set_perturbation(name, value)

    policy = policy_factory()
    policy_rng = np.random.default_rng(item.episode_seq)
    env_seed = extract_env_seed(item.episode_seq)

    obs, _ = env.reset(seed=env_seed)
    if isinstance(policy, ResettablePolicy):
        policy.reset(policy_rng)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
```

**Key Call Sites for `policy.act()`:**
- Line 207: `action = policy.act(obs)` — exact call where observation is passed.
- Observation is the dict returned by `env.reset()` or `env.step()`.
- Action must be `NDArray[np.float64]` with shape `(7,)` for the tabletop env.

**Policy Reset Contract:**
```python
# File: src/gauntlet/runner/worker.py:198–199
if isinstance(policy, ResettablePolicy):
    policy.reset(policy_rng)
```
- Runtime duck-type check via `isinstance(..., ResettablePolicy)`.
- Protocol is `@runtime_checkable` so the check works without importing the concrete class.
- **Only called once per episode, after `env.reset()` and before the first `act()` call.**
- RNG is `np.random.default_rng(item.episode_seq)` — decorrelated from the env seed but deterministic.

### Worker Initialization & Multiprocessing Contract

**Worker State:**
```python
# File: src/gauntlet/runner/worker.py:114–149
@dataclass
class WorkerInitArgs:
    """Init args for :func:`pool_initializer`.

    Both fields are zero-arg callables to side-step the pickle boundary:

    * ``env_factory`` — builds a :class:`TabletopEnv`. The MjModel inside
      is not picklable, so the env must be born inside the worker.
    * ``policy_factory`` — builds a fresh :class:`Policy` per episode.
      Stashed once and called per-item; future torch/GPU policies that
      cannot be pickled (e.g. HuggingFacePolicy) can therefore live
      entirely inside the worker.
    """

    env_factory: Callable[[], TabletopEnv]
    policy_factory: Callable[[], Policy]
```

**Pool Initializer:**
```python
# File: src/gauntlet/runner/worker.py:139–148
def pool_initializer(args: WorkerInitArgs) -> None:
    """Run once per worker process at pool startup.

    Loads the MJCF (one MjModel per worker) and caches the policy
    factory. Subsequent items reuse the env via
    :meth:`TabletopEnv.restore_baseline`; the model is never re-loaded.
    """
    env = args.env_factory()
    _WORKER_STATE["env"] = env
    _WORKER_STATE["policy_factory"] = args.policy_factory
```

**Pickling Semantics — CRITICAL:**
- The **policy factory itself must pickle**, not the policy instance.
- The factory is cached in the worker's module-level `_WORKER_STATE` dict.
- **Each policy is instantiated fresh inside the worker, once per episode.**
- This design explicitly accommodates policies with non-picklable internal state (e.g., torch models, GPU memory).

---

## 3. The Observation Shape the Environment Emits

### File: `src/gauntlet/env/tabletop.py`

**Observation Space Definition:**
```python
# File: src/gauntlet/env/tabletop.py:206–215
self.observation_space: spaces.Dict = spaces.Dict(
    {
        "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
        "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
        "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
    }
)
```

**Observation Type (gymnasium Dict):**
- Type alias: `_ObsType = dict[str, NDArray[Any]]` (line 93).
- Keys: `"cube_pos"`, `"cube_quat"`, `"ee_pos"`, `"gripper"`, `"target_pos"`.
- All values are `float64` numpy arrays.

**Observation Content Returned by `_build_obs()`:**
```python
# File: src/gauntlet/env/tabletop.py:592–603
def _build_obs(self) -> dict[str, NDArray[np.float64]]:
    cube_pos = np.array(self._data.xpos[self._cube_body_id], dtype=np.float64)
    cube_quat = np.array(self._data.xquat[self._cube_body_id], dtype=np.float64)
    ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
    gripper = np.array([self._gripper_state], dtype=np.float64)
    return {
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "ee_pos": ee_pos,
        "gripper": gripper,
        "target_pos": self._target_pos.copy(),
    }
```

**No image component.** Phase 1 uses proprioceptive state only (XYZ positions, quaternions, gripper state).

### Action Space

**Action Space Definition:**
```python
# File: src/gauntlet/env/tabletop.py:206
self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
```

**Action Layout:**
```python
# File: src/gauntlet/env/tabletop.py:14–21
"""
Action layout (``shape=(7,), dtype=float64, bounds=[-1, 1]``) matches
:data:`gauntlet.policy.scripted.DEFAULT_PICK_AND_PLACE_TRAJECTORY`::

    [dx, dy, dz, drx, dry, drz, gripper]

The first six entries are per-step end-effector twist commands scaled by
:attr:`TabletopEnv.MAX_LINEAR_STEP` / :attr:`TabletopEnv.MAX_ANGULAR_STEP`.
The gripper command snaps to a binary ``open (+1) / closed (-1)`` state
(snap, not ramp — chosen for determinism and simplicity).
"""
```

---

## 4. CLI Wiring

### File: `src/gauntlet/cli.py`

**`run` Subcommand:**
```python
# File: src/gauntlet/cli.py:159–218
@app.command("run")
def run(
    suite_path: Annotated[Path, typer.Argument(...)],
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Policy spec: 'random', 'scripted', or 'module.path:attr'.",
        ),
    ],
    out: Annotated[Path, typer.Option(...)],
    ...
) -> None:
    """Execute a suite and write episodes / report artefacts to ``--out``."""
```

**Policy Argument Resolution:**
```python
# File: src/gauntlet/cli.py:234–237
try:
    policy_factory = resolve_policy_factory(policy)
except PolicySpecError as exc:
    raise _fail(str(exc)) from exc
```
- Direct delegation to `resolve_policy_factory(policy)` from `gauntlet.policy.registry`.
- On error, wraps `PolicySpecError` (which is a `ValueError` subclass) as a CLI error.

**Acceptance Criterion for New Policy Names:**
1. Use `"random"` or `"scripted"` for built-in shortcuts (no code change needed in `registry.py`).
2. Use `"module.path:attr"` format for custom policies.
   - Example: `gauntlet run suite.yaml --policy my_pkg.policies:hf_vla_factory`
   - The `my_pkg.policies` module must export a zero-arg callable named `hf_vla_factory`.
   - That callable must return a `Policy` (i.e., have an `act(obs)` method).

---

## 5. Test Patterns

### File: `tests/test_policy.py`

**Test Structure:**
```python
# File: tests/test_policy.py:1–75
"""Tests for the policy protocol and reference implementations."""

class TestRandomPolicy:
    def test_instantiates(self) -> None:
    def test_rejects_nonpositive_dim(self) -> None:
    def test_action_has_correct_shape_and_dtype(self) -> None:
    def test_reproducible_with_same_seed(self) -> None:
    def test_satisfies_policy_protocols(self) -> None:
```

**Protocol Conformance Testing:**
```python
# File: tests/test_policy.py:71–74
def test_satisfies_policy_protocols(self) -> None:
    p = RandomPolicy(action_dim=3)
    assert isinstance(p, Policy)
    assert isinstance(p, ResettablePolicy)
```
- Uses duck-type checks via `isinstance()` against the `@runtime_checkable` protocols.

**Observation Fixture:**
```python
# File: tests/test_policy.py:17
_EMPTY_OBS: Observation = {"state": np.zeros(3, dtype=np.float64)}
```
- Simple mock observation dict for policy testing.

### File: `tests/test_runner.py`

**Factory Convention:**
```python
# File: tests/test_runner.py:31–74
# All factories are defined at module scope (must pickle under spawn).

def make_random_policy() -> RandomPolicy:
    """Module-level factory so it pickles under spawn."""
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)

def make_scripted_policy() -> ScriptedPolicy:
    """Default scripted trajectory factory."""
    return ScriptedPolicy()

def _counted_random_factory() -> RandomPolicy:
    """Random-policy factory that records call count in a module global."""
    _FACTORY_CALL_COUNTER[0] += 1
    return RandomPolicy(action_dim=_ACTION_DIM, seed=None)
```

**Integration Test Pattern:**
```python
# File: tests/test_runner.py:146–166
def test_random_policy_integration_smoke() -> None:
    suite = _make_suite(seed=3, episodes_per_cell=1, ...)
    runner = Runner(n_workers=1, env_factory=make_fast_env)
    episodes = runner.run(policy_factory=make_random_policy, suite=suite)

    assert len(episodes) == 2
    for ep in episodes:
        assert isinstance(ep.success, bool)
        assert isinstance(ep.terminated, bool)
        assert isinstance(ep.truncated, bool)
```

**Pickle Testing (Spawn Start Method):**
```python
# File: tests/test_runner.py:350–367
def test_lambdas_fail_loudly_under_spawn() -> None:
    """Documents the pickle contract for the multiworker path.

    Lambdas (and other unpicklable closures) cannot cross the spawn
    process boundary. Any caller that hands the Runner a lambda gets a
    pickling error from multiprocessing — that is the loud failure.
    """

    def local_factory() -> RandomPolicy:
        return RandomPolicy(action_dim=_ACTION_DIM, seed=None)

    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(local_factory)

    pickle.dumps(make_random_policy)  # module-level pickles fine
```

**Optional Dependency Pattern (if needed):**
- File `tests/test_policy.py` does not currently use `pytest.importorskip`.
- But the pattern exists in the broader pytest ecosystem if HuggingFace models are optional.
- Example: `torch = pytest.importorskip("torch")` at module scope gates tests requiring torch.

---

## 6. Dependency Layout

### File: `pyproject.toml`

**Core Dependencies:**
```toml
# File: pyproject.toml:22–31
dependencies = [
    "mujoco>=3.2,<4",
    "gymnasium>=1.0,<2",
    "pydantic>=2.7,<3",
    "typer>=0.12,<1",
    "numpy>=1.26,<3",
    "pandas>=2.2,<3",
    "jinja2>=3.1,<4",
    "pyyaml>=6.0,<7",
]
```
- Core does **NOT** include torch, transformers, or any HF dependencies.
- Per spec §6 design principle: "No `torch` in the core."

**Dev Dependencies:**
```toml
# File: pyproject.toml:40–48
[dependency-groups]
dev = [
    "pytest>=8.2,<9",
    "pytest-cov>=5.0,<7",
    "ruff>=0.6,<1",
    "mypy>=1.11,<2",
    "types-pyyaml>=6.0,<7",
    "pandas-stubs>=2.2,<3",
]
```
- Testing stack. No HF models here either.

**No Optional Dependencies Section (Yet):**
- `pyproject.toml` uses `[dependency-groups]` (uv-style), not `[project.optional-dependencies]`.
- **No HuggingFace or torch pinned anywhere.**
- HuggingFacePolicy users must install extras themselves (future Phase 2 work).

**Mypy Configuration (Strict Mode):**
```toml
# File: pyproject.toml:78–99
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
no_implicit_reexport = true
disallow_untyped_defs = true
disallow_any_generics = true
files = ["src", "tests"]

# MuJoCo bindings expose a large untyped C API surface.
# Per spec §6, Any is permitted at the MuJoCo FFI boundary only.
[[tool.mypy.overrides]]
module = ["mujoco", "mujoco.*"]
ignore_missing_imports = true

# Optional / not-yet-typed deps. Trim as upstream stubs land.
[[tool.mypy.overrides]]
module = ["gymnasium", "gymnasium.*"]
ignore_missing_imports = true
```
- **Strict mode enforced:** `disallow_untyped_defs = true`, `disallow_any_generics = true`.
- FFI boundaries (mujoco) permitted to use `Any`.
- If HF modules lack stubs, they will need an override section.

**Ruff Configuration:**
```toml
# File: pyproject.toml:56–72
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "UP",  # pyupgrade
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RUF", # ruff-specific
]
```
- Import sorting via `isort` (rule `I`).
- No exceptions needed for HF imports; standard rules apply.

---

## Touch-Point Summary

### Minimum Set of Files an Implementation Agent Must Touch

For HuggingFacePolicy to be usable, an agent **must** modify:

1. **Create new file:** `src/gauntlet/policy/huggingface.py`
   - Define `HuggingFacePolicy` class (satisfies `Policy` or `ResettablePolicy` protocol).
   - Optionally: add factory function `make_hf_policy()` at module scope for pickling.

2. **Modify (optional but recommended):** `src/gauntlet/policy/__init__.py`
   - Re-export `HuggingFacePolicy` for public API.
   - Can be used in `"module.path:attr"` form without changes.

3. **Modify (optional):** `pyproject.toml`
   - If adding HF as optional dependencies, add a `[project.optional-dependencies]` section.
   - Example:
     ```toml
     [project.optional-dependencies]
     huggingface = ["transformers>=4.30,<5", "torch>=2.0,<3", "pillow>=10"]
     ```
   - Keep core unaffected (no changes to main `dependencies`).

4. **Add tests (optional but recommended):** `tests/test_hf_policy.py`
   - Test instantiation, `act()` method, picklability (if using multiprocessing).
   - Can use `pytest.importorskip("transformers")` to gate tests on optional deps.

### Files an Implementation Agent Must NOT Touch

**Do NOT modify:**

1. **`src/gauntlet/policy/base.py`**
   - Core protocol definitions. Changing them breaks the entire harness.

2. **`src/gauntlet/policy/registry.py`**
   - The registry mechanism is designed to accept any `"module.path:attr"` string.
   - No need to hardcode HuggingFace in the registry.

3. **`src/gauntlet/runner/runner.py` and `src/gauntlet/runner/worker.py`**
   - The runner assumes all policies are picklable factories and instances (or their factories are).
   - HuggingFacePolicy's pickling strategy (factory-based instantiation) fits the existing contract; no runner changes needed.

4. **`src/gauntlet/env/tabletop.py`**
   - Observation shape is fixed (5 float64 keys). HuggingFacePolicy reads what it gets.

5. **`src/gauntlet/cli.py`**
   - Policy dispatch is generic. No CLI changes needed.

6. **`tests/test_policy.py`, `tests/test_runner.py`**
   - Reference implementations tested. New policy gets its own test file.

7. **Any mypy/ruff config in `pyproject.toml` that isn't optional-dependencies.**
   - The strict checking applies to all code. No carve-outs.

---

## Key Design Decisions & Constraints

### 1. Picklability is Structural

**The Verdict: PICKLING REQUIRED for n_workers >= 2.**

- The policy **instance** does not need to be picklable if instantiated inside the worker.
- The policy **factory** must be picklable to cross the `spawn` process boundary.
- HuggingFacePolicy can hold unpicklable state (torch tensors, GPU memory) **if and only if** the factory is a module-level function that instantiates the policy inside the worker.

**Mitigation:**
```python
# Module-level factory (picklable):
def make_hf_policy() -> HuggingFacePolicy:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("open-vla-v1")
    tokenizer = AutoTokenizer.from_pretrained("open-vla-v1")
    return HuggingFacePolicy(model, tokenizer)

# CLI usage:
# gauntlet run suite.yaml --policy my_pkg.policies:make_hf_policy
```

The factory is picklable (a module-level function name); the model/tokenizer are born inside the worker.

### 2. Policy Registry Mechanism

**The Verdict: NO EXPLICIT REGISTRY. Dispatch via `"module.path:attr"` string.**

- Built-in shortcuts: `"random"`, `"scripted"`.
- Custom policies use dynamic import: `"my_package.my_module:my_factory"`.
- No need to modify `registry.py`.

### 3. Observation Dict Shape

**The Verdict: 5 float64 keys, no images in Phase 1.**

- Observation is always `{"cube_pos": (3,), "cube_quat": (4,), "ee_pos": (3,), "gripper": (1,), "target_pos": (3,)}`.
- HuggingFacePolicy (VLA) is trained on images + language. **It will need to handle missing images gracefully** or this will cascade into a Phase 2 env redesign.
- For Phase 1, HuggingFacePolicy is structurally possible but may not be behaviorally useful without image observations.

---

## Summary for Agent Implementation

| Aspect | Finding |
|--------|---------|
| **Picklability** | REQUIRED for multiprocessing (`n_workers >= 2`). Factory must be module-level; policy instance can hold unpicklable state if born inside worker. |
| **Registry Mechanism** | NO explicit registry dict. Use `"module.path:attr"` string dispatch via `resolve_policy_factory()`. No code changes to registry needed. |
| **Observation Dict Shape** | Fixed 5-key dict (all float64): `cube_pos`, `cube_quat`, `ee_pos`, `gripper`, `target_pos`. No images. VLA models may require Phase 2 env changes to be useful. |

---

**Document Created:** 2026-04-22  
**Scope:** Read-only exploration. No code modifications.  
**Next Steps:** Implementation agent uses this map as a blueprint for HuggingFacePolicy development.
