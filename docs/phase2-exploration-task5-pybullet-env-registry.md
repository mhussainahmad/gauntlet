# Phase 2 Task 5 Exploration: PyBullet Env Registry & Protocol Extraction

**Date:** 2026-04-22  
**Branch:** phase-2/pybullet-sim  
**Base:** 832551f (main)

---

## Q1: TabletopEnv Public Surface

### Methods & Attributes from `src/gauntlet/env/tabletop.py`

**Initialization:**
```python
def __init__(
    self,
    *,
    max_steps: int = 200,
    n_substeps: int = 5,
) -> None:
```
Lines 151–216.

**Gym API (gymnasium.Env contract):**

```python
def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
```
Lines 385–468.

```python
def step(
    self,
    action: NDArray[np.float64],
) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
```
Lines 470–511.

```python
def close(self) -> None:
```
Lines 526–533.

```python
def render(self) -> NDArray[np.uint8]:
```
Lines 513–524.

**Perturbation API:**

```python
def set_perturbation(self, name: str, value: float) -> None:
```
Lines 317–339.

```python
def restore_baseline(self) -> None:
```
Lines 294–313.

**Space Attributes (gymnasium.Env):**

```python
self.action_space: spaces.Box = spaces.Box(...)  # lines 206
self.observation_space: spaces.Dict = spaces.Dict(...)  # lines 207–215
```

**Public Class Attributes:**
- `MAX_LINEAR_STEP: float = 0.05` (line 129)
- `MAX_ANGULAR_STEP: float = 0.1` (line 130)
- `GRASP_RADIUS: float = 0.05` (line 133)
- `TARGET_RADIUS: float = 0.05` (line 134)
- `GRIPPER_OPEN: float = 1.0` (line 137)
- `GRIPPER_CLOSED: float = -1.0` (line 138)
- `metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}` (line 126)

---

### Methods Called by `Runner` via Worker

From `src/gauntlet/runner/worker.py` lines 168–225 (`_execute_one`):

```python
# Line 189:
env.restore_baseline()

# Lines 190–191:
for name, value in item.perturbation_values.items():
    env.set_perturbation(name, value)

# Line 197:
obs, _ = env.reset(seed=env_seed)

# Lines 207–208:
obs, reward, terminated, truncated, info = env.step(action)
```

**No explicit calls to:**
- `render()` — only for debug/optional visualization
- `close()` — called in finally block by Runner, not critical-path

**Required methods for env-neutral protocol:**
1. `reset(seed: int | None, options: dict | None) -> (obs, info)`
2. `step(action) -> (obs, reward, terminated, truncated, info)`
3. `restore_baseline() -> None`
4. `set_perturbation(name: str, value: float) -> None`
5. `action_space` property
6. `observation_space` property

**Optional (not in critical path):**
- `render()`
- `close()`

---

### Potential Signature Conflicts

**None identified.** The public API is clean:
- No MuJoCo-specific renderer kwargs (e.g., `render_mode` is handled at instance level, not method param).
- `action_space` and `observation_space` are standard gymnasium attributes.
- Perturbation methods (`set_perturbation`, `restore_baseline`) are env-agnostic in signature.

**Result:** TabletopEnv's public surface matches a reasonable `GauntletEnv` protocol with **zero signature changes needed**.

---

## Q2: Current Env Dispatch & Registry

### Suite YAML Parsing

**File:** `src/gauntlet/suite/schema.py`

**Where `env:` is declared:**
```python
class Suite(BaseModel):
    """..."""
    name: str
    env: str  # Line 211
    episodes_per_cell: int
    seed: int | None = None
    axes: dict[str, AxisSpec]
```

**Validation (lines 227–235):**
```python
@field_validator("env")
@classmethod
def _env_supported(cls, v: str) -> str:
    if v not in SUPPORTED_ENVS:
        supported = ", ".join(sorted(SUPPORTED_ENVS))
        raise ValueError(
            f"env: Phase 1 only supports {{{supported}}}; got {v!r}",
        )
    return v
```

**Hardcoded env registry (line 49):**
```python
SUPPORTED_ENVS: Final[frozenset[str]] = frozenset({"tabletop"})
```

---

### Env Factory Construction

**File:** `src/gauntlet/runner/runner.py`

**Default factory (lines 79–84):**
```python
def _default_env_factory() -> TabletopEnv:
    """Default env factory: build a stock :class:`TabletopEnv`.

    Module-level so it pickles cleanly under the ``spawn`` start method.
    """
    return TabletopEnv()
```

**Runner.__init__ signature (lines 95–134):**
```python
def __init__(
    self,
    *,
    n_workers: int = 1,
    env_factory: Callable[[], TabletopEnv] | None = None,
    start_method: str = "spawn",
) -> None:
```

**Usage (line 133):**
```python
self._env_factory = env_factory if env_factory is not None else _default_env_factory
```

**No lookup by suite.env name.** The runner accepts an explicit `env_factory` parameter; there is no dispatch logic keying off `suite.env`.

---

### Questions Answered

**Is there a registry today?**

**NO.** Only:
1. A hardcoded `SUPPORTED_ENVS` frozenset in schema.py containing `{"tabletop"}`.
2. A hardcoded default factory `_default_env_factory()` in runner.py.
3. No mapping from env name string → factory function.

**How many lines to extract a registry?**

Roughly **10–20 lines of new code:**

1. Create a new module `src/gauntlet/env/registry.py`:
   ```python
   from typing import Callable
   from gauntlet.env.protocol import GauntletEnv  # (to be created)
   
   ENV_REGISTRY: dict[str, Callable[[], GauntletEnv]] = {
       "tabletop": lambda: TabletopEnv(),
       # "pybullet": lambda: PyBulletEnv(),
   }
   ```

2. Update `src/gauntlet/runner/runner.py` `run()` method (~5 lines):
   ```python
   from gauntlet.env.registry import ENV_REGISTRY
   
   if env_factory is None:
       if suite.env not in ENV_REGISTRY:
           raise ValueError(f"unknown env {suite.env!r}")
       env_factory = ENV_REGISTRY[suite.env]
   ```

3. Update `src/gauntlet/suite/schema.py` to validate against registry keys (already done via `SUPPORTED_ENVS`).

**Files touched:**
- `src/gauntlet/env/registry.py` (new)
- `src/gauntlet/runner/runner.py` (lines ~270–272, modify logic)
- `src/gauntlet/suite/schema.py` (import from registry instead of hardcoding)
- `src/gauntlet/env/__init__.py` (re-export protocol)

---

## Q3: Perturbation Axes & PyBullet Difficulty

### 7 Axes Inventory

From `src/gauntlet/env/perturbation/__init__.py` lines 81–89 and `axes.py`:

| # | Name | Type | Range | Sampler |
|---|------|------|-------|---------|
| 1 | `lighting_intensity` | CONTINUOUS | [0.3, 1.5] | `make_continuous_sampler` |
| 2 | `camera_offset_x` | CONTINUOUS | [-0.1, 0.1] | `make_continuous_sampler` |
| 3 | `camera_offset_y` | CONTINUOUS | [-0.1, 0.1] | `make_continuous_sampler` |
| 4 | `object_texture` | CATEGORICAL | {0.0, 1.0} | `make_categorical_sampler((0.0, 1.0))` |
| 5 | `object_initial_pose_x` | CONTINUOUS | [-0.15, 0.15] | `make_continuous_sampler` |
| 6 | `object_initial_pose_y` | CONTINUOUS | [-0.15, 0.15] | `make_continuous_sampler` |
| 7 | `distractor_count` | INT | [0, 10] | `make_int_sampler` |

---

### MuJoCo Implementation Details

From `src/gauntlet/env/tabletop.py` `_apply_one_perturbation` lines 346–381:

**1. `lighting_intensity` (lines 349–350):**
```python
elif name == "lighting_intensity":
    m.light_diffuse[0] = np.array([value, value, value], dtype=np.float64)
```
Mutates the first light's RGB diffuse component uniformly. MuJoCo's render pipeline scales all geoms by this multiplier.

**Difficulty: EASY**  
PyBullet can set light intensity via `changeLightColor()` or similar; the visual effect maps directly.

---

**2. `camera_offset_x` (lines 351–355):**
```python
elif name == "camera_offset_x":
    base = self._baseline["cam_pos_main"]
    m.cam_pos[self._main_cam_id] = np.array(
        [base[0] + value, base[1], base[2]], dtype=np.float64
    )
```
Shifts the main camera position X relative to the baseline snapshot taken at `__init__`. Baseline stored in `_baseline["cam_pos_main"]` (line 277).

**Difficulty: EASY**  
PyBullet supports camera position offsets directly; retrieve baseline at init, apply delta.

---

**3. `camera_offset_y` (lines 356–360):**
```python
elif name == "camera_offset_y":
    base = self._baseline["cam_pos_main"]
    m.cam_pos[self._main_cam_id] = np.array(
        [base[0], base[1] + value, base[2]], dtype=np.float64
    )
```
Identical logic to offset_x, but mutates Y instead.

**Difficulty: EASY**  
Same as offset_x.

---

**4. `object_texture` (lines 361–364):**
```python
elif name == "object_texture":
    choose_alt = round(float(value)) != 0
    mat_id = self._cube_material_alt_id if choose_alt else self._cube_material_default_id
    m.geom_matid[self._cube_geom_id] = mat_id
```
Swaps the cube's material ID between two pre-defined materials in the MJCF (`cube_mat` and `cube_alt_mat`, defined in `assets/tabletop.xml` lines 70–71). MuJoCo resolves matid → material properties (colour, friction, etc.) at render/physics time.

**Difficulty: MODERATE**  
PyBullet doesn't natively support material IDs; would need to:
1. Maintain two colour/texture definitions in code or asset.
2. Call `changeVisualShape()` to swap cube colour.
3. Ensure the two colours are visually distinct and pre-defined (not loaded from MJCF).

---

**5. `object_initial_pose_x` (lines 365–366):**
```python
elif name == "object_initial_pose_x":
    self._data.qpos[self._cube_qpos_adr + 0] = float(value)
```
Directly overwrite the cube body's initial X qpos after `reset()` randomises it. Happens *after* `mj_resetData()` but *before* `mj_forward()`. This allows perturbation to override the seed-driven random pose.

**Difficulty: EASY**  
PyBullet stores state in similar qpos/qvel arrays; just write to the cube's X position in the state array.

---

**6. `object_initial_pose_y` (lines 367–368):**
```python
elif name == "object_initial_pose_y":
    self._data.qpos[self._cube_qpos_adr + 1] = float(value)
```
Identical to offset_x, but mutates Y qpos index.

**Difficulty: EASY**  
Same as initial_pose_x.

---

**7. `distractor_count` (lines 369–381):**
```python
elif name == "distractor_count":
    count = round(float(value))
    for i, gid in enumerate(self._distractor_geom_ids):
        if i < count:
            base_rgba = self._baseline["distractor_rgba"][i].copy()
            base_rgba[3] = 1.0
            m.geom_rgba[gid] = base_rgba
            m.geom_contype[gid] = 1
            m.geom_conaffinity[gid] = 1
        else:
            m.geom_rgba[gid] = self._baseline["distractor_rgba"][i]
            m.geom_contype[gid] = int(self._baseline["distractor_contype"][i])
            m.geom_conaffinity[gid] = int(self._baseline["distractor_conaffinity"][i])
```

Enables/disables the first N (0 ≤ N ≤ 10) distractor geoms (pre-allocated in MJCF, named `distractor_0_geom` through `distractor_9_geom`). When enabled:
- Set rgba[3] (alpha) to 1.0 to show the geom.
- Set contype/conaffinity to 1 to enable collision.

When disabled:
- Restore baseline rgba (alpha typically 0.0, making it invisible).
- Restore baseline collision flags (disabled).

Baseline snapshots taken at init (lines 284–290).

**Difficulty: HARD**  
PyBullet requires either:
1. **Dynamic asset generation:** Re-load a URDF/SDF with N distractor objects at reset time (expensive, breaks determinism unless the procedural generation is seed-driven).
2. **Pre-allocated invisible objects:** Store 10 distractor bodies at init, toggle visibility and collision per episode. Requires PyBullet's `changeVisualShape()` + `changeDynamics()` or equivalent, and ensuring invisible bodies do not affect physics.
3. **Visual-only override:** Render distractor geometry post-hoc in observation (e.g., render pass), decoupled from physics. Complex, non-standard.

The MuJoCo version relies on MJCF-level pre-allocation and material/collision tweaks; PyBullet lacks a direct equivalent to MuJoCo's model-mutation API.

---

## Summary Table: Perturbation Axes PyBullet Difficulty

| Axis | Type | MuJoCo Method | PyBullet Difficulty | Reason |
|------|------|---------------|---------------------|--------|
| `lighting_intensity` | continuous | `m.light_diffuse[0] = [v,v,v]` | **EASY** | Direct light intensity API exists. |
| `camera_offset_x` | continuous | `m.cam_pos[id] = [base[0]+v, ...]` | **EASY** | Baseline-relative offset, standard cam control. |
| `camera_offset_y` | continuous | `m.cam_pos[id] = [..., base[1]+v, ...]` | **EASY** | Same as offset_x. |
| `object_texture` | categorical | Material ID swap via `m.geom_matid[id]` | **MODERATE** | No native material ID system; requires colour swap via API. |
| `object_initial_pose_x` | continuous | `qpos[adr+0] = value` | **EASY** | Direct qpos mutation. |
| `object_initial_pose_y` | continuous | `qpos[adr+1] = value` | **EASY** | Direct qpos mutation. |
| `distractor_count` | integer | Toggle visibility/collision of pre-allocated geoms. | **HARD** | Pre-allocation + visibility/collision toggle not straightforward; risk of physics/render desync. |

---

## Proposed GauntletEnv Protocol Signature

```python
from typing import Protocol, runtime_checkable
from collections.abc import Mapping
import gymnasium as gym
from numpy.typing import NDArray
import numpy as np

@runtime_checkable
class GauntletEnv(Protocol):
    """Backend-neutral tabletop environment interface.
    
    Any implementation (MuJoCo, PyBullet, Isaac Sim, Genesis, etc.) 
    must satisfy this protocol to be used as a drop-in replacement 
    in the Runner.
    """

    # Gymnasium spaces (required attributes).
    action_space: gym.spaces.Box
    observation_space: gym.spaces.Dict

    # Core gym.Env methods.
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict]:
        """Reset the environment.
        
        Args:
            seed: Deterministic seed for reproducibility.
            options: Optional reset-time configuration (unused in Phase 1).
        
        Returns:
            (observation_dict, info_dict)
        """
        ...

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict]:
        """Execute one control step.
        
        Args:
            action: Array of shape (7,), dtype float64, bounds [-1, 1].
        
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        ...

    def close(self) -> None:
        """Release resources (model, renderer, etc.)."""
        ...

    # Perturbation interface (Phase 2 contract).
    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a named scalar perturbation for the next reset().
        
        Must validate that `name` is one of the 7 canonical axis names:
        - lighting_intensity, camera_offset_x, camera_offset_y,
          object_texture, object_initial_pose_x, object_initial_pose_y,
          distractor_count
        
        Raises:
            ValueError: if name is unknown or value is out of range.
        """
        ...

    def restore_baseline(self) -> None:
        """Restore the model/scene to its unperturbed baseline state.
        
        Called by the Runner before applying the next episode's 
        perturbations. Must undo all mutations from prior perturbations.
        """
        ...

    # Optional (for debug/viz; not in critical path).
    def render(self) -> NDArray[np.uint8]:
        """Return an RGB frame for visualization."""
        ...
```

---

## Registry Extraction Footprint

**New files:**
- `src/gauntlet/env/registry.py` (~15 lines)
- `src/gauntlet/env/protocol.py` (~80 lines, the Protocol definition above)

**Modified files:**
- `src/gauntlet/runner/runner.py` (~5–10 lines, import and use registry in `run()`)
- `src/gauntlet/suite/schema.py` (~2 lines, import SUPPORTED_ENVS from registry)
- `src/gauntlet/env/__init__.py` (~3 lines, re-export protocol and registry)
- `src/gauntlet/runner/worker.py` (~0 lines, no change; already accepts generic factory)

**Total new code:** ~25–30 lines.  
**Total lines touched:** ~5 files, ~20–25 lines modified.

---

## 150-Word Executive Summary

**Q1: Does TabletopEnv's surface match a GauntletEnv protocol with ZERO signature changes?**

**YES.** TabletopEnv provides all required methods (`reset`, `step`, `close`, `render`) and attributes (`action_space`, `observation_space`) with clean signatures. Perturbation methods (`set_perturbation`, `restore_baseline`) are env-agnostic. No MuJoCo-specific kwargs or internal details leak into the public API. A Protocol can be extracted that TabletopEnv already satisfies without modification.

**Q2: Is there a registry today?**

**NO.** Only a hardcoded `SUPPORTED_ENVS = {"tabletop"}` frozenset in schema.py and a hardcoded default factory in runner.py. No mapping from env name string to factory. Extracting a registry requires ~25 lines across 5 files.

**Q3: Which perturbation axes look HARDEST to port to PyBullet?**

**`distractor_count`** is hardest. It relies on MuJoCo's model-mutation API to toggle pre-allocated geoms' visibility and collision state. PyBullet lacks a direct equivalent and would require either dynamic asset generation (expensive, non-deterministic) or complex visibility/collision toggling. All other axes map cleanly: lighting_intensity and camera offsets are direct, texture swap is moderate (colour API call), and pose overrides are trivial (qpos mutation).

