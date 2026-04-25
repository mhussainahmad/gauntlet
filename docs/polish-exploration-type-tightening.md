# Polish exploration: type tightening — eliminate `Any` outside FFI overrides

Status: exploration / pre-implementation
Owner: phase-2.5/type-tightening branch
Phase: 2.5 Task 15

## 1. Why this matters

`gauntlet` already runs `mypy --strict`, but `Any` is permitted broadly
today via per-module overrides for FFI-heavy packages (`mujoco`,
`pybullet`, `genesis`, `isaacsim`, `omni`, `pxr`, `rclpy`,
`imageio_ffmpeg`, etc.). Inside `src/gauntlet`'s pure-Python core,
occasional explicit `Any` annotations leak from helper code where a
concrete type is already known — the JSON-walking helpers in the HTML
report renderers, the worker-local cache dict in `runner.worker`, and
similar. Those leaks hide structure that the runtime already enforces;
narrowing them raises the "production-ready" bar without touching the
legitimate FFI seams.

This task is **annotation-only at runtime**. The only allowed runtime
changes are tightening a `dict[str, Any]` to `dict[str, ConcreteType]`
or replacing `Any` returns with the concrete returned type — both byte-
identical at runtime if the existing code was already returning what
its docstrings claim.

## 2. Audit invocation

```
uv run mypy --strict --warn-return-any --disallow-any-explicit src/gauntlet 2>&1
```

Run against `src/gauntlet` only (not `tests/`) because pytest fixtures
under hypothesis use `Callable[..., None]`-typed decorators that the
flag would also flag — those are out of scope for this task.

Baseline (April 2026, pre-PR): **145 errors in 33 files**.

## 3. Findings: classification of the 145 leaks

The audit lumps three structurally distinct cases under one error code.
The breakdown:

### 3a. `Any` from pydantic plugin synthetic methods (NOT purgeable)

`disallow_any_explicit` flags every `class Foo(BaseModel):` declaration
because pydantic's `dataclass_transform` mechanism synthesizes an
internal `__mypy-replace` method whose `**kwargs: Any` signature is
the "Any" mypy reports — there is no literal `Any` in our source.

Repro:

```python
# /tmp/repro.py
from pydantic import BaseModel
class Foo(BaseModel):
    name: str = "x"
```

```
$ mypy --strict --disallow-any-explicit /tmp/repro.py
/tmp/repro.py:3: error: Explicit "Any" is not allowed  [explicit-any]
```

Files affected (24 lines): `runner/episode.py`, `aggregate/schema.py`,
`report/schema.py`, `monitor/schema.py`, `ros2/schema.py`,
`suite/schema.py` — every pydantic `BaseModel` subclass triggers this.

This is unavoidable without code changes to pydantic stubs themselves.
Adding `# type: ignore[explicit-any]` on every BaseModel subclass would
be busywork that obscures real leaks.

**Verdict: stays. Documented in §5 below as the "deferred flag"
justification.**

### 3b. `Any` from FFI / Protocol seams (stays, per spec §6)

The rest of the audit list comes from legitimate FFI-boundary uses
that match the existing override pattern:

- `env/base.py`, `env/registry.py`, `env/__init__.py`,
  `env/{pybullet,isaac,genesis}/__init__.py` — `Callable[..., GauntletEnv]`
  factory signatures and `gym.spaces.Space[Any]` Protocol fields. The
  factory `...` parameter spec is treated as `Any`; replacing it with
  ParamSpec breaks the heterogeneous backend init signatures.
- `policy/base.py` — `Observation = Mapping[str, NDArray[Any]]`.
  Already documented as the FFI boundary; backends emit mixed dtypes.
- `monitor/__init__.py`, `policy/__init__.py`, `ros2/__init__.py` —
  `__getattr__(name) -> Any` for the lazy-import pattern. Standard
  PEP 562 idiom; narrowing would defeat lazy loading.
- `plugins.py` — third-party plugin loading via `importlib.metadata`.
  Plugin classes are genuinely opaque until loaded.
- `suite/loader.py` — `yaml.safe_load` returns `Any` (yaml stubs do
  not narrow). Cast to `dict[str, Any]` happens at the boundary.
- `ros2/publisher.py`, `ros2/recorder.py` — `rclpy.Node` typed as
  `Any` in the upstream stubs; the modules already document this in
  their module docstrings.
- `runner/video.py` — imageio `imwrite` typed as `Any` even when
  installed (the [video] override entry covers the not-installed
  case).
- `env/tabletop.py`, `env/genesis/tabletop_genesis.py`,
  `env/pybullet/tabletop_pybullet.py`, `env/isaac/tabletop_isaac.py`
  — backend wrappers around mujoco / genesis / pybullet / isaacsim.
  Each ships a module docstring documenting the FFI carve-out.
- `policy/huggingface.py`, `policy/lerobot.py` — adapters around
  torch / transformers / lerobot. Same pattern.

**Verdict: stays. These ARE the FFI seams the existing override
pattern is for.**

### 3c. `Any` in pure-Python helpers (PURGEABLE)

A small remaining slice is genuine leakage where a concrete type is
already known:

- `aggregate/html.py:_nan_to_none(value: Any) -> Any` — recursive
  walker over JSON-style nested values. The accepted shape is exactly
  `JsonValue = float | int | str | bool | None | dict[str, JsonValue] |
  list[JsonValue]`. Tightening lets call-sites benefit from the
  recursive type.
- `report/html.py:_nan_to_none(value: Any) -> Any` — duplicate of the
  above by deliberate decoupling of the two report packages. Same
  fix.
- `runner/worker.py:_WORKER_STATE: dict[str, Any]` — heterogeneous
  per-worker cache with exactly four documented keys (`env`,
  `policy_factory`, `trajectory_dir`, `video_config`). A `TypedDict`
  with `total=False` matches the runtime-known shape exactly.
- `suite/lhs.py:lhs_unit_cube` — returns bare `np.ndarray` (which
  mypy widens to `ndarray[Any, Any]`). The function builds the result
  from `np.arange + rng.uniform / int`, which numpy promotes to
  float64; the docstring already promises a float matrix in
  ``[0, 1)``. Narrow to `NDArray[np.float64]`.

**Verdict: purge in this PR.**

## 4. Per-subpackage diff plan

| Subpackage | Files touched | Change |
| ---- | ---- | ---- |
| `aggregate/` | `html.py` | Introduce module-level `_JsonValue` recursive type alias; retype `_nan_to_none`. |
| `report/` | `html.py` | Same as above (deliberately duplicated, not imported). |
| `runner/` | `worker.py` | Replace `_WORKER_STATE: dict[str, Any]` with a `TypedDict`. The `info: dict[str, Any]` from `env.step` stays — that is the gymnasium FFI boundary surfaced literally. |
| `suite/` | `lhs.py` | Tighten `lhs_unit_cube` return type from bare `np.ndarray` to `NDArray[np.float64]` (matches the runtime dtype the function actually returns and the contract the docstring already promises). |

Total purgeable lines: 4 explicit / implicit `Any` annotations. The
rest of the 145-leak baseline stays as documented above.

## 5. mypy flag changes

### 5a. Considered but deferred

- `disallow_any_explicit = true` — globally rejected because of the
  pydantic synthetic-method blocker (§3a). Re-evaluating requires
  either code changes upstream in pydantic stubs or per-pydantic-class
  ignore comments (a worse trade than the current state).
- `disallow_any_decorated = true` — passes cleanly on `src/gauntlet`
  alone but trips 42 errors on `tests/` (hypothesis decorators that
  type the wrapped function as `Callable[..., None]`). Adding a tests
  override would gain little — the source tree is already clean — and
  clutters `[tool.mypy]`. Defer.
- `disallow_any_unimported = true` — fails in 3 FFI files
  (`policy/huggingface.py`, `monitor/ae.py`, `monitor/train.py`)
  where the override grants `ignore_missing_imports`. Each needs a
  paired `disallow_any_unimported = false` to compose. Touching new
  override entries here pulls in scope outside this task; defer.

### 5b. Landing in this PR

None. The deliverable is the manual purge — enforcement of the "zero
explicit `Any` in pure-Python core" claim is verified by running the
audit invocation in §2 and confirming the only remaining leaks fall
under §3a (pydantic synthetic) and §3b (FFI seam).

## 6. Out of scope

- `src/gauntlet/cli.py` — sibling agent owns it under the
  `polish/gauntlet-diff` branch. Eight `Any` leaks remain there; this
  PR does not touch `cli.py`. The sibling agent can choose to address
  them separately or leave them.
- The `tests/` tree — hypothesis `@given(...)`-decorated test
  functions are typed `Callable[..., None]` by the mypy plugin. This
  is `disallow_any_decorated` territory and is deferred.
- The 24 `BaseModel` subclass declarations under §3a — mass-suppressing
  them with `# type: ignore[explicit-any]` adds noise without value;
  blocked by the upstream pydantic-stub design.

## 7. Behaviour preservation affirmation

Every change is annotation-only at runtime. `_WORKER_STATE` keeps the
same key set (`env`, `policy_factory`, `trajectory_dir`,
`video_config`) and the same `dict.get(...)`-with-`None`-fallback read
pattern. The `_nan_to_none` helpers keep the same recursion and the
same identity behaviour for non-float, non-container leaves.
`lhs_unit_cube` returns the same matrix it always did (numpy promotes
the arithmetic to float64 at runtime regardless of the annotation).
mypy `--strict`, ruff, and the in-scope pytest subset all pass on
every commit.

## 8. Post-purge audit result

Running the §2 invocation after the purge (against the rebased branch
which now includes the sibling's `polish/gauntlet-diff` PR #29):

```
$ uv run mypy --warn-return-any --disallow-any-explicit src/gauntlet
...
Found 145 errors in 31 files (checked 61 source files)
```

The pre-PR baseline (against `aa56142`, the commit before sibling
PR #29 merged) was 145 across 33 files (58 checked). Sibling PR #29
added 4 new leaks (all in `src/gauntlet/diff/diff.py`) and shifted
`cli.py` line numbers by +1 without changing its leak count. This PR
purged 4 leaks in pure-Python helpers, so:

```
145 (post-rebase) = 145 (baseline) + 4 (sibling diff/) − 4 (this PR)
```

The remaining 145 are all documented as either §3a (pydantic
synthetic) or §3b (FFI seam):

| File | Leaks | Bucket |
| ---- | ---- | ---- |
| `env/genesis/tabletop_genesis.py` | 18 | §3b — Genesis FFI |
| `env/isaac/tabletop_isaac.py` | 17 | §3b — Isaac Sim FFI |
| `policy/lerobot.py` | 13 | §3b — LeRobot FFI |
| `policy/huggingface.py` | 13 | §3b — HuggingFace FFI |
| `env/tabletop.py` | 13 | §3b — MuJoCo FFI |
| `env/pybullet/tabletop_pybullet.py` | 10 | §3b — PyBullet FFI |
| `cli.py` | 8 | §6 — sibling-owned, out of scope |
| `env/base.py` | 6 | §3b — gymnasium Protocol |
| `suite/loader.py` | 5 | §3b — yaml FFI |
| `report/schema.py` | 5 | §3a — pydantic synthetic |
| `env/registry.py` | 5 | §3b — `Callable[..., GauntletEnv]` |
| `diff/diff.py` | 4 | §6 — sibling-owned, out of scope |
| `runner/worker.py` | 3 | §3b — `NDArray[Any]` + numpy stub + gymnasium info |
| `ros2/recorder.py` | 3 | §3b — rclpy FFI |
| `suite/schema.py` | 2 | §3a — pydantic synthetic |
| `ros2/publisher.py` | 2 | §3b — rclpy FFI |
| `plugins.py` | 2 | §3b — entry-point FFI |
| `monitor/schema.py` | 2 | §3a — pydantic synthetic |
| `aggregate/schema.py` | 2 | §3a — pydantic synthetic |
| 12 single-leak files | 12 | mix of §3a / §3b |

Zero `Any` leaks remain in the pure-Python core helper code that this
task targets. Excluding the sibling-owned `cli.py` and `diff/`
(out of scope per §6) the in-scope-and-purgeable count is **0**.
