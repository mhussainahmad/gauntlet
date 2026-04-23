# Phase 2 RFC 003 — runtime drift detector (autoencoder + action entropy)

- **Status**: Draft
- **Phase**: 2, Task 3 (third item in `GAUNTLET_SPEC.md` §7: "Runtime distribution-shift detector: small autoencoder on observations, flags when deployed obs drift off-manifold. Action entropy monitoring.")
- **Author**: innovator / architect agent
- **Date**: 2026-04-22
- **Supersedes**: n/a
- **References**: `docs/phase2-rfc-001-huggingface-policy.md` (extras-group pattern, torch-free core rule, `render_in_obs` kwarg); `docs/phase2-rfc-002-lerobot-smolvla.md` (per-capability extras precedent).

---

## 1. Summary

Phase 2 §7 asks for a "small autoencoder on observations, flags when deployed obs drift off-manifold. Action entropy monitoring." This RFC scopes that to a post-episode analyser: fit a small torch autoencoder on trajectories from a known-good reference sweep, then score a candidate sweep's trajectories by mean per-step reconstruction error plus a per-dim action-std "entropy" proxy. To keep §6's "no torch in the core" rule intact we put the detector behind a new `[monitor]` extras group (torch-only — no `transformers`, no `pillow`, no `scikit-learn`) and ship it as a torch-free shell (`monitor.schema`, `monitor.entropy`) plus a torch-backed analysis core (`monitor.autoencoder`, `monitor.train`, `monitor.score`). The `Policy` protocol, the `Runner`'s bit-determinism contract, and the existing `Episode` / `Report` schemas are all unchanged; the one additive change is a `--record-trajectories <dir>` flag on `gauntlet run` (opt-in) that writes per-episode NPZ sidecars. A new `gauntlet monitor train` / `gauntlet monitor score` command group renders a `drift.json` sidecar next to `report.json`; the HTML template picks it up if it is present and falls back gracefully when it is not. State observations (14-D proprio on `TabletopEnv`) are the default AE target; image AE is an optional mode keyed off RFC-001's `render_in_obs=True`.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.monitor` — a post-episode drift analyser with two signals:
  - **Reconstruction-error drift**: mean per-step L2 reconstruction error from a small torch autoencoder trained on a reference sweep.
  - **Action entropy**: per-dim standard deviation of the 7-D action vector across the episode (continuous-deterministic-policy-friendly proxy; see §5).
- Keep `gauntlet.core` torch-free. Importing `gauntlet`, `gauntlet.policy`, `gauntlet.env`, `gauntlet.runner`, `gauntlet.report` must not transitively import `torch`. Importing `gauntlet.monitor.schema` and `gauntlet.monitor.entropy` must also be torch-free (so the HTML report can read `drift.json` on a torch-free install).
- Keep the public `Policy` / `ResettablePolicy` protocols, the `Episode` schema, the `Report` schema, and the `Runner`'s bit-determinism contract unchanged.
- Keep `mypy --strict` passing regardless of whether the `[monitor]` extra is installed.
- `uv sync --extra monitor` — one command to turn the detector on. Composes with `--extra hf` / `--extra lerobot`.
- Reference workflow is four commands (mirrors Phase 1's `run → report → compare`):
  1. `gauntlet run <ref_suite.yaml> --policy <known_good> --record-trajectories <ref_traj_dir> --out <ref_out>`
  2. `gauntlet monitor train <ref_traj_dir> --out <ae_dir>`
  3. `gauntlet run <eval_suite.yaml> --policy <candidate> --record-trajectories <eval_traj_dir> --out <eval_out>`
  4. `gauntlet monitor score <eval_out/episodes.json> <eval_traj_dir> --ae <ae_dir> --out <eval_out/drift.json>`

### Non-goals

- **Runtime (per-step) hooks inside `Runner.step`.** §7 explicitly names "runtime distribution-shift detector" but the spec §2 non-goals list rules out live dashboards for Phase 2, and per-step hooks add latency to parallel rollouts for zero observable benefit over post-episode analysis. Re-scoped to "post-episode, post-run" detection. See §4 for the integration-point decision.
- **Reference-distribution action-KL**. The stronger metric (per-dim KL between candidate action histogram and a reference action histogram) is listed as future work in §5; it requires keeping reference action trajectories around, which this RFC's reference-sweep workflow does support, but the metric itself is deferred to keep Task 3 scoped.
- **Nearest-neighbour-in-action-sequence-space** and other trajectory-distance metrics. Rejected outright (§5): `O(N²)` compute, and adds no signal that per-dim std + (future) action-KL cannot give.
- **Scikit-learn PCA baselines.** Considered briefly as a minimal-dep fallback (scikit-learn is ~30 MB, no C-extension surprises on ubuntu-latest). Rejected to keep the `[monitor]` extra to **one** heavy dep (torch).
- **Multi-reference-suite ensembling.** One `ae_dir` = one reference suite. A future "ensemble across multiple reference suites" story exists (useful for teams with distinct baselines per deployment site) but is explicitly deferred.
- **Real-time HTML streaming.** The drift view in `report.html` is static, matching the Phase 1 HTML contract (§4: "static HTML — no server needed to view").
- **Downloading AE weights in CI.** The AE is small enough (~10 KB for the state AE, ~5 MB for the image AE) that fitting one in a CI job from a 50-episode synthetic reference sweep is cheap; a real long-suite AE checkpoint is never fetched by tests.

## 3. Dependency placement decision

### Choice: **new `[monitor]` extra — torch-only, no other heavy deps.**

### Options considered

- **(A) Reuse `[hf]`.** `[hf]` already pulls `torch>=2.2,<3`. Nothing else in `[hf]` (transformers, timm, tokenizers, pillow) is needed for the state AE, and only pillow is needed for the optional image AE.
- **(B) New `[monitor]` extra = `torch` only.** Users who don't run VLAs can still compute drift scores. **Chosen.**
- **(C) Fold into `[dev]`.** Makes the detector uninstallable by end users; defeats the purpose of a deployment-time tool.
- **(D) Separate `gauntlet-monitor` distribution.** Same premature-distribution argument as RFC-001 §3 and RFC-002 §3 — rejected.

### Why (B)

Four facts drove this:

1. **`[hf]`'s surface is VLA-specific.** A team evaluating a diffusion policy or a hand-rolled model has no reason to install `transformers`, `timm`, or `tokenizers` just to get an autoencoder-backed drift score. The precedent set by RFC-002 (lerobot-specific extras live in their own group) applies here too: one extra per capability, composed via `uv sync --extra hf --extra monitor` when a user wants both.
2. **`torch` is the only non-stdlib, non-already-present dep.** The state AE is a 14→32→14 MLP; the image AE is a small CNN. `numpy` is already in core. `pandas` and `jinja2` are already in core. No `scikit-learn`, no `scipy`, no `matplotlib` — the drift view renders via the existing Chart.js CDN path.
3. **Version-pin independence.** `[hf]` floors `torch>=2.2,<3` because OpenVLA requires bf16 + `SDPA`. The monitor AE works on any `torch>=2.2`; tying it to `[hf]`'s floors means future OpenVLA releases that bump `transformers>=5` also bump the monitor extra, for no reason. Separate extras decouple the release cadence — the same argument RFC-002 §3 used for `[lerobot]`.
4. **The optional image-AE path needs `pillow`.** `[hf]` already pulls it, but we do **not** want to make `[monitor]` transitively require the VLA stack. `pillow` is ~5 MB and stdlib-friendly; adding it to `[monitor]` costs ~2s of install time on ubuntu-latest. The state AE mode (the default) needs only `torch`.

### `pyproject.toml` diff (fragments only — not applied in this RFC)

```toml
# EXISTING from RFC-001 and RFC-002 — unchanged.
[project.optional-dependencies]
hf = [
    "torch>=2.2,<3",
    "transformers>=4.40,<5",
    "timm>=0.9.10,<2",
    "tokenizers>=0.19,<1",
    "pillow>=10.0,<12",
]
lerobot = [
    "lerobot[smolvla]>=0.4,<1",
    "transformers>=4.40,<5",
    "pillow>=10.0,<12",
]

# NEW — lean extra for the drift detector. Torch is the only heavy dep.
# pillow is included because the optional image AE reads rendered frames
# saved by --record-trajectories as PNG-decodable arrays; on the state-AE
# default path it's unused. 5 MB of pillow is a cheaper price than the
# alternative (a torch-only extra that silently breaks when a user flips
# --mode image).
monitor = [
    "torch>=2.2,<3",
    "pillow>=10.0,<12",
]

# NEW — dev group analogue of hf-dev / lerobot-dev.
[dependency-groups]
monitor-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]

# NEW mypy override — let mypy import-check `src/gauntlet/monitor/*.py`
# even when torch is not installed (default CI job stays torch-free).
[[tool.mypy.overrides]]
module = ["torch", "torch.*", "PIL", "PIL.*"]
ignore_missing_imports = true
# (Already present from RFC-001 for torch; kept here for documentation —
#  the override is shared, not duplicated.)

# NEW pytest marker.
[tool.pytest.ini_options]
markers = [
    "hf: tests that require the [hf] extra (torch, transformers, PIL)",
    "lerobot: tests that require the [lerobot] extra (lerobot, torch, PIL)",
    "monitor: tests that require the [monitor] extra (torch, PIL)",
]
```

### CI structure

One new job, mirroring `hf-tests` and `lerobot-tests`:

- **Default job (unchanged)**: `uv sync` (no extras) → `ruff` → `mypy` → `pytest -m 'not hf and not lerobot and not monitor'`. This stays the continuous enforcement of §6 "no torch in the core" — now over three torch-requiring extras.
- **`hf-tests` (unchanged)**: `uv sync --extra hf --group hf-dev` → `pytest -m hf`.
- **`lerobot-tests` (unchanged)**: `uv sync --extra lerobot --group lerobot-dev` → `pytest -m lerobot`.
- **NEW `monitor-tests`**: `uv sync --extra monitor --group monitor-dev` → `pytest -m monitor`. Standard `ubuntu-latest` (no GPU) — the AE fits on a 50-episode synthetic reference sweep in <5 seconds on CPU.

All four jobs block merges.

## 4. Integration point: post-episode analyser (option B)

Three options were on the table.

- **(A) Runtime hook inside `Runner._execute_one`.** Compute reconstruction error per step, accumulate on the worker, attach to Episode. Rejected: (i) adds torch to the worker import graph, (ii) per-step AE forward is single-digit-ms but scales linearly with `suite.num_cells() * episodes_per_cell`, (iii) couples rollout throughput to the AE's device + dtype choices (a slow-device AE silently throttles parallel rollouts), (iv) §2 non-goal calls out "Web dashboard with live updates" — this is one step away from that.
- **(B) Post-episode, post-run analyser.** Trajectory data written to disk by the Runner; a separate `gauntlet monitor score` reads the trajectories + a pre-trained AE and emits `drift.json`. **Chosen.**
- **(C) Live dashboard / streaming over a socket.** Explicitly out of scope per spec §7 (Phase 2 HTML is still static per spec §4).

### Why (B)

- Zero change to the bit-determinism contract of the `Runner` (the NPZ sidecar is written downstream of `Episode` construction; Runner output ordering and seed derivation are untouched).
- Torch lives in exactly one module tree (`src/gauntlet/monitor/`). The Runner, the env, the report, and the CLI's `run` / `report` / `compare` paths stay torch-free.
- Reproducibility: a candidate sweep's trajectories + the reference AE checkpoint = a bit-reproducible `drift.json`. The NPZ sidecar carries the `seed` and `master_seed` so a user can reconstruct the candidate episode if they need to dig deeper.
- The Runner does grow one opt-in kwarg — `trajectory_dir: Path | None = None` — to wire the CLI flag through. This is the one caveat on "Runner untouched"; when the kwarg is `None` (the default) the Runner is byte-identical to its Phase 1 form.

### Why not (A)

The task brief flagged "cheap per-step hook" as a possible option; on measurement it isn't. A 14→32→14 MLP forward on CPU is ~50 µs, which is <1% of a 5 ms env step — tolerable in isolation. But the CNN image-AE forward at 224×224 is 5-10 ms on CPU, which doubles the step time. Users would then disable the runtime check for image mode and re-enable it for state mode, and we'd be maintaining two integration paths. Post-episode analysis sidesteps the whole question.

## 5. Action-entropy metric: per-dim action std within the trajectory

### Choice: **per-dim standard deviation of the 7-D action vector across the steps of one episode.**

Concretely, for an episode with `T` steps and action matrix `A ∈ ℝ^(T × 7)`:

```
action_std_per_dim_i = std(A[:, i]) for i ∈ {0..6}         # shape (7,)
action_entropy_scalar = mean(action_std_per_dim)           # scalar
```

Both are emitted in `drift.json` per episode (`per_episode[i].action_std_per_dim` and `per_episode[i].action_entropy`). The scalar is plotted; the per-dim vector is visible in the per-episode table for drill-down.

### Why this, over the three alternatives the task brief named

1. **Variance / std across steps within an episode** (chosen). Captures the concrete failure mode "policy saturates at ±1 on every step" — std drops to ~0. Captures the opposite failure "policy oscillates wildly" — std rises. Works identically for deterministic (`ScriptedPolicy`) and stochastic (`RandomPolicy`) policies. Requires **no reference distribution** (the AE side already needs one; doubling the reference-data requirement is a bad tradeoff). Trivially unit-testable — a synthetic trajectory with known std gives an exact expected value.
2. **Per-dim histogram divergence (KL) against a reference distribution.** Strictly more informative than std — captures distributional shape, not just spread — but requires building a reference action histogram per axis per dim and keeping it around. Flagged as future work: the reference-sweep workflow (§2) already produces the raw data, so the upgrade path is "add an `action_histogram.npz` emitted by `monitor train` and consumed by `monitor score`". Adding it in Task 3 doubles the state of `ae_dir` and forces another schema decision for zero additional signal over (1) on the tabletop env. Defer.
3. **Nearest-neighbour distance in action-sequence space.** Rejected outright: `O(N²)` over all reference episodes per scored episode, no closed-form win over std + KL, and introduces a tuning knob (`k`) with no obvious default.

### What the metric does NOT capture

Deliberate limitations (documented in the `drift.json` schema docstring so users don't over-trust the signal):

- A policy that is stuck at `action = [0.5, 0, 0, 0, 0, 0, open]` for every step of every episode has near-zero std; the AE reconstruction error will catch the *observation* staleness (the cube isn't being moved) but action-entropy alone reads "low variability" without distinguishing "stuck" from "confidently executing a short plan".
- Per-dim std is scale-dependent. Action dim 6 (gripper) is ±1 binary per TabletopEnv's snap semantics; dims 0-5 are `[-1, 1]`-bounded twist commands. A "high std on dim 6" signal usually means "the policy is toggling the gripper every step", which is a failure mode. The scalar mean across dims is reported so users can spot the obvious case; the per-dim vector lets them investigate.

## 6. Trajectory capture (the design question the task brief glossed)

### The gating fact

`Episode` today stores outcomes only: `seed`, `perturbation_config`, `success`, `terminated`, `truncated`, `step_count`, `total_reward`, `metadata`. There is no per-step obs array, no per-step action array. Reconstruction error needs obs arrays; action entropy needs action arrays. The RFC must decide where they come from.

### Options considered

- **(i) NPZ sidecar per episode, written by the Runner under an opt-in flag.** `gauntlet run --record-trajectories <dir>` produces `<dir>/cell_{cell_index:04d}_ep_{episode_index:04d}.npz` with arrays `obs_<key>` per proprio key, optional `image` (when `render_in_obs=True`), and `action`. Runner gains one kwarg (`trajectory_dir: Path | None = None`); when `None` the Runner is byte-identical. **Chosen.**
- **(ii) Re-run from seed inside `monitor score`.** Zero Runner change; `monitor score` reconstructs the trajectory by re-driving the env at `Episode.seed` with the original `policy_factory`. Rejected: doubles compute, and requires the user to re-supply the same `policy_factory` at score time — which is a non-trivial ask for VLA-sized policies and breaks the "score is cheap and offline" mental model.
- **(iii) Extend `Episode.metadata` with in-line trajectory arrays.** Rejected: 200-step × 14-D obs + 7-D action ≈ 4 KB per episode; a 10k-episode suite would inflate `episodes.json` to ~40 MB. Pydantic validation over nested arrays is also awkward.

### Why (i)

- Preserves `Episode` schema byte-for-byte (the "Episode unchanged" decision in §9 stays honest).
- Preserves the Runner's bit-determinism contract: the NPZ is a side-effect *after* the in-memory Episode is built, not *during* rollout. A run with `--record-trajectories <dir>` produces the same `Episode` list as a run without.
- Opt-in: disk cost is zero on users who don't need drift scoring. A 1000-episode state-only run writes ~200 MB (200 steps × 14 float64 × 1000 episodes); image-mode is ~3 GB (224 × 224 × 3 uint8 × 200 × 1000). The flag defaults to off precisely so the unaware user doesn't blow their disk.
- Consumer-friendly: `numpy.load(path)` is the whole read API; no pydantic, no JSON parsing, no schema version to negotiate.

### NPZ sidecar layout

One NPZ per episode. File name encodes identity so `monitor score` can match NPZs to Episode rows without a separate index:

```
<trajectory_dir>/cell_<cell_index:04d>_ep_<episode_index:04d>.npz
```

Arrays written (all shapes are `(T, ...)` where `T == episode.step_count`):

| Array                  | Dtype     | Shape            | Present when                             |
|------------------------|-----------|------------------|------------------------------------------|
| `obs_cube_pos`         | float64   | `(T, 3)`         | always                                   |
| `obs_cube_quat`        | float64   | `(T, 4)`         | always                                   |
| `obs_ee_pos`           | float64   | `(T, 3)`         | always                                   |
| `obs_gripper`          | float64   | `(T, 1)`         | always                                   |
| `obs_target_pos`       | float64   | `(T, 3)`         | always                                   |
| `obs_image`            | uint8     | `(T, H, W, 3)`   | only when env constructed with `render_in_obs=True` |
| `action`               | float64   | `(T, 7)`         | always                                   |
| `seed`                 | int64     | `()` (scalar)    | always                                   |
| `cell_index`           | int64     | `()` (scalar)    | always                                   |
| `episode_index`        | int64     | `()` (scalar)    | always                                   |

The scalar-index arrays let `monitor score` verify it has matched the right NPZ to the right Episode row (defensive check — the filename is the primary key, the scalars are the cross-check).

### Runner change required

Exactly one kwarg on `Runner.__init__` and one plumbing change in `_execute_one`:

```python
# Runner gains one kwarg; default None => zero behaviour change.
def __init__(
    self,
    *,
    n_workers: int = 1,
    env_factory: Callable[[], TabletopEnv] | None = None,
    start_method: str = "spawn",
    trajectory_dir: Path | None = None,       # NEW
) -> None: ...

# _execute_one accumulates obs / action per step (O(T) memory, one copy
# at end). When the Runner was built with trajectory_dir != None, the
# worker writes one NPZ per episode AFTER the Episode object is built,
# so the in-memory Episode is byte-identical to the no-flag path.
```

The worker writes its NPZs directly (no pool coordination needed); NPZ writes are atomic at the OS level on POSIX, and the filename includes `(cell_index, episode_index)` so there can be no collisions across workers. `multiprocessing.Pool` ordering semantics are unchanged.

## 7. Autoencoder architecture sketch

### `StateAutoencoder` (default, MLP, torch-only)

14-D proprio input. Architecture chosen to fit in <10 KB of weights:

```
input  -> Linear(14, 64) -> ReLU
        -> Linear(64, 32) -> ReLU          # encoder
        -> Linear(32, 8)                   # latent (8-D)
        -> Linear(8, 32) -> ReLU
        -> Linear(32, 64) -> ReLU
        -> Linear(64, 14)                  # reconstruction
```

Loss: `F.mse_loss(reconstruction, input)`. Trained with Adam (`lr=1e-3`), 50 epochs on the reference trajectory set (~50 episodes × 200 steps = 10k points), early-stopping when val MSE plateaus. Per-obs-key normalization stats (mean / std across the reference set) are computed at train time and saved alongside the weights so `monitor score` applies the same transform on candidate trajectories.

### `ImageAutoencoder` (optional, small CNN, torch-only)

224×224×3 uint8 → `float32 / 255.0` → CNN encoder → 128-D latent → CNN decoder → 224×224×3 `float32` → compared to the normalized input. Architecture kept small on purpose (~2 M parameters, fits in ~8 MB):

```
Encoder: Conv(3,32,s=2) -> Conv(32,64,s=2) -> Conv(64,128,s=2) -> Conv(128,128,s=2)
         -> Flatten -> Linear(to 128)
Decoder: Linear(128 -> flat) -> 4x ConvTranspose to 224x224x3 -> Sigmoid
```

Loss: `F.mse_loss` on the normalized tensor. `batch_size=32`, 20 epochs, same Adam config. Image mode requires `pillow` (for dtype-safe array round-tripping of uint8 frames) — hence `pillow` in the `[monitor]` extra.

### Shared interface

Both AEs expose the same public surface so `monitor.score` does not branch on type:

```python
class Autoencoder(Protocol):
    latent_dim: int
    input_spec: InputSpec            # "state" | "image" + shapes

    def encode(self, obs_batch: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def reconstruct(self, obs_batch: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def score(self, obs_batch: NDArray[np.float32]) -> NDArray[np.float64]: ...
    # score() returns one float per row — mean per-step L2 reconstruction error.
```

`monitor.autoencoder.load(ae_dir: Path) -> Autoencoder` is the single loader; it reads `ae_dir/config.json` to decide which subclass to instantiate, then loads `ae_dir/weights.pt` + `ae_dir/normalization.json`. Users interact with the returned object through `.score(obs_batch)` and never touch torch directly in their own code.

### Why not Lightning / timm

§6 hard rule: "small deps". `torch` alone gives us `nn.Module`, `Adam`, `DataLoader`, and `save`/`load`; that is 100% of the training pipeline the AE needs. `pytorch-lightning` would add ~30 MB of framework for fit-loop boilerplate we can write in ~40 lines. `timm` is not needed — the CNN is hand-written. Both are explicitly rejected.

## 8. Module layout

```
src/gauntlet/monitor/
├── __init__.py            # Re-exports DriftReport, PerEpisodeDrift, analyze,
│                          # load_autoencoder. __getattr__-guarded torch-dep
│                          # imports for DriftReport are kept in schema.py so
│                          # `from gauntlet.monitor import DriftReport` works
│                          # on a torch-free install.
├── schema.py              # TORCH-FREE. Pydantic DriftReport + PerEpisodeDrift.
├── entropy.py             # TORCH-FREE. Action-std / scalar entropy helpers.
├── trajectory.py          # TORCH-FREE. NPZ read + write helpers; small
│                          # module so Runner can import it without pulling
│                          # torch. Re-used by monitor.train and monitor.score.
├── autoencoder.py         # TORCH. StateAutoencoder / ImageAutoencoder classes
│                          # + save() / load() + a load_autoencoder dispatch.
├── train.py               # TORCH. fit() over a reference trajectory dir;
│                          # emits ae_dir/{weights.pt, normalization.json,
│                          # config.json}.
└── score.py               # TORCH. analyze(episodes_json_path, traj_dir,
│                          # ae_dir) -> DriftReport. Orchestrator.
```

Split rationale: `schema.py`, `entropy.py`, and `trajectory.py` are the three modules that the `Runner` (for `trajectory.py`) and the HTML renderer (for `schema.py`) need to import on the default — torch-free — install path. Keeping them in their own files, with no torch imports, means `import gauntlet.monitor.schema` stays clean regardless of whether the extra is installed. `autoencoder.py` / `train.py` / `score.py` lazy-import torch inside their module bodies and raise the install-hint `ImportError` when the extra is missing.

### Import-guard pattern (mirrors RFC-001 §3)

```python
# src/gauntlet/monitor/autoencoder.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # noqa: F401

_MONITOR_INSTALL_HINT = (
    "gauntlet.monitor.autoencoder requires the 'monitor' extra. Install with:\n"
    "    uv sync --extra monitor\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[monitor]'"
)

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(_MONITOR_INSTALL_HINT) from exc
```

The `try/except` is at module scope rather than inside `__init__` because the AE class body uses `nn.Module` as its base — the import must succeed before the class statement is parsed. `schema.py`, `entropy.py`, and `trajectory.py` do NOT have this guard; they are pure numpy + stdlib + pydantic.

## 9. CLI design

### New subcommand group: `gauntlet monitor train` and `gauntlet monitor score`

Rejected: folding drift scoring into `gauntlet report --ae <ae_dir>`. The `report` command must stay torch-free (the HTML renderer is torch-free and the extras-install cost for every HTML re-render is not worth the single-command convenience). Separation of concerns: `report` aggregates outcomes, `monitor score` runs torch inference.

```
gauntlet monitor train <trajectory_dir> --out <ae_dir>
    [--mode state|image]                # default: state
    [--latent-dim <int>]                # default: 8 (state) / 128 (image)
    [--epochs <int>]                    # default: 50 (state) / 20 (image)
    [--batch-size <int>]                # default: 256 (state) / 32 (image)
    [--device auto|cpu|cuda|cuda:N]     # default: auto
    [--val-split <float>]               # default: 0.1

gauntlet monitor score <episodes_json> <trajectory_dir> --ae <ae_dir> --out <drift_json>
    [--device auto|cpu|cuda|cuda:N]     # default: auto
    [--top-k <int>]                     # default: 10 — most-OOD episodes listed
```

### Additive flag on `gauntlet run`

```
gauntlet run <suite> --policy <p> --out <dir>
    [--record-trajectories <traj_dir>]  # NEW; default off — byte-identical
                                        # to Phase 1 when absent.
```

Passing `--record-trajectories` without the `[monitor]` extra is **not** an error — the Runner's NPZ write path uses `numpy.savez_compressed`, which is in the core deps. The AE side of the workflow (`monitor train` / `monitor score`) is what requires the extra.

## 10. Schema impact

### `Episode`: **unchanged.**

No new fields, no renamed fields, no type changes. `ConfigDict(extra="forbid")` stays. This is a hard design constraint: the trajectory data lives in NPZ sidecars (§6), not in the pydantic model.

### `Report`: **unchanged.**

No field changes. `report.json` is byte-identical before/after this RFC on identical inputs.

### New `DriftReport` pydantic model (in `src/gauntlet/monitor/schema.py`)

Sidecar artefact, serialised to `<out>/drift.json` by `gauntlet monitor score`. Not a field on `Report` — the HTML template checks for the sidecar's existence next to `report.json` and conditionally renders the drift panel. (This keeps the torch-free HTML path clean and makes the drift view opt-in without a schema change to `Report`.)

```python
# src/gauntlet/monitor/schema.py  (torch-free; stdlib + pydantic + numpy only)

class PerEpisodeDrift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_index: int
    episode_index: int
    seed: int                           # echoed from Episode.seed for cross-ref
    perturbation_config: dict[str, float]
    n_steps: int

    reconstruction_error_mean: float    # mean per-step L2 recon error
    reconstruction_error_max: float     # per-step max (outlier-surfacer)

    action_std_per_dim: list[float]     # length 7, per-dim std across steps
    action_entropy: float               # mean of action_std_per_dim


class DriftReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_name: str
    n_episodes: int
    ae_mode: Literal["state", "image"]
    ae_latent_dim: int
    ae_reference_suite: str | None      # name of the reference suite, if known

    # Baselines computed over the reference trajectories, stored on the
    # AE checkpoint's normalization.json — echoed here so the HTML report
    # can show "candidate mean X vs reference mean Y" side-by-side.
    reference_reconstruction_error_mean: float
    reference_reconstruction_error_p95: float

    # Candidate-sweep aggregates.
    candidate_reconstruction_error_mean: float
    candidate_reconstruction_error_p95: float
    candidate_action_entropy_mean: float

    per_episode: list[PerEpisodeDrift]
    top_ood_episodes: list[int]         # indices into per_episode, size top-k
```

### HTML report integration (scoped, opt-in)

Minimum viable surface — one additional panel in the existing template, conditionally rendered when `drift.json` is present next to `report.json`:

- One Chart.js bar chart: per-episode `reconstruction_error_mean`, with a dashed horizontal line at `reference_reconstruction_error_p95` (the OOD threshold).
- One table of the top-10 most-OOD episodes (from `top_ood_episodes`), columns: `cell_index`, `perturbation_config`, `reconstruction_error_mean`, `action_entropy`, `success`.

No second HTML file, no separate renderer command. If scope pressure mounts during implementation, defer the HTML panel to a follow-up task and ship `drift.json` as JSON-only.

## 11. Test plan

All torch-requiring tests live under `tests/monitor/` and are marked `@pytest.mark.monitor`.

### Unit tests (run in the default, torch-free job — `@pytest.mark.monitor` *not* required)

1. **Schema round-trip.** Construct `DriftReport` / `PerEpisodeDrift` from a hand-built dict, dump to JSON, load back, assert equality. Torch-free because `schema.py` is torch-free.
2. **Action-entropy metric on synthetic trajectories.** Construct a `(200, 7)` action matrix with known per-dim std; assert `compute_action_entropy(actions).action_std_per_dim ≈ known` (within fp tolerance). Degenerate cases: zero-length trajectory → `ValueError`; one-step trajectory → all-zero std (population std with `ddof=0`).
3. **Constant-action entropy is zero.** `A = np.ones((50, 7))` → `action_std_per_dim == [0]*7`, `action_entropy == 0`. Guards against "I accidentally computed std across dims instead of across time".
4. **Trajectory NPZ round-trip.** `trajectory.write_npz(path, obs_dict, actions, identity)` → `trajectory.read_npz(path)` → assert obs arrays, action array, and identity scalars all round-trip byte-identical. Torch-free.
5. **Runner opt-in flag.** Run a tiny suite with `Runner(..., trajectory_dir=tmp_path)`; assert exactly `num_episodes` NPZ files appear with the expected filename pattern; assert `Episode` list is byte-identical (via `model_dump_json`) to a run with `trajectory_dir=None`. No torch needed.
6. **Import guard.** Monkeypatch `sys.modules["torch"]` to `None`; assert `from gauntlet.monitor.autoencoder import StateAutoencoder` raises `ImportError` whose message contains `uv sync --extra monitor`. `schema.py`, `entropy.py`, and `trajectory.py` remain importable with torch absent (critical — verified explicitly in the same test).

### Monitor-extra tests (run in the new `monitor-tests` job — `@pytest.mark.monitor`)

7. **`StateAutoencoder` trains on synthetic data.** Generate 1000 random 14-D vectors drawn from `N(0, 1)`; train for 5 epochs; assert per-sample MSE on the training distribution falls by >10x from init to final. Deterministic: `torch.manual_seed(0)` at test start.
8. **Reconstruction error increases OOD.** Train the AE on samples drawn from `N(0, 1)`; score (a) a held-out batch from `N(0, 1)` and (b) a batch from `N(5, 1)`; assert `mean(score_b) > 10 * mean(score_a)`. The numerical multiplier is conservative — the real discriminative margin is much bigger.
9. **`ImageAutoencoder` trains on synthetic checkerboard.** Two-pixel checkerboard at 32×32 (scaled up to 224×224 in the test fixture); train for 3 epochs; assert reconstruction MSE drops monotonically. Tiny (shape-correctness-only) rather than quality-correctness — image AE real training is a workflow_dispatch job, not a CI job.
10. **`load_autoencoder(ae_dir)` dispatches on mode.** Train + save a state AE → `load(ae_dir).input_spec.kind == "state"`. Train + save an image AE → `.kind == "image"`. Corrupted `config.json` → clean `ValueError`.
11. **`analyze(episodes, traj_dir, ae_dir)` end-to-end on a small suite.** Tiny reference sweep (10 episodes, state mode) → train AE → candidate sweep with a deliberately OOD perturbation (e.g., `object_initial_pose_x=2.0`, way outside training range) → assert `DriftReport.candidate_reconstruction_error_mean > 2 * DriftReport.reference_reconstruction_error_mean`. The integration test for the whole pipeline.
12. **Top-k ordering.** With 20 candidate episodes of varying OOD-ness, assert `top_ood_episodes` contains the indices of the 10 highest `reconstruction_error_mean` in descending order of error.
13. **Normalization round-trip.** Train-time per-key mean/std is written to `normalization.json`; `load_autoencoder` re-applies it on score. Construct a reference sample that is bit-identical-to-training and assert its score < 1e-5.

### Explicitly out-of-scope for this task

- Real long-duration AE training on a full reference suite (hours of CPU time — `workflow_dispatch` only).
- Real image AE training over rendered frames (GPU-desirable — `workflow_dispatch` only).
- Performance benchmarks — the AE is small and the dataset is tiny; no perf budget to defend.
- A second "trajectory capture" mode for non-TabletopEnv envs. Tabletop is the only supported env in Phase 1 and most of Phase 2.

## 12. Open questions

Each has a reasonable default in parentheses so the implementation agent is not blocked.

- **OOD threshold convention.** Is an episode "OOD" when `reconstruction_error_mean > reference_p95`, `reference_p99`, or `reference_mean + 3σ`? (**Default: `reference_p95`. Documented as the bar on the bar chart; the per-episode table lets users see the raw number and decide for themselves. Surfacing a single threshold in `drift.json` (rather than three) keeps the schema small; users with stricter requirements re-score in Python from `per_episode`.**)
- **Image AE when `render_in_obs=False`.** `monitor train --mode image` against a trajectory dir that has no `obs_image` arrays — clean error or fall-back to state mode? (**Default: clean error. Falling back silently violates §6's "never hide failures". Error message points at `gauntlet run --render-in-obs` as the fix.**)
- **Reference-suite match check.** When the candidate's `suite_name` differs from the AE's reference `suite_name`, warn or error? (**Default: warn on stderr (matches `gauntlet compare`'s cross-suite warning pattern), record both names in `DriftReport.ae_reference_suite` / `DriftReport.suite_name` so the HTML panel can render them side-by-side.**)
- **GPU detection.** `--device auto` picks CUDA when `torch.cuda.is_available()`, else CPU. Do we also test MPS (Apple Silicon)? (**Default: CUDA or CPU. MPS support is a nice-to-have that adds a second device codepath; reject until a user asks.**)
- **Compressed vs uncompressed NPZ.** `savez` vs `savez_compressed`. Compressed is ~3x smaller for the proprio arrays, ~1.2x smaller for the image arrays (uint8 is hard to compress). Write cost is ~2x slower. (**Default: `savez_compressed`. Disk is almost always the bottleneck, not CPU; image arrays the compression nearly breaks even on, but proprio wins big.**)
- **Action-KL as a follow-up.** The reference-sweep workflow already produces the data. Does this RFC earmark the metric for a follow-up RFC, or leave it unscheduled? (**Default: earmark. Flag `monitor.entropy` as the home for the future `action_kl_per_dim(actions, reference_histogram)` helper; leave the schema hole empty until someone builds the histogram emitter.**)
- **Per-episode reconstruction-error histograms on disk.** For deep investigation, should `drift.json` carry per-step reconstruction error arrays per episode? (**Default: no. That's back-to-the-trajectory-sidecar territory. If users want per-step drift they can re-score with `monitor.score.analyze(..., return_per_step=True)` in Python. Keep `drift.json` small.**)
- **Multi-seed AE training.** Do we repeat AE training over multiple torch seeds and report mean/std of reference errors? (**Default: single-seed, `torch.manual_seed(0)` at train start, seed echoed into `ae_dir/config.json`. Multi-seed is a follow-up if reference MSE turns out noisy in practice.**)
- **Schema version field on `DriftReport`.** Do we stamp a version so future schema changes don't silently break old `drift.json` consumers? (**Default: no — follow the Phase 1 precedent that `Episode` and `Report` don't carry version fields. Add one in a follow-up the first time we make a non-additive change to `DriftReport`.**)

## 13. Rough implementation checklist

Sized as one §9-style PR, ~10 commits.

1. **`pyproject.toml`**: add `[project.optional-dependencies] monitor = ["torch>=2.2,<3", "pillow>=10.0,<12"]`, the `monitor-dev` dependency group, the new `monitor` pytest marker. Regenerate `uv.lock`.
2. **`src/gauntlet/monitor/schema.py`** (torch-free): `PerEpisodeDrift` and `DriftReport` pydantic models. `ConfigDict(extra="forbid")`. Round-trip tested in test case 1.
3. **`src/gauntlet/monitor/entropy.py`** (torch-free): `compute_action_entropy(actions: NDArray[np.float64]) -> ActionEntropyResult` returning the 7-D per-dim std + scalar mean. Input validation: shape `(T, 7)`, `T >= 1`, `dtype float`.
4. **`src/gauntlet/monitor/trajectory.py`** (torch-free): `write_npz(path, obs_dict, actions, *, cell_index, episode_index, seed)` and `read_npz(path) -> TrajectoryBundle`. `savez_compressed` on write; defensive scalar-arrays cross-check on read.
5. **Runner**: add `trajectory_dir: Path | None = None` kwarg to `Runner.__init__`; plumb through to `pool_initializer` via `WorkerInitArgs`. In `_execute_one`, accumulate obs and action lists per step; after the `Episode` is built, call `trajectory.write_npz` if `trajectory_dir is not None`. Memory: one O(T) list per rollout, freed at episode end — negligible on top of the existing MJCF model footprint.
6. **CLI**: add `--record-trajectories <dir>` flag to `gauntlet run`; wire to `Runner(trajectory_dir=...)`. No changes to `report` / `compare`.
7. **`src/gauntlet/monitor/autoencoder.py`** (torch): `StateAutoencoder`, `ImageAutoencoder`, the `Autoencoder` Protocol, `save(ae_dir)` and module-level `load(ae_dir) -> Autoencoder` dispatch. Module-scope `try: import torch` with the `_MONITOR_INSTALL_HINT`.
8. **`src/gauntlet/monitor/train.py`** (torch): `fit(trajectory_dir, *, mode, latent_dim, epochs, batch_size, device, val_split) -> Autoencoder`. Writes `ae_dir/{weights.pt, normalization.json, config.json}`. Deterministic: `torch.manual_seed` at entry.
9. **`src/gauntlet/monitor/score.py`** (torch): `analyze(episodes_json, trajectory_dir, ae_dir) -> DriftReport`. Orchestrator only — calls `load_autoencoder`, iterates NPZ sidecars, delegates action entropy to `entropy.py`, assembles the `DriftReport`. Top-k sort is `np.argsort(-per_episode_mse)[:top_k]`.
10. **CLI**: add `gauntlet monitor train` and `gauntlet monitor score` subcommands (new `app_monitor = typer.Typer(...)` registered as `app.add_typer(app_monitor, name="monitor")`). Argument parsing / error envelope matches the existing `run` / `report` / `compare` conventions. Both commands call into `monitor.train.fit` / `monitor.score.analyze`; both surface the same `_MONITOR_INSTALL_HINT` ImportError when the extra is missing.
11. **HTML template**: conditional drift panel in `src/gauntlet/report/templates/report.html.j2` that checks whether `drift.json` exists next to the rendered report path. If present, load it, render one Chart.js bar chart of per-episode reconstruction error + a top-k table. If absent, render nothing (no error, no empty section). This lives in `html.py`, not `analyze.py`, because it's a view-layer concern.
12. **Tests**: all thirteen cases from §11. Case 5 (Runner opt-in flag) is the most important — it enforces the "byte-identical Episode list with and without `--record-trajectories`" contract.
13. **CI**: add the `monitor-tests` job to `.github/workflows/ci.yml`. Confirm the default job now runs `pytest -m 'not hf and not lerobot and not monitor'`.
14. **`examples/evaluate_with_drift.py`**: the four-command workflow from §2, scripted end-to-end against `RandomPolicy` (reference) and `ScriptedPolicy` (candidate) — shows the full loop producing a `drift.json` + the HTML panel. Docstring calls out that this is a demonstration, not a benchmark: both reference and candidate are hand-written, so the drift signal is artificial-but-observable.
15. **`README.md`**: one bullet under the existing Quickstart: `uv sync --extra monitor` + a one-line pointer to `examples/evaluate_with_drift.py`.
16. **Local gate**: `uv run ruff check && uv run mypy && uv run pytest -m 'not hf and not lerobot and not monitor'` AND `uv sync --extra monitor --group monitor-dev && uv run pytest -m monitor`. Both must be green.

---

## Appendix A — External facts anchoring this RFC (as of April 2026)

- `numpy.savez_compressed` is a stable, in-core API; `numpy.load` returns a lazy `NpzFile` that avoids loading all arrays until access — scales to 10k-episode reference sweeps without running out of memory at score time.
- `torch>=2.2` is the floor shared with `[hf]`; `nn.Module`, `torch.optim.Adam`, `torch.utils.data.TensorDataset`, `torch.nn.functional.mse_loss`, and `torch.save` / `torch.load(weights_only=True)` are all stable on that floor.
- Pydantic v2 `ConfigDict(extra="forbid")` + `model_dump(mode="json")` + `model_validate` is the existing Phase 1 serialisation contract (see `src/gauntlet/runner/episode.py` and `src/gauntlet/report/schema.py`); `DriftReport` inherits it verbatim.
- `TabletopEnv`'s proprio obs is 14-D when flattened (`cube_pos 3 + cube_quat 4 + ee_pos 3 + gripper 1 + target_pos 3`); verified against `src/gauntlet/env/tabletop.py:_build_obs`.
- `TabletopEnv.render_in_obs` is the RFC-001-introduced kwarg that exposes a `(H, W, 3) uint8` rendered frame under `obs["image"]`. The image AE path reuses it as-is — no new env kwarg in this RFC.
- `Runner` owns the only entropy source in the rollout (`SeedSequence`-derived per-episode uint32); the NPZ sidecar writes happen after `Episode` construction, so the bit-determinism contract from `runner/worker.py:_execute_one` is preserved unchanged.
