# Gauntlet

**An evaluation harness for learned robot policies.**

> "How does this policy fail, and has the latest checkpoint regressed against the last one?"

Gauntlet is the robotics analogue of `pytest` + Sentry: pre-deployment regression testing plus structured failure-mode analytics, designed for policies that can't be debugged by reading code (because there is no code — just weights).

---

## 1. Why this exists

The deployment gap in learned robotics is quantifiable: a VLA policy that scores 95% in the lab can drop to 60% in production, and manufacturing typically requires >99.9% reliability. Mean success rate hides the structure of failures — teams don't know *which* perturbations break their policy until it's deployed and breaking.

Existing tools (SIMPLER, RoboEval, NVIDIA Isaac Lab-Arena, PolaRiS) are research artifacts — fragmented, tied to specific simulators or policy architectures, no shared standard. This project fills that gap with an opinionated, standards-track harness that:

1. Wraps any policy (OpenVLA, π0, SmolVLA, diffusion, custom) behind a uniform interface.
2. Runs it across a parameterized suite of perturbations (lighting, camera pose, object texture, clutter, initial conditions).
3. Produces structured reports that break down failures by perturbation axis — never hiding them in aggregate means.
4. Supports regression comparison between two policy checkpoints, so you know if your fine-tune actually improved things.

---

## 2. MVP scope (Phase 1 — what to build now)

Build a Python library + CLI called `gauntlet` that:

- Defines a minimal `Policy` adapter protocol that any model can implement in <20 lines.
- Provides one parameterized MuJoCo environment (pick-and-place on a tabletop) with five controllable perturbation axes: lighting intensity/color, camera pose offset, object texture swap, object initial pose, distractor object count.
- Runs evaluation sweeps across a YAML-defined perturbation grid, parallelized across processes, with full seed control for reproducibility.
- Collects structured episode results (success/failure, perturbation config, trajectory summary).
- Generates a JSON + HTML report with per-axis success breakdowns and failure clustering.
- Exposes `gauntlet run`, `gauntlet report`, and `gauntlet compare` via CLI.

### Non-goals for Phase 1

These are explicitly out of scope — do not build them now:

- Real-robot deployment or hardware interfaces.
- Runtime observability for deployed policies (that's Phase 2).
- Real-to-sim scene reconstruction / neural rendering (Phase 3).
- Support for simulators other than MuJoCo (Phase 2).
- Integration with ROS 2 (Phase 2).
- Web dashboard with live updates (Phase 1 HTML report is static).

---

## 3. Architecture

```
gauntlet/
├── policy/          # Policy adapter protocol + reference wrappers (Random, Scripted, HF-loaded)
├── env/             # Parameterized MuJoCo environments with perturbation axes
├── suite/           # Test suite definitions (YAML-configured perturbation grids)
├── runner/          # Parallel rollout orchestration + seed management
├── report/          # Structured failure analysis + HTML/JSON generation
└── cli.py           # gauntlet run / report / compare
```

### Core abstractions

- **`Policy`** — a `Protocol` with a single method: `act(obs: Observation) -> Action`. Reference implementations: `RandomPolicy`, `ScriptedPolicy`, `HuggingFacePolicy` (loads OpenVLA/SmolVLA checkpoints).
- **`PerturbationAxis`** — a named, seedable parameter (`lighting_intensity`, `camera_offset_x`, etc.) with a sampling distribution.
- **`Suite`** — a collection of `PerturbationAxis` objects + their grid specification, loaded from YAML.
- **`Episode`** — one rollout. Has `seed`, `perturbation_config`, `success: bool`, `trajectory`, `metadata`.
- **`Report`** — takes a list of `Episode` results + produces per-axis breakdowns, failure clusters, and an HTML artifact.

---

## 4. Tech stack (decisions already made — do not re-litigate)

- **Python 3.11+**
- **Package manager**: `uv` (not poetry, not pip directly)
- **Simulator**: MuJoCo via official `mujoco` Python bindings (not `dm_control`, not `mujoco-py`)
- **Env interface**: `gymnasium` (not legacy `gym`)
- **Config**: `pydantic` v2 for all config + result schemas
- **CLI**: `typer`
- **Parallelism**: `multiprocessing` (not `ray` for MVP — keep dependencies minimal)
- **Data**: `numpy`, `pandas`
- **Report HTML**: `jinja2` templates + `Chart.js` from CDN (static HTML, no build step for Phase 1)
- **Testing**: `pytest` + `pytest-cov`
- **Linting/typing**: `ruff` + `mypy --strict`
- **CI**: GitHub Actions

---

## 5. Phase 1 tasks (in this order)

Each task should land as its own PR-sized chunk with tests.

1. **Project scaffold** — `uv init`, `pyproject.toml` with all deps pinned, `README.md` stub, `ruff` + `mypy` config, GitHub Actions running tests + lint on push.
2. **`Policy` protocol + reference policies** — define the `Policy` protocol in `policy/base.py`; implement `RandomPolicy` and `ScriptedPolicy` (hard-coded pick-and-place trajectory). Tests: policies can be instantiated and produce actions of the right shape.
3. **MuJoCo tabletop env** — build `env/tabletop.py` with a simple pick-and-place scene (one cube, one target zone). Standard `gymnasium.Env` interface. Deterministic from a seed.
4. **Perturbation axes** — implement five axes as `PerturbationAxis` classes: `Lighting`, `CameraOffset`, `ObjectTexture`, `ObjectInitialPose`, `DistractorCount`. Each applies to the env before `reset()`.
5. **Suite schema + YAML loader** — `pydantic` model for a `Suite`; parse YAML like:
   ```yaml
   name: tabletop-basic-v1
   env: tabletop
   axes:
     lighting: {low: 0.3, high: 1.5, steps: 4}
     camera_offset_x: {low: -0.05, high: 0.05, steps: 3}
   episodes_per_cell: 10
   ```
6. **`Runner`** — takes a `Policy`, a `Suite`, returns a list of `Episode` results. Runs cells in parallel across processes. Fully seeded — rerunning with the same seed reproduces identical episodes.
7. **`Episode` + `Report` schemas** — `pydantic` models. `Report` computes: overall success rate, per-axis marginal success rates, failure clusters (which axis-combinations have >2x baseline failure rate), and a per-cell heatmap data structure.
8. **HTML report generator** — `jinja2` template that renders: summary card, per-axis bar charts (Chart.js), heatmap of 2D axis combinations, failure mode table. Self-contained HTML — no server needed to view.
9. **CLI** — `gauntlet run <suite.yaml> --policy <path> --out <dir>`, `gauntlet report <results.json>`, `gauntlet compare <results_a.json> <results_b.json>`.
10. **End-to-end example** — `examples/evaluate_random_policy.py` that runs the full loop on the tabletop env and produces an HTML report, plus a README quickstart that reproduces it in three commands.

---

## 6. Design principles

These are hard rules for this codebase and should be enforced on every change.

- **Type everything.** `mypy --strict` must pass. No `Any` except at FFI boundaries with MuJoCo.
- **Reproducibility is non-negotiable.** Every rollout must be reproducible from `(suite_name, axis_config, seed)`. If you can't reproduce a failure, the system is broken.
- **Never aggregate away failures.** Every report must show the breakdown before the mean. The default view is "which axes broke this policy," not "95%."
- **Minimal policy adapter.** If wrapping a new policy takes more than 20 lines, the `Policy` protocol is wrong — fix the protocol, don't work around it.
- **No wall-of-numbers reports.** A human should be able to look at the HTML report for 10 seconds and know what to fix.
- **Small deps.** Every new dependency must justify itself. No `ray`, no `torch` in the core (policy wrappers can import torch, but `gauntlet.core` cannot).

---

## 7. Out of scope, but worth noting for later phases

**Phase 2** (after MVP is used by at least one external team):

- Runtime distribution-shift detector: small autoencoder on observations, flags when deployed obs drift off-manifold. Action entropy monitoring.
- Additional simulators: Isaac Sim, Genesis, PyBullet adapters.
- Trajectory replay tool — take a failed episode, let a developer modify one variable and re-simulate.
- ROS 2 integration for running on real robots with logging.

**Phase 3**:

- Real-to-sim scene reconstruction (gaussian splatting from robot cameras into an eval environment).
- Fleet-wide failure mode clustering from deployed robots.
- Hosted service / web dashboard.

---

## 8. References

Context for the design — not required reading to build the MVP, but they inform the direction:

- a16z, *The physical AI deployment gap* (Jan 2026) — https://www.a16z.news/p/the-physical-ai-deployment-gap
- Silicon Valley Robotics Center, *State of Robotics 2026* — https://www.roboticscenter.ai/state-of-robotics-2026
- SIMPLER (Berkeley/Stanford) — https://simpler-env.github.io/
- RoboEval — https://robo-eval.github.io/
- NVIDIA Isaac Lab-Arena (Feb 2026 pre-alpha) — https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/
- PolaRiS (Dec 2025) — https://arxiv.org/html/2512.16881

---

## 9. Kickoff prompt

When handing this spec to a contributor (human or agent), use this opening:

> Read `GAUNTLET_SPEC.md`. Start with Phase 1 task 1 (project scaffold). After it's complete and tests pass, pause and summarize what you built. Then proceed to task 2. Do not skip ahead or bundle tasks. Before each task, restate the task number and goal; after each task, show what files changed and confirm all tests + lint + mypy pass.
