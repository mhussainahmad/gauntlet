# Phase 2 Task 3 Integration Map: Drift Detector Module

## 1. Episode Schema: Trajectory Preservation

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/runner/episode.py`, lines 38–60

**Episode Pydantic Model (verbatim):**
```python
class Episode(BaseModel):
    """Result of one rollout. Pure data; no behaviour."""

    model_config = ConfigDict(extra="forbid")

    # Identity / reproducibility.
    suite_name: str
    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]

    # Outcome.
    success: bool
    terminated: bool
    truncated: bool
    step_count: int
    total_reward: float

    # Extensible bag for future fields (trajectory summary, timing, master
    # seed echo, etc.). Task 7 will add reporting helpers that read keys
    # from here without changing the schema above.
    metadata: dict[str, float | int | str | bool] = Field(default_factory=dict)
```

**Finding:** Episode carries **summaries only** — final outcome (`success`, `total_reward`, `step_count`) but **no observation or action trajectory data**. Observations and actions are discarded after `_execute_one` completes.

**Worker behavior** (`/home/hussain/dev/gauntlet/src/gauntlet/runner/worker.py`, lines 168–225):
- Loop at lines 206–210 steps the env and accumulates reward, but does **not** store obs/action sequences.
- Episode is constructed at line 213–225, recording only scalar aggregates.
- Per line 5–6 docstring: "Phase 1 task 7 (Report) extends it with reporting helpers but must NOT rename or drop any existing field" — the schema is locked for stability.

**Trajectory loss implication:** A drift detector that scores per-step reconstruction error must capture observations **during** rollout, not post-hoc from `episodes.json`. Options:
1. **Streaming to disk during `run`** — worker writes per-step obs/action to a separate `.trajectories.jsonl` or `.trajectories.npz` artifact.
2. **In-process collection in Report** — extend Episode.metadata with summary (e.g., `"obs_entropy"`) computed at step time by wrapping the worker loop.
3. **Dual Episode + Trajectory schema** — create a new `Trajectory` pydantic model for per-step data, stored separately, keyed by Episode identity.

---

## 2. Report Schema and Serialization

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/report/schema.py`, lines 122–142

**Report Pydantic Model (verbatim):**
```python
class Report(BaseModel):
    """Top-level failure-analysis report.

    Built by :func:`gauntlet.report.analyze.build_report` from a list
    of :class:`gauntlet.runner.Episode` results. The field order is
    deliberately breakdown-first (§6).
    """

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    n_episodes: int
    n_success: int
    per_axis: list[AxisBreakdown]
    per_cell: list[CellBreakdown]
    failure_clusters: list[FailureCluster]
    heatmap_2d: dict[str, Heatmap2D]
    overall_success_rate: float
    overall_failure_rate: float
    cluster_multiple: float
```

**Serialization contract:**
- Report is built deterministically from episodes in `build_report()` (`/home/hussain/dev/gauntlet/src/gauntlet/report/analyze.py`, lines 1–39).
- CLI `run` writes **both** `episodes.json` and `report.json` (lines 254–259 of `/home/hussain/dev/gauntlet/src/gauntlet/cli.py`):
  - `episodes_path = out / "episodes.json"` — array of Episode dicts, append-friendly but never re-built.
  - `report_json_path = out / "report.json"` — singular Report dict, replaced entirely.
- CLI `report` subcommand (lines 276–306) auto-detects input type: if list → rebuild from episodes, if dict → load Report as-is.

**Extension point:** Report uses `ConfigDict(extra="forbid")`, so adding new fields requires schema change. Two options:
1. **Add drift_scores field to Report** — `per_episode_drift_scores: dict[int, float]` (keyed by episode index), or nested under CellBreakdown.
2. **Emit separate drift.json artifact** — keep Report unchanged, write `drift_scores.json` alongside `episodes.json` and `report.json`, loaded by HTML viewer separately.

**HTML consumption** (`/home/hussain/dev/gauntlet/src/gauntlet/report/html.py`, lines 70–97):
- `render_html(report)` dumps Report via `report.model_dump(mode="json")` and passes to Jinja2 template.
- Template receives both the Report object and the JSON dump (line 86).
- NaN cells are converted to None for JSON validity (lines 47–67).

---

## 3. CLI `run` and `report` Flow

**Location:** `/home/hussain/dev/gauntlet/src/gauntlet/cli.py`, lines 159–306

**`run` subcommand signature** (lines 160–218):
```python
@app.command("run")
def run(
    suite_path: Annotated[Path, ...],
    policy: Annotated[str, typer.Option("--policy", "-p", ...)],
    out: Annotated[Path, typer.Option("--out", "-o", ...)],
    n_workers: Annotated[int, typer.Option("--n-workers", "-w", min=1)] = 1,
    seed_override: Annotated[int | None, typer.Option(...)] = None,
    no_html: Annotated[bool, typer.Option("--no-html", ...)] = False,
    env_max_steps: Annotated[int | None, typer.Option(..., hidden=True)] = None,
) -> None:
```

**Current artifact flow** (lines 239–268):
1. Load suite → resolve policy factory → create Runner.
2. `episodes = runner.run(policy_factory, suite)` (line 245).
3. `report = build_report(episodes)` (line 250).
4. Write to disk:
   - `episodes.json` (line 258) — serialized Episode list.
   - `report.json` (line 259) — serialized Report.
   - `report.html` (line 262) — only if `not no_html`.

**`report` subcommand** (lines 276–306):
- Input: path to either `episodes.json` or `report.json`.
- Auto-detect (line 296, calls `_load_report_or_episodes`): if list of dicts, validate as Episode list and rebuild Report; if dict, validate as Report.
- Output: single HTML file (default next to input, or `--out <path>`).

**Extension point for drift scoring:**

Option A: **`--drift-ae <path>` flag on `run`** (compute during rollout):
- Requires capturing trajectories in memory or streaming to disk during `_execute_one`.
- Drift scores computed per-episode before Report is built.
- Scores stored in Episode.metadata or separate artifact.
- Clean, but requires worker-side trajectory collection.

Option B: **Separate `drift` subcommand** (compute post-hoc):
- Input: `episodes.json` + `--drift-ae <path>`.
- Output: `drift_scores.json` alongside or nested in output Report.
- Can reuse episodes.json; no re-running needed.
- Looser coupling, but requires external model loading.

Option C: **Extend `run` with implicit drift scoring** (if AE found in suite dir):
- If `suite.yaml` references `suite_name.ae.pt` in same dir, auto-load and score.
- Drift scores appended to Episode.metadata or Report via rebuild.
- Most transparent but requires convention enforcement.

---

## 4. Dependency Footprint and torch Import Status

**Location:** `/home/hussain/dev/gauntlet/pyproject.toml`, lines 22–31

**Core dependencies (verbatim):**
```toml
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

**torch import audit:** Grep of `/home/hussain/dev/gauntlet/src/gauntlet/**/*.py` confirms **zero torch imports** in core. ✓

**Current extras:** None defined in pyproject.toml. Per spec §6, torch must be optional and imported only by `src/gauntlet/monitor/` (drift AE) or specialized policies.

**Integration:** Add `[project.optional-dependencies]` section to pyproject.toml:
```toml
[project.optional-dependencies]
monitor = ["torch>=2.0,<3", "torchvision>=0.15,<1"]
```
- Users: `pip install gauntlet[monitor]` to enable drift detector.
- Core import remains clean; lazy import in monitor/__init__.py.

---

## 5. Artifact Layout and .gitignore

**Location:** `/home/hussain/dev/gauntlet/examples/suites/`

**Current structure:**
```
examples/
  suites/
    tabletop-smoke.yaml
    tabletop-basic-v1.yaml
```

**No model artifact convention yet.** Proposal:
- Store reference AE weights in `examples/suites/models/` (e.g., `examples/suites/models/tabletop.ae.pt`).
- Or inline path in suite YAML: `drift_ae_checkpoint: examples/suites/models/tabletop.ae.pt` (optional field).
- Keep trajectories separate: `{out}/trajectories.jsonl` (per-step obs/action pairs keyed by episode index).

**Location:** `/home/hussain/dev/gauntlet/.gitignore`, lines 30–41

**Current exclusions (verbatim):**
```gitignore
# Gauntlet artifacts (results, generated reports)
out/
results/
*.results.json

# Local agent / Claude Code infrastructure (runtime state, not project code)
.claude/
.claude-flow/
.swarm/
.mcp.json
CLAUDE.md
```

**Proposal to add:**
```gitignore
# Model artifacts (drift detector AE checkpoints, large)
*.pt
*.pth
*.safetensors

# Generated trajectories (optional; can be verbosity)
*.trajectories.jsonl
*.trajectories.npz
```

---

## Critical Findings

### Critical Finding #1: Does `Episode` carry full trajectories?

**NO.** Episode stores outcome summaries (`success`, `step_count`, `total_reward`) only. Observations and actions are **discarded during rollout** inside `_execute_one()` (worker.py:206–225). They are not stored in metadata or any other field.

**Consequence for drift detector:** A reconstruction-error-based autoencoder cannot score offline from `episodes.json` alone. You must either:
1. Capture obs/action sequences **during** the `run` command (via worker modification or streaming artifact).
2. Accept per-episode summary proxies (e.g., episode-level action entropy from sampled policy RNG entropy).
3. Rerun episodes with trajectory capture enabled (slow, redundant).

### Critical Finding #2: Is torch already a core import?

**NO.** Zero torch imports detected in `src/gauntlet/`. Core is pure numpy/mujoco/pydantic, per spec §6. ✓

---

## Minimum-Blast-Radius Integration

The drift detector module (`src/gauntlet/monitor/`) should integrate as follows:

1. **Create new directory:** `/home/hussain/dev/gauntlet/src/gauntlet/monitor/` (empty dir, will hold autoencoder trainer + scorer).

2. **Extend Episode.metadata (non-breaking):** Add optional `"action_entropy"` (float) or `"obs_summary"` fields computed during step. This piggybacks the existing metadata dict; no Episode schema change needed.

3. **New `drift_scores.json` artifact:** Emit alongside `episodes.json` and `report.json` in CLI `run` output. Structure: `{"episode_index": {cell_index, ep_index, drift_score}, ...}`.

4. **Separate CLI subcommand or flag:** `gauntlet drift-score <episodes.json> --model <ae.pt> --out <drift_scores.json>`. Decoupled from `run`, so existing workflows are unaffected. Report subcommand can optionally consume drift scores if present.

5. **No Report schema change (initially):** Drift scores live in metadata or separate artifact. If needed later, add optional `drift_scores_by_episode: dict[int, float]` to Report as a non-breaking extension (requires schema change).

6. **Dependency:** Add optional `[monitor]` extra to pyproject.toml with torch; only imported if user opts in.

**Rationale:** Keeps core clean, sideways integration, no breaking changes, allows post-hoc scoring from episodes.json without re-running.

