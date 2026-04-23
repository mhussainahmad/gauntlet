# Phase 2 RFC-004: Trajectory Replay

Status: draft
Owner: architect (phase-2/trajectory-replay)
Branched from: `main` @ `832551f`
Related spec: `GAUNTLET_SPEC.md` §4 (Episode schema), §6 (reproducibility is
non-negotiable), §7 Phase 2 ("Trajectory replay tool — take a failed
episode, let a developer modify one variable and re-simulate.")

---

## 1. Summary

Add a fourth subcommand, `gauntlet replay`, that takes a single Episode
from a prior run, lets a developer override one (or more) perturbation
axis values, and re-simulates exactly that rollout — same seed, same
policy, same env pipeline — producing a new `Episode` alongside the
original.

The core move is pedestrian: the existing `_execute_one` function in
`gauntlet/runner/worker.py` is already a pure
`(env, policy_factory, WorkItem) -> Episode`. Replay is "build a
`WorkItem` whose fields match the original Episode exactly, swap the
value(s) in `perturbation_values`, run it". No Runner changes, no pool,
no new schema for the result.

The one area that does require care is **seed reconstruction** —
`SeedSequence.spawn` is path-dependent on the original
`(n_cells, episodes_per_cell)` topology, not just on the target node's
coordinates. This RFC proposes a minimally additive `Episode.metadata`
change to record that topology so replay is self-sufficient.

With that, the acceptance criterion — **zero-override replay is
bit-identical to the original Episode** — is achievable on today's
Runner without any refactor.

---

## 2. Goals / Non-goals

### Goals

- `gauntlet replay` subcommand: given `episodes.json` + `suite.yaml` +
  `--policy <spec>` + `--episode-id` + zero-or-more `--override`, emit
  one replayed Episode paired with the original.
- Zero-override replay is bit-identical to the referenced Episode
  (same `seed`, `total_reward`, `step_count`, `success`, `terminated`,
  `truncated`, `perturbation_config`).
- Override validation is strict-to-declared-suite-axes by name and
  strict-to-declared-axis-envelope by value — misspelt axes and
  out-of-range values fail loud with a helpful message before the env
  is ever constructed.
- Multi-axis overrides are applied combined (one replay, not a sweep).
- In-process only (`n_workers = 1`). Replay is fundamentally a
  single-episode, interactive operation.

### Non-goals

- Trajectory recording / step-by-step observation capture. The spec §4
  defines Episode fields as terminal summaries (success, reward, step
  count); per-step traces are a separate feature owned by a sibling
  RFC and are not a prerequisite for replay. The `--record-trajectories`
  hook is forward-referenced but not specified here (see §11).
- Parameter sweeps. That is what `gauntlet run` exists for. Passing
  two `--override` flags produces ONE replayed Episode with both
  overrides applied, not a 2D grid.
- Re-running an entire cell's worth of episodes. Replay is
  single-Episode only. A batch-replay CLI is a future extension.
- HTML artefact for the replay result. The paired JSON output is the
  deliverable; `gauntlet report` / `gauntlet compare` can be pointed
  at the output file after the fact.
- Hot-swapping the policy. `--policy` is required and identical to
  `run`'s resolver; "what if I'd used a different policy on the same
  seed" is a legitimate use of replay, but this is just "pass a
  different `--policy` string" — not a new feature.

---

## 3. Episode identity

### Current state (read from `runner/episode.py`)

Every Episode already carries:

- `suite_name: str`
- `cell_index: int` — zero-based ordinal in `suite.cells()` enumeration order
- `episode_index: int` — zero-based index within the cell
- `seed: int` — the uint32 handed to `TabletopEnv.reset(seed=...)`
- `perturbation_config: dict[str, float]`
- `metadata["master_seed"]: int` — the master seed (literal, or the
  auto-generated `SeedSequence.entropy` when `Suite.seed` was `None`)

The triple `(suite_name, cell_index, episode_index)` is stable,
unique within a run, and deterministic across re-runs of the same
suite. **No new identity field is needed.**

### CLI identity surface

`--episode-id "C:E"` (e.g. `--episode-id "12:3"`), parsed as
`(cell_index, episode_index)`. `suite_name` is asserted against
`suite.yaml`'s `name` at load time; mismatch is a hard error. Rationale
for not introducing a hash-based opaque id:

- Users find a failure by eyeballing `episodes.json` or the HTML
  report, both of which already surface `cell_index` /
  `episode_index`. An opaque hash would force a lookup step.
- `"C:E"` is trivially copy-pasteable and stable under `jq` filtering.

### Topology echo (minimally additive schema change)

For bit-identical replay, we must reconstruct the exact
`SeedSequence` node the original worker used. The derivation in
`Runner._build_work_items` is:

```
master   = SeedSequence(suite.seed)            # or fresh() when seed is None
cell_seqs    = master.spawn(n_cells)           # <-- path-dependent on n_cells
episode_seqs = cell_seqs[cell_idx].spawn(eps_per_cell)
episode_seq  = episode_seqs[ep_idx]
```

The spawned child at `[cell_idx, ep_idx]` depends on the topology
`(n_cells, episodes_per_cell)` of the original run, not only on the
coordinates. To make replay robust against edits to the suite YAML
between the run and the replay (e.g. the developer adds an axis value
or tweaks `episodes_per_cell`), we record that topology on the
Episode:

- `metadata["n_cells"]: int` — `suite.num_cells()` at run time
- `metadata["episodes_per_cell"]: int` — `suite.episodes_per_cell`

Both are tiny, additive, and exactly the two integers needed to
reconstruct the spawn tree from `metadata["master_seed"]` without
re-reading the suite. `Episode.model_config = ConfigDict(extra="forbid")`
is already set at the top level, but `metadata` is a `dict[str, float |
int | str | bool]` that accepts arbitrary keys, so this is a pure
data-only change — no model_validator edits.

Alternative considered: trust the original `suite.yaml` to be
unchanged and read `(n_cells, episodes_per_cell)` from it. Rejected —
if the suite YAML is edited between run and replay (a normal workflow
when axes are being iterated on), the spawn tree silently desyncs and
the replayed `seed` drifts, silently breaking bit-identity. Echoing
the topology into the Episode is the cheaper contract.

---

## 4. Replay workflow

### CLI signature

```
gauntlet replay EPISODES_JSON \
  --suite SUITE_YAML \
  --policy SPEC \
  --episode-id C:E \
  [--override AXIS=VALUE ...] \
  [--out REPLAY_JSON] \
  [--env-max-steps N]
```

- `EPISODES_JSON` — positional, path to the `episodes.json` emitted
  by a prior `gauntlet run`.
- `--suite` — path to the suite YAML the run used. Required: we need
  `suite.name` (asserted against the Episode's `suite_name`) and
  `suite.axes` (insertion order is load-bearing; see §6).
- `--policy` — same resolver as `run` (`"random"`, `"scripted"`,
  `"module.path:attr"`); delegates to `policy.registry.resolve_policy_factory`.
- `--episode-id C:E` — the cell and episode indices of the target.
- `--override AXIS=VALUE` — repeatable. AXIS must be one of the axis
  names declared in `suite.axes`; VALUE is parsed as `float` (the
  axis's scalar type on the wire).
- `--out` — optional; defaults to `<episodes_json_dir>/replay.json`.
- `--env-max-steps` — hidden test hook mirroring `run`'s flag.

No `-w`/`--n-workers`: replay is in-process.

### Runtime semantics

Order of operations inside `replay_command`:

1. Parse `episode_id` as `"cell:episode"` → `(cell_index, episode_index)`.
   Any other shape is an error.
2. Load `episodes.json` (reuse `cli._read_json` +
   `cli._episodes_from_dicts`) and find the Episode with matching
   `(cell_index, episode_index)`. Multiple matches inside one run
   are impossible by the Runner's contract; a missing match is an
   error that lists the available `(c:e)` pairs (capped to the first
   10 for terseness).
3. Load `suite.yaml` via `suite.load_suite`. Assert
   `suite.name == target.suite_name`; mismatch is an error.
4. Parse each `--override` as `AXIS=VALUE`. Validate (§5).
5. Reconstruct the SeedSequence node for the target:
   - `master_seed = target.metadata["master_seed"]` (int)
   - `n_cells = target.metadata["n_cells"]` (int; fall back to
     `suite.num_cells()` with a warning-to-stderr if absent — i.e.
     the Episode predates this RFC)
   - `eps_per_cell = target.metadata["episodes_per_cell"]` (int;
     fall back to `suite.episodes_per_cell` with the same warning)
   - `master = SeedSequence(master_seed)`
   - `episode_seq = master.spawn(n_cells)[cell_index].spawn(eps_per_cell)[episode_index]`
6. Build `perturbation_values` = `target.perturbation_config`
   overridden by each `--override`. **Iteration order is
   `target.perturbation_config.keys()`, not the order the user
   typed the overrides on the CLI** (see §6 for why). Taking the
   order from the Episode itself (not the suite YAML) makes replay
   robust against suite-YAML reorderings between run and replay,
   the same way §3 makes it robust against topology edits.
7. Build a `WorkItem` with `suite_name`, `cell_index`,
   `episode_index`, `perturbation_values`, `episode_seq`, and
   `master_seed` matching the original.
8. Call `_execute_one(env, policy_factory, item)` in-process with a
   fresh `TabletopEnv`. Close the env in a `finally`.
9. Emit the paired JSON (§7) to `--out`.

Every failure mode emits `error: ...` to stderr and raises
`typer.Exit(1)`, matching the existing CLI conventions in
`gauntlet/cli.py`.

---

## 5. Override validation

Strategy: **strict-to-declared-suite-axes by name, bounded-to-axis-
envelope by value**.

Two-layer validation, both done up-front before env construction:

- **Name**: `override.axis in suite.axes` (i.e. declared in the
  suite YAML). An axis that exists in `env.perturbation.AXIS_NAMES`
  but is *not* in `suite.axes` is rejected — replaying against an
  axis the original run never varied would mean extending the
  perturbation surface beyond what the user ran, which is a
  different experiment (and what `gauntlet run` is for). On rejection,
  emit the full list of legal axis names taken from `suite.axes` so
  the user can re-type.
- **Value**: the parsed float must lie within
  `[axes[name].low, axes[name].high]` (inclusive on both sides) for
  continuous / int axes, or be present in `axes[name].values` (with
  a 1e-9 tolerance for float compare) for categorical axes.
  Crucially, the value is **not restricted to the grid points
  enumerated by `AxisSpec.enumerate()`** — the whole point of replay
  is exploring *off* the original grid. Restricting values to grid
  points would defeat the feature.

For axes that have a hard env-side bound outside the YAML (only
`distractor_count` today: `TabletopEnv.set_perturbation` clamps to
`[0, N_DISTRACTOR_SLOTS]`), the CLI re-asserts the env's constraint
on top of the suite envelope so the user sees the error at the CLI
boundary, not buried inside a worker traceback.

Alternative considered: permit overrides for any env kwarg
(`TabletopEnv._KNOWN_AXIS_NAMES`). Rejected for two reasons:

- The suite YAML is the authoritative statement of "what this run
  varies"; allowing the CLI to silently cross that boundary
  bifurcates the config contract.
- Axes outside the suite's declared set would make the replayed
  Episode's `perturbation_config` carry keys that were never in any
  cell of the original run, breaking the per-cell grouping contract
  downstream code (`report.analyze`, `cli.compare`) relies on.

Users that want to explore off-grid axes should add them to the
suite YAML and re-run. Replay is for "what if value V, same suite".

---

## 6. Axis application order (bit-identity invariant)

`_apply_one_perturbation` in `env/tabletop.py` is **not orthogonal
across axes**. Specifically, `camera_offset_x` and `camera_offset_y`
both write the *full* `cam_pos` 3-vector relative to baseline:

```python
elif name == "camera_offset_x":
    m.cam_pos[cam] = [base[0] + value, base[1], base[2]]
elif name == "camera_offset_y":
    m.cam_pos[cam] = [base[0], base[1] + value, base[2]]
```

If both are applied, the second one wins (the first's Y contribution
is reset to `base[1]`, or vice-versa). The Runner applies perturbations
in `item.perturbation_values` iteration order, which in turn is
`dict(cell.values)` order, which in turn is `suite.axes` insertion
order from `suite.cells()`.

**Replay must preserve that exact iteration order** to reproduce the
original rollout bit-for-bit. The user's `--override` CLI order is
NOT the order axes are applied; axes are always applied in the
order they appear in `target.perturbation_config`, which is the
order the original run applied them (because `SuiteCell.values`
preserves `suite.axes` insertion order through `dict(cell.values)`
in `Runner._build_work_items`, which then round-trips through JSON
and `Episode.model_validate`). The build step is:

```
for axis_name in target.perturbation_config:
    perturbation_values[axis_name] = overrides.get(axis_name,
                                                   target.perturbation_config[axis_name])
```

Sourcing the order from the Episode rather than from the suite YAML
is deliberate: it makes replay robust against a developer
reordering `axes:` in the YAML between the run and the replay.
Validation of override NAMES still consults `suite.axes` (§5) — it
is only application ORDER that keys off the Episode.

This is a one-line invariant but a load-bearing one. It is called
out in §9 as a hard-tested reproducibility contract.

(A pre-existing latent bug in the tabletop env — the
`camera_offset_x/y` non-commutativity — is orthogonal to replay and
is not this RFC's job to fix. Replay only has to preserve whatever
the Runner did.)

---

## 7. Runner integration

Decision: **reuse `_execute_one` directly, no Runner invocation, no
synthetic Suite**. Rationale:

- `_execute_one(env, policy_factory, item)` is a pure function over
  `(env, policy_factory, WorkItem)`. It is already shared between
  the Runner's in-process and pool paths; invoking it a third way
  from `replay` keeps the producer of Episodes at a single code
  path.
- Building a one-cell synthetic Suite and handing it to `Runner.run`
  would NOT produce the same seed, because `master.spawn(1)[0]` at
  the top of a synthetic tree is a different node from
  `master.spawn(n_cells)[cell_idx]` in the original tree.
  Reconstructing the original node directly from
  `(master_seed, n_cells, episodes_per_cell, cell_index, episode_index)`
  is the only honest path to bit-identity.
- A one-episode replay has no use for the multiprocessing pool. The
  pool's value is amortising MJCF load across hundreds of episodes
  per worker, not one.

The replay module thus constructs its own env via the default
`TabletopEnv` factory (respecting `--env-max-steps`), its own policy
factory via `resolve_policy_factory`, and calls `_execute_one`
exactly once. No changes to `runner/runner.py` or `runner/worker.py`.

`_execute_one`, `WorkItem`, `WorkerInitArgs`, `extract_env_seed`,
`pool_initializer`, `run_work_item` are already in
`gauntlet/runner/worker.__all__`; `_execute_one` is not — it is
prefixed with `_` but is the public primitive we rely on. We export
it from `gauntlet.runner` (add to `runner/__init__.py`'s `__all__`)
as `execute_one`, documenting that replay is its second caller. This
is the smallest surface-area change to the runner module.

Picklability note: replay runs in-process, so
`policy_factory` and `env_factory` do not cross a process boundary.
Users can plug in non-pickleable policies here (including ones that
would break `-w >= 2`), which is a mild silver lining of the
single-process design.

---

## 8. Output format

Emit **one JSON object** to `--out`, schema:

```json
{
  "episode_id": "12:3",
  "suite_name": "tabletop-basic-v1",
  "policy": "scripted",
  "overrides": {"lighting_intensity": 1.2},
  "original": <Episode JSON>,
  "replayed": <Episode JSON>
}
```

Design choices:

- `original` and `replayed` are both full `Episode` objects (via
  `Episode.model_dump(mode="json")`). Downstream tooling can load
  either with `Episode.model_validate` without a special path.
- `overrides` echoes the exact `{axis: value}` the user passed. Empty
  dict when no `--override` was given.
- `policy` echoes the user's spec string for audit trail. The CLI
  does NOT try to record which policy produced the original episode
  — that information was never captured in Phase 1 and is out of
  scope here.
- No new Pydantic model. The paired object is a `dict[str, Any]`
  built by the CLI and round-tripped via
  `json.dumps(..., allow_nan=False)` through the same
  `_write_json` helper `run` already uses.
- Rationale for NOT stuffing replay metadata into
  `replayed.metadata`: the replayed Episode should be a normal
  Episode, indistinguishable from one produced by `run`, so that
  `gauntlet report <replay_out.replayed>` Just Works if a user
  extracts it. Replay-specific metadata (overrides, original link)
  lives at the envelope level.

A helper `gauntlet replay ... --extract replayed > ep.json` flag is
explicitly **not** added in this RFC — `jq '.replayed' replay.json`
is enough.

---

## 9. Reproducibility guarantee + test plan

**Hard acceptance criterion**: zero-override replay produces an
Episode where every field equals the original, bit-for-bit, for all
axis combinations exercised in `tests/test_runner.py`.

Equality check is field-by-field (success, terminated, truncated,
step_count, total_reward, seed, perturbation_config,
metadata["master_seed"]) via pytest's default `==` — floats are
compared exactly because the rollout is fully deterministic from the
seed.

Test plan (`tests/test_replay.py`):

1. `test_zero_override_bit_identity`: run a small 2×3 suite with the
   scripted policy, pick an arbitrary episode, replay with no
   overrides, assert `original == replayed` on the full Episode
   (Pydantic model equality, no field exclusions). After commit 1
   of §12, the replayed WorkItem carries identical `master_seed`,
   `n_cells`, and `episodes_per_cell`, so `_execute_one` emits an
   identical `metadata` dict. Any difference here is a real bug,
   not an expected artefact.
2. `test_zero_override_every_cell`: parametrise over all
   `(cell_index, episode_index)` pairs in a tiny 2×2 suite,
   assert bit-identity for each. Guards against off-by-one errors
   in the spawn-tree reconstruction.
3. `test_axis_application_order_preserved`: construct a suite with
   both `camera_offset_x` AND `camera_offset_y` declared (in that
   insertion order), replay with zero overrides, assert
   `replayed.seed == original.seed` AND
   `replayed.total_reward == original.total_reward`. This pins the
   §6 invariant: whatever the runner does, replay preserves.
4. `test_single_override_changes_outcome`: replay with an override
   that materially changes behaviour (e.g. `distractor_count=10`
   on an otherwise empty scene) and assert the replayed seed
   matches (same reset entropy) but `step_count` or `total_reward`
   differ from original. This checks the override path actually
   applies.
5. `test_override_keeps_non_overridden_axes_from_original`: two-axis
   suite, override only one axis, assert the non-overridden axis
   in `replayed.perturbation_config` equals the original's value for
   that axis.
6. `test_multiple_overrides_applied_in_suite_order`: two axes where
   order matters (`camera_offset_x` then `camera_offset_y`), pass
   overrides on the CLI in the OPPOSITE order, assert the replayed
   Episode matches the one produced by the same values passed via
   `gauntlet run`.
7. `test_replay_rejects_unknown_axis`: `--override not_an_axis=1.0`
   exits non-zero with a message that names the legal axes from
   the suite.
8. `test_replay_rejects_out_of_envelope_value`: `--override
   lighting_intensity=99.9` where the suite declares
   `{low: 0.3, high: 1.5}` exits non-zero before env construction
   (monkeypatch the env factory to raise to confirm it is not called).
9. `test_replay_rejects_suite_name_mismatch`: episodes.json and
   suite.yaml disagree on `suite_name` → exit 1.
10. `test_replay_rejects_unknown_episode_id`: `--episode-id 99:99`
    when the run has 2 cells × 3 episodes → exit 1 with a list of
    available pairs.
11. `test_replay_legacy_episode_without_topology_metadata`: simulate
    an Episode missing `n_cells` / `episodes_per_cell` in metadata
    (an episodes.json produced before this RFC). Replay falls back
    to `suite.num_cells()` / `suite.episodes_per_cell` with a
    stderr warning, and bit-identity still holds if the suite YAML
    is unchanged. Guards the upgrade path.
12. `test_replay_roundtrips_output_format`: parse the emitted
    `replay.json` with `Episode.model_validate(payload["original"])`
    and `Episode.model_validate(payload["replayed"])`, assert no
    validation errors. Pins the "replayed Episode is a normal
    Episode" property from §8.

All tests use `n_workers=1` and the hidden `env_max_steps` knob to
keep wall-clock under the existing test-suite budget (Task 6's tests
already use the same trick).

---

## 10. Module layout

New submodule `src/gauntlet/replay/`:

```
src/gauntlet/replay/
├── __init__.py     # public surface: replay_one, ReplayResult
├── replay.py       # in-process driver; calls _execute_one
└── overrides.py    # parse / validate --override flags
```

- `replay.py` holds `replay_one(target: Episode, suite: Suite,
  policy_factory: Callable[[], Policy], overrides: dict[str, float],
  env_factory: Callable[[], TabletopEnv] | None = None) -> Episode`.
  Pure function; no CLI dependency. This is the primitive a future
  library user (e.g. a notebook) calls directly.
- `overrides.py` holds `parse_override(spec: str) -> tuple[str, float]`
  (for `AXIS=VALUE`) and `validate_overrides(overrides, suite) ->
  None` (§5 rules). Split out so CLI and tests can exercise the
  validator without constructing a full CLI context.
- `__init__.py` re-exports `replay_one` and a small
  `ReplayResult` TypedDict for the paired-JSON envelope.

CLI glue (`cli.replay`) lives in `src/gauntlet/cli.py` alongside
`run` / `report` / `compare`, consistent with the single-CLI-module
pattern established in Task 9.

One minor edit to `src/gauntlet/runner/__init__.py`: re-export
`execute_one` (currently `_execute_one`, prefixed) as a public
primitive, with a docstring line noting replay is its second caller.
This is the only change to the existing runner module.

No new top-level dependency.

---

## 11. Open questions (with defaults)

1. **Should `Episode.metadata` carry `n_cells` / `episodes_per_cell`?**
   Default: **yes** (§3). Additive, tiny, and makes replay robust
   against suite-YAML edits between run and replay. Alternative
   (read topology from the suite at replay time) works when the
   suite is unchanged, which is the common case, so the legacy
   fallback in §9 test 11 covers pre-RFC Episodes. Decision requires
   one commit's worth of change to `runner.runner._build_work_items`
   and the §9 test to go from warn-and-fall-back to hard-require.

2. **Should `--record-trajectories <dir>` be added to `replay` now?**
   Default: **no, defer**. Phase 1 Episodes do not capture
   per-step traces and there is no sibling RFC on `main` that
   adds the Runner-level hook for them. If and when a sibling RFC
   lands that feature, extending `replay` to consume it is a
   one-line `--record-trajectories` flag that forwards to the same
   hook; nothing in this RFC forecloses that. Mentioning it here
   keeps a forward-reference without coupling the two efforts.

3. **Should `--episode-id` accept an integer global ordinal
   (`cell_index * eps_per_cell + episode_index`) in addition to
   `"C:E"`?**
   Default: **no**. The global ordinal couples to
   `episodes_per_cell`, which is a run-time quantity; one-suite-edit
   later the same integer resolves to a different episode. `"C:E"`
   is topology-free.

4. **Should replay emit an HTML artefact for side-by-side
   original-vs-replayed?**
   Default: **no**. JSON is the contract; an HTML companion is a
   follow-up RFC if and when demand shows up. Replay is fundamentally
   an interactive, one-shot operation — the report surface is for
   aggregate analysis.

5. **Should replay's output file be appendable (a `.jsonl`) so a
   user can scripts-loop `replay` across many episodes and diff
   them later?**
   Default: **no**. Each replay writes a single self-contained JSON
   object; composing them is `jq -s` territory. If a batch-replay
   use case emerges, it gets its own subcommand (`gauntlet
   replay-sweep` or similar) and its own RFC.

6. **Should the replay CLI fail when the replayed policy differs
   from the run policy?** We don't know what policy produced the
   original episodes (§8). Default: **no check**, document in
   `--help` that the policy arg is what will be simulated, no
   retrospective matching. If Phase-2 teams add a `policy_spec`
   field to Episode metadata later (reasonable), a soft warning
   on mismatch is a two-line add.

---

## 12. Implementation checklist

Six commits, each leaving `pytest`, `ruff`, and `mypy --strict`
green. Branch: `phase-2/trajectory-replay`.

1. **Echo topology into `Episode.metadata`**. Add `n_cells: int`
   and `episodes_per_cell: int` fields to `runner.worker.WorkItem`
   (the dataclass is the chokepoint that carries data from Runner
   into `_execute_one`). Edit `runner.runner._build_work_items` to
   populate them from `len(cells)` and `suite.episodes_per_cell`.
   Edit `_execute_one` to copy them into `Episode.metadata`
   alongside the existing `master_seed`. Landing them on the
   WorkItem (not reading inline in `_execute_one`) matters because
   `replay_one` (commit 4) reconstructs a WorkItem from an existing
   Episode and needs somewhere to put the topology values back. One
   new test in `tests/test_runner.py` asserting their presence and
   correctness. This lands first so the rest of the RFC can key
   off the new metadata.

2. **Publicise `execute_one`**. Remove the underscore prefix in
   `runner/worker.py` OR re-export it as `execute_one` from
   `runner/__init__.py` without renaming in-place (less churn).
   Add a docstring line naming replay as the second caller. No
   behaviour change; existing runner tests should pass unmodified.

3. **`gauntlet.replay.overrides`**. `parse_override` +
   `validate_overrides` implementing §5. Isolated unit tests in
   `tests/test_replay_overrides.py` covering: happy path,
   unknown-axis, out-of-envelope, categorical-value-not-in-set,
   malformed `"foo"` / `"foo=bar=baz"` / `"=1.0"` syntax,
   `distractor_count` int-clamp at the env boundary.

4. **`gauntlet.replay.replay_one`**. The library primitive.
   Reconstructs the SeedSequence node, builds the WorkItem, calls
   `execute_one`. Handles the legacy-Episode-without-topology
   fallback (§9 test 11). Tests for bit-identity (§9 tests 1–6)
   live here and go direct to `replay_one` without the CLI.

5. **`gauntlet replay` CLI subcommand**. `cli.replay` function
   wiring `load_suite`, `resolve_policy_factory`, `replay_one`,
   and the paired-JSON output (§8). Tests via `typer.testing.CliRunner`
   covering: happy path (end-to-end: `run` a tiny suite, pick an
   episode, `replay` it, diff), suite-name mismatch, unknown
   episode id, unknown axis, out-of-envelope value.

6. **Docs + quickstart snippet**. Extend `README.md` with a
   three-line `gauntlet replay` quickstart mirroring Task 10's
   existing `run`/`report` quickstart, and one-paragraph section
   in the spec's Phase-2-done subsection (to be tracked separately
   from this RFC). Update `examples/evaluate_random_policy.py` if
   helpful — purely additive, no behaviour change.

Acceptance: `pytest -q` clean; `ruff check` clean; `mypy --strict`
clean; `gauntlet replay --help` renders; the zero-override replay
round-trip from §9 test 1 passes on CI.
