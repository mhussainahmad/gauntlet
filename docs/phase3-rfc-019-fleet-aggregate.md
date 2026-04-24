# Phase 3 RFC 019 — Fleet-wide failure-mode clustering

Status: Accepted
Phase: 3
Task: 19
Spec reference: `GAUNTLET_SPEC.md` §7 ("Fleet-wide failure mode clustering
from deployed robots").

## 1. Why this exists

Today `gauntlet report` analyses a single evaluation run. In production, a
team runs gauntlet across N policies × M checkpoints × K hardware configs
and ends up with a directory of `episodes.json` + `report.json` artefacts.
The questions that matter at fleet scale are:

* "Across all 47 runs, which axis-combinations cause failures most often?"
* "Which checkpoint regressed against its predecessor on the lighting axis
  specifically?"
* "Is the failure cluster `lighting=0.3 ∧ texture=brushed_metal` a
  one-off, or does it appear in a majority of our checkpoints?"

This RFC introduces `gauntlet aggregate <dir>` — a cross-run roll-up that
consumes existing per-run artefacts and produces a fleet-level
meta-report (`fleet_report.json` + `fleet_report.html`).

## 2. Non-goals

* No schema changes to `Episode` or `Report`. The aggregator is a pure
  consumer of the existing surface — backwards-compat is sacred.
* No new external dependencies. Reuses `pydantic`, `jinja2`, `typer`,
  `pathlib`, `Chart.js` (CDN).
* No deployed-robot ingestion. The fleet view is built from the same
  static `report.json` files `gauntlet run` already emits; collecting
  those from real hardware is out of scope here (covered by the existing
  ROS 2 integration).
* No web service. The HTML artefact is self-contained and openable from
  the filesystem, like every other report in the codebase.

## 3. Discovery model

`aggregate_directory(dir)` recursively globs for files literally named
`report.json` under the supplied root. This matches the
`gauntlet run --out <dir>` convention (`<dir>/report.json`), which every
in-tree example and CI artefact uses. Running the aggregator over a
parent directory therefore Just Works for the common
`<runs>/<run_id>/report.json` layout.

Files that fail to parse as a `Report` are reported with their path in
the error message — the failure surface is one `typer.Exit(1)` per
malformed run, not a silent skip. (Rationale: a fleet report that
silently ignores half its inputs would be misleading by §6 "never
aggregate away failures".)

`source_file` on each `FleetRun` is stored *relative to the scan root*,
not absolute, so the resulting `fleet_report.json` is portable (the
common workflow is `aggregate; tar -czvf fleet.tar.gz fleet_report.* runs/`).

## 4. Aggregation semantics

### 4.1 Per-axis aggregation

`per_axis_aggregate: dict[str, AxisBreakdown]` is keyed by axis name and
union-aggregates across runs:

* For each axis name appearing in any run, union the value-keys across
  all runs that carry that axis.
* `counts[v] = Σ counts_run[v]`, `successes[v] = Σ successes_run[v]`,
  `rates[v] = successes[v] / counts[v]` (or `NaN` if `counts[v] == 0`,
  which shouldn't happen but the schema tolerates it).
* Float keys are normalised through the same `_norm` helper used in
  `gauntlet.report.analyze` (9-decimal rounding) so 1e-15 jitter doesn't
  split a bucket. Reused (not duplicated) — see §8.

A run that doesn't carry an axis simply doesn't contribute; the union
order across runs is determined by first-appearance order, mirroring
`_ordered_axis_names`.

### 4.2 Persistent failure clusters

A "persistent" cluster is one whose fingerprint
`tuple(sorted((name, _norm(value)) for name, value in cluster.axes.items()))`
appears in `>= persistence_threshold * n_runs` distinct runs. The
threshold defaults to 0.5 ("majority of runs"); pass `--persistence-threshold`
to override.

Comparison uses `>=`, not `>`. A test pins this boundary explicitly: a
cluster appearing in exactly `ceil(threshold * n_runs)` runs is
included. The intuition: 0.5 means "at least half"; rounding up matches
the "majority" reading.

For each persistent cluster the fleet-level `n_episodes`, `n_success`,
`failure_rate`, and `lift` are *re-computed against the fleet baseline
failure rate*, not just copied from any one run. Sum
`n_episodes`/`n_success` across runs where the fingerprint appeared,
recompute `failure_rate = (n_episodes - n_success) / n_episodes`, and
`lift = failure_rate / fleet_baseline_failure_rate`. Otherwise the
"persistent clusters" surface is a one-run snapshot, which is
misleading. The fingerprint sort makes the match invariant under
per-run axis insertion order.

### 4.3 Cross-run success distribution

`cross_run_success_distribution: dict[str, list[float]]` is keyed by
`suite_name` and lists each run's `overall_success_rate` for that
suite, in the input-discovery order (which is `sorted(glob)`, hence
deterministic). One bucket per distinct `suite_name` in the directory —
mixed-suite directories are allowed (see §5) and produce a multi-bucket
distribution.

## 5. Heterogeneous suites

Spec §7 motivates the feature with "47 runs across N policies × M
checkpoints", which usually means one suite per directory. Nothing
forbids a directory containing runs from multiple suites, though, so:

* Mixed-suite directories are *allowed*, not an error.
* `cross_run_success_distribution` is keyed by `suite_name`, so
  per-suite trend lines stay separated.
* A `[warn]` line is written to stderr when more than one distinct
  `suite_name` appears, naming the mix. Doesn't affect exit code.
* `per_axis_aggregate` and `persistent_failure_clusters` cross
  suite boundaries on purpose — the user opted into a
  cross-suite aggregation by pointing the tool at a mixed directory.

## 6. CLI surface

```
gauntlet aggregate <dir> --out <output_dir>
                          [--persistence-threshold 0.5]
                          [--html / --no-html]
```

* `<dir>` — root to scan. Recursive glob for `report.json`.
* `--out` — output directory; created if missing. Mirrors `gauntlet run`.
* `--persistence-threshold` — fraction in `[0, 1]`. Default 0.5.
* `--html / --no-html` — render `fleet_report.html`. Default ON,
  matching `gauntlet run`.

Outputs:

* `fleet_report.json` — serialised `FleetReport`.
* `fleet_report.html` — self-contained HTML (Chart.js from CDN).

The subcommand lives next to `gauntlet compare` in `cli.py`. No other
subcommands are touched. No existing flag changes meaning.

## 7. HTML layout sketch

```
+------------------------------------------------------------+
| Gauntlet fleet report — <root>                            |
+------------------------------------------------------------+
| Summary card                                              |
|   N runs:   47                                            |
|   Total episodes: 4,213                                   |
|   Mean success:  72.4% (σ 8.1%)                           |
|   Persistence threshold: ≥ 50% of runs                    |
+------------------------------------------------------------+
| Persistent failure clusters (lead, per §6)                |
|   axis combination | n | failure rate | lift | seen_in    |
|   ────────────────────────────────────────────────────────|
|   lighting=0.3 ∧ texture=brushed | 312 | 86%  | 4.1× | 28 |
|   ...                                                     |
+------------------------------------------------------------+
| Per-axis aggregate (Chart.js bar charts)                  |
|   <one canvas per axis name>                              |
+------------------------------------------------------------+
| Per-suite success-rate distribution                       |
|   small-multiple table: suite_name | n_runs | mean | std  |
+------------------------------------------------------------+
| Per-run table (collapsed under <details>)                 |
|   run_id | suite | policy | n_eps | success_rate         |
+------------------------------------------------------------+
```

The cluster table leads, per spec §6. The per-run table is collapsed
under `<details>` — same "no wall of numbers" rule the per-cell table
follows in the per-run report.

## 8. Implementation sketch

* `src/gauntlet/aggregate/schema.py` — `FleetRun`, `FleetReport`. Both
  use `ConfigDict(extra="forbid", ser_json_inf_nan="strings")`,
  matching hotfix #18.
* `src/gauntlet/aggregate/analyze.py` — `discover_run_files`,
  `aggregate_reports`, `aggregate_directory`. Imports `_norm` from
  `gauntlet.report.analyze` (or, if `_norm` is too private, a
  re-export under a public name — TBD at implementation time;
  duplicating the constant would risk drift).
* `src/gauntlet/aggregate/html.py` + a sibling
  `templates/fleet_report.html.jinja` loaded via
  `PackageLoader("gauntlet.aggregate", "templates")`. Same autoescape
  config as the existing report HTML — XSS surface for `source_file`,
  `policy_label`, and axis names is non-trivial, so the security tests
  live in `tests/test_fleet_aggregate.py` mirroring
  `tests/test_security_html_report.py`.
* `src/gauntlet/cli.py` — one new `gauntlet aggregate` subcommand,
  placed adjacent to `compare`. Uses the existing `_write_json`,
  `_read_json`, `_fail`, `_echo_err` helpers — no duplication.

## 9. Open questions / deferred

* Per-run regression deltas (Q: "which checkpoint regressed against
  its predecessor on the lighting axis specifically") could be added
  as a follow-up by chronologically ordering runs and computing
  pairwise compare. Out of scope for this RFC; the
  `cross_run_success_distribution` already exposes the raw data
  needed to compute that downstream.
* A "fleet compare" (diff two fleet reports) is a natural extension
  but not part of this task.
* Streaming aggregation over a JSONL directory (instead of one
  `report.json` per run) is a possible follow-up if the per-run
  artefact count grows past O(thousands).
