# Fleet aggregation: cross-run failure-mode clustering

`gauntlet aggregate <dir>` walks every `report.json` under a directory
and produces two artefacts:

| Path                           | Contents                                                        |
| ------------------------------ | --------------------------------------------------------------- |
| `<out>/fleet_report.json`      | Per-axis roll-up + persistent-failure clusters across all runs. |
| `<out>/fleet_clustering.json`  | Failure-mode clusters across runs (Phase 3 Task 19, T19).       |

`fleet_report.json` is the existing meta-report (RFC §7); this page
documents the **clustering** artefact added in T19.

## When to use it

Use `gauntlet aggregate` for **fleet-level cross-run analysis** — the
case where a single per-run `report.json` is not enough:

* **Multiple checkpoints of the same policy** (e.g. successive
  fine-tunes on the same suite) — does the same axis-config keep
  breaking the policy across versions?
* **Multiple operators / collection sites** — does the same failure
  mode show up across teams running the same suite, or are the
  failures site-specific?
* **Multiple tasks under the same evaluation harness** — when the same
  policy is benched on `tabletop_pick`, `tabletop_place`, and
  `tabletop_pour`, do the failures cluster by lighting / texture (a
  perception axis the operator can fix) or by task (a policy axis
  that needs a re-train)?

The single-run `gauntlet report` answers "where does THIS run break?".
`gauntlet aggregate` answers "where does the FLEET break, and which
breakage modes recur?"

## CLI surface

```
gauntlet aggregate <dir> --out <out-dir>
                         [--cluster-output <path>]
                         [--max-clusters N]
                         [--persistence-threshold P]
                         [--html / --no-html]
```

| Flag                       | Default                          | Purpose                                                                       |
| -------------------------- | -------------------------------- | ----------------------------------------------------------------------------- |
| `<dir>`                    | (required)                       | Recursively scanned for files literally named `report.json`.                  |
| `--out`                    | (required)                       | Output directory; receives `fleet_report.json` (+ `.html`).                   |
| `--cluster-output`         | `<out>/fleet_clustering.json`    | Where to write the failure-mode clustering JSON. Pass to override.            |
| `--max-clusters`           | `8`                              | Hard cap on cluster count; enforced by single-linkage agglomerative merge.    |
| `--persistence-threshold`  | `0.5`                            | Existing meta-report knob — fraction of runs a cluster must appear in.        |
| `--html` / `--no-html`     | `--html`                         | Render `fleet_report.html` alongside the JSON.                                |

`<dir>`, `--out`, and `--cluster-output` are all routed through
`gauntlet.security.safe_join` at the boundary — symlink-out-of-tree
candidates inside the scan root are filtered, and an explicit
absolute-path-injection in `--cluster-output` is rejected.

## Failure-signature schema

Every cell that appears in a per-run `failure_clusters` list contributes
one **failure signature** to the clustering pipeline:

```python
{
    "axes": {
        "lighting_intensity": 0.3,    # axis name -> normalised float value
        "object_texture":     0.0,
    },
    "wilson_lower_bound": 0.45,        # rounded to a 0.05 grid
    "behavioural_metric_summary": {    # rounded to a 0.1 grid; None preserved
        "mean_jerk_rms":          1.2,
        "mean_path_length_ratio": 1.5,
        "mean_time_to_success":   None,   # measurement absent for this run
    },
}
```

Source fields by component:

* **`axes`** — `FailureCluster.axes` (which is itself an axis-pair from
  the per-run report's `failure_clusters` list). Values are normalised
  to 9 decimal places so 1e-15 floating-point drift cannot split a
  bucket.
* **`wilson_lower_bound`** — `FailureCluster.ci_low` (B-03 Wilson 95% CI
  on the failure rate). Rounded to a 0.05 grid so two near-identical
  cells (`0.41` vs `0.43`) collapse, while distinctly different cells
  (`0.40` vs `0.55`) stay apart. Pre-B-03 reports without `ci_low` are
  re-computed from `(n_episodes, n_success)`.
* **`behavioural_metric_summary`** — three behavioural fields lifted
  from `FailureCluster` (B-02 / B-21):
  `mean_jerk_rms`, `mean_path_length_ratio`, `mean_time_to_success`.
  Rounded to a 0.1 grid. **`None` is preserved as `None`** — a missing
  measurement is not the same data point as a measured-but-zero one,
  so two clusters with absent telemetry stay distinct from two clusters
  with `0.0` telemetry.

The signature is the input to two passes:

1. **Hash-bucket pass** — equal signatures collapse into the same
   bucket. Runs that share a fingerprint go straight into the same
   cluster.
2. **Agglomerative merge pass** — when the bucket count exceeds
   `--max-clusters`, the closest bucket pairs are merged via
   single-linkage on a composite distance until the cap is met.

The composite distance is the sum of three normalised components
(equally weighted, total in `[0, 3]`):

* **Axis Hamming** — fraction of (axis-name, value) pairs that differ.
* **Wilson L1** — absolute difference of grid-rounded Wilson lower
  bounds.
* **Behavioural L1** — mean absolute behavioural-metric delta; `None`
  on either side contributes a soft mismatch (`0.5`) instead of a
  hard one (`1.0`).

Rank order matters more than absolute scale here — the metric is only
used to pick the next pair to merge.

## JSON output schema

`fleet_clustering.json` is a single object:

```jsonc
{
  "clusters": [
    {
      "cluster_id": 0,
      "member_run_ids": ["run-a", "run-b", "run-c"],
      "representative_failure_signature": { /* see above */ },
      "cross_run_consistency": 0.75
    }
    // ...
  ],
  "n_runs": 4,
  "n_unique_failures": 12,
  "silhouette": 0.42
}
```

Field semantics:

| Field                   | Type            | Meaning                                                                                                                                                     |
| ----------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clusters`              | array           | Sorted by descending `len(member_run_ids)`, then by signature for stable ties. Empty when no run has any failure cluster.                                   |
| `cluster_id`            | int             | Zero-indexed, dense — `0..n_clusters-1`. Stable across re-runs against the same input.                                                                      |
| `member_run_ids`        | array of string | Sorted, deduped. Each entry is the run-id (the parent directory name of the contributing `report.json`, mirroring `FleetRun.run_id`).                       |
| `representative_failure_signature` | object | The **medoid** signature in the cluster — the one with minimum summed intra-cluster distance. Not an arbitrary first-encountered member.                |
| `cross_run_consistency` | float in `[0, 1]` | Fraction of total fleet runs that hit this failure mode. `1.0` = every run; `0.05` = a long-tail mode that bites one run in twenty.                       |
| `n_runs`                | int             | Total number of `report.json` files scanned. Zero is allowed and short-circuits clustering.                                                                 |
| `n_unique_failures`     | int             | Number of distinct signatures **before** the agglomerative-merge pass capped the count to `--max-clusters`. Lets a caller see "47 modes, top 8 returned".   |
| `silhouette`            | float or null   | Mean Rousseeuw silhouette over the post-merge clusters in `[-1, 1]`, or `null` when undefined (zero / one cluster, one unique signature).                   |

## Edge cases

The clustering layer is built around these documented short-circuits
(pinned by `tests/test_aggregate_fleet_clustering.py`):

* **Empty directory** — no `report.json` files anywhere. Returns
  `n_runs=0`, `clusters=[]`, `silhouette=null`. No exception.
* **Only clean runs** — `report.json` files exist but none has a
  failure cluster. Returns `n_runs > 0`, `clusters=[]`,
  `silhouette=null`.
* **Single run** — only one `report.json`. Clustering still runs and
  emits one cluster per signature, but `silhouette` is `null`
  (Rousseeuw silhouette is undefined with fewer than two clusters).
* **`max_clusters < 1`** — programmer error, `ValueError` raised by
  the underlying API. The CLI surfaces it as a Typer parse-time
  rejection (`min=1`).

## Anti-features

This module is **not** a substitute for the per-run failure-cluster
analysis in `gauntlet.report.analyze`. The fleet clustering is built on
top of per-run clusters — it never re-derives them from the underlying
episodes. If a single run has no failure clusters (a tiny suite, a
clean policy, or a B-26 pruning that fired before any failures
materialised), that run contributes zero signatures to the fleet
analysis, regardless of how many episodes it carried.

The signature includes only three behavioural metrics
(`mean_jerk_rms`, `mean_path_length_ratio`, `mean_time_to_success`)
and not the full B-02 / B-21 / B-30 / B-37 surface — adding more
fields would explode the bucket count toward "every cell its own
cluster" without buying additional separation power. Future work could
make the field set configurable, but the CLI default is intentionally
conservative.
