# Performance benchmarks

A small family of `scripts/bench_*.py` benchmarks captures reproducible
numbers for the most performance-sensitive operations in `gauntlet`.
The benches are deliberately stdlib-plus-numpy-only (no `pyperf`,
`pytest-benchmark`, or other added dependencies) so they can run in any
default-job environment without an extras install.

Each script accepts a `--quick` flag for fast smoke runs and prints a
single-line JSON summary as the *last* line of stdout, so a CI job can
extract the result with `tail -n 1`.

## Scripts

| Script | What it measures | `--quick` knobs |
|--------|------------------|-----------------|
| `scripts/bench_rollout.py` | `TabletopEnv` construction time + per-step latency (p50/p95/p99) + per-episode wall, no `Runner` wrapping. | 5 episodes x 20 steps |
| `scripts/bench_runner.py` | `Runner.run` wall time on a synthetic 9-cell suite at `n_workers in {1, 2, 4, 8}`; reports speedup vs `n_workers=1`. | `n_workers in {1, 2}`, 9-item suite, 15 steps/episode |
| `scripts/bench_suite_loader.py` | `load_suite_from_string` mean + p95 wall over 1-axis x 2-steps, 3-axis x 5-steps, 5-axis x 10-steps synthetic suites. | 20 repetitions per case |
| `scripts/bench_report.py` | `build_report` wall on synthetic Episode lists at N=10, 100, 1000, 10000. | drops the N=10000 case |

## How to run

```bash
# Single bench, quick mode (sub-second wall on a laptop):
uv run python scripts/bench_rollout.py --quick

# Full sweep (default knobs):
uv run python scripts/bench_rollout.py
uv run python scripts/bench_runner.py
uv run python scripts/bench_suite_loader.py
uv run python scripts/bench_report.py

# Capture the JSON line for downstream tooling:
uv run python scripts/bench_report.py --quick | tail -n 1 > /tmp/bench_report.json
```

The full (non-`--quick`) `bench_runner` sweep includes `n_workers=8`
and may spawn 8 MuJoCo subprocesses; on a constrained machine the
script catches per-worker-count failures and lands them as `null`
in the JSON output rather than aborting the whole sweep.

## Canonical numbers

The numbers below come from a `--quick` run of every bench inside this
worktree. They are intended as a baseline for spotting regressions
(>2x slowdown is the rule of thumb that should trigger PR-review
discussion); they are not absolute targets.

* `<machine-spec>`: Linux 6.17.0-22-generic, AMD Ryzen 7 4800H (16 logical
  CPUs, 1.4-2.9 GHz), Python 3.11.13.
* `<date>`: 2026-04-24.
* All numbers from `--quick` mode; full-sweep numbers will be larger.

### `bench_rollout --quick`

| Metric | Value |
|--------|-------|
| `construct_ms` | 22.3 ms |
| `step_p50_ms` | 0.071 ms |
| `step_p95_ms` | 0.083 ms |
| `step_p99_ms` | 0.090 ms |
| `step_mean_ms` | 0.073 ms |
| `episode_mean_ms` | 1.55 ms |

5 episodes x 20 steps = 100 step samples.

### `bench_runner --quick`

| `n_workers` | wall (ms) | speedup vs n=1 |
|-------------|-----------|----------------|
| 1 | 35.7 | 1.00x |
| 2 | 394.9 | 0.09x |

The `n=2` measurement is dominated by `spawn`-process startup +
per-worker MuJoCo MJCF compile (one `pool_initializer` call per
worker), not by step throughput. On the 9-item smoke suite there is
not enough work per worker to amortise startup; the full-sweep numbers
(default knobs) put the cross-over closer to `n_workers=4`. This is
expected and documented in the script's module docstring.

### `bench_suite_loader --quick`

| Shape | cells | YAML bytes | mean (ms) | p95 (ms) |
|-------|------:|-----------:|----------:|---------:|
| 1-axis x 2-steps | 2 | 133 | 0.43 | 0.43 |
| 3-axis x 5-steps | 125 | 260 | 0.73 | 0.74 |
| 5-axis x 10-steps | 100,000 | 383 | 1.08 | 1.12 |

20 repetitions per case. Wall time scales with axis count, not with
grid volume — the loader does not enumerate the Cartesian product
(`Suite.cells` is a generator). If a future change makes parse time
grow with cell count (e.g. eager enumeration in a validator) the 5x10
case would shoot up by ~3 orders of magnitude and surface here first.

### `bench_report --quick`

| `n_episodes` | per_cell groups | failure clusters | build (ms) |
|-------------:|----------------:|-----------------:|-----------:|
| 10 | 10 | 0 | 0.19 |
| 100 | 60 | 2 | 0.85 |
| 1000 | 100 | 3 | 6.14 |

Cluster + heatmap iteration dominates at large N; the jump from N=100
to N=1000 is roughly 7x for a 10x volume increase (still sub-linear
because the per-cell grid saturates at 100 unique cells). A regression
that pushes N=1000 above ~15 ms — or makes the dropped N=10000 case
take more than ~150 ms in a full-sweep run — is the signal.

## Regression-monitoring strategy

The benches are reproducible single-process scripts; they are not
wired into pytest (the full pytest suite is laptop-hostile per the
project's testing policy) and they do not currently gate CI.

Suggested cadence:

1. Re-run all four `--quick` benches against `main` whenever a PR
   touches `src/gauntlet/runner/`, `src/gauntlet/suite/`,
   `src/gauntlet/report/`, or `src/gauntlet/env/tabletop.py`.
2. Compare each metric against the canonical numbers above. Flag in PR
   review any metric that is >2x worse than the canonical value on the
   same machine class.
3. Update the canonical numbers above when an intentional algorithmic
   change shifts the baseline (e.g. the planned HNSW-style cluster
   acceleration would lower `bench_report` numbers — re-baseline once
   the PR lands, in the same commit, with a one-line justification).

The scripts are write-once, read-often: the JSON summary at the end of
each script's stdout makes it cheap to plumb into a future CI job
(`tail -n 1 | jq '.step_p95_ms'` or similar) when the cadence above
proves insufficient.
