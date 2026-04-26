# Performance benchmarks

A small family of `scripts/bench_*.py` benchmarks captures reproducible
numbers for the most performance-sensitive operations in `gauntlet`.
The benches are deliberately stdlib-plus-numpy-only (no `pyperf`,
`pytest-benchmark`, or other added dependencies) so they can run in any
default-job environment without an extras install.

Each script accepts a `--quick` flag for fast smoke runs and prints a
single-line JSON summary as the *last* line of stdout, so a CI job can
extract the result with `tail -n 1`. Each script also writes a JSON
sidecar file (`<bench_name>.json` by default; `--out PATH` overrides)
containing the full pretty-printed summary so downstream tooling does
not have to parse stdout.

## Scripts

| Script | What it measures | `--quick` knobs |
|--------|------------------|-----------------|
| `scripts/bench_rollout.py` | Per-backend tabletop rollout: env construction time, per-step latency (p50/p95/p99/mean), per-episode wall, and episodes/second. Defaults to the always-installed MuJoCo `tabletop` backend; PyBullet / Genesis / Isaac selectable via `--backend` and skip-not-fail when their extras are not installed. | 5 episodes x 20 steps |
| `scripts/bench_render_latency.py` | Per-frame `obs["image"]` wall-clock on `TabletopEnv(render_in_obs=True)` at 84x84, 224x224, and 480x640. Skip-not-fail when the offscreen GL context cannot be created (no EGL / OSMesa / display). | 10 frames per size, 2 warmup |
| `scripts/bench_runner.py` | `Runner.run` wall on a tabletop suite at `n_workers in {1, 2, 4, 8}` (clamped to `os.cpu_count()`); reports speedup vs n=1 and parallel efficiency (`speedup / n_workers`). Default suite is a synthetic 6-cell shape; `--from-suite PATH` switches to a bundled YAML (e.g. `examples/suites/tabletop-smoke.yaml` => 24 episodes — the spec's reference sweep). | `n_workers in {1, 2}`, 6-item suite, 15 steps/episode |
| `scripts/bench_suite_loader.py` | `load_suite_from_string` mean + p95 wall over 1-axis x 2-steps, 3-axis x 5-steps, 5-axis x 10-steps synthetic suites. With `--bundled` also parses every `examples/suites/*.yaml` (skip-not-fail per file when the YAML's backend extra is missing). | 20 reps per case |
| `scripts/bench_report.py` | `build_report` wall on synthetic Episode lists at N=10, 100, 1000, 10000. | drops the N=10000 case |

## Phase 2.5 T12 benchmark suite (`scripts/perf/bench_*.py`)

A second, complementary family of benchmarks lives under
`scripts/perf/`. The Polish-task family above (`scripts/bench_*.py`)
deliberately bypasses the Runner to isolate env-primitive regressions;
the T12 family runs *through* the Runner so the full stack — Runner
construction, episode dispatch, seed derivation, Episode building —
is in the measured wall. A regression in one layer surfaces in only
one of the two families, narrowing a bisect.

| Script | What it measures |
|--------|------------------|
| `scripts/perf/bench_rollout.py` | Per-backend rollout throughput via `gauntlet.runner.Runner` driving the `RandomPolicy` baseline against a 1-cell tabletop suite. Reports `episodes_per_sec` plus per-step latency `step_p50_ms` / `step_p95_ms` / `step_p99_ms`. Backends: `mujoco` (always-on), `pybullet` / `genesis` / `isaac` (skip-not-fail when their extra is missing). |
| `scripts/perf/bench_render.py` | Per-step offscreen-render latency on `TabletopEnv(render_in_obs=True)` at 224x224. Reports `frames_per_sec` plus `render_step_p50_ms` / `render_step_p95_ms` / `render_step_p99_ms`. MuJoCo only (the only always-on render backend per repo conventions); skip-not-fail when the offscreen GL context cannot be created. |
| `scripts/perf/bench_suite_loader.py` | `load_suite` parse wall on a synthetic `--cells`-cell suite plus the B-40 `compute_suite_provenance_hash` wall (the canonical-payload AST hash that gates the result cache). Reports `load_time_ms` and `ast_hash_time_ms`. |
| `scripts/perf/bench_runner_scaling.py` | `Runner.run` wall sweep across `--workers 1,2,4,8` (clamped to `os.cpu_count()`) on a 6-cell tabletop suite. Reports the speedup curve plus an Amdahl's-law fit (`amdahl_serial_frac`, `amdahl_parallel_frac`). |

Each T12 script writes a flat JSON sidecar with these contract fields:

* metric-specific keys (per the table above)
* `version` — `gauntlet.__version__`
* `timestamp` — ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`)
* `git_commit` — `git rev-parse HEAD` (or `null` outside a checkout)
* `partial` — `true` when `KeyboardInterrupt` cut the run short

The CLI surface is uniform: `--seed` (default `42`), `--output PATH`
(default `benchmarks/<name>.json`), and metric-specific knobs. Every
script handles `KeyboardInterrupt` by emitting a partial sidecar with
`partial: true` rather than crashing without output.

```bash
# Each T12 bench, full sweep:
python scripts/perf/bench_rollout.py --backend mujoco --episodes 50 \
    --output benchmarks/rollout.json
python scripts/perf/bench_render.py --steps 200 \
    --output benchmarks/render.json
python scripts/perf/bench_suite_loader.py --cells 1000 \
    --output benchmarks/suite_loader.json
python scripts/perf/bench_runner_scaling.py \
    --output benchmarks/scaling.json
```

The T12 scripts are smoke-tested by `tests/test_bench_smoke.py`,
marked `@pytest.mark.slow` so the default pytest run skips them; opt
in with `pytest -m slow tests/test_bench_smoke.py`. The smoke tests
invoke each CLI with the smallest viable budget (`--episodes 2`,
`--steps 5`, `--cells 10`, `--workers 1`) and assert the resulting
JSON sidecar matches the contract above.

The T12 benchmarks are **not** regression-gated in CI — they are
reference tooling for users and maintainers, run manually whenever
`src/gauntlet/runner/`, `src/gauntlet/suite/`, or
`src/gauntlet/env/tabletop.py` see a non-trivial change. Sidecar
JSON files land under `benchmarks/` and are gitignored; the scripts
and this documentation are versioned, the actual numbers are not.

### Reproducibility notes

* **Seed.** Default `--seed 42` everywhere. The runner derives all
  per-episode seeds from this single value (see
  `src/gauntlet/runner/runner.py` "Seed derivation"), so a re-run on
  the same machine class with the same seed produces bit-identical
  Episodes — and bench wall numbers reproduce within the noise floor
  documented under "Variance to expect" below.
* **Network.** No T12 script calls out to the network. All synthetic
  suites are constructed in-memory; the loader bench stages a tiny
  YAML to a tmp file under the output's parent directory and removes
  it on exit.
* **Host hardware shape.** Wall numbers shift with CPU class, Python
  build, and (for `bench_render`) the offscreen GL stack. Compare
  numbers only across runs on the same host. The provenance fields
  (`version` / `timestamp` / `git_commit`) stamp each sidecar so a
  downstream consumer can correlate a number with its tree; capture
  the host CPU / Python / mujoco-wheel triple alongside if you start
  a numbers spreadsheet.

## How to run

```bash
# Single bench, quick mode (sub-second wall on a laptop):
uv run --no-sync python scripts/bench_rollout.py --quick

# Full sweep (default knobs):
uv run --no-sync python scripts/bench_rollout.py
uv run --no-sync python scripts/bench_render_latency.py
uv run --no-sync python scripts/bench_runner.py
uv run --no-sync python scripts/bench_suite_loader.py
uv run --no-sync python scripts/bench_report.py

# Per-backend rollout (skip-not-fail when extra missing):
uv run --no-sync python scripts/bench_rollout.py --backend tabletop-pybullet
uv run --no-sync python scripts/bench_rollout.py --backend tabletop-genesis
uv run --no-sync python scripts/bench_rollout.py --backend tabletop-isaac

# Runner scaling against the bundled smoke suite (24 episodes):
uv run --no-sync python scripts/bench_runner.py \
    --from-suite examples/suites/tabletop-smoke.yaml

# Suite-loader sweep including every bundled YAML:
uv run --no-sync python scripts/bench_suite_loader.py --bundled

# Capture the JSON sidecar at a custom location:
uv run --no-sync python scripts/bench_report.py --out /tmp/bench_report.json

# Capture only the JSON line for downstream tooling:
uv run --no-sync python scripts/bench_report.py --quick | tail -n 1 > /tmp/bench_report.json
```

The full (non-`--quick`) `bench_runner` sweep includes `n_workers=8`
and may spawn 8 MuJoCo subprocesses; on a constrained machine the
script catches per-worker-count failures and lands them as `null`
in the JSON output rather than aborting the whole sweep.

## Methodology notes

* **Timing primitive.** All benches use `time.perf_counter()`. No new
  dependencies (no `pyperf`, no `pytest-benchmark`).
* **Warmup.** `bench_render_latency` runs `--warmup` (default 3) frames
  before measurement so MuJoCo's offscreen GL context (shader compile,
  FBO allocation) is amortised before the first measured frame. The
  rollout / runner / report / loader benches do not warm up — they
  intentionally include first-call cost so a cold-start regression
  (e.g. an import-time hit added to `gauntlet.report`) lands in the
  numbers.
* **Sample size.** Defaults are tuned to finish each bench in seconds
  on a laptop while still giving a stable p99 (n=100 step samples for
  rollout, n=30 frames for render). The bar this monitors is >2x
  regressions, not sub-millisecond drift.
* **Variance to expect.** Across runs on the same machine, expect
  10-30% wall jitter on every metric — JIT/cache warmup, kernel
  scheduling, and (for parallel scaling) MuJoCo's per-worker MJCF
  compile dominate sub-bench-window noise. A single >2x regression
  is the signal worth investigating; a 1.3x bump may just be a noisy
  re-run.
* **What NOT to compare across machines.** Absolute wall numbers are
  *only* comparable when machine class, Python version, mujoco wheel,
  and (for render) GL backend match. Use these numbers to spot local
  regressions, not to advertise hardware-relative throughput.

## Canonical numbers

The numbers below come from a fresh run of every bench inside this
worktree. They are intended as a baseline for spotting regressions
(>2x slowdown is the rule of thumb that should trigger PR-review
discussion); they are not absolute targets.

* `<machine-spec>`: Linux 6.17.0-22-generic x86_64, AMD Ryzen 7 4800H
  (16 logical CPUs, 1.4-2.9 GHz), Python 3.11.13.
* `<date>`: 2026-04-25.
* `--quick` numbers are tagged where used; full-sweep numbers are
  larger.

### `bench_rollout --quick` (tabletop / MuJoCo)

| Metric | Value |
|--------|-------|
| `construct_ms` | ~30 ms |
| `step_p50_ms` | 0.09 ms |
| `step_p95_ms` | 0.14 ms |
| `step_p99_ms` | 0.16 ms |
| `step_mean_ms` | 0.10 ms |
| `episode_mean_ms` | 2.04 ms |
| `episodes_per_sec` | ~390 ep/s |

5 episodes x 20 steps = 100 step samples. `episodes_per_sec` is the
total measured-window throughput (5 episodes / total wall) and is the
headline number to compare against future runs.

PyBullet / Genesis / Isaac backends are skipped on a default-job
environment (their extras are not installed); the bench prints
`skipped: backend 'tabletop-X' not importable (extra not installed)`
and exits 0. JSON sidecar still written (with `"skipped": true`).

### `bench_render_latency` (full sweep, 30 frames per size)

| Size (HxW) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | fps_mean |
|-----------|---------:|---------:|---------:|----------:|---------:|
| 84 x 84   | 18.7 | 23.9 | 25.2 | 19.0 | 52.6 |
| 224 x 224 | 20.5 | 23.2 | 35.1 | 21.0 | 47.7 |
| 480 x 640 | 25.0 | 28.9 | 45.4 | 26.1 | 38.4 |

Render latency is dominated by MuJoCo's offscreen GL readback (the
per-step CPU cost of `step()` itself stays sub-millisecond — see
`bench_rollout` above). Wall scales sub-linearly with pixel count
because the framebuffer copy is bandwidth-limited well below the
peak the system supports. A regression that pushes the 224x224 mean
above ~50 ms is the signal.

If your machine cannot create an offscreen GL context (typical
container / CI without EGL or OSMesa) the bench prints
`skipped: <reason>` and exits 0; the JSON sidecar records
`"skipped": true` with the underlying exception text.

### `bench_runner --from-suite examples/suites/tabletop-smoke.yaml` (24 episodes)

| `n_workers` | wall (ms) | speedup vs n=1 | efficiency |
|-------------|----------:|---------------:|-----------:|
| 1 | 119.6 | 1.00x | 1.000 |
| 2 | 548.7 | 0.22x | 0.109 |
| 4 | 695.1 | 0.17x | 0.043 |
| 8 | 771.2 | 0.16x | 0.019 |

Worker counts are clamped to `os.cpu_count()` (16 logical cores on
the reference machine — no clamping triggered).

The 24-episode smoke suite is **too small** to amortise the per-worker
spawn cost: each MuJoCo worker pays a fixed `pool_initializer` MJCF
compile + offscreen GL setup of ~200-400 ms, while the 24 actual
work items each finish in ~1-2 ms. The 1-worker number is therefore
the single number worth tracking against this suite; the higher
worker counts are documented here so the regression-monitor catches a
Pool-startup blowup but they are not a recommendation to use n=8 for
24 episodes. For a real workload (hundreds-to-thousands of episodes)
the cross-over moves left and parallel scaling becomes useful — see
the `bench_runner.py` default synthetic-suite mode for the
finer-grained scaling sweep.

### `bench_runner --quick` (synthetic suite, 6 items)

| `n_workers` | wall (ms) | speedup vs n=1 |
|-------------|-----------|----------------|
| 1 | ~36 | 1.00x |
| 2 | ~395 | 0.09x |

Same behaviour at smaller scale: spawn overhead dominates a 6-item
suite. This is by design — `--quick` is for "did the script still
import + run end-to-end?" smoke checks, not for parallel-scaling data.

### `bench_suite_loader` (full sweep + `--bundled`)

Synthetic shapes (200 reps each):

| Shape | cells | YAML bytes | mean (ms) | p95 (ms) |
|-------|------:|-----------:|----------:|---------:|
| 1-axis x 2-steps | 2 | 133 | 0.46 | 0.61 |
| 3-axis x 5-steps | 125 | 260 | 0.87 | 1.11 |
| 5-axis x 10-steps | 100,000 | 383 | 1.29 | 1.62 |

Bundled `examples/suites/*.yaml` (200 reps each; backend-extra-only
files skipped):

| File | cells | YAML bytes | mean (ms) | p95 (ms) |
|------|------:|-----------:|----------:|---------:|
| `tabletop-basic-v1.yaml` | 144 | 658 | 1.35 | 1.57 |
| `tabletop-lhs-smoke.yaml` | 32 | 939 | 1.44 | 1.68 |
| `tabletop-smoke.yaml` | 6 | 646 | 0.92 | 1.20 |
| `tabletop-sobol-smoke.yaml` | 32 | 1,349 | 1.76 | 2.50 |

Skipped (`extra not installed`): `tabletop-genesis-smoke.yaml`,
`tabletop-isaac-smoke.yaml`, `tabletop-pybullet-smoke.yaml`.

Wall time scales with axis count, not with grid volume — the loader
does not enumerate the Cartesian product (`Suite.cells` is a generator).
If a future change makes parse time grow with cell count (e.g. eager
enumeration in a validator) the 5x10 case would shoot up by ~3 orders
of magnitude and surface here first.

### `bench_report` (full sweep, includes N=10000)

| `n_episodes` | per_cell groups | failure clusters | build (ms) |
|-------------:|----------------:|-----------------:|-----------:|
| 10 | 10 | 0 | 0.23 |
| 100 | 60 | 2 | 0.93 |
| 1000 | 100 | 3 | 6.82 |
| 10000 | 100 | 0 | 69.61 |

Cluster + heatmap iteration dominates at large N; the jump from
N=1000 to N=10000 is roughly 10x for a 10x volume increase
(approximately linear once the per-cell grid saturates at 100 unique
cells). A regression that pushes N=1000 above ~15 ms — or makes the
N=10000 case take more than ~150 ms — is the signal.

## Regression-monitoring strategy

The benches are reproducible single-process scripts; they are not
wired into pytest (the full pytest suite is laptop-hostile per the
project's testing policy) and they do not currently gate CI.

Suggested cadence:

1. Re-run all five `--quick` benches against `main` whenever a PR
   touches `src/gauntlet/runner/`, `src/gauntlet/suite/`,
   `src/gauntlet/report/`, or `src/gauntlet/env/tabletop.py`.
2. Compare each metric against the canonical numbers above. Flag in PR
   review any metric that is >2x worse than the canonical value on the
   same machine class.
3. Update the canonical numbers above when an intentional algorithmic
   change shifts the baseline (e.g. the planned HNSW-style cluster
   acceleration would lower `bench_report` numbers — re-baseline once
   the PR lands, in the same commit, with a one-line justification).

The scripts are write-once, read-often: the JSON sidecar (`*.json`)
plus the trailing JSON line on stdout makes it cheap to plumb into a
future CI job (`jq '.step_p95_ms' bench_rollout.json` or
`tail -n 1 | jq '.step_p95_ms'`) when the cadence above proves
insufficient.

## Known caveats

* **`bench_runner` parallel efficiency on small suites is misleading.**
  The 24-episode smoke suite is dominated by per-worker spawn cost.
  Use the 1-worker wall as the headline number for that sweep, and
  reach for a larger `--episodes-per-cell` (or a non-smoke suite) when
  measuring scaling proper.
* **`bench_render_latency` requires offscreen GL.** Headless containers
  without EGL or OSMesa skip-not-fail. The bench does **not** install
  the GL stack; surface a CI follow-up if a target environment needs it.
* **PyBullet / Genesis / Isaac throughput numbers are not captured here.**
  The repo's default-job environment does not install those extras;
  install the matching extra (`uv sync --extra pybullet` etc.) and
  re-run `bench_rollout --backend tabletop-X` to capture them locally.
* **Numbers above are local-laptop only.** Different CPU class, Python
  build, or mujoco wheel will shift the absolutes; only intra-machine
  deltas are meaningful.
