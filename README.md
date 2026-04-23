# Gauntlet

> An evaluation harness for learned robot policies.

Gauntlet answers a single question for VLA / diffusion / scripted policies:

> *"How does this policy fail, and has the latest checkpoint regressed against the last one?"*

It wraps any policy behind a uniform adapter, runs it across a parameterized
suite of MuJoCo perturbations (lighting, camera pose, textures, clutter,
initial conditions), and produces a structured report that **breaks failures
down by axis** instead of hiding them in an aggregate mean.

See [`GAUNTLET_SPEC.md`](./GAUNTLET_SPEC.md) for the full design.

---

## Status

Phase 1 MVP is feature-complete: tabletop env, perturbation axes, parallel
runner, breakdown-first HTML report, and the `gauntlet run / report /
compare` CLI. Phase 2 starts on real-policy adapters and runtime
observability. The public surface is still unstable.

## Backends

Gauntlet ships two simulator backends. The Suite YAML's `env:` key is
the dispatch: `tabletop` uses MuJoCo (default, ships in the core
install); `tabletop-pybullet` uses PyBullet and lives behind an
optional extra.

| `env:` slug          | Simulator | Install                                  | Observations |
|----------------------|-----------|------------------------------------------|--------------|
| `tabletop`           | MuJoCo    | `uv sync` (core)                         | State + render-on-demand |
| `tabletop-pybullet`  | PyBullet  | `uv sync --extra pybullet`               | State-only (§6.2) |

The two backends share action/observation spaces byte-for-byte and the
canonical 7 perturbation axes. They are **not** numerically identical:
same policy + same seed on `tabletop` vs `tabletop-pybullet` produces
semantically similar but numerically different trajectories. Running
`gauntlet compare` across backends measures simulator drift, not
policy regression; the CLI requires `--allow-cross-backend` to proceed.

See [`docs/phase2-rfc-005-pybullet-adapter.md`](./docs/phase2-rfc-005-pybullet-adapter.md)
for the full PyBullet backend design.

## Quickstart

Three commands reproduce the end-to-end example against the bundled
smoke suite (3 lighting intensities x 2 cube textures x 4 episodes = 24
rollouts; finishes in seconds on a laptop):

```bash
uv sync
uv run gauntlet run examples/suites/tabletop-smoke.yaml --policy random --out out/
open out/report.html  # macOS: open ; Linux: xdg-open ; Windows: start
```

Artefacts land in `out/`: `episodes.json` (one record per rollout),
`report.json` (analysed breakdowns), and `report.html` — a self-contained
report leading with the failure-clusters table, then per-axis bar charts,
then 2D heatmaps of axis combinations. The smoke suite is intentionally
tiny; for the canonical 4-axis x 144-cell x 1440-rollout shape, swap the
YAML path for `examples/suites/tabletop-basic-v1.yaml`. See
[`GAUNTLET_SPEC.md`](./GAUNTLET_SPEC.md) for the full design and
[`examples/evaluate_random_policy.py`](./examples/evaluate_random_policy.py)
for the equivalent invocation via the public Python API.

## Development

```bash
# Sync deps (creates .venv, installs everything in pyproject + dev group).
uv sync

# Lint, type-check, test.
uv run ruff check .
uv run mypy
uv run pytest
```

## Project layout

```
src/gauntlet/
  policy/   # Policy adapter protocol + reference wrappers
  env/      # Parameterized MuJoCo envs with perturbation axes
  suite/    # YAML-defined perturbation grid suites
  runner/   # Parallel rollout orchestration + seed management
  report/   # Failure analysis + HTML/JSON generation
  cli.py    # gauntlet run / report / compare
```

## License

MIT.
