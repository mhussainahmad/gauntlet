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

Pre-alpha. Phase 1 (MVP scaffold + tabletop env) is in progress. The
public surface is unstable.

## Quickstart (target — not all wired up yet)

```bash
uv sync
uv run gauntlet run suites/tabletop-basic-v1.yaml --policy random --out out/
uv run gauntlet report out/results.json
```

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
