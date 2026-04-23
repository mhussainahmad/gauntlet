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

### Using a real VLA

- Install the HF extras: `uv sync --extra hf` (pulls torch / transformers / pillow; core installs stay torch-free).
- See [`examples/evaluate_openvla.py`](./examples/evaluate_openvla.py) for the ≤20-line OpenVLA-7B factory.
- Image-conditioned policies need a rendered frame — construct `TabletopEnv(render_in_obs=True)` so `obs["image"]` is emitted.
- SmolVLA: `uv sync --extra lerobot`; see [`examples/evaluate_smolvla.py`](./examples/evaluate_smolvla.py).
- SmolVLA-base is pretrained on SO-100 (6-D joint) whereas TabletopEnv is 7-D EE-twist+gripper — zero-shot success is ~0% by embodiment mismatch; fine-tune on TabletopEnv-compatible data for meaningful evaluation.

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
