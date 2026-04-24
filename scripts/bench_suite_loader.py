"""YAML suite-loader benchmark.

Times :func:`gauntlet.suite.load_suite_from_string` on three synthetic
suite shapes:

* 1 axis x 2 steps,
* 3 axes x 5 steps,
* 5 axes x 10 steps.

Surfaces O(grid-size) regressions in the loader / pydantic validation
path early — the loader itself does not enumerate the Cartesian product
(``Suite.cells`` is a generator), so parse time should grow with axis
count, not with grid volume.

Usage:
    uv run python scripts/bench_suite_loader.py [--repetitions N]
                                                [--quick]

The final stdout line is a single JSON object so a CI job can ``tail -n 1``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

# Import the env package so the ``tabletop`` slug is registered before
# the loader runs ``_reject_purely_visual_suites`` -> ``get_env_factory``.
import gauntlet.env  # noqa: F401  (side-effect import: registers tabletop)
from gauntlet.suite import load_suite_from_string

__all__ = ["main"]


# Five state-effecting axes available on the tabletop env. Picked from
# AXIS_NAMES; all five are continuous so the same {low,high,steps} shape
# is reused across every synthetic spec.
_AXIS_POOL: tuple[str, ...] = (
    "object_initial_pose_x",
    "object_initial_pose_y",
    "lighting_intensity",
    "camera_offset_x",
    "camera_offset_y",
)

# (n_axes, steps_per_axis) tuples driving the sweep. The synthetic
# suite size grows from 2 cells (1 x 2) to 100k cells (5 x 10), but
# the loader never materialises the grid so wall-time should track
# axis count, not the product.
_DEFAULT_CASES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 5),
    (5, 10),
)
_DEFAULT_REPS: int = 200
_QUICK_REPS: int = 20


def _build_yaml(*, n_axes: int, steps: int) -> str:
    """Return a synthetic tabletop suite YAML with the given shape.

    Uses the first ``n_axes`` axes from :data:`_AXIS_POOL`; each axis
    is continuous with ``low: 0.0, high: 1.0, steps: N``. Suite-level
    knobs (name / env / seed / episodes_per_cell) are constant so the
    only thing that varies across cases is the axis count and the
    per-axis ``steps`` integer.
    """
    if n_axes < 1 or n_axes > len(_AXIS_POOL):
        raise ValueError(f"n_axes must be in [1, {len(_AXIS_POOL)}]; got {n_axes}")
    lines = [
        "name: bench-loader",
        "env: tabletop",
        "seed: 0",
        "episodes_per_cell: 1",
        "axes:",
    ]
    for axis in _AXIS_POOL[:n_axes]:
        lines.extend(
            [
                f"  {axis}:",
                "    low: 0.0",
                "    high: 1.0",
                f"    steps: {steps}",
            ]
        )
    return "\n".join(lines) + "\n"


def _bench_one_case(*, n_axes: int, steps: int, reps: int) -> dict[str, Any]:
    """Time ``reps`` parses of one synthetic suite shape.

    Returns a per-case dict with the wall-time mean + p95, the resulting
    cell count (for context), and the YAML byte length (for context).
    """
    yaml_text = _build_yaml(n_axes=n_axes, steps=steps)
    deltas: list[float] = []
    suite = None
    for _ in range(reps):
        t0 = time.perf_counter()
        suite = load_suite_from_string(yaml_text)
        deltas.append(time.perf_counter() - t0)
    if suite is None:  # reps == 0; defensive
        raise RuntimeError("loader bench ran zero repetitions")

    deltas.sort()
    n_cells = suite.num_cells()
    mean_ms = sum(deltas) / len(deltas) * 1000.0
    # Nearest-rank p95 (no interpolation; matches bench_rollout style).
    p95_idx = max(0, min(len(deltas) - 1, round(0.95 * len(deltas)) - 1))
    p95_ms = deltas[p95_idx] * 1000.0
    return {
        "n_axes": n_axes,
        "steps_per_axis": steps,
        "n_cells": n_cells,
        "yaml_bytes": len(yaml_text),
        "reps": reps,
        "mean_ms": round(mean_ms, 4),
        "p95_ms": round(p95_ms, 4),
    }


def main(*, reps: int, quick: bool) -> dict[str, Any]:
    """Run the sweep across the three synthetic suite shapes."""
    print(f"bench_suite_loader: reps={reps} quick={quick}")

    cases: list[dict[str, Any]] = []
    for n_axes, steps in _DEFAULT_CASES:
        result = _bench_one_case(n_axes=n_axes, steps=steps, reps=reps)
        cases.append(result)
        print(
            f"  {n_axes}-axis x {steps}-steps  ({result['n_cells']} cells, "
            f"{result['yaml_bytes']} B):  "
            f"mean={result['mean_ms']:.3f} ms  p95={result['p95_ms']:.3f} ms"
        )

    summary: dict[str, Any] = {
        "name": "bench_suite_loader",
        "quick": quick,
        "reps": reps,
        "cases": cases,
    }
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark load_suite_from_string across synthetic suite shapes.",
    )
    parser.add_argument("--repetitions", type=int, default=_DEFAULT_REPS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke run: {_QUICK_REPS} repetitions per case. Overrides --repetitions.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    reps_to_use = _QUICK_REPS if args.quick else args.repetitions
    main(reps=reps_to_use, quick=args.quick)
