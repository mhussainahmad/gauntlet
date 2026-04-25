"""YAML suite-loader benchmark.

Times :func:`gauntlet.suite.load_suite_from_string` on three synthetic
suite shapes:

* 1 axis x 2 steps,
* 3 axes x 5 steps,
* 5 axes x 10 steps.

Plus, with ``--bundled``, also times :func:`gauntlet.suite.load_suite`
parsing the bundled YAML files under ``examples/suites/*.yaml`` to
capture real-world parse latency on the canonical suite shapes the
project ships.

Surfaces O(grid-size) regressions in the loader / pydantic validation
path early — the loader itself does not enumerate the Cartesian product
(``Suite.cells`` is a generator), so parse time should grow with axis
count, not with grid volume.

Usage:
    uv run --no-sync python scripts/bench_suite_loader.py [--repetitions N]
                                                          [--bundled]
                                                          [--quick]
                                                          [--out PATH]

Outputs:
    * Text table to stdout.
    * Single-line JSON summary as the *last* line of stdout
      (CI can ``tail -n 1``).
    * JSON sidecar file ``bench_suite_loader.json``
      (override with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Import the env package so the ``tabletop`` slug is registered before
# the loader runs ``_reject_purely_visual_suites`` -> ``get_env_factory``.
import gauntlet.env  # noqa: F401  (side-effect import: registers tabletop)
from gauntlet.suite import load_suite, load_suite_from_string

__all__ = ["main"]


# Default location of the bundled example suites, relative to the repo
# root (the script's parent's parent). Resolved lazily so the script
# can also be invoked from outside the repo.
_BUNDLED_SUITES_DIR: Path = Path(__file__).resolve().parent.parent / "examples" / "suites"


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


def _bench_bundled_suite(*, path: Path, reps: int) -> dict[str, Any] | None:
    """Time ``reps`` parses of one bundled suite YAML.

    Returns ``None`` when the file fails to parse (missing extra for a
    backend like ``tabletop-genesis`` whose loader gate raises) so the
    sweep skips-not-fails on unsupported bundled suites without
    aborting the whole bench.
    """
    yaml_text = path.read_text(encoding="utf-8")
    deltas: list[float] = []
    suite = None
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            suite = load_suite(path)
            deltas.append(time.perf_counter() - t0)
    except Exception as exc:
        print(f"  {path.name}: skipped ({type(exc).__name__}: {exc})")
        return None

    if suite is None:  # reps == 0; defensive
        return None

    deltas.sort()
    n_cells = suite.num_cells()
    mean_ms = sum(deltas) / len(deltas) * 1000.0
    p95_idx = max(0, min(len(deltas) - 1, round(0.95 * len(deltas)) - 1))
    p95_ms = deltas[p95_idx] * 1000.0
    return {
        "path": str(path),
        "name": path.name,
        "n_cells": n_cells,
        "yaml_bytes": len(yaml_text),
        "reps": reps,
        "mean_ms": round(mean_ms, 4),
        "p95_ms": round(p95_ms, 4),
    }


def _emit_sidecar(summary: dict[str, Any], out_path: Path) -> None:
    """Write the summary dict to ``out_path`` as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")


def main(
    *,
    reps: int,
    quick: bool,
    bundled: bool,
    out_path: Path,
) -> dict[str, Any]:
    """Run the sweep across the synthetic suite shapes (and bundled YAMLs)."""
    print(f"bench_suite_loader: reps={reps} quick={quick} bundled={bundled}")

    cases: list[dict[str, Any]] = []
    for n_axes, steps in _DEFAULT_CASES:
        result = _bench_one_case(n_axes=n_axes, steps=steps, reps=reps)
        cases.append(result)
        print(
            f"  {n_axes}-axis x {steps}-steps  ({result['n_cells']} cells, "
            f"{result['yaml_bytes']} B):  "
            f"mean={result['mean_ms']:.3f} ms  p95={result['p95_ms']:.3f} ms"
        )

    bundled_cases: list[dict[str, Any]] = []
    if bundled:
        if not _BUNDLED_SUITES_DIR.is_dir():
            print(f"  bundled mode: directory missing ({_BUNDLED_SUITES_DIR}); skipping")
        else:
            yaml_paths = sorted(_BUNDLED_SUITES_DIR.glob("*.yaml"))
            print(f"  bundled mode: {len(yaml_paths)} YAML files in {_BUNDLED_SUITES_DIR}")
            for path in yaml_paths:
                bundled_result = _bench_bundled_suite(path=path, reps=reps)
                if bundled_result is None:
                    continue
                bundled_cases.append(bundled_result)
                print(
                    f"  {bundled_result['name']:<35} ({bundled_result['n_cells']:>6} cells, "
                    f"{bundled_result['yaml_bytes']:>4} B):  "
                    f"mean={bundled_result['mean_ms']:.3f} ms  "
                    f"p95={bundled_result['p95_ms']:.3f} ms"
                )

    summary: dict[str, Any] = {
        "name": "bench_suite_loader",
        "quick": quick,
        "bundled": bundled,
        "reps": reps,
        "cases": cases,
        "bundled_cases": bundled_cases,
    }
    _emit_sidecar(summary, out_path)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark load_suite_from_string across synthetic suite shapes "
            "(and the bundled examples/suites/*.yaml files when --bundled)."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=_DEFAULT_REPS)
    parser.add_argument(
        "--bundled",
        action="store_true",
        help=(
            "Also parse every YAML file under examples/suites/*.yaml. "
            "Skip-not-fail per-file when a backend extra is missing."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke run: {_QUICK_REPS} repetitions per case. Overrides --repetitions.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_suite_loader.json"),
        help="Sidecar JSON output path. Default: bench_suite_loader.json in cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    reps_to_use = _QUICK_REPS if args.quick else args.repetitions
    main(
        reps=reps_to_use,
        quick=args.quick,
        bundled=args.bundled,
        out_path=args.out,
    )
    sys.exit(0)
