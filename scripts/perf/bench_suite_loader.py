"""Suite-loader benchmark — Phase 2.5 T12.

Times :func:`gauntlet.suite.load_suite` parsing a synthetic suite of
``--cells`` cells, then independently times the B-40 provenance-hash
computation (``compute_suite_provenance_hash``) which uses an AST-style
canonical hash of the parsed Suite.

The synthetic suite is shaped to land at exactly ``--cells`` cells by
choosing the smallest 1-axis ``steps`` count whose product hits the
target — ``cells = 1000`` becomes a 1-axis ``steps: 1000`` suite, which
exercises the loader's pydantic validation cost without inflating the
YAML size to multi-megabytes.

CLI:
    python scripts/perf/bench_suite_loader.py --cells 1000 \\
        --output benchmarks/suite_loader.json

Reports a flat JSON dict with:
    * ``load_time_ms``      — mean parse wall over ``--repetitions`` iters
    * ``ast_hash_time_ms``  — mean B-40 provenance-hash wall over the same
    * ``cells``             — actual cell count of the synthesised suite
    * ``yaml_bytes``        — size of the synthesised YAML
    * ``version`` / ``timestamp`` / ``git_commit`` — provenance
    * ``partial`` (bool)    — true if KeyboardInterrupt cut the run short
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import emit_sidecar, percentile, provenance_fields

import gauntlet.env  # noqa: F401  (registers tabletop slug for the loader)
from gauntlet.runner.provenance import compute_suite_provenance_hash
from gauntlet.suite import load_suite

__all__ = ["main"]


# Number of measurement reps per metric. 50 is small enough to stay
# under one wall-clock second at default knobs and large enough for a
# stable mean.
_DEFAULT_REPS: int = 50


def _build_synthetic_yaml(*, cells: int) -> str:
    """Return a 1-axis tabletop suite YAML hitting exactly ``cells`` cells.

    Uses ``object_initial_pose_x`` (continuous, state-effecting) so the
    loader's purely-visual-suite linter short-circuits cleanly. The
    1-axis shape is the worst case for parse latency vs cell count
    (max steps for a given cell budget); a multi-axis shape would let
    the per-axis cost amortise across the cells.
    """
    if cells < 1:
        raise ValueError(f"cells must be positive, got {cells}")
    return (
        "name: bench-suite-loader-t12\n"
        "env: tabletop\n"
        "seed: 42\n"
        "episodes_per_cell: 1\n"
        "axes:\n"
        "  object_initial_pose_x:\n"
        "    low: -0.1\n"
        "    high: 0.1\n"
        f"    steps: {cells}\n"
    )


def main(
    *,
    cells: int,
    reps: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    """Run the bench, write the sidecar JSON, return the summary."""
    print(f"bench_suite_loader (T12): cells={cells} reps={reps} seed={seed} output={output}")

    yaml_text = _build_synthetic_yaml(cells=cells)
    # The loader takes a path argument; staging the YAML to a tmp file
    # keeps the bench faithful to the production code path (file-on-
    # disk read + parse + validate). We re-use the same file across
    # all reps so the OS page cache effect is constant — the bench
    # measures pydantic / yaml.safe_load cost, not disk I/O.
    tmp_path = output.parent / ".bench_suite_loader_tmp.yaml"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(yaml_text, encoding="utf-8")

    load_deltas: list[float] = []
    hash_deltas: list[float] = []
    actual_cells = 0
    partial = False
    try:
        # Load once outside the timing loop to confirm the cell count
        # matches the request and to warm any module-level pydantic
        # caches the loader touches on first use.
        warmup_suite = load_suite(tmp_path)
        actual_cells = warmup_suite.num_cells()
        compute_suite_provenance_hash(warmup_suite)

        for _ in range(reps):
            t0 = time.perf_counter()
            suite = load_suite(tmp_path)
            load_deltas.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            compute_suite_provenance_hash(suite)
            hash_deltas.append(time.perf_counter() - t0)
    except KeyboardInterrupt:
        partial = True
        print("interrupted: emitting partial results (partial=true)")
    finally:
        # Clean up the staged tmp YAML; if cleanup itself fails (read-
        # only filesystem, race) just warn — the bench output is still
        # valid.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"  warning: failed to remove tmp YAML {tmp_path} ({exc})")

    load_deltas.sort()
    hash_deltas.sort()
    load_mean_ms = (sum(load_deltas) / len(load_deltas) * 1000.0) if load_deltas else 0.0
    hash_mean_ms = (sum(hash_deltas) / len(hash_deltas) * 1000.0) if hash_deltas else 0.0
    load_p95_ms = percentile(load_deltas, 95.0) * 1000.0
    hash_p95_ms = percentile(hash_deltas, 95.0) * 1000.0

    print(
        f"  load_time_mean={load_mean_ms:.3f} ms  load_p95={load_p95_ms:.3f} ms  "
        f"(n={len(load_deltas)}, cells={actual_cells}, yaml_bytes={len(yaml_text)})"
    )
    print(
        f"  ast_hash_time_mean={hash_mean_ms:.3f} ms  "
        f"ast_hash_p95={hash_p95_ms:.3f} ms  (n={len(hash_deltas)})"
    )

    summary: dict[str, object] = {
        "name": "bench_suite_loader",
        "cells_requested": cells,
        "cells": actual_cells,
        "reps": reps,
        "seed": seed,
        "yaml_bytes": len(yaml_text),
        "skipped": False,
        "skip_reason": None,
        "partial": partial,
        "load_time_ms": round(load_mean_ms, 4),
        "load_time_p95_ms": round(load_p95_ms, 4),
        "ast_hash_time_ms": round(hash_mean_ms, 4),
        "ast_hash_time_p95_ms": round(hash_p95_ms, 4),
        **provenance_fields(),
    }
    emit_sidecar(summary, output)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 T12: suite-loader benchmark (parse + B-40 AST hash).",
    )
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=_DEFAULT_REPS,
        help=f"Measurement reps per metric. Default: {_DEFAULT_REPS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed. Default: 42 (T12 convention).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/suite_loader.json"),
        help="JSON output path. Default: benchmarks/suite_loader.json under cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        cells=args.cells,
        reps=args.repetitions,
        seed=args.seed,
        output=args.output,
    )
    sys.exit(0)
