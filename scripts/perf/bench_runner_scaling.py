"""Runner parallel-scaling benchmark — Phase 2.5 T12.

Sweeps :class:`gauntlet.runner.Runner` across ``--workers`` worker
counts on a fixed tabletop suite and reports the speedup curve plus
an Amdahl's-law fit (the serial fraction ``s`` and parallel fraction
``p = 1 - s``). Worker counts are clamped to ``os.cpu_count()`` so a
2-core CI runner does not over-subscribe; the JSON output records both
the requested and the effective sweep.

Amdahl's-law fit:
    speedup(N) = 1 / (s + (1 - s) / N)

The fit minimises the squared error of ``log(measured_speedup) -
log(amdahl_speedup(N, s))`` across the worker counts; ``s`` is
constrained to ``[0, 1]`` and surfaced in the JSON output. A bench
result with ``s`` close to 1.0 means the workload is dominated by the
serial portion (per-worker spawn cost on a small suite), while
``s`` close to 0 means near-linear scaling.

CLI:
    python scripts/perf/bench_runner_scaling.py \\
        --output benchmarks/scaling.json
    python scripts/perf/bench_runner_scaling.py \\
        --workers 1,2,4,8 --output benchmarks/scaling.json

Reports a flat JSON dict with:
    * ``walls_ms``           — per-worker-count wall in ms (dict)
    * ``speedup_vs_n1``      — per-worker-count speedup (dict)
    * ``amdahl_serial_frac`` — fitted serial fraction
    * ``amdahl_parallel_frac`` — fitted parallel fraction (1 - serial)
    * ``version`` / ``timestamp`` / ``git_commit`` — provenance
    * ``partial`` (bool)     — true if KeyboardInterrupt cut the run short
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import emit_sidecar, provenance_fields

import gauntlet.env  # noqa: F401  (registers tabletop slug)
from gauntlet.env import TabletopEnv
from gauntlet.env.base import GauntletEnv
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite import Suite, load_suite_from_string

__all__ = ["main"]


# Tabletop env action dim: [dx, dy, dz, drx, dry, drz, gripper].
_TABLETOP_ACTION_DIM: int = 7

# Default sweep — covers single-process baseline plus 2 / 4 / 8 worker
# counts. The 8-worker point is clamped down on smaller CI runners via
# ``os.cpu_count()``.
_DEFAULT_WORKER_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

# Default per-cell episode count + max-step cap. Picked so the 1-worker
# baseline takes ~hundreds of ms (enough that spawn overhead does not
# dominate the parallel data points).
_DEFAULT_EPISODES_PER_CELL: int = 4
_DEFAULT_MAX_STEPS: int = 30


def _clamp_worker_counts(counts: tuple[int, ...]) -> tuple[int, ...]:
    """Clamp each worker count to ``os.cpu_count()``, dedupe, preserve order.

    A 4-core runner asked to do n_workers=8 gets clamped to 4; the JSON
    records both the requested and the effective tuple so a downstream
    consumer sees the truncation.
    """
    cap = os.cpu_count() or 1
    seen: set[int] = set()
    clamped: list[int] = []
    for n in counts:
        eff = max(1, min(n, cap))
        if eff not in seen:
            seen.add(eff)
            clamped.append(eff)
    return tuple(clamped)


def _build_suite_yaml(*, episodes_per_cell: int, seed: int) -> str:
    """Return a 2-axis tabletop suite YAML for the scaling sweep.

    2 x 3 grid = 6 cells, ``episodes_per_cell`` drives volume. Two
    state-effecting continuous axes so the loader's purely-visual-
    suite linter short-circuits cleanly on every backend.
    """
    return (
        "name: bench-runner-scaling-t12\n"
        "env: tabletop\n"
        f"seed: {seed}\n"
        f"episodes_per_cell: {episodes_per_cell}\n"
        "axes:\n"
        "  object_initial_pose_x:\n"
        "    low: -0.1\n"
        "    high: 0.1\n"
        "    steps: 3\n"
        "  object_initial_pose_y:\n"
        "    low: -0.1\n"
        "    high: 0.1\n"
        "    steps: 2\n"
    )


def _bench_one_worker_count(
    *,
    suite: Suite,
    n_workers: int,
    max_steps: int,
) -> float:
    """Time one ``Runner.run`` invocation at the given worker count."""
    env_factory: Callable[[], GauntletEnv] = cast(
        Callable[[], GauntletEnv],
        partial(TabletopEnv, max_steps=max_steps),
    )
    runner = Runner(n_workers=n_workers, env_factory=env_factory)
    start = time.perf_counter()
    episodes = runner.run(
        policy_factory=partial(RandomPolicy, action_dim=_TABLETOP_ACTION_DIM),
        suite=suite,
    )
    wall = time.perf_counter() - start
    expected = suite.num_cells() * suite.episodes_per_cell
    if len(episodes) != expected:
        raise RuntimeError(
            f"runner returned {len(episodes)} episodes; expected {expected}",
        )
    return wall


def _amdahl_fit(speedups: dict[int, float]) -> float:
    """Return the serial fraction ``s`` that best fits the speedup curve.

    Amdahl: ``speedup(N) = 1 / (s + (1 - s) / N)``.

    Derivation of ``s`` from any single (N, S) pair where N > 1:

        S * (s + (1 - s) / N) = 1
        s * S + (1 - s) * S / N = 1
        s * (S - S / N) = 1 - S / N
        s = (1 - S / N) / (S - S / N) = (N - S) / (S * (N - 1))

    With multiple samples we average the per-N estimate (clamped to
    ``[0, 1]``) — robust to a single noisy point and cheap to compute
    without numpy.optimize. A single point at N=1 is uninformative
    (every Amdahl curve passes through (1, 1)) and is dropped.
    """
    estimates: list[float] = []
    for n, sp in speedups.items():
        if n <= 1 or sp <= 0.0:
            continue
        denom = sp * (float(n) - 1.0)
        if denom == 0.0:
            continue
        s_est = (float(n) - sp) / denom
        # Clamp to [0, 1] — out-of-range estimates can come from
        # super-linear measured speedups (caching effects) or sub-1.0
        # speedups (spawn overhead dominates).
        s_est = max(0.0, min(1.0, s_est))
        estimates.append(s_est)
    if not estimates:
        return 1.0  # fully serial when nothing to fit on
    # Geometric mean of clamped estimates — a single outlier (one bad
    # rep at high N) gets pulled toward the median rather than dragging
    # the linear average. Equivalent to mean-of-logs; constants are
    # well-defined for the [0, 1] domain after clamping (we add a
    # tiny epsilon to avoid log(0) when every point clamps to 0).
    eps = 1e-9
    log_mean = sum(math.log(max(e, eps)) for e in estimates) / float(len(estimates))
    return math.exp(log_mean)


def main(
    *,
    worker_counts: tuple[int, ...],
    episodes_per_cell: int,
    max_steps: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    """Run the sweep, write the sidecar JSON, return the summary."""
    effective = _clamp_worker_counts(worker_counts)
    print(
        f"bench_runner_scaling (T12): worker_counts={list(effective)} "
        f"(requested={list(worker_counts)}) episodes_per_cell={episodes_per_cell} "
        f"max_steps={max_steps} seed={seed} output={output}"
    )

    suite_yaml = _build_suite_yaml(episodes_per_cell=episodes_per_cell, seed=seed)
    suite = load_suite_from_string(suite_yaml)
    n_items = suite.num_cells() * suite.episodes_per_cell
    print(f"  suite: {suite.num_cells()} cells x {suite.episodes_per_cell} eps = {n_items} items")

    walls: dict[int, float | None] = {}
    partial = False
    try:
        for n in effective:
            try:
                wall = _bench_one_worker_count(suite=suite, n_workers=n, max_steps=max_steps)
                walls[n] = wall
                print(f"  n_workers={n}: {wall * 1000.0:.1f} ms")
            except Exception as exc:  # bench resilient — continue on per-N failures
                walls[n] = None
                print(f"  n_workers={n}: FAILED ({type(exc).__name__}: {exc})")
    except KeyboardInterrupt:
        partial = True
        print("interrupted: emitting partial results (partial=true)")

    baseline = walls.get(1)
    speedups: dict[int, float] = {}
    for n, w in walls.items():
        if baseline is None or w is None or w <= 0.0:
            continue
        speedups[n] = baseline / w
        if n != 1:
            print(f"  speedup n={n} vs n=1: {speedups[n]:.3f}x")

    serial_frac = _amdahl_fit(speedups)
    parallel_frac = 1.0 - serial_frac
    print(f"  Amdahl fit: serial_frac={serial_frac:.4f}  parallel_frac={parallel_frac:.4f}")

    summary: dict[str, object] = {
        "name": "bench_runner_scaling",
        "requested_worker_counts": list(worker_counts),
        "worker_counts": list(effective),
        "cpu_count": os.cpu_count() or 1,
        "episodes_per_cell": episodes_per_cell,
        "max_steps": max_steps,
        "seed": seed,
        "n_items": n_items,
        "skipped": False,
        "skip_reason": None,
        "partial": partial,
        "walls_ms": {
            str(n): (round(w * 1000.0, 4) if w is not None else None) for n, w in walls.items()
        },
        "speedup_vs_n1": {str(n): round(s, 4) for n, s in speedups.items()},
        "amdahl_serial_frac": round(serial_frac, 6),
        "amdahl_parallel_frac": round(parallel_frac, 6),
        **provenance_fields(),
    }
    emit_sidecar(summary, output)
    print(json.dumps(summary))
    return summary


def _parse_worker_counts(value: str) -> tuple[int, ...]:
    """Parse ``"1,2,4,8"`` into ``(1, 2, 4, 8)``.

    Argparse helper. Rejects empty strings and non-positive values with
    a clear ``argparse.ArgumentTypeError``.
    """
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--workers must be a non-empty comma-separated list of positive ints"
        )
    out: list[int] = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid worker count {p!r}: {exc}") from exc
        if n < 1:
            raise argparse.ArgumentTypeError(
                f"worker count must be >= 1; got {n}",
            )
        out.append(n)
    return tuple(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 T12: Runner parallel-scaling benchmark with Amdahl's-law fit.",
    )
    parser.add_argument(
        "--workers",
        type=_parse_worker_counts,
        default=_DEFAULT_WORKER_COUNTS,
        help="Comma-separated worker counts. Default: 1,2,4,8.",
    )
    parser.add_argument(
        "--episodes-per-cell",
        type=int,
        default=_DEFAULT_EPISODES_PER_CELL,
    )
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed. Default: 42 (T12 convention).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/scaling.json"),
        help="JSON output path. Default: benchmarks/scaling.json under cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        worker_counts=tuple(args.workers),
        episodes_per_cell=args.episodes_per_cell,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
    )
    sys.exit(0)
