"""Parallel-scaling benchmark for :class:`gauntlet.runner.Runner`.

Times :meth:`Runner.run` on a synthetic in-memory tabletop suite at
``n_workers in {1, 2, 4, 8}`` (or just ``{1, 2}`` under ``--quick``) and
reports the wall-clock for each worker count plus the speedup vs N=1.

The suite is built via :func:`gauntlet.suite.load_suite_from_string` so
the bench is self-contained — no coupling to the example YAML files.
The env factory is :func:`functools.partial` over :class:`TabletopEnv`,
mirroring ``examples/evaluate_random_policy.py``; both factories live at
module scope so the ``spawn`` start method can pickle them.

Usage:
    uv run python scripts/bench_runner.py [--episodes-per-cell N]
                                          [--max-steps N] [--seed S]
                                          [--quick]

The final stdout line is a single JSON object so a CI job can ``tail -n 1``.

Notes:
    The ``n_workers in {4, 8}`` measurements include a try/except guard
    so a laptop-hostile environment (containerised CI with limited
    cores, MuJoCo GL flakes under a stressed pool) does not abort the
    whole bench. A failed point lands as a ``null`` in the JSON output.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from typing import Any

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite import Suite, load_suite_from_string

__all__ = ["main"]


# Action dimension exposed by TabletopEnv: [dx, dy, dz, drx, dry, drz, gripper].
_TABLETOP_ACTION_DIM: int = 7

# Default sweep is small enough to finish on a laptop while still
# producing >1 work item per worker at N=4. Default suite layout:
#   2 axes x 3 steps each = 6 cells; episodes_per_cell drives volume.
_DEFAULT_EPISODES_PER_CELL: int = 3
_DEFAULT_MAX_STEPS: int = 30
_DEFAULT_WORKER_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

# Quick variant: still 6 cells but only 1 episode each => 6 work items,
# only N=1,2 measured (N=4 / N=8 would be dominated by spawn overhead
# and produce noisy/sub-1.0 speedups on this small a workload).
_QUICK_EPISODES_PER_CELL: int = 1
_QUICK_MAX_STEPS: int = 15
_QUICK_WORKER_COUNTS: tuple[int, ...] = (1, 2)


def _build_suite_yaml(*, episodes_per_cell: int, seed: int) -> str:
    """Return a 2-axis x 3-steps tabletop suite as a YAML string.

    Uses two continuous, state-effecting axes
    (``object_initial_pose_x`` / ``object_initial_pose_y``) so the
    loader's :func:`_reject_purely_visual_suites` short-circuits cleanly
    on every backend, including the cosmetic-axis-aware ones.
    """
    return (
        "name: bench-runner\n"
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
        "    steps: 3\n"
    )


def _bench_one_worker_count(
    *,
    suite: Suite,
    n_workers: int,
    max_steps: int,
) -> float:
    """Time one ``Runner.run`` invocation at the given worker count.

    Returns the wall-clock seconds for the call. Exceptions propagate;
    the caller wraps the high-N points in try/except so partial results
    still land in the JSON output.
    """
    runner = Runner(
        n_workers=n_workers,
        env_factory=partial(TabletopEnv, max_steps=max_steps),
    )
    start = time.perf_counter()
    episodes = runner.run(
        policy_factory=partial(RandomPolicy, action_dim=_TABLETOP_ACTION_DIM),
        suite=suite,
    )
    wall = time.perf_counter() - start
    # Sanity check the Runner actually produced what we expect — a
    # silent zero-episode return would otherwise look like a great
    # speedup. ``num_cells * episodes_per_cell`` is the contract.
    expected = suite.num_cells() * suite.episodes_per_cell
    if len(episodes) != expected:
        raise RuntimeError(
            f"runner returned {len(episodes)} episodes; expected {expected}",
        )
    return wall


def main(
    *,
    episodes_per_cell: int,
    max_steps: int,
    seed: int,
    worker_counts: tuple[int, ...],
    quick: bool,
) -> dict[str, Any]:
    """Run the scaling sweep and return the summary dict.

    Returns the same dict that gets printed as the trailing JSON line.
    """
    print(
        f"bench_runner: episodes_per_cell={episodes_per_cell} max_steps={max_steps} "
        f"seed={seed} worker_counts={list(worker_counts)} quick={quick}"
    )

    suite_yaml = _build_suite_yaml(episodes_per_cell=episodes_per_cell, seed=seed)
    suite = load_suite_from_string(suite_yaml)
    n_items = suite.num_cells() * suite.episodes_per_cell
    print(
        f"  synthetic suite: {suite.num_cells()} cells x {episodes_per_cell} eps = {n_items} items"
    )

    walls: dict[int, float | None] = {}
    for n in worker_counts:
        try:
            wall = _bench_one_worker_count(suite=suite, n_workers=n, max_steps=max_steps)
            walls[n] = wall
            print(f"  n_workers={n}: {wall * 1000.0:.1f} ms")
        except Exception as exc:  # resilient bench, surface and continue
            walls[n] = None
            print(f"  n_workers={n}: FAILED ({type(exc).__name__}: {exc})")

    baseline = walls.get(1)
    speedups: dict[int, float | None] = {}
    for n, w in walls.items():
        if baseline is None or w is None or w <= 0.0:
            speedups[n] = None
        else:
            speedups[n] = round(baseline / w, 4)
            if n != 1:
                print(f"  speedup n={n} vs n=1: {speedups[n]}x")

    summary: dict[str, Any] = {
        "name": "bench_runner",
        "quick": quick,
        "episodes_per_cell": episodes_per_cell,
        "max_steps": max_steps,
        "seed": seed,
        "worker_counts": list(worker_counts),
        "n_items": n_items,
        "wall_ms": {
            str(n): (round(w * 1000.0, 4) if w is not None else None) for n, w in walls.items()
        },
        "speedup_vs_n1": {str(n): s for n, s in speedups.items()},
    }
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Runner.run parallel scaling on a synthetic tabletop suite.",
    )
    parser.add_argument("--episodes-per-cell", type=int, default=_DEFAULT_EPISODES_PER_CELL)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Smoke run: episodes_per_cell={_QUICK_EPISODES_PER_CELL}, "
            f"max_steps={_QUICK_MAX_STEPS}, worker_counts={list(_QUICK_WORKER_COUNTS)}. "
            "Overrides --episodes-per-cell / --max-steps."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    if args.quick:
        eps = _QUICK_EPISODES_PER_CELL
        steps = _QUICK_MAX_STEPS
        counts = _QUICK_WORKER_COUNTS
    else:
        eps = args.episodes_per_cell
        steps = args.max_steps
        counts = _DEFAULT_WORKER_COUNTS
    main(
        episodes_per_cell=eps,
        max_steps=steps,
        seed=args.seed,
        worker_counts=counts,
        quick=args.quick,
    )
