"""Parallel-scaling benchmark for :class:`gauntlet.runner.Runner`.

Times :meth:`Runner.run` on a tabletop suite at ``n_workers in
{1, 2, 4, 8}`` (clamped to ``os.cpu_count()``; or just ``{1, 2}`` under
``--quick``) and reports the wall-clock for each worker count plus the
speedup vs N=1 and the parallel efficiency (speedup / n_workers).

Two suite-source modes are supported:

* default (synthetic, 2 axes x 3 steps = 6 cells) — self-contained, no
  coupling to the bundled YAML;
* ``--from-suite PATH`` — load the bundled smoke suite (e.g.
  ``examples/suites/tabletop-smoke.yaml``, which yields 24 episodes).
  This matches the spec for the parallel-scaling sweep.

The env factory is :func:`functools.partial` over :class:`TabletopEnv`,
mirroring ``examples/evaluate_random_policy.py``; both factories live at
module scope so the ``spawn`` start method can pickle them.

Usage:
    uv run --no-sync python scripts/bench_runner.py [--episodes-per-cell N]
                                                    [--max-steps N] [--seed S]
                                                    [--from-suite PATH]
                                                    [--quick] [--out PATH]

Outputs:
    * Text table to stdout.
    * Single-line JSON summary as the *last* line of stdout
      (CI can ``tail -n 1``).
    * JSON sidecar file ``bench_runner.json`` (override with ``--out``).

Notes:
    The ``n_workers in {4, 8}`` measurements include a try/except guard
    so a laptop-hostile environment (containerised CI with limited
    cores, MuJoCo GL flakes under a stressed pool) does not abort the
    whole bench. A failed point lands as a ``null`` in the JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

from gauntlet.env import TabletopEnv
from gauntlet.env.base import GauntletEnv
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite import Suite, load_suite, load_suite_from_string

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


def _clamp_worker_counts(counts: tuple[int, ...]) -> tuple[int, ...]:
    """Clamp each worker count to ``os.cpu_count()``, dedupe, preserve order.

    The default sweep includes ``n_workers=8``; on a 4-core CI runner we
    don't want to oversubscribe. Returns the dedup'd, order-preserved
    tuple of clamped counts.
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
    # ``partial(TabletopEnv, ...)`` returns ``partial[TabletopEnv]``;
    # the Runner expects ``Callable[[], GauntletEnv]``. The cast mirrors
    # the deliberate widening at ``gauntlet.env.__init__``'s
    # ``register_env`` call (TabletopEnv satisfies GauntletEnv structurally).
    env_factory_cast: Callable[[], GauntletEnv] = cast(
        Callable[[], GauntletEnv],
        partial(TabletopEnv, max_steps=max_steps),
    )
    runner = Runner(
        n_workers=n_workers,
        env_factory=env_factory_cast,
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


def _emit_sidecar(summary: dict[str, Any], out_path: Path) -> None:
    """Write the summary dict to ``out_path`` as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")


def main(
    *,
    episodes_per_cell: int,
    max_steps: int,
    seed: int,
    worker_counts: tuple[int, ...],
    quick: bool,
    suite_path: Path | None,
    out_path: Path,
) -> dict[str, Any]:
    """Run the scaling sweep and return the summary dict.

    Returns the same dict that gets printed as the trailing JSON line.
    """
    effective_counts = _clamp_worker_counts(worker_counts)
    print(
        f"bench_runner: episodes_per_cell={episodes_per_cell} max_steps={max_steps} "
        f"seed={seed} worker_counts={list(effective_counts)} quick={quick} "
        f"suite_path={suite_path}"
    )

    if suite_path is not None:
        suite = load_suite(suite_path)
        # Bundled suite drives episodes_per_cell from the YAML; the CLI
        # arg is ignored so the printed summary stays accurate.
        eps_per_cell_used = suite.episodes_per_cell
        suite_label = f"bundled suite {suite_path}"
    else:
        suite_yaml = _build_suite_yaml(episodes_per_cell=episodes_per_cell, seed=seed)
        suite = load_suite_from_string(suite_yaml)
        eps_per_cell_used = episodes_per_cell
        suite_label = "synthetic suite"
    n_items = suite.num_cells() * suite.episodes_per_cell
    print(f"  {suite_label}: {suite.num_cells()} cells x {eps_per_cell_used} eps = {n_items} items")

    walls: dict[int, float | None] = {}
    for n in effective_counts:
        try:
            wall = _bench_one_worker_count(suite=suite, n_workers=n, max_steps=max_steps)
            walls[n] = wall
            print(f"  n_workers={n}: {wall * 1000.0:.1f} ms")
        except Exception as exc:  # resilient bench, surface and continue
            walls[n] = None
            print(f"  n_workers={n}: FAILED ({type(exc).__name__}: {exc})")

    baseline = walls.get(1)
    speedups: dict[int, float | None] = {}
    efficiencies: dict[int, float | None] = {}
    for n, w in walls.items():
        if baseline is None or w is None or w <= 0.0:
            speedups[n] = None
            efficiencies[n] = None
        else:
            sp = baseline / w
            speedups[n] = round(sp, 4)
            efficiencies[n] = round(sp / float(n), 4)
            if n != 1:
                print(f"  speedup n={n} vs n=1: {speedups[n]}x  (efficiency {efficiencies[n]})")

    summary: dict[str, Any] = {
        "name": "bench_runner",
        "quick": quick,
        "episodes_per_cell": eps_per_cell_used,
        "max_steps": max_steps,
        "seed": seed,
        "worker_counts": list(effective_counts),
        "requested_worker_counts": list(worker_counts),
        "cpu_count": os.cpu_count() or 1,
        "suite_path": str(suite_path) if suite_path is not None else None,
        "n_items": n_items,
        "wall_ms": {
            str(n): (round(w * 1000.0, 4) if w is not None else None) for n, w in walls.items()
        },
        "speedup_vs_n1": {str(n): s for n, s in speedups.items()},
        "efficiency_vs_n1": {str(n): e for n, e in efficiencies.items()},
    }
    _emit_sidecar(summary, out_path)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Runner.run parallel scaling on a tabletop suite.",
    )
    parser.add_argument("--episodes-per-cell", type=int, default=_DEFAULT_EPISODES_PER_CELL)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from-suite",
        type=Path,
        default=None,
        help=(
            "Optional bundled-suite YAML path (e.g. examples/suites/tabletop-smoke.yaml). "
            "When set, --episodes-per-cell is ignored and the YAML's value is used."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Smoke run: episodes_per_cell={_QUICK_EPISODES_PER_CELL}, "
            f"max_steps={_QUICK_MAX_STEPS}, worker_counts={list(_QUICK_WORKER_COUNTS)}. "
            "Overrides --episodes-per-cell / --max-steps."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_runner.json"),
        help="Sidecar JSON output path. Default: bench_runner.json in cwd.",
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
        suite_path=args.from_suite,
        out_path=args.out,
    )
    sys.exit(0)
