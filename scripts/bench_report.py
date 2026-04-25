"""Report-aggregation benchmark.

Times :func:`gauntlet.report.build_report` on synthetic
:class:`gauntlet.runner.Episode` lists of N=10, 100, 1000, 10000 (or
just 10 / 100 / 1000 under ``--quick``). Surfaces O(N^2) clustering /
heatmap regressions early — the analyse module groups by (cell_index,
axis-pair value combos), so a quadratic blowup would land first here.

The synthetic episodes use 2 axes (so the failure-cluster /
heatmap_2d code paths are exercised) with 10 distinct values per axis
(so per_cell groups stay small relative to N at large N). Success
flips deterministically off a seeded :class:`numpy.random.Generator`
to keep the run reproducible across machines.

Usage:
    uv run --no-sync python scripts/bench_report.py [--seed S] [--quick]
                                                    [--out PATH]

Outputs:
    * Text table to stdout.
    * Single-line JSON summary as the *last* line of stdout
      (CI can ``tail -n 1``).
    * JSON sidecar file ``bench_report.json`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from gauntlet.report import build_report
from gauntlet.runner import Episode

__all__ = ["main"]


# Two state-effecting axes give the analysis full coverage:
# * per_axis -> 2 breakdowns
# * per_cell -> N_X * N_Y groups
# * failure_clusters -> 1 unordered pair iterated
# * heatmap_2d       -> 1 entry built
_AXIS_X: str = "object_initial_pose_x"
_AXIS_Y: str = "object_initial_pose_y"
_N_VALUES_PER_AXIS: int = 10  # 100 cells total in the (X, Y) grid
_SUCCESS_PROB: float = 0.7  # leaves a non-zero baseline failure rate

_DEFAULT_SIZES: tuple[int, ...] = (10, 100, 1000, 10000)
_QUICK_SIZES: tuple[int, ...] = (10, 100, 1000)


def _build_episodes(*, n: int, rng: np.random.Generator) -> list[Episode]:
    """Synthesise ``n`` :class:`Episode` records on a 10x10 axis grid.

    Cell index = ``y_idx * N_VALUES + x_idx`` so episodes that share a
    perturbation_config also share a cell_index — this matches the
    Runner's contract and lets ``_per_cell_breakdowns`` group correctly.
    """
    x_values = np.linspace(-0.1, 0.1, _N_VALUES_PER_AXIS).tolist()
    y_values = np.linspace(-0.1, 0.1, _N_VALUES_PER_AXIS).tolist()

    episodes: list[Episode] = []
    # Pre-sample all randomness in vectorised form — keeps construction
    # time out of the bench window. The bench window is the
    # ``build_report`` call itself, not this synth.
    cell_x_idx = rng.integers(0, _N_VALUES_PER_AXIS, size=n).tolist()
    cell_y_idx = rng.integers(0, _N_VALUES_PER_AXIS, size=n).tolist()
    success_flags = (rng.random(size=n) < _SUCCESS_PROB).tolist()

    for i in range(n):
        x_idx = int(cell_x_idx[i])
        y_idx = int(cell_y_idx[i])
        cell_index = y_idx * _N_VALUES_PER_AXIS + x_idx
        episodes.append(
            Episode(
                suite_name="bench-report",
                cell_index=cell_index,
                episode_index=i,
                seed=int(i),
                perturbation_config={
                    _AXIS_X: x_values[x_idx],
                    _AXIS_Y: y_values[y_idx],
                },
                success=bool(success_flags[i]),
                terminated=bool(success_flags[i]),
                truncated=not bool(success_flags[i]),
                step_count=10,
                total_reward=1.0 if success_flags[i] else 0.0,
            )
        )
    return episodes


def _bench_one_size(*, n: int, rng: np.random.Generator) -> dict[str, Any]:
    """Synthesise + time one :func:`build_report` call at size ``n``."""
    episodes = _build_episodes(n=n, rng=rng)
    t0 = time.perf_counter()
    report = build_report(episodes)
    wall = time.perf_counter() - t0
    return {
        "n_episodes": n,
        "n_cells_in_report": len(report.per_cell),
        "n_failure_clusters": len(report.failure_clusters),
        "build_ms": round(wall * 1000.0, 4),
    }


def _emit_sidecar(summary: dict[str, Any], out_path: Path) -> None:
    """Write the summary dict to ``out_path`` as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")


def main(
    *,
    sizes: tuple[int, ...],
    seed: int,
    quick: bool,
    out_path: Path,
) -> dict[str, Any]:
    """Run the sweep and return the summary dict."""
    print(f"bench_report: sizes={list(sizes)} seed={seed} quick={quick}")

    cases: list[dict[str, Any]] = []
    for n in sizes:
        # Fresh RNG per case so each size's synth is independent of
        # the others (avoids an earlier large case shifting the random
        # stream the smaller cases see).
        rng = np.random.default_rng(seed + n)
        result = _bench_one_size(n=n, rng=rng)
        cases.append(result)
        print(
            f"  n={result['n_episodes']:>5}  "
            f"build={result['build_ms']:.3f} ms  "
            f"per_cell={result['n_cells_in_report']}  "
            f"clusters={result['n_failure_clusters']}"
        )

    summary: dict[str, Any] = {
        "name": "bench_report",
        "quick": quick,
        "seed": seed,
        "cases": cases,
    }
    _emit_sidecar(summary, out_path)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark report.build_report at varying episode-list sizes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(f"Smoke run: drops the N=10000 case (sweep becomes {list(_QUICK_SIZES)})."),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_report.json"),
        help="Sidecar JSON output path. Default: bench_report.json in cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    sizes_to_use = _QUICK_SIZES if args.quick else _DEFAULT_SIZES
    main(
        sizes=sizes_to_use,
        seed=args.seed,
        quick=args.quick,
        out_path=args.out,
    )
    sys.exit(0)
