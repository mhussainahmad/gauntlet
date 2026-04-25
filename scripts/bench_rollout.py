"""Single-process rollout throughput for the gauntlet tabletop envs.

Measures the lowest-level rollout primitive: a tabletop env factory +
:class:`gauntlet.policy.RandomPolicy` driven by a bare Python loop, no
:class:`gauntlet.runner.Runner` wrapping. Four timings are reported:

* env construction (single perf_counter window around the constructor),
* per-step latency (one perf_counter delta per env.step), summarised as
  median / p50 / p95 / p99,
* per-episode wall-clock (one window per episode),
* episodes/second throughput (total measured episodes / total measured wall).

The script intentionally bypasses the Runner so a regression isolated
to ``Runner._build_work_items`` or the multiprocessing path does not
mask a regression in the env primitive (and vice versa).

Backends:
    The MuJoCo ``tabletop`` backend is the default and is always installed
    (it ships in the core dependency set). PyBullet / Genesis / Isaac are
    importable only when their extras are installed; the script
    skip-not-fails (prints ``skipped: <reason>`` and exits 0) when the
    requested backend's import gate raises.

Usage:
    uv run --no-sync python scripts/bench_rollout.py [--backend NAME]
                                                     [--episodes N]
                                                     [--steps-per-episode N]
                                                     [--seed S] [--quick]
                                                     [--out PATH]

Outputs:
    * Text table to stdout.
    * Single-line JSON summary as the *last* line of stdout
      (CI can ``tail -n 1``).
    * JSON sidecar file ``bench_rollout.json`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from gauntlet.env import TabletopEnv
from gauntlet.env.base import GauntletEnv
from gauntlet.policy import RandomPolicy

__all__ = ["main"]


# Supported backend slugs. ``tabletop`` is the MuJoCo default (always
# installed); the other three skip-not-fail when their extras are not
# importable. Names match the slugs the Suite loader's ``env:`` key
# accepts (RFC-005 §3.4).
_BACKEND_NAMES: tuple[str, ...] = (
    "tabletop",
    "tabletop-pybullet",
    "tabletop-genesis",
    "tabletop-isaac",
)


def _resolve_backend_factory(name: str) -> Callable[..., GauntletEnv] | None:
    """Return the env factory for ``name`` or ``None`` if its extra is missing.

    Importing ``gauntlet.env.pybullet`` / ``.genesis`` / ``.isaac`` is
    side-effecting: each subpackage's ``__init__`` calls ``register_env``
    when its native dependency is importable. We catch ``ImportError``
    (and any subclass — ``ModuleNotFoundError`` is what stdlib raises)
    and return None so the caller can skip-not-fail.
    """
    if name == "tabletop":
        # Always available — registered at ``gauntlet.env`` import time.
        # TabletopEnv satisfies GauntletEnv structurally; the cast mirrors
        # the deliberate widening at ``gauntlet.env.__init__``'s
        # ``register_env`` call.
        return cast(Callable[..., GauntletEnv], TabletopEnv)
    try:
        if name == "tabletop-pybullet":
            import gauntlet.env.pybullet
        elif name == "tabletop-genesis":
            import gauntlet.env.genesis
        elif name == "tabletop-isaac":
            import gauntlet.env.isaac  # noqa: F401
        else:
            raise ValueError(f"unknown backend {name!r}; choose from {_BACKEND_NAMES}")
    except ImportError:
        return None
    # The subpackage import registered the slug; resolve via the registry
    # so any future renaming flows through one place.
    from gauntlet.env.registry import get_env_factory

    try:
        return get_env_factory(name)
    except ValueError:
        return None


# Action dimension exposed by TabletopEnv: [dx, dy, dz, drx, dry, drz, gripper].
_TABLETOP_ACTION_DIM: int = 7

# Default sweep — modest enough to finish in seconds on a laptop while
# still giving a stable p99. ``--quick`` collapses both knobs.
_DEFAULT_EPISODES: int = 30
_DEFAULT_STEPS: int = 50
_QUICK_EPISODES: int = 5
_QUICK_STEPS: int = 20


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of an already-sorted list.

    Uses the nearest-rank rule (no interpolation). Cheap and adequate for
    the regression-monitoring use case — we care about >2x slowdowns,
    not sub-millisecond drift in the interpolated estimate. Returns 0.0
    on an empty input so the JSON output stays well-formed even when a
    smoke run captures zero samples.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, round(pct / 100.0 * n)))
    return sorted_values[rank - 1]


def _bench_episode(
    env: GauntletEnv,
    policy: RandomPolicy,
    *,
    seed: int,
    max_steps: int,
) -> tuple[float, list[float]]:
    """Drive one episode, returning (wall_seconds, per_step_seconds_list).

    ``policy.reset`` is called per episode so its RNG is reset to the
    same node the Runner would hand it (decorrelated env stream is
    already seeded via ``env.reset(seed=...)``).
    """
    policy.reset(np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    step_deltas: list[float] = []
    ep_start = time.perf_counter()
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = policy.act(obs)
        t0 = time.perf_counter()
        obs, _reward, terminated, truncated, info = env.step(action)
        step_deltas.append(time.perf_counter() - t0)
        if len(step_deltas) >= max_steps:
            break
    ep_wall = time.perf_counter() - ep_start
    del info  # keep mypy happy without `_ = info`
    return ep_wall, step_deltas


def _emit_sidecar(summary: dict[str, Any], out_path: Path) -> None:
    """Write the summary dict to ``out_path`` as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")


def main(
    *,
    backend: str,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    quick: bool,
    out_path: Path,
) -> dict[str, Any]:
    """Run the benchmark and return the summary dict.

    Returns the same dict that gets printed as the trailing JSON line so
    callers (tests, future scripts) can re-use the result without
    re-parsing stdout.
    """
    print(
        f"bench_rollout: backend={backend} episodes={episodes} "
        f"steps_per_episode={steps_per_episode} seed={seed} quick={quick}"
    )

    factory = _resolve_backend_factory(backend)
    if factory is None:
        summary: dict[str, Any] = {
            "name": "bench_rollout",
            "quick": quick,
            "skipped": True,
            "skip_reason": f"backend {backend!r} not importable (extra not installed)",
            "backend": backend,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "seed": seed,
        }
        print(f"skipped: backend {backend!r} not importable (extra not installed)")
        _emit_sidecar(summary, out_path)
        print(json.dumps(summary))
        return summary

    construct_start = time.perf_counter()
    env = factory(max_steps=steps_per_episode)
    construct_seconds = time.perf_counter() - construct_start
    print(f"  env construction: {construct_seconds * 1000.0:.2f} ms")

    policy = RandomPolicy(action_dim=_TABLETOP_ACTION_DIM, seed=seed)

    all_step_deltas: list[float] = []
    episode_walls: list[float] = []
    measured_window_start = time.perf_counter()
    try:
        for ep_idx in range(episodes):
            ep_seed = seed + ep_idx
            ep_wall, step_deltas = _bench_episode(
                env, policy, seed=ep_seed, max_steps=steps_per_episode
            )
            episode_walls.append(ep_wall)
            all_step_deltas.extend(step_deltas)
    finally:
        env.close()
    measured_window_seconds = time.perf_counter() - measured_window_start

    all_step_deltas.sort()
    p50_ms = _percentile(all_step_deltas, 50.0) * 1000.0
    p95_ms = _percentile(all_step_deltas, 95.0) * 1000.0
    p99_ms = _percentile(all_step_deltas, 99.0) * 1000.0
    mean_step_ms = (
        (sum(all_step_deltas) / len(all_step_deltas) * 1000.0) if all_step_deltas else 0.0
    )
    mean_episode_ms = (sum(episode_walls) / len(episode_walls) * 1000.0) if episode_walls else 0.0
    eps_per_sec = (
        (len(episode_walls) / measured_window_seconds) if measured_window_seconds > 0.0 else 0.0
    )

    print(
        f"  per-step latency: p50={p50_ms:.3f} ms  p95={p95_ms:.3f} ms  "
        f"p99={p99_ms:.3f} ms  mean={mean_step_ms:.3f} ms  (n={len(all_step_deltas)})"
    )
    print(f"  per-episode wall: mean={mean_episode_ms:.2f} ms  (n={len(episode_walls)})")
    print(f"  throughput:       {eps_per_sec:.2f} episodes/sec")

    summary = {
        "name": "bench_rollout",
        "quick": quick,
        "skipped": False,
        "skip_reason": None,
        "backend": backend,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "seed": seed,
        "step_samples": len(all_step_deltas),
        "construct_ms": round(construct_seconds * 1000.0, 4),
        "step_p50_ms": round(p50_ms, 4),
        "step_p95_ms": round(p95_ms, 4),
        "step_p99_ms": round(p99_ms, 4),
        "step_mean_ms": round(mean_step_ms, 4),
        "episode_mean_ms": round(mean_episode_ms, 4),
        "episodes_per_sec": round(eps_per_sec, 4),
    }
    _emit_sidecar(summary, out_path)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-process tabletop rollout throughput.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tabletop",
        choices=list(_BACKEND_NAMES),
        help=(
            "Env slug. ``tabletop`` is the always-installed MuJoCo backend; "
            "the others skip-not-fail when their extras are not installed."
        ),
    )
    parser.add_argument("--episodes", type=int, default=_DEFAULT_EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=_DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Smoke run: {_QUICK_EPISODES} episodes x {_QUICK_STEPS} steps. "
            "Overrides --episodes / --steps-per-episode."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_rollout.json"),
        help="Sidecar JSON output path. Default: bench_rollout.json in cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    eps = _QUICK_EPISODES if args.quick else args.episodes
    steps = _QUICK_STEPS if args.quick else args.steps_per_episode
    main(
        backend=args.backend,
        episodes=eps,
        steps_per_episode=steps,
        seed=args.seed,
        quick=args.quick,
        out_path=args.out,
    )
    sys.exit(0)
