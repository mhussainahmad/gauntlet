"""Single-process rollout throughput for the MuJoCo tabletop env.

Measures the lowest-level rollout primitive: ``TabletopEnv`` +
:class:`gauntlet.policy.RandomPolicy` driven by a bare Python loop, no
:class:`gauntlet.runner.Runner` wrapping. Three timings are reported:

* env construction (single perf_counter window around the constructor),
* per-step latency (one perf_counter delta per env.step), summarised as
  median / p50 / p95 / p99,
* full-episode wall-clock (one window per episode).

The script intentionally bypasses the Runner so a regression isolated
to ``Runner._build_work_items`` or the multiprocessing path does not
mask a regression in the env primitive (and vice versa).

Usage:
    uv run python scripts/bench_rollout.py [--episodes N]
                                           [--steps-per-episode N]
                                           [--seed S] [--quick]

The final stdout line is a single JSON object so a CI job can ``tail -n 1``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy

__all__ = ["main"]


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
    env: TabletopEnv,
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


def main(*, episodes: int, steps_per_episode: int, seed: int, quick: bool) -> dict[str, Any]:
    """Run the benchmark and return the summary dict.

    Returns the same dict that gets printed as the trailing JSON line so
    callers (tests, future scripts) can re-use the result without
    re-parsing stdout.
    """
    print(
        f"bench_rollout: episodes={episodes} steps_per_episode={steps_per_episode} "
        f"seed={seed} quick={quick}"
    )

    construct_start = time.perf_counter()
    env = TabletopEnv(max_steps=steps_per_episode)
    construct_seconds = time.perf_counter() - construct_start
    print(f"  env construction: {construct_seconds * 1000.0:.2f} ms")

    policy = RandomPolicy(action_dim=_TABLETOP_ACTION_DIM, seed=seed)

    all_step_deltas: list[float] = []
    episode_walls: list[float] = []
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

    all_step_deltas.sort()
    p50_ms = _percentile(all_step_deltas, 50.0) * 1000.0
    p95_ms = _percentile(all_step_deltas, 95.0) * 1000.0
    p99_ms = _percentile(all_step_deltas, 99.0) * 1000.0
    mean_step_ms = (
        (sum(all_step_deltas) / len(all_step_deltas) * 1000.0) if all_step_deltas else 0.0
    )
    mean_episode_ms = (sum(episode_walls) / len(episode_walls) * 1000.0) if episode_walls else 0.0

    print(
        f"  per-step latency: p50={p50_ms:.3f} ms  p95={p95_ms:.3f} ms  "
        f"p99={p99_ms:.3f} ms  mean={mean_step_ms:.3f} ms  (n={len(all_step_deltas)})"
    )
    print(f"  per-episode wall: mean={mean_episode_ms:.2f} ms  (n={len(episode_walls)})")

    summary: dict[str, Any] = {
        "name": "bench_rollout",
        "quick": quick,
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
    }
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-process MuJoCo tabletop rollout throughput.",
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
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    eps = _QUICK_EPISODES if args.quick else args.episodes
    steps = _QUICK_STEPS if args.quick else args.steps_per_episode
    main(episodes=eps, steps_per_episode=steps, seed=args.seed, quick=args.quick)
