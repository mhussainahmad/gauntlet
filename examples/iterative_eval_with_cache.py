"""Iterative evaluation with the per-Episode rollout cache.

Usage:
    uv run python examples/iterative_eval_with_cache.py [--out OUT_DIR]
                                                        [--cache-dir CACHE_DIR]
                                                        [--suite SUITE_YAML]
                                                        [--n-workers N]

Demonstrates the iterative-dev workflow that motivates the cache:

1. First invocation: every cell is a miss; the Runner rolls each one
   and stores the resulting :class:`Episode` to ``--cache-dir``.
2. Second invocation (no code or suite change): every cell is a hit;
   the Runner returns the cached Episode without re-rolling — the
   second run takes a fraction of the first's wall-clock time.

For the smoke MuJoCo Tabletop env the gain is small (rollouts are
already millisecond-cheap); for VLA policies (`evaluate_smolvla.py`,
`evaluate_openvla.py`) where per-rollout time is 30+ seconds, the
same opt-in flag turns a 10-hour rerun into a sub-second cache hit.
See ``docs/polish-exploration-incremental-cache.md`` §1 for the
domain-win analysis.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main"]


# Per-episode env step cap. Threaded into Runner(max_steps=...) AND into
# the env factory so the cache key matches what the env actually does.
_SMOKE_MAX_STEPS = 20
_TABLETOP_ACTION_DIM = 7

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out"
_DEFAULT_CACHE: Path = _REPO_ROOT / "out" / ".gauntlet-cache"


def _build_env_factory(max_steps: int) -> Callable[[], TabletopEnv]:
    """Picklable env factory; ``functools.partial`` is spawn-friendly."""
    return partial(TabletopEnv, max_steps=max_steps)


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    """Picklable policy factory; the Runner re-seeds per episode."""
    return partial(RandomPolicy, action_dim=action_dim)


def _run_once(
    *,
    suite: Suite,
    cache_dir: Path,
    n_workers: int,
    max_steps: int,
    label: str,
) -> tuple[list[Episode], float, dict[str, int]]:
    """Run the suite once and return (episodes, wall_time_s, cache_stats)."""
    runner = Runner(
        n_workers=n_workers,
        env_factory=_build_env_factory(max_steps),
        cache_dir=cache_dir,
        max_steps=max_steps,
        # policy_id is the contract: same string -> same cache key. We
        # tag with the example's name so two different examples with
        # the same suite + RandomPolicy do not silently collide.
        policy_id=f"iterative-eval:{label}:RandomPolicy",
    )
    started = time.perf_counter()
    episodes = runner.run(
        policy_factory=_build_policy_factory(_TABLETOP_ACTION_DIM),
        suite=suite,
    )
    elapsed = time.perf_counter() - started
    return episodes, elapsed, runner.cache_stats()


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    cache_dir: Path = _DEFAULT_CACHE,
    n_workers: int = 1,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """Run the suite twice and print the cache hit-rate progression.

    Args:
        suite_path: YAML suite to evaluate. Defaults to the smoke suite.
        out_dir: Output directory; created if missing.
        cache_dir: Cache root; created if missing. Reuse across runs to
            see the second-invocation speedup.
        n_workers: Worker processes. ``1`` triggers the in-process fast
            path (still uses the cache).
        max_steps: Per-episode env step cap. Threaded into the env
            factory AND the cache key so the two stay in sync.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)

    print(
        f"Evaluating {suite.name} ({suite.num_cells()} cells x "
        f"{suite.episodes_per_cell} episodes) with cache_dir={cache_dir}"
    )

    # First invocation — cold cache.
    first_episodes, first_time, first_stats = _run_once(
        suite=suite,
        cache_dir=cache_dir,
        n_workers=n_workers,
        max_steps=max_steps,
        label="first",
    )
    print(
        f"  first run: {len(first_episodes)} episodes in {first_time:.2f}s "
        f"(cache: hits={first_stats['hits']} misses={first_stats['misses']} "
        f"puts={first_stats['puts']})"
    )

    # Second invocation — same suite, same cache_dir, same policy_id.
    second_episodes, second_time, second_stats = _run_once(
        suite=suite,
        cache_dir=cache_dir,
        n_workers=n_workers,
        max_steps=max_steps,
        label="first",  # same label -> same policy_id -> hits
    )
    speedup = first_time / second_time if second_time > 0 else float("inf")
    print(
        f"  second run: {len(second_episodes)} episodes in {second_time:.2f}s "
        f"(cache: hits={second_stats['hits']} misses={second_stats['misses']} "
        f"puts={second_stats['puts']})"
    )
    print(f"  speedup vs first run: {speedup:.1f}x")

    # Episode lists must be byte-identical across the two invocations
    # (same suite, same policy_id, same env config -> same cache keys).
    for a, b in zip(first_episodes, second_episodes, strict=True):
        assert a.model_dump() == b.model_dump(), (
            f"cache returned a stale Episode for cell={a.cell_index} "
            f"episode={a.episode_index}; this is a cache bug."
        )
    print("  cached Episodes match the freshly-rolled ones bit-for-bit.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative evaluation with the rollout cache.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Suite YAML to evaluate (default: {_DEFAULT_SUITE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help=f"Cache directory (default: {_DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Worker processes (default: 1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_SMOKE_MAX_STEPS,
        help=f"Per-episode step cap (default: {_SMOKE_MAX_STEPS}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        suite_path=args.suite,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        n_workers=args.n_workers,
        max_steps=args.max_steps,
    )
