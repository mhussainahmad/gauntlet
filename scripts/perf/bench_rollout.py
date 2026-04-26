"""Rollout-throughput benchmark — Phase 2.5 T12.

Drives :class:`gauntlet.runner.Runner` with the :class:`RandomPolicy`
baseline against a 1-cell tabletop suite (one cell per backend) for
``--episodes`` rollouts and reports throughput + per-step latency.

Why through the Runner (not the bare env loop, like ``scripts/bench_rollout.py``):
the Polish-task ``bench_rollout.py`` deliberately bypasses the Runner to
isolate env-primitive regressions; this T12 bench complements it by
covering the full stack — Runner build, episode dispatch, seed
derivation, Episode construction. A regression in either layer surfaces
in only one of the two scripts, narrowing a bisect.

Backends:
    ``mujoco`` is always available (the core dependency set ships it as
    ``tabletop``). PyBullet / Genesis / Isaac selectable via ``--backend``;
    each prints a clean ``skipped: <reason>`` and exits 0 when its extra
    is not installed (so this bench can run unattended on a default-extras
    checkout without aborting). Maps user-facing slug → registry slug:

        mujoco   -> tabletop
        pybullet -> tabletop-pybullet
        genesis  -> tabletop-genesis
        isaac    -> tabletop-isaac

CLI:
    python scripts/perf/bench_rollout.py --backend mujoco --episodes 50 \\
        --output benchmarks/rollout.json

Outputs (flat JSON dict written to ``--output``):
    * ``episodes_per_sec``  — measured throughput
    * ``step_mean_ms``      — mean per-step latency (ms)
    * ``step_p50_ms`` / ``step_p95_ms`` / ``step_p99_ms`` — percentiles
    * ``version`` / ``timestamp`` / ``git_commit`` — provenance
    * ``partial`` (bool)    — true if KeyboardInterrupt cut the run short
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Ensure the scripts/perf/ directory is on sys.path so the
# ``_common`` sibling resolves both as a script (``python scripts/...``)
# and as an importable module (the smoke test invokes the script via
# subprocess, but uv's editable-install path already puts the repo root
# on sys.path so the absolute import works there too).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import emit_sidecar, percentile, provenance_fields

import gauntlet.env  # noqa: F401  (side-effect: registers tabletop slug)
from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory
from gauntlet.policy import RandomPolicy
from gauntlet.runner import Runner
from gauntlet.suite import load_suite_from_string

__all__ = ["main"]


# Tabletop env action dim: [dx, dy, dz, drx, dry, drz, gripper].
_TABLETOP_ACTION_DIM: int = 7

# Map user-facing backend slugs to env-registry slugs. The user-facing
# names (``mujoco`` / ``pybullet`` / ``genesis`` / ``isaac``) are
# memorable; the registry slugs (``tabletop`` / ``tabletop-X``) are
# what the Suite YAML and the env registry actually hold.
_BACKEND_TO_SLUG: dict[str, str] = {
    "mujoco": "tabletop",
    "pybullet": "tabletop-pybullet",
    "genesis": "tabletop-genesis",
    "isaac": "tabletop-isaac",
}

# Optional-extra backends gate on a side-effecting import (each
# subpackage's ``__init__`` calls ``register_env`` only when its
# native dep is importable). We try the import and skip-not-fail.
_BACKEND_IMPORT_PATH: dict[str, str] = {
    "pybullet": "gauntlet.env.pybullet",
    "genesis": "gauntlet.env.genesis",
    "isaac": "gauntlet.env.isaac",
}


def _try_register_backend(backend: str) -> str | None:
    """Trigger the side-effecting import for an optional backend.

    Returns ``None`` on success and a skip-reason string on
    ``ImportError`` (i.e. extra not installed). For the always-on
    ``mujoco`` backend this is a no-op.
    """
    if backend == "mujoco":
        return None
    module_path = _BACKEND_IMPORT_PATH[backend]
    try:
        __import__(module_path)
    except ImportError as exc:
        return f"backend {backend!r} not importable ({type(exc).__name__}: {exc})"
    return None


def _build_singleton_suite_yaml(*, env_slug: str, episodes: int, seed: int) -> str:
    """Return a 1-axis x 1-step suite YAML for the requested backend.

    Single-cell suite isolates rollout throughput (no per-cell compile
    overhead amortised across the sweep). ``episodes_per_cell`` drives
    volume.
    """
    return (
        "name: bench-rollout-t12\n"
        f"env: {env_slug}\n"
        f"seed: {seed}\n"
        f"episodes_per_cell: {episodes}\n"
        "axes:\n"
        "  object_initial_pose_x:\n"
        "    low: -0.05\n"
        "    high: 0.05\n"
        "    steps: 1\n"
    )


def main(
    *,
    backend: str,
    episodes: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    """Run the bench, write the sidecar JSON, return the summary dict."""
    print(f"bench_rollout (T12): backend={backend} episodes={episodes} seed={seed} output={output}")

    skip_reason = _try_register_backend(backend)
    if skip_reason is not None:
        summary: dict[str, object] = {
            "name": "bench_rollout",
            "backend": backend,
            "episodes": episodes,
            "seed": seed,
            "skipped": True,
            "skip_reason": skip_reason,
            "partial": False,
            "episodes_per_sec": 0.0,
            "step_mean_ms": 0.0,
            "step_p50_ms": 0.0,
            "step_p95_ms": 0.0,
            "step_p99_ms": 0.0,
            **provenance_fields(),
        }
        print(f"skipped: {skip_reason}")
        emit_sidecar(summary, output)
        print(json.dumps(summary))
        return summary

    env_slug = _BACKEND_TO_SLUG[backend]
    suite_yaml = _build_singleton_suite_yaml(env_slug=env_slug, episodes=episodes, seed=seed)
    suite = load_suite_from_string(suite_yaml)

    # The env factory is the registry hit for the slug; the cast widens
    # the registry's specific factory type to the Runner's
    # ``Callable[[], GauntletEnv]`` shape (TabletopEnv et al. satisfy
    # GauntletEnv structurally — same widening ``gauntlet.env.__init__``
    # does at registration time).
    env_factory: Callable[[], GauntletEnv] = get_env_factory(env_slug)

    runner = Runner(n_workers=1, env_factory=env_factory)

    # Per-step latency is sampled by re-rolling one episode "by hand" at
    # the end so we get sub-step latency stats without a Runner-internal
    # hook. The Runner pass measures throughput end-to-end (the metric
    # operators care about); the hand-roll pass measures per-step
    # latency (the metric a regression in env.step would surface in).

    partial = False
    eps_count: int = 0
    runner_wall_seconds: float = 0.0
    try:
        runner_start = time.perf_counter()
        # NOTE: lambda is safe ONLY because n_workers=1 (the in-process
        # fast path skips the pickle round-trip). If this script is ever
        # extended to expose --workers as a CLI flag, switch to
        # ``functools.partial(RandomPolicy, action_dim=..., seed=...)``
        # so the spawn pool can pickle the factory across the process
        # boundary; see scripts/perf/bench_runner_scaling.py for the
        # canonical pattern.
        results = runner.run(
            policy_factory=lambda: RandomPolicy(action_dim=_TABLETOP_ACTION_DIM, seed=seed),
            suite=suite,
        )
        runner_wall_seconds = time.perf_counter() - runner_start
        eps_count = len(results)
    except KeyboardInterrupt:
        partial = True
        print("interrupted: emitting partial results (partial=true)")

    # Hand-roll one episode to capture per-step deltas.
    step_deltas: list[float] = []
    if not partial:
        env = env_factory()
        try:
            policy = RandomPolicy(action_dim=_TABLETOP_ACTION_DIM, seed=seed)
            obs, _info = env.reset(seed=seed)
            terminated = False
            truncated = False
            # Episode-level cap: avoid an unbounded run-away if the env's
            # internal max_steps is misconfigured. ``len(step_deltas) <
            # 4096`` is a conservative ceiling — TabletopEnv's default
            # max_steps is far below this.
            while not (terminated or truncated) and len(step_deltas) < 4096:
                action = policy.act(obs)
                t0 = time.perf_counter()
                obs, _r, terminated, truncated, _info = env.step(action)
                step_deltas.append(time.perf_counter() - t0)
        except KeyboardInterrupt:
            partial = True
            print("interrupted: emitting partial results (partial=true)")
        finally:
            env.close()

    step_deltas.sort()
    p50_ms = percentile(step_deltas, 50.0) * 1000.0
    p95_ms = percentile(step_deltas, 95.0) * 1000.0
    p99_ms = percentile(step_deltas, 99.0) * 1000.0
    mean_step_ms = (sum(step_deltas) / len(step_deltas) * 1000.0) if step_deltas else 0.0
    eps_per_sec = (float(eps_count) / runner_wall_seconds) if runner_wall_seconds > 0.0 else 0.0

    print(
        f"  episodes_per_sec={eps_per_sec:.2f}  step_mean={mean_step_ms:.3f} ms  "
        f"p50={p50_ms:.3f}  p95={p95_ms:.3f}  p99={p99_ms:.3f}  (n={len(step_deltas)})"
    )

    summary = {
        "name": "bench_rollout",
        "backend": backend,
        "episodes": episodes,
        "episodes_completed": eps_count,
        "seed": seed,
        "skipped": False,
        "skip_reason": None,
        "partial": partial,
        "step_samples": len(step_deltas),
        "runner_wall_ms": round(runner_wall_seconds * 1000.0, 4),
        "episodes_per_sec": round(eps_per_sec, 4),
        "step_mean_ms": round(mean_step_ms, 4),
        "step_p50_ms": round(p50_ms, 4),
        "step_p95_ms": round(p95_ms, 4),
        "step_p99_ms": round(p99_ms, 4),
        **provenance_fields(),
    }
    emit_sidecar(summary, output)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 T12: rollout-throughput benchmark via Runner.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=sorted(_BACKEND_TO_SLUG.keys()),
        help=(
            "Env backend. ``mujoco`` is always installed; the others "
            "skip-not-fail when their extras are not installed."
        ),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed. Default: 42 (T12 convention).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/rollout.json"),
        help="JSON output path. Default: benchmarks/rollout.json under cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
    )
    sys.exit(0)
