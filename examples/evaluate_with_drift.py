"""Three-step drift-detection workflow — README's quickstart under the hood.

Usage:
    uv sync --extra monitor
    uv run python examples/evaluate_with_drift.py [--out OUT_DIR]

Flow (matches RFC §2):

1. Reference sweep: run :class:`ScriptedPolicy` with
   ``--record-trajectories`` to build a known-good reference corpus.
2. Train: fit a :class:`StateAutoencoder` on the reference corpus.
3. Score: run :class:`RandomPolicy` as a deliberately-noisier candidate,
   dump its trajectories, score them against the reference AE, emit
   ``drift.json``.

This is a demonstration, not a benchmark: the reference and candidate
policies are both hand-written, so the reported drift signal is
artificial-but-observable — ``RandomPolicy`` produces per-step actions
that are noticeably further from the scripted trajectories the AE was
trained on.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy, ScriptedPolicy
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

# Module-level paths + factories (picklable under ``spawn``).
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out" / "drift_demo"
_TABLETOP_ACTION_DIM: int = 7
_SMOKE_MAX_STEPS: int = 20

__all__ = ["main"]


def _build_env_factory(max_steps: int) -> Callable[[], TabletopEnv]:
    """Module-level env factory; ``partial`` over the class pickles under spawn."""
    return partial(TabletopEnv, max_steps=max_steps)


def _scripted_factory() -> ScriptedPolicy:
    """Reference policy — the AE learns to reconstruct its trajectories."""
    return ScriptedPolicy()


def _random_factory() -> RandomPolicy:
    """Candidate policy — deliberately higher entropy than the reference."""
    return RandomPolicy(action_dim=_TABLETOP_ACTION_DIM, seed=None)


def _run(
    *,
    suite: Suite,
    policy_factory: Callable[[], ScriptedPolicy | RandomPolicy],
    traj_dir: Path,
    max_steps: int,
    n_workers: int,
) -> list[Episode]:
    """Run one sweep with trajectory capture enabled."""
    runner = Runner(
        n_workers=n_workers,
        env_factory=_build_env_factory(max_steps),
        trajectory_dir=traj_dir,
    )
    return runner.run(policy_factory=policy_factory, suite=suite)


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 1,
    max_steps: int = _SMOKE_MAX_STEPS,
) -> None:
    """End-to-end drift demo. Writes ``drift.json`` under ``out_dir``."""
    # Lazy-import the torch-backed modules so ``python examples/...
    # --help`` works on a torch-free install.
    from gauntlet.monitor.score import score_drift
    from gauntlet.monitor.train import train_ae

    suite: Suite = load_suite(suite_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference sweep.
    ref_traj = out_dir / "reference_trajectories"
    _run(
        suite=suite,
        policy_factory=_scripted_factory,
        traj_dir=ref_traj,
        max_steps=max_steps,
        n_workers=n_workers,
    )

    # 2. Train the AE.
    ae_dir = out_dir / "autoencoder"
    train_ae(ref_traj, out_dir=ae_dir, epochs=30, batch_size=64, seed=0)

    # 3. Candidate sweep + score.
    cand_traj = out_dir / "candidate_trajectories"
    cand_episodes = _run(
        suite=suite,
        policy_factory=_random_factory,
        traj_dir=cand_traj,
        max_steps=max_steps,
        n_workers=n_workers,
    )
    import json

    episodes_json = out_dir / "episodes.json"
    episodes_json.write_text(
        json.dumps([ep.model_dump(mode="json") for ep in cand_episodes], indent=2) + "\n",
        encoding="utf-8",
    )
    drift = score_drift(episodes_json, cand_traj, ae_dir, top_k=5)
    (out_dir / "drift.json").write_text(drift.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote drift.json under {out_dir} "
        f"(reference_mean={drift.reference_reconstruction_error_mean:.4f}, "
        f"candidate_mean={drift.candidate_reconstruction_error_mean:.4f})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=_SMOKE_MAX_STEPS)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        suite_path=args.suite,
        out_dir=args.out,
        n_workers=args.n_workers,
        max_steps=args.max_steps,
    )
