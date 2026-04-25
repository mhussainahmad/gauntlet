"""Evaluate a Decision Transformer baseline against the tabletop smoke suite.

**Bring your own checkpoint.** No public Decision Transformer
checkpoint is trained on Gauntlet's TabletopEnv. The intended
workflow (B-16) is:

1. Dump trajectories with ``Runner(trajectory_dir=..., trajectory_format="parquet")``
   (the [parquet] extra, B-23).
2. Train a small DT externally on those trajectories — gauntlet has
   explicitly stayed out of the training story (see B-16 anti-feature
   note in ``docs/backlog.md``).
3. Point ``--model-id`` here at the resulting HF Hub repo or local
   checkpoint path; the
   :class:`~gauntlet.policy.dt.DecisionTransformerPolicy` adapter loads
   it and runs inference against the smoke suite.

This file is **not** executed by the test suite. Install the extras
and run::

    uv sync --extra dt
    uv run python examples/evaluate_dt.py --model-id <your-dt-repo> --out out-dt/
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

from gauntlet.env import TabletopEnv
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main", "make_dt_policy"]


_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out-dt"
_DEFAULT_TARGET_RETURN: float = 100.0
_DEFAULT_CONTEXT_LENGTH: int = 20

_MODEL_ID_HOLDER: dict[str, str] = {}


def make_dt_policy() -> object:
    """Module-level factory — picklable, spawn-safe (spec §6)."""
    from gauntlet.policy.dt import DecisionTransformerPolicy

    return DecisionTransformerPolicy(
        model_id=_MODEL_ID_HOLDER["model_id"],
        device="cpu",
        target_return=_DEFAULT_TARGET_RETURN,
        context_length=_DEFAULT_CONTEXT_LENGTH,
    )


def _build_env_factory() -> Callable[[], TabletopEnv]:
    return partial(TabletopEnv, render_in_obs=True, render_size=(224, 224))


def main(
    *,
    model_id: str,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 1,
) -> None:
    _MODEL_ID_HOLDER["model_id"] = model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)
    runner = Runner(n_workers=n_workers, env_factory=_build_env_factory())
    episodes: list[Episode] = runner.run(policy_factory=make_dt_policy, suite=suite)
    print(f"Completed {len(episodes)} episodes -> {out_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Decision Transformer baseline.")
    parser.add_argument("--model-id", required=True, help="HF Hub repo ID or local path")
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        model_id=args.model_id,
        suite_path=args.suite,
        out_dir=args.out,
        n_workers=args.n_workers,
    )
