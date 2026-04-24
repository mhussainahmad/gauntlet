"""Run :class:`RandomPolicy` against the tabletop smoke suite + record MP4s.

Usage:
    uv run python examples/evaluate_random_policy_with_video.py \
        [--out OUT_DIR] [--suite SUITE_YAML] [--n-workers N] \
        [--video-fps FPS] [--only-failures]

This is the Polish "rollout MP4 video recording" partner of
``examples/evaluate_random_policy.py``. The Runner is constructed
with ``record_video=True`` and ``render_in_obs=True`` is wired into
the env factory so every rollout produces a per-episode MP4
alongside the usual ``episodes.json`` / ``report.json`` /
``report.html`` triple. The HTML report carries inline ``<video>``
thumbnails in the failure-clusters table — open
``out/report.html`` and click the thumbnail to play the failed
rollout in-browser.

Requires the optional ``[video]`` extra:

    uv sync --extra video
    # or:  pip install "gauntlet[video]"

The ``[video]`` extra pulls ``imageio[ffmpeg]``, which bundles a
static ffmpeg binary — no system ffmpeg install required.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from gauntlet.env import TabletopEnv
from gauntlet.policy import RandomPolicy
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import Suite, load_suite

__all__ = ["main"]


_TABLETOP_ACTION_DIM: int = 7
_SMOKE_MAX_STEPS: int = 20

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE: Path = _REPO_ROOT / "examples" / "suites" / "tabletop-smoke.yaml"
_DEFAULT_OUT: Path = _REPO_ROOT / "out"


def _build_env_factory(max_steps: int) -> Callable[[], TabletopEnv]:
    """Return a picklable env factory with ``render_in_obs=True``.

    The Runner enforces that ``obs["image"]`` is present after the
    first reset whenever ``record_video=True``; constructing the env
    with ``render_in_obs=True`` here is therefore a hard requirement
    of this example. ``functools.partial`` over the class is the
    canonical spawn-friendly factory.
    """
    return partial(TabletopEnv, max_steps=max_steps, render_in_obs=True)


def _build_policy_factory(action_dim: int) -> Callable[[], RandomPolicy]:
    return partial(RandomPolicy, action_dim=action_dim)


def _episodes_to_json(episodes: list[Episode]) -> str:
    payload: list[dict[str, Any]] = [ep.model_dump(mode="json") for ep in episodes]
    cleaned = _nan_to_none(payload)
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _report_to_json(report: Report) -> str:
    cleaned = _nan_to_none(report.model_dump(mode="json"))
    return json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(
    *,
    suite_path: Path = _DEFAULT_SUITE,
    out_dir: Path = _DEFAULT_OUT,
    n_workers: int = 2,
    max_steps: int = _SMOKE_MAX_STEPS,
    video_fps: int = 30,
    only_failures: bool = False,
) -> None:
    """Run the full evaluation pipeline + write per-episode MP4s.

    Args:
        suite_path: YAML suite to load.
        out_dir: Output directory; created if missing. Receives
            ``episodes.json``, ``report.json``, ``report.html`` AND a
            ``videos/`` subdirectory with one MP4 per recorded episode.
        n_workers: Worker processes. Defaults to 2.
        max_steps: Hard cap on per-episode env steps.
        video_fps: Output framerate for the MP4 encoder.
        only_failures: When ``True``, only ``success=False`` episodes
            get an MP4 (saves disk on long sweeps).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suite: Suite = load_suite(suite_path)

    runner = Runner(
        n_workers=n_workers,
        env_factory=_build_env_factory(max_steps),
        record_video=True,
        # Default ``video_dir=None`` would write to ``Path("videos")``
        # (CWD-relative); pin it under the output dir so the HTML
        # report's relative ``<video src="videos/...mp4">`` embed
        # works out-of-the-box.
        video_dir=out_dir / "videos",
        video_fps=video_fps,
        record_only_failures=only_failures,
    )
    episodes: list[Episode] = runner.run(
        policy_factory=_build_policy_factory(_TABLETOP_ACTION_DIM),
        suite=suite,
    )

    report: Report = build_report(episodes)

    (out_dir / "episodes.json").write_text(_episodes_to_json(episodes), encoding="utf-8")
    (out_dir / "report.json").write_text(_report_to_json(report), encoding="utf-8")
    write_html(report, out_dir / "report.html")

    n_videos = sum(1 for ep in episodes if ep.video_path is not None)
    print(
        f"Wrote {len(episodes)} episodes / {len(report.per_cell)} cells / "
        f"{n_videos} MP4s -> {out_dir} "
        f"(success: {report.overall_success_rate * 100:.1f}%)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RandomPolicy + tabletop smoke suite + per-episode MP4 recording.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Suite YAML (default: {_DEFAULT_SUITE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=2,
        help="Worker processes (default: 2).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_SMOKE_MAX_STEPS,
        help=f"Per-episode step cap (default: {_SMOKE_MAX_STEPS}).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="MP4 framerate (default: 30).",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Only write MP4s for episodes that ended with success=False.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        suite_path=args.suite,
        out_dir=args.out,
        n_workers=args.n_workers,
        max_steps=args.max_steps,
        video_fps=args.video_fps,
        only_failures=args.only_failures,
    )
