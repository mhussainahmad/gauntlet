"""Render-latency benchmark for the MuJoCo tabletop env.

Measures per-frame wall-clock for ``obs["image"]`` rendering on
:class:`gauntlet.env.TabletopEnv` constructed with ``render_in_obs=True``
at three canonical sizes:

* 84 x 84   (small policy-input thumbnail, e.g. CNN feature extractor),
* 224 x 224 (ImageNet-style backbone input),
* 480 x 640 (VGA-class observation for downstream perception).

The bench is single-process and stays in MuJoCo's offscreen GL context
(``mujoco.MjrContext``) — exactly the same render path the Runner exposes
to a policy that reads ``obs["image"]``. If the offscreen context cannot
be initialised (no EGL / OSMesa / display) the script skips cleanly with
a printed reason and exits 0 — the spec requires skip-not-fail so this
bench can run unattended on a headless container or a graphics-broken
laptop without taking down a CI pipeline.

Usage:
    uv run --no-sync python scripts/bench_render_latency.py [--frames N]
                                                            [--warmup N]
                                                            [--seed S]
                                                            [--quick]

Outputs:
    * Text table to stdout (sizes x p50/p95/p99/mean).
    * Single-line JSON summary as the *last* line of stdout
      (CI can ``tail -n 1``).
    * JSON sidecar file ``bench_render_latency.json`` written next to
      the working directory (override with ``--out PATH``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["main"]


# Three canonical render sizes (height, width). Picked to span the
# common policy-input regimes from a CNN thumbnail to a VGA frame.
_DEFAULT_SIZES: tuple[tuple[int, int], ...] = (
    (84, 84),
    (224, 224),
    (480, 640),
)

# Sample counts; --quick collapses both. Frames per size of 30 keeps the
# full run sub-second per size on a laptop while still giving a stable
# p99 (30 samples is enough for >2x regression detection, the bar this
# bench is designed for).
_DEFAULT_FRAMES: int = 30
_DEFAULT_WARMUP: int = 3
_QUICK_FRAMES: int = 10
_QUICK_WARMUP: int = 2

# Action dim for the random control loop (TabletopEnv: dx,dy,dz,drx,dry,drz,gripper).
_TABLETOP_ACTION_DIM: int = 7


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Returns 0.0 on an empty input so the JSON output stays well-formed
    even when a smoke run captures zero samples.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, round(pct / 100.0 * n)))
    return sorted_values[rank - 1]


def _bench_one_size(
    *,
    height: int,
    width: int,
    frames: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    """Time ``frames`` renders at the given resolution.

    Constructs a fresh :class:`TabletopEnv` per size so the offscreen
    GL context matches the requested framebuffer size; reuses a single
    env across the warmup + measurement loop so MuJoCo's per-context
    setup (shader compile, FBO allocation) is amortised before the
    first measured frame.
    """
    # Lazy import — keeps the skip-not-fail path on top of the module
    # body cheap, even when MuJoCo is broken in some weird subtle way.
    from gauntlet.env import TabletopEnv

    env = TabletopEnv(render_in_obs=True, render_size=(height, width))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    construct_start = time.perf_counter()
    obs, _info = env.reset(seed=seed)
    construct_seconds = time.perf_counter() - construct_start
    # Sanity: image really is the requested HxWx3.
    image = obs["image"]
    if image.shape != (height, width, 3):
        env.close()
        raise RuntimeError(
            f"render returned shape {image.shape}; expected ({height}, {width}, 3)",
        )

    try:
        # Warmup — first few step() calls touch the GL context for the
        # first time after reset; we don't want that in the percentiles.
        for _ in range(warmup):
            action = rng.uniform(-1.0, 1.0, size=_TABLETOP_ACTION_DIM)
            env.step(action.astype(np.float32))

        for _ in range(frames):
            action = rng.uniform(-1.0, 1.0, size=_TABLETOP_ACTION_DIM)
            t0 = time.perf_counter()
            obs, _r, _term, _trunc, _info = env.step(action.astype(np.float32))
            # Touch the image once so any lazy CPU readback is included
            # in the measurement window.
            _ = obs["image"].shape
            deltas.append(time.perf_counter() - t0)
    finally:
        env.close()

    deltas.sort()
    p50_ms = _percentile(deltas, 50.0) * 1000.0
    p95_ms = _percentile(deltas, 95.0) * 1000.0
    p99_ms = _percentile(deltas, 99.0) * 1000.0
    mean_ms = (sum(deltas) / len(deltas) * 1000.0) if deltas else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0
    return {
        "height": height,
        "width": width,
        "frames": frames,
        "warmup": warmup,
        "construct_ms": round(construct_seconds * 1000.0, 4),
        "step_with_render_p50_ms": round(p50_ms, 4),
        "step_with_render_p95_ms": round(p95_ms, 4),
        "step_with_render_p99_ms": round(p99_ms, 4),
        "step_with_render_mean_ms": round(mean_ms, 4),
        "fps_mean": round(fps, 2),
    }


def _try_skip_if_render_unavailable() -> str | None:
    """Return a skip reason if the env cannot be constructed with rendering.

    Tries to construct + reset a tiny ``TabletopEnv(render_in_obs=True)``;
    any exception (typically a MuJoCo GL / EGL / OSMesa import failure on
    a headless machine) is converted into a skip reason. Returns None
    when rendering is available.
    """
    try:
        from gauntlet.env import TabletopEnv

        env = TabletopEnv(render_in_obs=True, render_size=(64, 64))
        try:
            env.reset(seed=0)
        finally:
            env.close()
    except Exception as exc:  # broad catch: any GL/EGL failure path
        return f"{type(exc).__name__}: {exc}"
    return None


def _emit_sidecar(summary: dict[str, Any], out_path: Path) -> None:
    """Write the summary dict to ``out_path`` as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")


def main(
    *,
    sizes: tuple[tuple[int, int], ...],
    frames: int,
    warmup: int,
    seed: int,
    quick: bool,
    out_path: Path,
) -> dict[str, Any]:
    """Run the bench across all sizes and return the summary dict."""
    print(
        f"bench_render_latency: sizes={list(sizes)} frames={frames} warmup={warmup} "
        f"seed={seed} quick={quick}"
    )

    skip_reason = _try_skip_if_render_unavailable()
    if skip_reason is not None:
        summary: dict[str, Any] = {
            "name": "bench_render_latency",
            "quick": quick,
            "skipped": True,
            "skip_reason": skip_reason,
            "cases": [],
        }
        print(f"skipped: {skip_reason}")
        _emit_sidecar(summary, out_path)
        print(json.dumps(summary))
        return summary

    cases: list[dict[str, Any]] = []
    for height, width in sizes:
        result = _bench_one_size(
            height=height,
            width=width,
            frames=frames,
            warmup=warmup,
            seed=seed,
        )
        cases.append(result)
        print(
            f"  {height:>4} x {width:<4}: "
            f"p50={result['step_with_render_p50_ms']:.3f} ms  "
            f"p95={result['step_with_render_p95_ms']:.3f} ms  "
            f"p99={result['step_with_render_p99_ms']:.3f} ms  "
            f"mean={result['step_with_render_mean_ms']:.3f} ms  "
            f"({result['fps_mean']:.1f} fps)"
        )

    summary = {
        "name": "bench_render_latency",
        "quick": quick,
        "skipped": False,
        "skip_reason": None,
        "frames": frames,
        "warmup": warmup,
        "seed": seed,
        "cases": cases,
    }
    _emit_sidecar(summary, out_path)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark obs['image'] render latency for the MuJoCo tabletop env.",
    )
    parser.add_argument("--frames", type=int, default=_DEFAULT_FRAMES)
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Smoke run: {_QUICK_FRAMES} frames per size, "
            f"{_QUICK_WARMUP} warmup frames. Overrides --frames / --warmup."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench_render_latency.json"),
        help="Sidecar JSON output path. Default: bench_render_latency.json in cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    used_frames = _QUICK_FRAMES if args.quick else args.frames
    used_warmup = _QUICK_WARMUP if args.quick else args.warmup
    main(
        sizes=_DEFAULT_SIZES,
        frames=used_frames,
        warmup=used_warmup,
        seed=args.seed,
        quick=args.quick,
        out_path=args.out,
    )
    sys.exit(0)
