"""Offscreen-render latency benchmark — Phase 2.5 T12.

Measures per-step render latency on
:class:`gauntlet.env.TabletopEnv` constructed with
``render_in_obs=True`` so each ``env.step`` call materialises an
``obs["image"]`` frame through MuJoCo's offscreen GL context.

MuJoCo is the only always-on render backend per repo conventions
(see RFC-005 / RFC-007 / RFC-009 — the alternative backends either
have no offscreen-render path or require a CUDA workstation), so
this bench targets MuJoCo and prints ``skipped: <reason>`` + exits
0 when the offscreen GL context cannot be created (typical headless
container without EGL or OSMesa).

CLI:
    python scripts/perf/bench_render.py --steps 200 \\
        --output benchmarks/render.json

Reports a flat JSON dict with:
    * ``frames_per_sec``      — measured throughput
    * ``render_step_mean_ms`` — mean wall per render-step
    * ``render_step_p50_ms`` / ``..._p95_ms`` / ``..._p99_ms`` — percentiles
    * ``version`` / ``timestamp`` / ``git_commit`` — provenance
    * ``partial`` (bool)      — true if KeyboardInterrupt cut the run short
    * ``skipped`` (bool)      — true when GL context unavailable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import emit_sidecar, percentile, provenance_fields

__all__ = ["main"]


# Tabletop env action dim: [dx, dy, dz, drx, dry, drz, gripper].
_TABLETOP_ACTION_DIM: int = 7

# Default render resolution — 224x224 is the ImageNet-class backbone
# input size and matches what most policy adapters in this repo expect.
# Kept fixed (not a CLI flag) because the spec asks for a single bench
# number; the existing scripts/bench_render_latency.py script already
# sweeps multiple sizes for users who want that breakdown.
_DEFAULT_RENDER_SIZE: tuple[int, int] = (224, 224)

# Warmup frames before measurement — MuJoCo's first render-after-reset
# pays the FBO + shader compile cost. Three frames is enough to
# amortise it without inflating the bench wall.
_WARMUP_FRAMES: int = 3


def _try_skip_if_render_unavailable() -> str | None:
    """Return a skip reason if a tiny offscreen render fails, else ``None``.

    Tries to construct + reset a 64x64 render-enabled env; any exception
    becomes a skip reason. Mirrors the existing ``bench_render_latency.py``
    skip path so the two scripts behave the same on a headless host.
    """
    try:
        from gauntlet.env import TabletopEnv

        env = TabletopEnv(render_in_obs=True, render_size=(64, 64))
        try:
            env.reset(seed=0)
        finally:
            env.close()
    except Exception as exc:  # broad: any GL/EGL/OSMesa failure path
        return f"{type(exc).__name__}: {exc}"
    return None


def main(
    *,
    steps: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    """Run the render bench, write the sidecar JSON, return the summary."""
    print(
        f"bench_render (T12): steps={steps} seed={seed} "
        f"render_size={_DEFAULT_RENDER_SIZE} output={output}"
    )

    skip_reason = _try_skip_if_render_unavailable()
    if skip_reason is not None:
        summary: dict[str, object] = {
            "name": "bench_render",
            "steps": steps,
            "seed": seed,
            "render_height": _DEFAULT_RENDER_SIZE[0],
            "render_width": _DEFAULT_RENDER_SIZE[1],
            "skipped": True,
            "skip_reason": skip_reason,
            "partial": False,
            "frames_per_sec": 0.0,
            "render_step_mean_ms": 0.0,
            "render_step_p50_ms": 0.0,
            "render_step_p95_ms": 0.0,
            "render_step_p99_ms": 0.0,
            **provenance_fields(),
        }
        print(f"skipped: {skip_reason}")
        emit_sidecar(summary, output)
        print(json.dumps(summary))
        return summary

    # Lazy import: keeps the skip-not-fail path on top of the module
    # body cheap when MuJoCo is broken in some weird subtle way.
    from gauntlet.env import TabletopEnv

    h, w = _DEFAULT_RENDER_SIZE
    env = TabletopEnv(render_in_obs=True, render_size=(h, w))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    partial = False
    measured_window_seconds = 0.0
    try:
        obs, _info = env.reset(seed=seed)
        # Sanity: the env actually returned the requested HxWx3 frame.
        image = obs["image"]
        if image.shape != (h, w, 3):
            raise RuntimeError(
                f"render returned shape {image.shape}; expected ({h}, {w}, 3)",
            )

        # Warmup — first frames touch the GL context for the first time
        # post-reset; we don't want them in the percentiles.
        for _ in range(_WARMUP_FRAMES):
            action = rng.uniform(-1.0, 1.0, size=_TABLETOP_ACTION_DIM)
            env.step(action.astype(np.float32))

        measured_start = time.perf_counter()
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=_TABLETOP_ACTION_DIM)
            t0 = time.perf_counter()
            obs, _r, terminated, truncated, _info = env.step(action.astype(np.float32))
            # Touch the image once so any lazy CPU readback is included
            # in the measurement window.
            _ = obs["image"].shape
            deltas.append(time.perf_counter() - t0)
            if terminated or truncated:
                # Reset between-episodes resets are NOT counted in the
                # render window — they would skew the measurement.
                env.reset(seed=seed)
        measured_window_seconds = time.perf_counter() - measured_start
    except KeyboardInterrupt:
        partial = True
        print("interrupted: emitting partial results (partial=true)")
    finally:
        env.close()

    deltas.sort()
    p50_ms = percentile(deltas, 50.0) * 1000.0
    p95_ms = percentile(deltas, 95.0) * 1000.0
    p99_ms = percentile(deltas, 99.0) * 1000.0
    mean_ms = (sum(deltas) / len(deltas) * 1000.0) if deltas else 0.0
    fps = (float(len(deltas)) / measured_window_seconds) if measured_window_seconds > 0.0 else 0.0

    print(
        f"  frames_per_sec={fps:.2f}  step_mean={mean_ms:.3f} ms  "
        f"p50={p50_ms:.3f}  p95={p95_ms:.3f}  p99={p99_ms:.3f}  (n={len(deltas)})"
    )

    summary = {
        "name": "bench_render",
        "steps": steps,
        "seed": seed,
        "render_height": h,
        "render_width": w,
        "skipped": False,
        "skip_reason": None,
        "partial": partial,
        "samples": len(deltas),
        "measured_wall_ms": round(measured_window_seconds * 1000.0, 4),
        "frames_per_sec": round(fps, 4),
        "render_step_mean_ms": round(mean_ms, 4),
        "render_step_p50_ms": round(p50_ms, 4),
        "render_step_p95_ms": round(p95_ms, 4),
        "render_step_p99_ms": round(p99_ms, 4),
        **provenance_fields(),
    }
    emit_sidecar(summary, output)
    print(json.dumps(summary))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 T12: offscreen-render latency benchmark on TabletopEnv.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed. Default: 42 (T12 convention).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/render.json"),
        help="JSON output path. Default: benchmarks/render.json under cwd.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(
        steps=args.steps,
        seed=args.seed,
        output=args.output,
    )
    sys.exit(0)
