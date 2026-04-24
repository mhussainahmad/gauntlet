"""Run a multi-camera tabletop rollout and inspect the per-camera obs.

This script exists as the reference wiring for the opt-in multi-camera
observation surface introduced in
``docs/polish-exploration-multi-camera.md``. It demonstrates the
canonical three-camera layout (wrist + side + top) used by SmolVLA-
style multi-view policies.

Usage:
    uv run python examples/evaluate_multi_camera.py [--out OUT_DIR]

The script does not run a full evaluation suite — it constructs a
single ``TabletopEnv``, resets it, takes one zero-action step, and
dumps each camera's first-frame PNG to ``OUT_DIR`` so you can sanity-
check the camera placement before plugging multi-cam into a real
suite (`gauntlet run`). Three artefacts land in ``OUT_DIR`` (default
``out-multicam/``):

* ``wrist.png`` — 64x96 wrist-mounted view.
* ``top.png``   — 96x96 overhead.
* ``side.png``  — 64x96 side view.

Real evaluation pipelines would feed ``obs["images"]`` into a multi-
view policy directly:

    obs, _ = env.reset(seed=0)
    action = policy.act(obs["images"]["wrist"], obs["images"]["top"], ...)

The legacy ``obs["image"]`` key is also populated, aliased to the
**first** camera's frame (here ``wrist``), so existing single-camera
consumers like the runner's video recorder keep working unchanged
(see RFC §3 backwards-compatibility table).

Headless GL: ``MUJOCO_GL=egl`` or ``MUJOCO_GL=osmesa`` is required
for offscreen rendering on a CI worker. Set it in the environment
before running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gauntlet.env import CameraSpec, TabletopEnv

__all__ = ["main"]


# Three canonical viewpoints — the same shape SmolVLA / ACT / Diffusion
# Policy training rigs use. Sizes deliberately differ to exercise the
# per-spec renderer cache; in production callers usually use a uniform
# 224x224 to match VLA pre-training.
_DEFAULT_CAMERAS: list[CameraSpec] = [
    CameraSpec(name="wrist", pose=(0.15, -0.15, 0.55, 1.0, 0.0, 0.5), size=(64, 96)),
    CameraSpec(name="top", pose=(0.0, 0.0, 1.2, 0.0, 0.0, 0.0), size=(96, 96)),
    CameraSpec(name="side", pose=(0.7, 0.0, 0.6, 1.2, 1.5, 0.0), size=(64, 96)),
]

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_OUT: Path = _REPO_ROOT / "out-multicam"


def _save_png(frame: np.ndarray, path: Path) -> None:
    """Write a uint8 (H, W, 3) frame to *path* via PIL.

    PIL is already a transitive dep of mujoco / opencv; if it is not
    installed in this env the script falls back to a raw .npy dump
    so the example can still demonstrate the obs shape.
    """
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), frame)
        print(f"PIL unavailable; wrote {path.with_suffix('.npy').name} instead of {path.name}")
        return
    Image.fromarray(frame).save(path)


def main(*, out_dir: Path = _DEFAULT_OUT) -> None:
    """Build a multi-cam env, render one frame per camera, dump to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    env = TabletopEnv(cameras=_DEFAULT_CAMERAS)
    try:
        obs, _ = env.reset(seed=0)

        # Sanity-check the contract. ``obs["images"]`` is the per-cam
        # dict; ``obs["image"]`` is the legacy first-cam alias.
        assert "images" in obs, "multi-cam obs missing 'images' key"
        images = obs["images"]
        for spec in _DEFAULT_CAMERAS:
            assert spec.name in images, f"missing camera frame {spec.name!r}"
            frame = images[spec.name]
            assert frame.dtype == np.uint8
            assert frame.shape == (*spec.size, 3)
            _save_png(frame, out_dir / f"{spec.name}.png")
            print(f"  {spec.name}: {frame.shape} {frame.dtype} -> {out_dir / (spec.name + '.png')}")

        # The ``image`` alias must match the FIRST cam byte-for-byte
        # (RFC §3) — production consumers depend on this.
        first = _DEFAULT_CAMERAS[0].name
        assert np.array_equal(obs["image"], images[first])
        print(f"  alias: obs['image'] == obs['images'][{first!r}]  (verified)")

        # And one zero-action step preserves the shape contract.
        obs_step, *_ = env.step(np.zeros(7, dtype=np.float64))
        assert "images" in obs_step
        for spec in _DEFAULT_CAMERAS:
            assert obs_step["images"][spec.name].shape == (*spec.size, 3)
        print("  step(): contract preserved")
    finally:
        env.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate the multi-camera observation surface on TabletopEnv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for per-cam PNGs (default: {_DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(out_dir=args.out)
