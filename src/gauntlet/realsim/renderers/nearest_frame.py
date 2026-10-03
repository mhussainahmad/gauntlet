"""Zero-dependency nearest-training-frame renderer.

Closes the Phase 3 renderer story without taking on a torch / gsplat
dependency: returns the RGB image from whichever training
:class:`CameraFrame` has its 4x4 SE(3) pose closest to the requested
viewpoint, nearest-resized to the requested intrinsics' resolution.

What this renderer is good for
------------------------------
* Smoke-testing the realsim pipeline end-to-end without a GPU.
* Sanity-checking a scene's frame coverage — if every requested viewpoint
  ends up returning the same training frame, your input set does not span
  the perturbation envelope you think it does.
* A baseline for renderer comparisons. Any real reconstructor (gsplat,
  NeRF, surfel-rendering) MUST beat this — if it does not, something
  upstream is wrong.

What it is **not**
------------------
* Novel-view synthesis. Returning a training frame is a *lookup*, not
  a reconstruction. Do not pretend this generalises to perturbations
  off the captured manifold.
* Photometric calibration. Exposure and white-balance follow the
  closest frame's; cross-frame consistency is whatever the input had.

Scene-root note
---------------
:class:`CameraFrame.path` is relative to the directory the Scene was
loaded from. The Protocol's :meth:`render` contract does not carry that
root, so the renderer takes ``scene_root`` in its constructor and
resolves frame paths against it. The zero-arg factory registered under
``nearest-frame`` defaults to the process cwd — fine for the CLI
``gauntlet realsim render`` subcommand that ``chdir``s into the scene
directory; tests and library users that bypass the registry should pass
``NearestFrameRenderer(scene_root=...)`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover -- typing-only import
    from gauntlet.realsim.schema import CameraFrame, CameraIntrinsics, Pose, Scene


__all__ = ["NearestFrameRenderer"]


def _pose_translation(pose: Pose) -> NDArray[np.float64]:
    """Return the translation column of a 4x4 row-major SE(3) matrix."""
    m = np.asarray(pose.matrix, dtype=np.float64)
    return np.array([m[0, 3], m[1, 3], m[2, 3]], dtype=np.float64)


def _pose_rotation(pose: Pose) -> NDArray[np.float64]:
    """Return the 3x3 rotation sub-block of a 4x4 row-major SE(3) matrix."""
    m = np.asarray(pose.matrix, dtype=np.float64)
    return np.ascontiguousarray(m[:3, :3], dtype=np.float64)


def _pose_distance(a: Pose, b: Pose) -> float:
    """Return a scalar SE(3) pose distance between *a* and *b*.

    Uses translation L2 + a rotation-angle term scaled to metres. The
    rotation-angle is derived from ``trace(R_a^T R_b)`` — works on the
    raw 3x3 block even if it has drifted slightly off SO(3), which the
    Pose schema explicitly tolerates. The exact weighting (0.05 m per
    radian) is empirical; it is never surfaced to the user, only used
    for argmin selection.
    """
    t_a = _pose_translation(a)
    t_b = _pose_translation(b)
    translation_l2 = float(np.linalg.norm(t_a - t_b))
    r_a = _pose_rotation(a)
    r_b = _pose_rotation(b)
    # cos(theta) = (tr(R_a^T R_b) - 1) / 2 for orthonormal R; clip to
    # [-1, 1] so a slightly-off rotation does not surface as a NaN.
    cos_theta = float(np.clip((np.trace(r_a.T @ r_b) - 1.0) / 2.0, -1.0, 1.0))
    angle_rad = float(np.arccos(cos_theta))
    return translation_l2 + 0.05 * angle_rad


def _resize_nearest(
    image: NDArray[np.uint8],
    target_height: int,
    target_width: int,
) -> NDArray[np.uint8]:
    """Nearest-neighbour resize an HxWx3 ``uint8`` image to a new HxW.

    Uses pure-numpy index gathers — no third-party imaging dep. This is
    intentionally crude: the nearest-frame renderer is a baseline, not
    a quality target.
    """
    src_h, src_w = image.shape[:2]
    if src_h == target_height and src_w == target_width:
        return np.ascontiguousarray(image, dtype=np.uint8)
    row_idx = (np.arange(target_height) * src_h // target_height).astype(np.int64)
    col_idx = (np.arange(target_width) * src_w // target_width).astype(np.int64)
    return np.ascontiguousarray(image[np.ix_(row_idx, col_idx)], dtype=np.uint8)


class NearestFrameRenderer:
    """:class:`gauntlet.realsim.RealSimRenderer` returning the closest
    training frame's image.

    Args:
        scene_root: directory to resolve :class:`CameraFrame.path`
            against. Defaults to the process cwd, which matches the
            convention used by :func:`gauntlet.realsim.load_scene`.
    """

    def __init__(self, scene_root: Path | str | None = None) -> None:
        self._scene_root: Path = Path(scene_root) if scene_root is not None else Path.cwd()

    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> NDArray[np.uint8]:
        """Return the closest-frame RGB image resized to *intrinsics*.

        Returns:
            HxWx3 ``uint8`` array with ``H == intrinsics.height`` and
            ``W == intrinsics.width``.
        """
        frame = _pick_nearest_frame(scene, viewpoint)
        image = _load_frame_image(frame, self._scene_root)
        return _resize_nearest(image, intrinsics.height, intrinsics.width)


def _pick_nearest_frame(scene: Scene, viewpoint: Pose) -> CameraFrame:
    if not scene.frames:
        raise ValueError("NearestFrameRenderer: scene has zero frames; nothing to render from")
    distances = [_pose_distance(viewpoint, frame.pose) for frame in scene.frames]
    return scene.frames[int(np.argmin(distances))]


def _load_frame_image(frame: CameraFrame, scene_root: Path) -> NDArray[np.uint8]:
    """Decode a :class:`CameraFrame`'s image to an HxWx3 ``uint8`` array.

    Resolves ``frame.path`` against *scene_root* and decodes via PIL.
    PIL is imported lazily so the renderer module's import cost stays
    near-zero for callers that never invoke :meth:`render`.
    """
    image_path = (scene_root / frame.path).resolve()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover -- core dep, very unlikely
        raise ImportError(
            "NearestFrameRenderer requires Pillow. "
            "It ships with the realsim ingest pipeline; install via "
            "`pip install gauntlet` or `pip install pillow` to bring it in."
        ) from exc
    with Image.open(image_path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
