"""Image observations for the Isaac Sim tabletop backend (experimental).

Opt-in through ``IsaacSimTabletopEnv(render_in_obs=True)``. Adds one RGB
camera, a key light and two cube materials to the scene, and maps the
three cosmetic axes onto them:

* ``lighting_intensity`` scales the key light's ``inputs:intensity``;
* ``camera_offset_x`` / ``camera_offset_y`` shift the camera position
  (the aim point stays on the table centre);
* ``object_texture`` swaps the cube between the two materials, using the
  same colours as the MuJoCo cube (``cube_mat`` / ``cube_alt_mat``).

**Not hardware-verified.** Written against the Isaac Sim 5.0 sources
(``isaac-sim/IsaacSim`` at tag ``v5.0.0``):
``isaacsim.sensors.camera.Camera`` (``resolution=(width, height)``,
scalar-first ``wxyz`` orientation in the "world" camera axes, +X
forward / +Z up; ``initialize()`` after ``World.reset()``;
``get_rgba()`` returns ``H x W x 4``), ``isaacsim.core.api.materials.
PreviewSurface(prim_path, color)`` with RGB in ``[0, 1]``,
``VisualCuboid.apply_visual_material``, and
``isaacsim.core.utils.prims.create_prim(..., "DistantLight",
attributes=...)``. CI exercises it only against a fake ``isaacsim``
namespace; it has not been run on an RTX workstation. Pixel values are
not comparable to the other backends (different renderer).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

__all__ = ["IsaacTabletopRenderer", "look_at_quat_world"]

# Camera position matches the MuJoCo ``main`` camera; Isaac aims it at
# the table centre rather than reproducing MuJoCo's xyaxes framing.
CAMERA_POS: Final[NDArray[np.float64]] = np.array([0.6, -0.6, 0.8], dtype=np.float64)
CAMERA_TARGET: Final[NDArray[np.float64]] = np.array([0.0, 0.0, 0.42], dtype=np.float64)
LIGHT_BASE_INTENSITY: Final[float] = 3000.0
CUBE_COLORS: Final[tuple[NDArray[np.float64], NDArray[np.float64]]] = (
    np.array([0.8, 0.2, 0.2], dtype=np.float64),  # cube_mat
    np.array([0.2, 0.6, 0.2], dtype=np.float64),  # cube_alt_mat
)
_WARMUP_RENDERS: Final[int] = 4


def look_at_quat_world(
    eye: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``wxyz`` quaternion pointing a "world"-axes camera (+X forward, +Z up) at *target*."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left)
    true_up = np.cross(forward, left)
    rot = np.stack([forward, left, true_up], axis=1)  # columns: camera x, y, z in world
    return _rotmat_to_quat_wxyz(rot)


def _rotmat_to_quat_wxyz(m: NDArray[np.float64]) -> NDArray[np.float64]:
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quat = np.asarray(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


class IsaacTabletopRenderer:
    """Camera + light + cube materials for :class:`IsaacSimTabletopEnv`.

    Construct after ``SimulationApp`` is up and the scene prims exist.
    """

    def __init__(self, world: Any, cube: Any, render_size: tuple[int, int]) -> None:
        # Kit-dependent imports must follow SimulationApp construction.
        from isaacsim.core.api.materials import PreviewSurface
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.sensors.camera import Camera

        self._world = world
        self._cube = cube
        self._height, self._width = render_size
        self._light: Any = create_prim(
            "/World/gauntlet_key_light",
            "DistantLight",
            attributes={"inputs:intensity": LIGHT_BASE_INTENSITY},
        )
        self._materials: tuple[Any, Any] = (
            PreviewSurface(prim_path="/World/Looks/gauntlet_cube", color=CUBE_COLORS[0]),
            PreviewSurface(prim_path="/World/Looks/gauntlet_cube_alt", color=CUBE_COLORS[1]),
        )
        self._camera: Any = Camera(
            prim_path="/World/gauntlet_camera",
            resolution=(self._width, self._height),
            position=CAMERA_POS.copy(),
            orientation=look_at_quat_world(CAMERA_POS, CAMERA_TARGET),
        )
        self._initialized = False

    def after_world_reset(self) -> None:
        """Initialise the camera's render product (once, after the first ``World.reset``)."""
        if not self._initialized:
            self._camera.initialize()
            self._initialized = True

    def apply(
        self, *, light_intensity: float, cam_offset: NDArray[np.float64], texture: int
    ) -> None:
        """Push the current cosmetic-axis values into the scene."""
        self._light.GetAttribute("inputs:intensity").Set(
            LIGHT_BASE_INTENSITY * float(light_intensity)
        )
        eye = CAMERA_POS + np.array([cam_offset[0], cam_offset[1], 0.0], dtype=np.float64)
        self._camera.set_world_pose(
            position=eye,
            orientation=look_at_quat_world(eye, CAMERA_TARGET + np.r_[cam_offset, 0.0]),
            camera_axes="world",
        )
        self._cube.apply_visual_material(self._materials[1 if texture else 0])

    def read(self) -> NDArray[np.uint8]:
        """Render the current scene and return it as ``[H, W, 3]`` ``uint8``.

        Always renders first: the camera annotator holds the most recently
        rendered frame, so reading after a scene change without rendering
        would return a stale image. Renders a few more times if the sensor
        has not produced its first frame yet.
        """
        for _ in range(_WARMUP_RENDERS + 1):
            self._world.render()
            rgba = self._camera.get_rgba()
            if rgba is not None and np.asarray(rgba).size > 0:
                break
        else:
            raise RuntimeError(
                "Isaac camera produced no frame after warm-up renders; "
                "check that the RTX renderer is available"
            )
        frame = np.asarray(rgba)[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(np.asarray(frame, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
        if frame.shape != (self._height, self._width, 3):
            raise RuntimeError(
                f"Isaac camera returned {frame.shape}, expected {(self._height, self._width, 3)}"
            )
        return np.ascontiguousarray(frame)
