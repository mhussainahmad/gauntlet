"""Planar push variant of the tabletop env (``env: tabletop-push``).

Same scene, action space, observation space and perturbation axes as
:class:`~gauntlet.env.tabletop.TabletopEnv`; only the way the cube can
move differs:

* **No grasp.** The gripper command is accepted (the action stays 7-D)
  but never attaches the cube. The cube moves only through contact.
* **Colliding end-effector.** The EE sphere, a non-colliding marker in
  pick-and-place, collides here, so the policy pushes the cube by
  driving into it.
* **Settled success.** Success needs the cube's XY within
  :attr:`TARGET_RADIUS` of the target *and* its planar speed below
  :attr:`SETTLE_SPEED`. A cube that slides through the target and out
  the other side is a failure (overshoot), not a success.

Push brings failure modes pick-and-place does not have: overshoot, the
cube rotating off the pushing face and slipping sideways, and
approaching from the wrong side. Contact between the end-effector and
the cube (or a swapped-in object on the cube body) is the task, so the
collision / near-collision / peak-force telemetry skips it; contact with
a distractor still counts.
"""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from gauntlet.env.base import CameraSpec
from gauntlet.env.tabletop import TabletopEnv

__all__ = ["TabletopPushEnv"]


class TabletopPushEnv(TabletopEnv):
    """Push the cube into the target zone without grasping it."""

    SETTLE_SPEED: float = 0.02
    """Max planar cube speed (m/s) for an in-target cube to count as placed."""

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
        render_in_obs: bool = False,
        render_size: tuple[int, int] = (224, 224),
        cameras: list[CameraSpec] | None = None,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            n_substeps=n_substeps,
            render_in_obs=render_in_obs,
            render_size=render_size,
            cameras=cameras,
        )
        m = self._model
        ee_geom = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom"))
        if ee_geom < 0:
            raise RuntimeError("expected geom 'ee_geom' missing from MJCF")
        self._set_geom_collision(ee_geom, 1, 1)
        cube_body = int(m.geom_bodyid[self._cube_geom_id])
        self._expected_contact_pairs = frozenset(
            (min(ee_geom, g), max(ee_geom, g))
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == cube_body
        )

    def _update_grasp_state(self) -> None:
        self._grasped = False

    def _is_success(self, cube_pos: NDArray[np.float64]) -> bool:
        if self._xy_distance(cube_pos, self._target_pos) > self.TARGET_RADIUS:
            return False
        vadr = self._cube_qvel_adr
        planar_speed = float(np.linalg.norm(self._data.qvel[vadr : vadr + 2]))
        return planar_speed < self.SETTLE_SPEED
