"""PyBullet tabletop backend — state-only first cut (RFC-005 §5 through §8).

Parity with :class:`gauntlet.env.tabletop.TabletopEnv` at the
observation / action / perturbation-axis interface level (RFC-005 §7.2),
with three deliberate differences documented on the respective
surfaces:

* **Numerical non-parity** across backends (§7.4). Same seed + same
  policy → numerically different trajectories vs MuJoCo. Semantically
  similar, not numerically identical. ``gauntlet compare`` across
  backends measures simulator drift, not policy regression.
* **Quat wire-format** (§7.3). PyBullet returns xyzw everywhere;
  ``_build_obs`` converts once so ``cube_quat`` is always MuJoCo wxyz.
* **Visual-only axes** (§6 / §6.2). ``lighting_intensity`` and
  ``object_texture`` mutate the scene (texture swap applies
  immediately; lighting is stored on ``self._light_intensity`` for the
  follow-up rendering RFC) but do not change state-only observations.

Scene layout: per RFC §5.1, built from PyBullet primitives
(createMultiBody + loadTexture). Only ``plane.urdf`` from
``pybullet_data`` is loaded as a URDF — everything else is authored as
a box/cylinder shape so the repo ships no custom URDF.

Per-axis branches (RFC §13 item 10) are not wired here; the step-9
scope is the baseline env body. ``set_perturbation`` queues + validates,
``reset`` applies the queue through ``_apply_pending_perturbations``
which currently raises :class:`NotImplementedError` if anything is
queued. Step 10 lands the seven branches.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pybullet_data
from gymnasium import spaces
from numpy.typing import NDArray

import pybullet as p

__all__ = ["PyBulletTabletopEnv"]


# Scene constants — mirror the MuJoCo reference values exactly so
# baseline behaviour is semantically comparable (numerics still differ,
# see §7.4).
_TABLE_TOP_Z: float = 0.42
_CUBE_HALF: float = 0.025
_CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF
_TABLE_HALF_X: float = 0.5
_TABLE_HALF_Y: float = 0.5
_TABLE_HALF_Z: float = 0.02  # top-plate thickness / 2 (matches MuJoCo 0.04 box top).
_CUBE_INIT_HALFRANGE: float = 0.15
_TARGET_HALFRANGE: float = 0.2
_EE_REST_OFFSET_Z: float = 0.15  # reset places EE hovering this far above the cube.
_EE_VISUAL_RADIUS: float = 0.01

_FIXED_TIMESTEP: float = 1.0 / 240.0
_SOLVER_ITERATIONS: int = 10
_GRAVITY_Z: float = -9.81

_N_DISTRACTOR_SLOTS: int = 10
_DISTRACTOR_HALF: float = 0.02
_DISTRACTOR_BASELINE_XY = np.array(
    [
        # A small ring around the workspace — only revealed by distractor_count.
        (0.30, 0.30), (-0.30, 0.30), (0.30, -0.30), (-0.30, -0.30),
        (0.35, 0.00), (-0.35, 0.00), (0.00, 0.35), (0.00, -0.35),
        (0.25, 0.10), (-0.25, -0.10),
    ],
    dtype=np.float64,
)

_LARGE_CONSTRAINT_FORCE: float = 1000.0


def _xyzw_to_wxyz(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """PyBullet native (x, y, z, w) → MuJoCo (w, x, y, z) order. See §7.3."""
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=np.float64)


def _wxyz_to_xyzw(q: NDArray[np.float64]) -> list[float]:
    """MuJoCo (w, x, y, z) → PyBullet (x, y, z, w)."""
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


class PyBulletTabletopEnv:
    """PyBullet state-only tabletop pick-and-place env.

    Scene (RFC-005 §5.1): plane + table + cube + EE-mocap body +
    10 pre-loaded distractors (visibility-toggled) + visual target disc.

    Action / observation spaces are byte-compatible with
    :class:`gauntlet.env.tabletop.TabletopEnv` (§7.2); no kinematic arm
    or IK — the EE is a fixed constraint to a tiny kinematic body,
    driven each step by ``p.changeConstraint`` (§7.1).

    Attributes
    ----------
    MAX_LINEAR_STEP, MAX_ANGULAR_STEP, GRASP_RADIUS, TARGET_RADIUS,
    GRIPPER_OPEN, GRIPPER_CLOSED :
        Control / grasp / success constants (parity with
        :class:`~gauntlet.env.tabletop.TabletopEnv`).
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
        }
    )
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset(
        {"lighting_intensity", "object_texture"}
    )

    MAX_LINEAR_STEP: float = 0.05
    MAX_ANGULAR_STEP: float = 0.1
    GRASP_RADIUS: float = 0.05
    TARGET_RADIUS: float = 0.05
    GRIPPER_OPEN: float = 1.0
    GRIPPER_CLOSED: float = -1.0

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")

        self._max_steps = max_steps
        self._n_substeps = n_substeps

        # ---- spaces (identical to TabletopEnv) ----
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(
            {
                "cube_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "cube_quat": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float64
                ),
                "ee_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "gripper": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float64
                ),
                "target_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        # ---- per-session PyBullet client ----
        self._client: int = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )

        # ---- cached construction-time state that survives resetSimulation ----
        # Texture paths must be absolute so PyBullet can re-loadTexture after
        # every scene rebuild (restore_baseline does a full resetSimulation).
        assets_root = files("gauntlet.env.pybullet") / "assets"
        with (
            (assets_root / "cube_default.png").open("rb") as _fh_default,
            (assets_root / "cube_alt.png").open("rb") as _fh_alt,
        ):
            # Touch files so importlib resolves them from inside a wheel.
            _fh_default.read(1)
            _fh_alt.read(1)
        self._tex_default_path = str(assets_root / "cube_default.png")
        self._tex_alt_path = str(assets_root / "cube_alt.png")

        # ---- body IDs populated per-episode by _build_scene ----
        # (re-created after each resetSimulation — do not cache across that call)
        self._plane_id: int = -1
        self._table_id: int = -1
        self._cube_id: int = -1
        self._ee_body_id: int = -1
        self._ee_constraint_id: int = -1
        self._target_id: int = -1
        self._distractor_ids: list[int] = []
        self._tex_default_id: int = -1
        self._tex_alt_id: int = -1

        # ---- pending perturbations queue (applied by reset) ----
        self._pending_perturbations: dict[str, float] = {}

        # ---- per-episode runtime state ----
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
        self._grasped: bool = False
        self._gripper_state: float = self.GRIPPER_OPEN
        self._step_count: int = 0
        self._success: bool = False

        # Cosmetic-axis scratchpads — consumed by the rendering RFC, §6.1.
        self._light_intensity: float = 1.0

        self._build_scene()

    # ----------------------------------------------------------------- scene

    def _build_scene(self) -> None:
        """(Re)create every body in the scene.

        Called from :meth:`__init__` and from :meth:`restore_baseline`
        (after a ``resetSimulation``). Body IDs are re-fetched and
        stored; do NOT cache IDs across a ``restore_baseline`` call.
        """
        cid = self._client
        p.resetSimulation(physicsClientId=cid)
        p.setPhysicsEngineParameter(
            fixedTimeStep=_FIXED_TIMESTEP,
            numSolverIterations=_SOLVER_ITERATIONS,
            numSubSteps=0,
            deterministicOverlappingPairs=1,
            physicsClientId=cid,
        )
        p.setGravity(0.0, 0.0, _GRAVITY_Z, physicsClientId=cid)

        # Re-load textures after resetSimulation (resetSimulation invalidates
        # texture IDs the same way it invalidates body IDs).
        self._tex_default_id = p.loadTexture(self._tex_default_path, physicsClientId=cid)
        self._tex_alt_id = p.loadTexture(self._tex_alt_path, physicsClientId=cid)

        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)

        # ---- Table: kinematic box at _TABLE_TOP_Z ----
        table_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HALF_Z],
            physicsClientId=cid,
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HALF_Z],
            rgbaColor=[0.7, 0.55, 0.35, 1.0],
            physicsClientId=cid,
        )
        self._table_id = p.createMultiBody(
            baseMass=0.0,  # kinematic
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.0, 0.0, _TABLE_TOP_Z - _TABLE_HALF_Z],
            physicsClientId=cid,
        )

        # ---- Cube ----
        cube_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[_CUBE_HALF, _CUBE_HALF, _CUBE_HALF],
            physicsClientId=cid,
        )
        cube_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[_CUBE_HALF, _CUBE_HALF, _CUBE_HALF],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=cid,
        )
        self._cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=[0.0, 0.0, _CUBE_REST_Z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],  # xyzw identity
            physicsClientId=cid,
        )
        p.changeDynamics(
            self._cube_id, -1, lateralFriction=1.0, physicsClientId=cid
        )
        p.changeVisualShape(
            self._cube_id, -1,
            textureUniqueId=self._tex_default_id,
            physicsClientId=cid,
        )

        # ---- EE kinematic mocap-analogue + fixed constraint ----
        ee_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=_EE_VISUAL_RADIUS,
            rgbaColor=[0.2, 0.5, 1.0, 0.6],
            physicsClientId=cid,
        )
        self._ee_body_id = p.createMultiBody(
            baseMass=0.0,  # kinematic
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=ee_vis,
            basePosition=[0.0, 0.0, _CUBE_REST_Z + _EE_REST_OFFSET_Z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=cid,
        )
        self._ee_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._ee_body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[0.0, 0.0, _CUBE_REST_Z + _EE_REST_OFFSET_Z],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=cid,
        )

        # ---- Distractor pool (10 slots, hidden + ghosted at baseline) ----
        self._distractor_ids = []
        for xy in _DISTRACTOR_BASELINE_XY:
            d_col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[_DISTRACTOR_HALF, _DISTRACTOR_HALF, _DISTRACTOR_HALF],
                physicsClientId=cid,
            )
            d_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[_DISTRACTOR_HALF, _DISTRACTOR_HALF, _DISTRACTOR_HALF],
                rgbaColor=[0.6, 0.6, 0.9, 0.0],  # alpha 0 at baseline
                physicsClientId=cid,
            )
            body_id = p.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=d_col,
                baseVisualShapeIndex=d_vis,
                basePosition=[float(xy[0]), float(xy[1]), _TABLE_TOP_Z + _DISTRACTOR_HALF],
                physicsClientId=cid,
            )
            # Ghost at baseline — no collision participation.
            p.setCollisionFilterGroupMask(
                body_id, -1, collisionFilterGroup=0, collisionFilterMask=0,
                physicsClientId=cid,
            )
            self._distractor_ids.append(body_id)

        # ---- Target (visual-only cylinder) ----
        target_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.TARGET_RADIUS,
            length=0.002,
            rgbaColor=[0.15, 0.85, 0.2, 0.7],
            physicsClientId=cid,
        )
        self._target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_vis,
            basePosition=[0.0, 0.0, _TABLE_TOP_Z + 0.001],
            physicsClientId=cid,
        )

    # ----------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Deterministic reset — ``seed`` is the only entropy source.

        Order (RFC-005 §3.3): ``restore_baseline`` → seed-driven state
        randomisation → apply pending perturbations → clear queue.
        """
        del options
        # Full scene rebuild so no state leaks between episodes.
        self.restore_baseline()
        self._rng = np.random.default_rng(seed)

        # Cube XY randomisation.
        cube_xy = self._rng.uniform(
            low=-_CUBE_INIT_HALFRANGE,
            high=_CUBE_INIT_HALFRANGE,
            size=2,
        ).astype(np.float64)
        cube_pos = [float(cube_xy[0]), float(cube_xy[1]), _CUBE_REST_Z]
        p.resetBasePositionAndOrientation(
            self._cube_id,
            cube_pos,
            [0.0, 0.0, 0.0, 1.0],  # xyzw identity
            physicsClientId=self._client,
        )
        p.resetBaseVelocity(
            self._cube_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._client,
        )

        # Target XY randomisation (independent of cube).
        target_xy = self._rng.uniform(
            low=-_TARGET_HALFRANGE,
            high=_TARGET_HALFRANGE,
            size=2,
        ).astype(np.float64)
        self._target_pos = np.array(
            [target_xy[0], target_xy[1], _TABLE_TOP_Z], dtype=np.float64
        )
        p.resetBasePositionAndOrientation(
            self._target_id,
            [float(target_xy[0]), float(target_xy[1]), _TABLE_TOP_Z + 0.001],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )

        # EE baseline: hovering _EE_REST_OFFSET_Z above cube start.
        self._ee_pos = np.array(
            [cube_xy[0], cube_xy[1], _CUBE_REST_Z + _EE_REST_OFFSET_Z],
            dtype=np.float64,
        )
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
        p.resetBasePositionAndOrientation(
            self._ee_body_id,
            self._ee_pos.tolist(),
            _wxyz_to_xyzw(self._ee_quat),
            physicsClientId=self._client,
        )
        # Re-seat the constraint target to the new EE pose so step 1's
        # changeConstraint builds on a coherent starting point.
        p.changeConstraint(
            self._ee_constraint_id,
            jointChildPivot=self._ee_pos.tolist(),
            jointChildFrameOrientation=_wxyz_to_xyzw(self._ee_quat),
            maxForce=_LARGE_CONSTRAINT_FORCE,
            physicsClientId=self._client,
        )

        # Per-episode perturbations — step 10 wires the branches.
        if self._pending_perturbations:
            self._apply_pending_perturbations()
            self._pending_perturbations = {}

        # Episode flags.
        self._step_count = 0
        self._grasped = False
        self._gripper_state = self.GRIPPER_OPEN
        self._success = False

        return self._build_obs(), self._build_info()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[
        dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]
    ]:
        """Advance one control step (parity with MuJoCo TabletopEnv pipeline)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        self._apply_ee_command(a[0:3], a[3:6])
        self._gripper_state = (
            self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
        )
        self._update_grasp_state()

        # Physics substeps.
        for _ in range(self._n_substeps):
            p.stepSimulation(physicsClientId=self._client)

        if self._grasped:
            self._snap_cube_to_ee()

        self._step_count += 1
        cube_pos_tup, _ = p.getBasePositionAndOrientation(
            self._cube_id, physicsClientId=self._client
        )
        cube_pos = np.asarray(cube_pos_tup, dtype=np.float64)
        if self._xy_distance(cube_pos, self._target_pos) <= self.TARGET_RADIUS:
            self._success = True

        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps
        reward = -float(self._xy_distance(cube_pos, self._target_pos))
        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a named scalar on the env (applied by the next :meth:`reset`).

        Validation rules match :class:`gauntlet.env.tabletop.TabletopEnv`:
        unknown-axis names raise :class:`ValueError`; ``distractor_count``
        must be in ``[0, 10]``.
        """
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        if name == "distractor_count":
            count = round(float(value))
            if count < 0 or count > _N_DISTRACTOR_SLOTS:
                raise ValueError(
                    f"distractor_count must be in [0, {_N_DISTRACTOR_SLOTS}]; "
                    f"got {count}"
                )
        self._pending_perturbations[name] = float(value)

    def restore_baseline(self) -> None:
        """Return the env to its post-``__init__`` observable state.

        Implementation: full :func:`pybullet.resetSimulation` + scene
        rebuild. The expensive path, but correct and simple; the cost
        (~a handful of ms) is tolerable at our step budget (§12 Q4).
        Does NOT clear the pending-perturbation queue — that is the
        input to the next :meth:`reset`.
        """
        self._build_scene()
        # Cosmetic-axis scratchpads revert to baseline values.
        self._light_intensity = 1.0

    def close(self) -> None:
        """Release the PyBullet client. Idempotent."""
        import contextlib

        if self._client >= 0:
            # pybullet.error is raised on double-disconnect; suppress it.
            with contextlib.suppress(p.error):
                p.disconnect(physicsClientId=self._client)
            self._client = -1

    # --------------------------------------------------------------- helpers

    def _apply_ee_command(
        self,
        linear: NDArray[np.float64],
        angular: NDArray[np.float64],
    ) -> None:
        """Translate + rotate the EE by small deltas and update the constraint."""
        delta_lin = linear * self.MAX_LINEAR_STEP
        self._ee_pos = self._ee_pos + delta_lin

        axis_angle = angular * self.MAX_ANGULAR_STEP
        angle = float(np.linalg.norm(axis_angle))
        if angle > 0.0:
            axis = axis_angle / angle
            dq_wxyz = np.array(
                [
                    np.cos(angle / 2.0),
                    float(axis[0] * np.sin(angle / 2.0)),
                    float(axis[1] * np.sin(angle / 2.0)),
                    float(axis[2] * np.sin(angle / 2.0)),
                ],
                dtype=np.float64,
            )
            # Quaternion multiplication (wxyz * wxyz) — Hamilton product.
            self._ee_quat = _quat_mul_wxyz(dq_wxyz, self._ee_quat)
            n = float(np.linalg.norm(self._ee_quat))
            if n > 0.0:
                self._ee_quat = self._ee_quat / n

        p.changeConstraint(
            self._ee_constraint_id,
            jointChildPivot=self._ee_pos.tolist(),
            jointChildFrameOrientation=_wxyz_to_xyzw(self._ee_quat),
            maxForce=_LARGE_CONSTRAINT_FORCE,
            physicsClientId=self._client,
        )

    def _update_grasp_state(self) -> None:
        """Snap grasp flag based on gripper command + EE-cube proximity."""
        if self._gripper_state == self.GRIPPER_OPEN:
            self._grasped = False
            return
        cube_pos_tup, _ = p.getBasePositionAndOrientation(
            self._cube_id, physicsClientId=self._client
        )
        cube_pos = np.asarray(cube_pos_tup, dtype=np.float64)
        dist = float(np.linalg.norm(self._ee_pos - cube_pos))
        if dist <= self.GRASP_RADIUS:
            self._grasped = True
        # Already-grasped cubes stay grasped while gripper is closed;
        # _snap_cube_to_ee pulls them back each step.

    def _snap_cube_to_ee(self) -> None:
        """Override cube pose + velocity so it tracks the EE exactly."""
        p.resetBasePositionAndOrientation(
            self._cube_id,
            self._ee_pos.tolist(),
            _wxyz_to_xyzw(self._ee_quat),
            physicsClientId=self._client,
        )
        p.resetBaseVelocity(
            self._cube_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._client,
        )

    def _apply_pending_perturbations(self) -> None:
        """Dispatch each queued (name, value) to the per-axis branch.

        Branches land in step 10 (RFC-005 §13 item 10). Until then,
        any perturbation queued on a step-9 env triggers this hard
        error — better than silently running on a baseline scene.
        """
        raise NotImplementedError(
            "PyBullet backend perturbation branches land in RFC-005 §13 "
            "step 10. Construct TabletopEnv for MuJoCo-side perturbations, "
            "or hold this sweep until step 10 lands."
        )

    def _build_obs(self) -> dict[str, NDArray[np.float64]]:
        cube_pos_tup, cube_quat_xyzw = p.getBasePositionAndOrientation(
            self._cube_id, physicsClientId=self._client
        )
        cube_pos = np.asarray(cube_pos_tup, dtype=np.float64)
        cube_quat_wxyz = _xyzw_to_wxyz(cube_quat_xyzw)  # §7.3 conversion
        ee_pos = self._ee_pos.copy()
        gripper = np.array([self._gripper_state], dtype=np.float64)
        return {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat_wxyz,
            "ee_pos": ee_pos,
            "gripper": gripper,
            "target_pos": self._target_pos.copy(),
        }

    def _build_info(self) -> dict[str, Any]:
        return {
            "success": self._success,
            "grasped": self._grasped,
            "step": self._step_count,
        }

    @staticmethod
    def _xy_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))


def _quat_mul_wxyz(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` in wxyz order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
