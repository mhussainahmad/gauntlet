"""Isaac Sim tabletop backend — state-only first cut (RFC-009).

Parity with :class:`gauntlet.env.tabletop.TabletopEnv`,
:class:`gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv`,
and :class:`gauntlet.env.genesis.tabletop_genesis.GenesisTabletopEnv`
(state-only ``render_in_obs=False`` shape) at the observation /
action / perturbation-axis interface level (RFC-009 §3 + §6), with
the deliberate differences RFC-009 §7 documents:

* **Numerical non-parity across backends** (§7.3). Same seed + same
  policy -> numerically different trajectories on MuJoCo vs PyBullet
  vs Genesis vs Isaac Sim. Semantically similar, not numerically
  identical. ``gauntlet compare`` across backends measures simulator
  drift.
* **State-only first cut** (§6). Cosmetic axes
  (``lighting_intensity``, ``object_texture``, ``camera_offset_{x,y}``)
  queue + validate + store on shadow attributes and are members of
  :attr:`VISUAL_ONLY_AXES` pending a follow-up rendering RFC.
* **GPU-runtime requirement.** ``isaacsim`` wraps NVIDIA Omniverse
  Kit; constructing this env on a CPU-only machine raises a Kit
  bootstrap error from inside :class:`isaacsim.SimulationApp`.
  RFC-009 §4.4 / §8 documents the test strategy: CI uses a fake
  ``isaacsim`` namespace injected into ``sys.modules`` so the
  contract is exercised without ever launching real Kit.

Scene layout (RFC-009 §5): plane + fixed table-top + dynamic cube +
kinematic-by-teleport EE + visual target marker + 10 pre-allocated
teleport-away distractors. No URDF / USD robot asset — the EE is a
small ``VisualCuboid`` whose pose is overwritten via
``set_world_pose`` each control step. Same shape Genesis (RFC-007 §5)
and PyBullet (RFC-005 §7.1) chose.

Step 5 lands the body of ``reset`` / ``step`` / ``_build_obs`` (full
state-only rollout loop); step 6 lands the seven perturbation
branches inside ``set_perturbation`` / ``_apply_pending_perturbations``
+ the ``restore_baseline`` body.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import isaacsim  # bootstrap Kit (lazy via SimulationApp on construct)
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["IsaacSimTabletopEnv"]


# Scene / control constants mirror the other backends exactly so a
# given (seed, axis_config) produces semantically comparable rollouts
# across simulators even if the floating-point trajectories diverge
# (RFC-009 §7.3).
_TABLE_TOP_Z: float = 0.42
_CUBE_HALF: float = 0.025
_CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF

_TABLE_HALF_X: float = 0.5
_TABLE_HALF_Y: float = 0.5
_TABLE_HALF_Z: float = 0.02

_CUBE_INIT_HALFRANGE: float = 0.15
_TARGET_HALFRANGE: float = 0.2
_EE_REST_OFFSET_Z: float = 0.15
_EE_VISUAL_HALF: float = 0.01

_N_DISTRACTOR_SLOTS: int = 10
_DISTRACTOR_HALF: float = 0.02

# Positions match the Genesis and PyBullet adapters so the
# "three-distractor ring" that ``distractor_count`` = 3 produces is
# at identical (x, y) across backends. Disabled distractors are
# teleported to ``z = -10.0`` (below the ground plane and out of every
# physically-plausible EE reach) — same teleport-away semantic the
# Genesis adapter uses.
_DISTRACTOR_BASELINE_XY: NDArray[np.float64] = np.array(
    [
        (0.30, 0.30),
        (-0.30, 0.30),
        (0.30, -0.30),
        (-0.30, -0.30),
        (0.35, 0.00),
        (-0.35, 0.00),
        (0.00, 0.35),
        (0.00, -0.35),
        (0.25, 0.10),
        (-0.25, -0.10),
    ],
    dtype=np.float64,
)
_DISTRACTOR_REST_Z: float = _TABLE_TOP_Z + _DISTRACTOR_HALF
_DISTRACTOR_HIDDEN_Z: float = -10.0  # below the plane, out of reach


def _axis_angle_to_quat(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues-style axis-angle -> wxyz quat. Zero angle returns identity.

    Matches the Genesis adapter's helper byte-for-byte. Used by
    :meth:`IsaacSimTabletopEnv._apply_ee_command` to compose a small
    per-step rotation increment into the EE's current orientation.
    """
    angle = float(np.linalg.norm(axis_angle))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis_angle / angle
    s = float(np.sin(angle * 0.5))
    c = float(np.cos(angle * 0.5))
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` in wxyz order. Matches Genesis."""
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


def _normalize_quat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


class IsaacSimTabletopEnv:
    """Isaac Sim state-only tabletop pick-and-place env.

    Scene (RFC-009 §5.1): ground plane + fixed table + dynamic cube +
    kinematic EE + visual target marker + 10 pre-allocated
    teleport-away distractors.

    Action / observation spaces are shape-compatible with
    :class:`gauntlet.env.tabletop.TabletopEnv` (state-only); no
    kinematic arm, no IK — the EE is a kinematic prim teleported via
    ``set_world_pose`` each control step.

    Attributes
    ----------
    MAX_LINEAR_STEP, MAX_ANGULAR_STEP, GRASP_RADIUS, TARGET_RADIUS,
    GRIPPER_OPEN, GRIPPER_CLOSED :
        Control / grasp / success constants (parity with the other
        backends).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

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
    # State-only first cut (RFC-009 §6.6). The four cosmetic axes
    # store on shadow attributes during ``set_perturbation`` /
    # ``_apply_pending_perturbations`` and produce no obs-dict delta
    # until a follow-up rendering RFC lands. The Suite loader's
    # ``_reject_purely_visual_suites`` guard fires off this set, so a
    # YAML naming only cosmetic axes on ``tabletop-isaac`` is
    # rejected at load time. Same shape Genesis had pre-RFC-008 and
    # PyBullet had pre-RFC-006.
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
        }
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
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")

        self._max_steps = max_steps

        # ---- spaces (5-key state obs, identical to Genesis state-only) ----
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, gym.spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # ---- Kit bootstrap + scene construction ----
        # Step 5 lands the body. The scaffold instantiates SimulationApp
        # (headless), pulls World, and registers the prim handles so
        # subsequent reset/step calls find them. Done in __init__ so the
        # ~5-s Kit boot is paid once per worker process (matches the
        # Genesis ``gs.init`` amortisation pattern, RFC-009 §4.6).
        self._kit_app: Any = isaacsim.SimulationApp({"headless": True})

        # Imported here, not at module top, because importing them
        # before SimulationApp is constructed raises inside Kit. The
        # adapter pattern matches Isaac Sim's official examples.
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import (
            DynamicCuboid,
            FixedCuboid,
            VisualCuboid,
        )

        self._world: Any = World()

        # ---- Scene primitives ----
        # Table-top: kinematic FixedCuboid sized to match the other
        # backends.
        self._table: Any = self._world.scene.add(
            FixedCuboid(
                prim_path="/World/table",
                position=np.array([0.0, 0.0, _TABLE_TOP_Z - _TABLE_HALF_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _TABLE_HALF_X, 2.0 * _TABLE_HALF_Y, 2.0 * _TABLE_HALF_Z],
                    dtype=np.float64,
                ),
            )
        )
        # The cube — DynamicCuboid (rigid body, mass-driven).
        self._cube: Any = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/cube",
                position=np.array([0.0, 0.0, _CUBE_REST_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF], dtype=np.float64
                ),
            )
        )
        # Visual target marker — VisualCuboid (no collision, no mass).
        self._target: Any = self._world.scene.add(
            VisualCuboid(
                prim_path="/World/target",
                position=np.array([0.0, 0.0, _TABLE_TOP_Z + 0.001], dtype=np.float64),
                size=np.array(
                    [2.0 * self.TARGET_RADIUS, 2.0 * self.TARGET_RADIUS, 0.002],
                    dtype=np.float64,
                ),
            )
        )
        # EE — VisualCuboid (kinematic, teleported each step).
        self._ee: Any = self._world.scene.add(
            VisualCuboid(
                prim_path="/World/ee",
                position=np.array([0.0, 0.0, _CUBE_REST_Z + _EE_REST_OFFSET_Z], dtype=np.float64),
                size=np.array(
                    [2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF],
                    dtype=np.float64,
                ),
            )
        )
        # Distractors — pre-allocate all 10 at hidden Z; the
        # ``distractor_count`` axis teleports the first ``count`` to
        # rest_z in ``_apply_pending_perturbations``.
        self._distractors: list[Any] = []
        for i in range(_N_DISTRACTOR_SLOTS):
            xy = _DISTRACTOR_BASELINE_XY[i]
            d = self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/distractor_{i}",
                    position=np.array(
                        [float(xy[0]), float(xy[1]), _DISTRACTOR_HIDDEN_Z], dtype=np.float64
                    ),
                    size=np.array(
                        [2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF],
                        dtype=np.float64,
                    ),
                )
            )
            self._distractors.append(d)

        # ---- pending perturbations queue (drained by reset) ----
        self._pending_perturbations: dict[str, float] = {}

        # ---- per-episode runtime state ----
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._step_count: int = 0
        self._grasped: bool = False
        self._gripper_state: float = self.GRIPPER_OPEN
        self._success: bool = False
        self._target_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

        # Visual-axis shadows — set_perturbation stores here even on the
        # state-only first cut so the follow-up rendering RFC can read
        # them without a second pass over the adapter (matches the same
        # shadow-pattern Genesis pre-RFC-008 used).
        self._light_intensity: float = 1.0
        self._cam_offset: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._texture_choice: int = 0

        self._closed: bool = False

    # --------------------------------------------------------------- helpers

    def _prim_pos(self, prim: Any) -> NDArray[np.float64]:
        """Read a prim's world position via ``get_world_pose``.

        Isaac Sim 5.x returns ``(position, orientation)`` from
        ``get_world_pose``; we discard the orientation and return a
        copy so downstream mutation of the obs dict cannot leak into
        Kit's internal state.
        """
        pos, _ = prim.get_world_pose()
        return np.asarray(pos, dtype=np.float64).reshape(3)

    def _prim_quat(self, prim: Any) -> NDArray[np.float64]:
        """Read a prim's world orientation (wxyz). RFC-009 §7.5."""
        _, quat = prim.get_world_pose()
        return np.asarray(quat, dtype=np.float64).reshape(4)

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        return {
            "cube_pos": self._prim_pos(self._cube),
            "cube_quat": self._prim_quat(self._cube),
            "ee_pos": self._prim_pos(self._ee),
            "gripper": np.array([self._gripper_state], dtype=np.float64),
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

    def _apply_ee_command(
        self,
        linear: NDArray[np.float64],
        angular: NDArray[np.float64],
    ) -> None:
        """Translate + rotate the kinematic EE by small per-step deltas.

        Same control surface the Genesis adapter uses (see
        ``GenesisTabletopEnv._apply_ee_command``): position is updated
        by ``MAX_LINEAR_STEP`` * normalised input; orientation
        accumulates a small rotation increment via
        :func:`_axis_angle_to_quat` + Hamilton product.
        """
        cur_pos = self._prim_pos(self._ee)
        new_pos = cur_pos + linear * self.MAX_LINEAR_STEP

        axis_angle = angular * self.MAX_ANGULAR_STEP
        new_quat = self._prim_quat(self._ee)
        if float(np.linalg.norm(axis_angle)) > 0.0:
            dq = _axis_angle_to_quat(axis_angle)
            new_quat = _normalize_quat(_quat_mul(dq, new_quat))

        self._ee.set_world_pose(
            position=new_pos.astype(np.float64, copy=False),
            orientation=new_quat.astype(np.float64, copy=False),
        )

    def _update_grasp_state(self) -> None:
        """Snap grasp flag based on gripper command + EE-cube proximity."""
        if self._gripper_state == self.GRIPPER_OPEN:
            self._grasped = False
            return
        ee = self._prim_pos(self._ee)
        cube = self._prim_pos(self._cube)
        dist = float(np.linalg.norm(ee - cube))
        if dist <= self.GRASP_RADIUS:
            self._grasped = True

    def _snap_cube_to_ee(self) -> None:
        """Overwrite cube pose with the EE pose post-physics (grasp sim)."""
        ee_pos = self._prim_pos(self._ee)
        ee_quat = self._prim_quat(self._ee)
        self._cube.set_world_pose(position=ee_pos, orientation=ee_quat)

    # ----------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Deterministic reset.

        ``seed`` is the only entropy source. Ordering per RFC-005 §3.2
        (the Protocol's contract): ``restore_baseline`` -> seed-driven
        randomisation -> apply queued perturbations -> clear queue.
        """
        del options
        self._rng = np.random.default_rng(seed)

        self.restore_baseline()

        # Re-seed cube XY (identity quat).
        cube_xy = self._rng.uniform(
            low=-_CUBE_INIT_HALFRANGE, high=_CUBE_INIT_HALFRANGE, size=2
        ).astype(np.float64)
        self._cube.set_world_pose(
            position=np.array([cube_xy[0], cube_xy[1], _CUBE_REST_Z], dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

        # Re-seed target XY (independent of cube).
        target_xy = self._rng.uniform(
            low=-_TARGET_HALFRANGE, high=_TARGET_HALFRANGE, size=2
        ).astype(np.float64)
        self._target_pos = np.array([target_xy[0], target_xy[1], _TABLE_TOP_Z], dtype=np.float64)
        self._target.set_world_pose(
            position=np.array([target_xy[0], target_xy[1], _TABLE_TOP_Z + 0.001], dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

        # Reset EE to hover above cube start.
        self._ee.set_world_pose(
            position=np.array(
                [cube_xy[0], cube_xy[1], _CUBE_REST_Z + _EE_REST_OFFSET_Z], dtype=np.float64
            ),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

        # World reset — Kit-side invalidation of physics velocities so
        # the next step is a clean slate.
        self._world.reset()

        # Apply queued perturbations on top of seed-driven state. Step
        # 6 lands the seven branches; until then this is a no-op
        # unless a future commit wires set_perturbation. RFC-005 §3.2.
        if self._pending_perturbations:
            self._apply_pending_perturbations()
            self._pending_perturbations = {}

        self._step_count = 0
        self._grasped = False
        self._gripper_state = self.GRIPPER_OPEN
        self._success = False

        return self._build_obs(), self._build_info()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Pipeline (matches Genesis): clip -> update EE pose ->
        update grasp state -> ``world.step`` -> if grasped, snap cube
        to EE -> build obs / reward / flags.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        self._apply_ee_command(a[0:3], a[3:6])

        self._gripper_state = self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
        self._update_grasp_state()

        # ``render=False`` keeps the headless step path fast — no
        # rasterisation cost on the state-only first cut.
        self._world.step(render=False)

        if self._grasped:
            self._snap_cube_to_ee()

        self._step_count += 1
        cube_pos = self._prim_pos(self._cube)
        if self._xy_distance(cube_pos, self._target_pos) <= self.TARGET_RADIUS:
            self._success = True

        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps
        reward = -float(self._xy_distance(cube_pos, self._target_pos))
        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a named scalar on the env (applied by the next :meth:`reset`).

        Validation rules match the other backends:

        * Unknown axis -> :class:`ValueError` with the canonical
          message naming the axis.
        * ``distractor_count`` must round to an integer in
          ``[0, _N_DISTRACTOR_SLOTS]``.

        Cosmetic axes (``lighting_intensity``, ``camera_offset_x``,
        ``camera_offset_y``, ``object_texture``) are accepted but
        only mutate the shadow attributes during the apply phase
        (state-only first cut, RFC-009 §6.6).
        """
        if name not in type(self).AXIS_NAMES:
            raise ValueError(
                f"unknown perturbation axis {name!r}; known axes: {sorted(type(self).AXIS_NAMES)}"
            )
        if name == "distractor_count":
            count = round(float(value))
            if count < 0 or count > _N_DISTRACTOR_SLOTS:
                raise ValueError(
                    f"distractor_count must be in [0, {_N_DISTRACTOR_SLOTS}]; got {count}"
                )
        self._pending_perturbations[name] = float(value)

    def restore_baseline(self) -> None:
        """Restore the env to its post-``__init__`` observable state.

        Hide every distractor at ``_DISTRACTOR_HIDDEN_Z`` (the
        ``distractor_count`` branch then re-reveals the first N during
        ``_apply_pending_perturbations``) and reset the cosmetic
        shadow attributes.  Idempotent — a second call is a no-op
        observational delta against the first.
        """
        for i, d in enumerate(self._distractors):
            xy = _DISTRACTOR_BASELINE_XY[i]
            d.set_world_pose(
                position=np.array(
                    [float(xy[0]), float(xy[1]), _DISTRACTOR_HIDDEN_Z], dtype=np.float64
                ),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        # Visual-axis shadows revert to neutral so a prior episode's
        # cosmetic perturbation doesn't leak.
        self._light_intensity = 1.0
        self._cam_offset = np.zeros(2, dtype=np.float64)
        self._texture_choice = 0

    def _apply_pending_perturbations(self) -> None:
        """Apply ``self._pending_perturbations`` to the scene.

        Walks each queued axis through :meth:`_apply_one_perturbation`.
        Pose-overriding axes (``object_initial_pose_{x,y}``) must
        run after the random cube XY write in :meth:`reset`, which
        is the order :meth:`reset` already guarantees (random write
        first, then call this method).
        """
        for name, value in self._pending_perturbations.items():
            self._apply_one_perturbation(name, value)

    def _apply_one_perturbation(self, name: str, value: float) -> None:
        """Per-axis branch dispatcher (RFC-009 §6 table)."""
        if name == "lighting_intensity":
            # Cosmetic on the state-only first cut (§6.1). Stored on
            # the shadow attribute for the follow-up rendering RFC.
            self._light_intensity = float(value)
            return

        if name == "camera_offset_x":
            # Cosmetic — camera pose is a rendering-time concept (§6.2).
            self._cam_offset = self._cam_offset.copy()
            self._cam_offset[0] = float(value)
            return

        if name == "camera_offset_y":
            self._cam_offset = self._cam_offset.copy()
            self._cam_offset[1] = float(value)
            return

        if name == "object_texture":
            # Cosmetic — material binding is a rendering-time concept
            # (§6.3). Stored on the shadow attribute.
            self._texture_choice = 1 if round(float(value)) != 0 else 0
            return

        if name == "object_initial_pose_x":
            # State-affecting (§6.4). Override the random cube X write
            # from reset; preserve the random Y already in place.
            cur_pos = self._prim_pos(self._cube)
            new_pos = np.array([float(value), float(cur_pos[1]), _CUBE_REST_Z], dtype=np.float64)
            self._cube.set_world_pose(
                position=new_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            return

        if name == "object_initial_pose_y":
            cur_pos = self._prim_pos(self._cube)
            new_pos = np.array([float(cur_pos[0]), float(value), _CUBE_REST_Z], dtype=np.float64)
            self._cube.set_world_pose(
                position=new_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
            return

        if name == "distractor_count":
            # State-affecting (§6.5). Teleport-away semantic — first
            # ``count`` distractors come up to rest_z; the remaining
            # ones stay at hidden_z (restore_baseline already put them
            # all there). Documented deviation from MuJoCo's
            # visibility+collision toggle.
            count = round(float(value))
            for i, d in enumerate(self._distractors):
                xy = _DISTRACTOR_BASELINE_XY[i]
                z = _DISTRACTOR_REST_Z if i < count else _DISTRACTOR_HIDDEN_Z
                d.set_world_pose(
                    position=np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                )
            return

        # Unknown axis names are blocked at set_perturbation; the
        # else-fallthrough is unreachable by contract.
        raise ValueError(f"internal: unknown perturbation axis {name!r}")

    def close(self) -> None:
        """Release Kit + simulation resources. Idempotent.

        Best-effort drop. ``SimulationApp`` exposes ``.close()`` from
        Isaac Sim 4.x onwards; we guard against double-close with a
        flag because ``SimulationApp`` is a process-global singleton
        and a second close on the same handle can wedge Kit.
        """
        import contextlib

        if self._closed:
            return
        # Kit may raise opaque C errors during shutdown — best-effort
        # drop matches the PyBullet adapter's contextlib.suppress
        # pattern.
        with contextlib.suppress(Exception):
            self._kit_app.close()
        self._closed = True
