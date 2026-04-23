"""Genesis tabletop backend — state + image obs (RFC-007 + RFC-008).

Parity with :class:`gauntlet.env.tabletop.TabletopEnv` and
:class:`gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv` at
the observation / action / perturbation-axis interface level
(RFC-007 §3 + §6, RFC-008 §3), with the deliberate differences
RFC-007 §7 documents:

* **Numerical non-parity across backends** (§7.3). Same seed + same
  policy -> numerically different trajectories on MuJoCo vs PyBullet
  vs Genesis. Semantically similar, not numerically identical.
  ``gauntlet compare`` across backends measures simulator drift.
* **Same-process bit-determinism** (§7.1). Two ``env.reset(seed=s)``
  + 20 ``env.step`` calls in the same process produce bit-identical
  obs. Verified empirically in the exploration pass.
* **Post-RFC-008 cosmetic-axis parity** — ``lighting_intensity``,
  ``object_texture``, ``camera_offset_{x,y}`` now mutate
  ``obs["image"]`` when ``render_in_obs=True``. :attr:`VISUAL_ONLY_AXES`
  is empty; the Suite loader's cosmetic-only rejection is a no-op on
  this backend.

Scene layout (RFC-007 §5 + RFC-008 §3.5): everything is built from
Genesis primitives (``gs.morphs.Box``, ``gs.morphs.Plane``,
``gs.morphs.Cylinder``) - no URDF, no MJCF. Two cubes pre-allocated
at build time (``_cube_red`` / ``_cube_green``) so ``object_texture``
can teleport-swap without a scene rebuild; ``self._cube`` always
aliases the active one. Keeps ``scene.build()`` at the ~5 s minimum
and ships no Genesis-specific asset in the repo.

Kinematic-EE pattern (RFC-007 §5): no ``createConstraint`` analogue
in Genesis, so the EE body is a gravity-compensated dynamic rigid
whose pose is overwritten via ``entity.set_pos()`` + ``entity.set_quat()``
every control step. Same-behaviour as PyBullet's ``p.changeConstraint``
loop.

Rendering path (RFC-008): optional pinhole camera added at
``__init__`` when ``render_in_obs=True``. ``_render_obs_image`` flushes
pending rigid transforms via the private
``scene.visualizer.rasterizer._context.update_rigid()`` hop — without
it, post-``reset`` renders would see stale poses because no
``scene.step()`` has run since the teleport (exploration §2.4). The
``lighting_intensity`` axis reaches the default directional light via
``_context._scene.directional_light_nodes[0].light.intensity``; pinned
by ``genesis-world<0.5``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

import genesis as gs

__all__ = ["GenesisTabletopEnv"]


# Scene / control constants mirror MuJoCo's TabletopEnv and PyBullet's
# backend exactly so a given (seed, axis_config) produces semantically
# comparable rollouts across all three backends even if the
# floating-point trajectories diverge (RFC-007 §7.3).
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

# Positions match :mod:`gauntlet.env.pybullet.tabletop_pybullet` so the
# "three-distractor ring" that :func:`distractor_count` = 3 produces is
# at identical (x, y) across backends. The only difference is the
# teleport-away semantics on Genesis: disabled distractors live at
# ``z = _DISTRACTOR_HIDDEN_Z`` (below the ground plane, out of every
# camera frustum and out of every physically-plausible EE reach).
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


# Camera pose — semantically matches MuJoCo's main camera
# (``assets/tabletop.xml``: ``pos="0.6 -0.6 0.8"``, target = table-top
# centre ``(0, 0, 0.42)``) and the PyBullet analogue RFC-006 §3.4 derived
# from the same source. These are the exact values landed on
# ``PyBulletTabletopEnv`` in RFC-006. Cross-backend numerical pixel
# parity is explicitly NOT a goal (RFC-007 §7.3) — semantic parity
# (same pose, same layout, same light direction) only.
_CAM_EYE_BASELINE: tuple[float, float, float] = (0.6, -0.6, 0.8)
_CAM_TARGET: tuple[float, float, float] = (0.0, 0.0, 0.42)
_CAM_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)
_CAM_FOV: float = 45.0
_CAM_NEAR: float = 0.01
_CAM_FAR: float = 5.0

# Genesis's default directional light (populated by
# ``VisOptions.lights`` on scene construction) ships at
# ``intensity=5.0`` on the 0.4.x line. The ``lighting_intensity`` axis
# is a scalar multiplier on this baseline — axis value 1.0 leaves the
# scene unchanged. Semantic match to MuJoCo's light stack and to
# PyBullet's ``lightDiffuseCoeff`` (RFC-006 §3.3).
_BASELINE_LIGHT_INTENSITY: float = 5.0

_DEFAULT_RENDER_SIZE: tuple[int, int] = (224, 224)


def _axis_angle_to_quat(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues-style axis-angle -> wxyz quat. Zero angle returns identity.

    Used by :meth:`GenesisTabletopEnv._apply_ee_command` to build a
    small per-step rotation increment to compose into the EE's current
    orientation.
    """
    angle = float(np.linalg.norm(axis_angle))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis_angle / angle
    s = float(np.sin(angle * 0.5))
    c = float(np.cos(angle * 0.5))
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton quaternion product for wxyz-ordered inputs."""
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


def _ensure_genesis_initialised() -> None:
    """Initialise Genesis's module-level backend state once per process.

    ``gs.init`` is idempotent within a process (verified in the
    exploration pass) — repeated calls after the first are no-ops. We
    guard against double-init explicitly so test fixtures that spin
    multiple envs up and down in one session pay the ~4-s init cost
    only once (RFC-007 §4.6).
    """
    if getattr(gs, "_initialized", False):
        return
    gs.init(backend=gs.cpu, logging_level="warning")


class GenesisTabletopEnv:
    """Genesis state-only tabletop pick-and-place env.

    Scene (RFC-007 §5.1): plane + fixed table + dynamic cube +
    gravity-compensated kinematic EE + visual target cylinder +
    10 pre-allocated teleport-away distractors.

    Action / observation spaces are shape-compatible with
    :class:`gauntlet.env.tabletop.TabletopEnv` (§3); no kinematic arm,
    no IK — the EE is a gravity-compensated rigid whose pose is
    teleported via ``set_pos`` + ``set_quat`` each control step.

    Attributes
    ----------
    MAX_LINEAR_STEP, MAX_ANGULAR_STEP, GRASP_RADIUS, TARGET_RADIUS,
    GRIPPER_OPEN, GRIPPER_CLOSED :
        Control / grasp / success constants (parity with
        :class:`~gauntlet.env.tabletop.TabletopEnv` and
        :class:`~gauntlet.env.pybullet.tabletop_pybullet.PyBulletTabletopEnv`).
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
    # Post-RFC-008: all four cosmetic axes are observable via
    # ``obs["image"]`` when ``render_in_obs=True``, so the honesty
    # carve-out from RFC-007 §6 closes and the Suite loader's
    # ``_reject_purely_visual_suites`` becomes a no-op on
    # ``tabletop-genesis``. Exact parity with ``TabletopEnv`` and
    # (post-RFC-006) ``PyBulletTabletopEnv``.
    #
    # Honesty caveat: a user running a cosmetic-only sweep with
    # ``render_in_obs=False`` still gets pairwise-identical state
    # observations across cells — not a bug, same property MuJoCo and
    # PyBullet share. The axes still store on their shadow attributes
    # and reach the render path only when rendering is on.
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()

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
        render_in_obs: bool = False,
        render_size: tuple[int, int] = _DEFAULT_RENDER_SIZE,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")
        if render_in_obs:
            h, w = render_size
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"render_size must be a (height, width) of positive ints; got {render_size}"
                )

        self._max_steps = max_steps
        self._n_substeps = n_substeps
        self._render_in_obs = bool(render_in_obs)
        self._render_size: tuple[int, int] = (int(render_size[0]), int(render_size[1]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, gym.spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
        if self._render_in_obs:
            h, w = self._render_size
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        # ---- per-process Genesis global init (idempotent) ----
        _ensure_genesis_initialised()

        # ---- scene + entities ----
        # ``SimOptions(dt=0.01, substeps=5)`` matches the MuJoCo reference
        # 50 Hz outer loop (200 steps * 5 substeps * 10 ms = 10 s episode).
        # ``show_viewer=False`` keeps construction headless — required on
        # CI runners without a display.
        self._scene: Any = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=self._n_substeps),
        )
        self._scene.add_entity(gs.morphs.Plane())
        # Fixed table — won't move, acts as the rigid support surface for
        # the cube.
        self._scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, _TABLE_TOP_Z - _TABLE_HALF_Z),
                size=(2.0 * _TABLE_HALF_X, 2.0 * _TABLE_HALF_Y, 2.0 * _TABLE_HALF_Z),
                fixed=True,
            )
        )
        # Target disc — visual-only, collision off. Position moves per
        # reset; fixed=False so ``set_pos`` is accepted on the instance.
        # gravity_compensation via material ensures it doesn't fall when
        # the pose is set mid-episode.
        self._target: Any = self._scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.0, 0.0, _TABLE_TOP_Z + 0.001),
                radius=self.TARGET_RADIUS,
                height=0.002,
                collision=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        # Two cubes — red (default texture) and green (alt texture) —
        # pre-allocated at build time because Genesis 0.4.x cannot swap
        # surface colours post-build (exploration §5). The active cube
        # is teleported to the rest pose; the inactive one lives at
        # ``_DISTRACTOR_HIDDEN_Z`` — same teleport-away pattern
        # ``distractor_count`` uses (RFC-007 §6.5).
        #
        # BOTH cubes carry ``gravity_compensation=1.0`` so the inactive
        # cube stays parked at ``z=-10`` across an episode. This is a
        # deliberate deviation from the single-cube state-only
        # predecessor (which let the cube fall and rest on the table)
        # — the audit in RFC-008 §3.5 confirms no existing test
        # exercises free-fall dynamics; success is XY-only and
        # ``_snap_cube_to_ee`` overrides cube pose during grasp anyway.
        # Documented here so a future reader doesn't "simplify" it away.
        self._cube_red: Any = self._scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, _CUBE_REST_Z),
                size=(2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF),
            ),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.surfaces.ColorTexture(color=(1.0, 0.2, 0.2)),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self._cube_green: Any = self._scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, _DISTRACTOR_HIDDEN_Z),
                size=(2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF, 2.0 * _CUBE_HALF),
            ),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.surfaces.ColorTexture(color=(0.2, 1.0, 0.2)),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        # ``self._cube`` is the authoritative handle for every downstream
        # operation (``_cube_pos``, ``_cube_quat``, ``_snap_cube_to_ee``,
        # reset's XY teleport). ``self._cube_alt`` aliases the hidden
        # cube. Both pointers are swapped atomically in
        # ``_apply_one_perturbation("object_texture", ...)``.
        self._cube: Any = self._cube_red
        self._cube_alt: Any = self._cube_green
        # EE body — gravity-compensated dynamic rigid, driven by
        # ``set_pos`` / ``set_quat`` each step. Collision kept on so the
        # EE-cube proximity check in ``_update_grasp_state`` sees a
        # physical touch if it happens (tested: still works).
        self._ee: Any = self._scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, _CUBE_REST_Z + _EE_REST_OFFSET_Z),
                size=(2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF, 2.0 * _EE_VISUAL_HALF),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        # Pre-allocate all 10 distractors at their rest positions; the
        # ``distractor_count`` axis teleports the tail (count..10) down
        # to ``_DISTRACTOR_HIDDEN_Z``.
        self._distractors: list[Any] = []
        for i in range(_N_DISTRACTOR_SLOTS):
            xy = _DISTRACTOR_BASELINE_XY[i]
            d = self._scene.add_entity(
                gs.morphs.Box(
                    pos=(float(xy[0]), float(xy[1]), _DISTRACTOR_HIDDEN_Z),
                    size=(2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF, 2.0 * _DISTRACTOR_HALF),
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
            self._distractors.append(d)

        # Optional render camera — only added when ``render_in_obs=True``
        # so the state-only default path has zero extra scene-graph
        # work. Genesis's ``add_camera(res=(W, H))`` takes width-first;
        # our ``render_size`` is (height, width) to match MuJoCo, so we
        # swap here. RFC-008 §3.6.
        self._camera: Any | None
        if self._render_in_obs:
            h, w = self._render_size
            self._camera = self._scene.add_camera(
                res=(w, h),
                pos=_CAM_EYE_BASELINE,
                lookat=_CAM_TARGET,
                up=_CAM_UP,
                fov=_CAM_FOV,
                near=_CAM_NEAR,
                far=_CAM_FAR,
                GUI=False,
            )
        else:
            self._camera = None

        # Build fuses the scene and compiles kernels — first call is
        # ~4-5 s on CPU; subsequent builds in the same process are <1 s
        # (RFC-007 §Q4). Do not call per-episode.
        self._scene.build()

        # ---- pending perturbations queue (applied by reset) ----
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
        # them without a second pass over the adapter.
        self._light_intensity: float = 1.0
        self._cam_offset: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._texture_choice: int = 0

    # ---------------------------------------------------------------- helpers

    def _ee_pos(self) -> NDArray[np.float64]:
        return np.asarray(self._ee.get_pos().cpu().numpy(), dtype=np.float64)

    def _ee_quat(self) -> NDArray[np.float64]:
        return np.asarray(self._ee.get_quat().cpu().numpy(), dtype=np.float64)

    def _cube_pos(self) -> NDArray[np.float64]:
        return np.asarray(self._cube.get_pos().cpu().numpy(), dtype=np.float64)

    def _cube_quat(self) -> NDArray[np.float64]:
        return np.asarray(self._cube.get_quat().cpu().numpy(), dtype=np.float64)

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        obs: dict[str, NDArray[Any]] = {
            "cube_pos": self._cube_pos(),
            "cube_quat": self._cube_quat(),
            "ee_pos": self._ee_pos(),
            "gripper": np.array([self._gripper_state], dtype=np.float64),
            "target_pos": self._target_pos.copy(),
        }
        if self._render_in_obs:
            obs["image"] = self._render_obs_image()
        return obs

    def _render_obs_image(self) -> NDArray[np.uint8]:
        """Render the scene camera into a uint8 ``(H, W, 3)`` array.

        Deterministic headless CPU Rasterizer path (RFC-008 §3.3). The
        ``update_rigid`` call flushes any pending ``set_pos`` /
        ``set_quat`` transforms into the Rasterizer context — without
        it, post-``reset`` renders would see stale entity poses because
        no ``scene.step()`` has run since the teleports (exploration
        §2.4). ``scene.step()`` advances sim time 50 ms and corrupts
        reset-time state for any free body under gravity, so we flush
        transforms without stepping.

        The private ``_context`` hop is honesty-flagged in RFC-008 §4.1
        / §10 Q5 and pinned by ``genesis-world<0.5`` (RFC-007 §3).
        """
        assert self._camera is not None, (
            "_render_obs_image called with render_in_obs=False; _build_obs "
            "must gate this call on self._render_in_obs"
        )
        self._scene.visualizer.rasterizer._context.update_rigid()
        rgb = self._camera.render(rgb=True)[0]
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))

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
        """Translate + rotate the kinematic EE by small per-step deltas."""
        cur_pos = self._ee_pos()
        new_pos = cur_pos + linear * self.MAX_LINEAR_STEP
        self._ee.set_pos((float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))

        axis_angle = angular * self.MAX_ANGULAR_STEP
        if float(np.linalg.norm(axis_angle)) > 0.0:
            dq = _axis_angle_to_quat(axis_angle)
            cur_quat = self._ee_quat()
            new_quat = _normalize_quat(_quat_mul(dq, cur_quat))
            self._ee.set_quat(
                (float(new_quat[0]), float(new_quat[1]), float(new_quat[2]), float(new_quat[3]))
            )

    def _update_grasp_state(self) -> None:
        """Snap grasp flag based on gripper command + EE-cube proximity."""
        if self._gripper_state == self.GRIPPER_OPEN:
            self._grasped = False
            return
        ee = self._ee_pos()
        cube = self._cube_pos()
        dist = float(np.linalg.norm(ee - cube))
        if dist <= self.GRASP_RADIUS:
            self._grasped = True

    def _snap_cube_to_ee(self) -> None:
        """Overwrite cube pose with the EE pose post-physics (grasp sim)."""
        ee_pos = self._ee_pos()
        ee_quat = self._ee_quat()
        self._cube.set_pos((float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])))
        self._cube.set_quat(
            (float(ee_quat[0]), float(ee_quat[1]), float(ee_quat[2]), float(ee_quat[3]))
        )

    # --------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Deterministic reset.

        ``seed`` is the only entropy source. Ordering per RFC-005 §3.2:
        restore baseline -> re-seed -> apply queued perturbations ->
        clear queue.
        """
        del options
        self._rng = np.random.default_rng(seed)

        self.restore_baseline()

        # Re-seed cube XY (identity quat, zero velocity).
        cube_xy = self._rng.uniform(
            low=-_CUBE_INIT_HALFRANGE, high=_CUBE_INIT_HALFRANGE, size=2
        ).astype(np.float64)
        self._cube.set_pos((float(cube_xy[0]), float(cube_xy[1]), _CUBE_REST_Z))
        self._cube.set_quat((1.0, 0.0, 0.0, 0.0))

        # Re-seed target XY (independent of cube).
        target_xy = self._rng.uniform(
            low=-_TARGET_HALFRANGE, high=_TARGET_HALFRANGE, size=2
        ).astype(np.float64)
        self._target_pos = np.array([target_xy[0], target_xy[1], _TABLE_TOP_Z], dtype=np.float64)
        self._target.set_pos((float(target_xy[0]), float(target_xy[1]), _TABLE_TOP_Z + 0.001))

        # Reset EE to hover above cube start.
        self._ee.set_pos((float(cube_xy[0]), float(cube_xy[1]), _CUBE_REST_Z + _EE_REST_OFFSET_Z))
        self._ee.set_quat((1.0, 0.0, 0.0, 0.0))

        # Apply any per-episode perturbations queued via
        # ``set_perturbation``. Landed in RFC-007 §12 step 9; currently
        # a no-op unless a future commit populates the branches.
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

        Pipeline: clip -> update EE pose -> update grasp state ->
        ``scene.step`` (n_substeps times via ``SimOptions.substeps``) ->
        if grasped, snap cube to EE -> build obs / reward / flags.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        self._apply_ee_command(a[0:3], a[3:6])

        self._gripper_state = self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
        self._update_grasp_state()

        # SimOptions.substeps already fuses n_substeps micro-steps into
        # one ``scene.step`` call, so we only call step() once per control
        # tick. Matches the MuJoCo ``for _ in range(n_substeps): mj_step``
        # in total physics time per control tick.
        self._scene.step()

        if self._grasped:
            self._snap_cube_to_ee()

        self._step_count += 1
        cube_pos = self._cube_pos()
        if self._xy_distance(cube_pos, self._target_pos) <= self.TARGET_RADIUS:
            self._success = True

        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps
        reward = -float(self._xy_distance(cube_pos, self._target_pos))
        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue an axis-value pair for the next reset.

        Raises
        ------
        ValueError
            If ``name`` is not one of :attr:`AXIS_NAMES`. For
            ``distractor_count``, also if ``round(value)`` is outside
            ``[0, _N_DISTRACTOR_SLOTS]`` — same integer-range check
            PyBullet enforces.
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
        """Restore the scene to its post-__init__ observational state.

        First-cut body: reset dynamic-body positions to their baseline
        (pre-reset) values so a subsequent ``reset(seed=...)`` sees a
        clean slate. No model-level fields to snapshot (Genesis's scene
        is immutable post-build at the entity-count level); each
        perturbation branch that lands later is responsible for
        reverting its own side-effect.

        Distractor teleport-away uses ``_DISTRACTOR_HIDDEN_Z`` as the
        "off" position — ``restore_baseline`` hides all of them; the
        ``distractor_count`` branch then re-reveals the first N in
        ``_apply_pending_perturbations``.
        """
        for i, d in enumerate(self._distractors):
            xy = _DISTRACTOR_BASELINE_XY[i]
            d.set_pos((float(xy[0]), float(xy[1]), _DISTRACTOR_HIDDEN_Z))

        # Unswap the cube pointer if a prior episode flipped it. The
        # now-hidden green cube is explicitly re-parked at
        # ``(0, 0, _DISTRACTOR_HIDDEN_Z)`` so it doesn't drift into the
        # next episode's frustum. Red stays wherever it was — reset's
        # XY teleport overwrites its position a moment later.
        if self._cube is not self._cube_red:
            self._cube_green.set_pos((0.0, 0.0, _DISTRACTOR_HIDDEN_Z))
            self._cube, self._cube_alt = self._cube_red, self._cube_green

        # Visual-axis shadows — restore to their neutral defaults so a
        # prior episode's cosmetic perturbation doesn't leak.
        self._light_intensity = 1.0
        self._cam_offset = np.zeros(2, dtype=np.float64)
        self._texture_choice = 0

        # Render-side baselines — push the neutral shadow values into
        # the pyrender context + camera pose so the next render sees
        # the unperturbed scene. Guarded on ``render_in_obs`` so
        # state-only envs never touch the private ``_context`` hop.
        if self._render_in_obs:
            self._apply_light_intensity()
            self._apply_camera_pose()

    def _apply_pending_perturbations(self) -> None:
        """Apply ``self._pending_perturbations`` to the scene.

        The seven branches (RFC-007 §6): the three state-affecting
        axes (``object_initial_pose_{x,y}``, ``distractor_count``)
        mutate entity state; the four cosmetic axes store on
        visual-only shadow attributes. The cosmetic-axis branches
        exist on day one so a follow-up rendering RFC (RFC-008)
        only needs to read these shadows, not rewire the dispatch.
        """
        for name, value in self._pending_perturbations.items():
            self._apply_one_perturbation(name, value)

    def _apply_camera_pose(self) -> None:
        """Apply ``self._cam_offset`` to the render camera.

        No-op when ``render_in_obs=False`` (no camera attached). The
        offset adds to ``_CAM_EYE_BASELINE``; the target and up vector
        stay fixed — pan, not orbit.
        """
        assert self._camera is not None, (
            "_apply_camera_pose requires render_in_obs=True — the "
            "caller must gate on self._render_in_obs"
        )
        eye = (
            _CAM_EYE_BASELINE[0] + float(self._cam_offset[0]),
            _CAM_EYE_BASELINE[1] + float(self._cam_offset[1]),
            _CAM_EYE_BASELINE[2],
        )
        self._camera.set_pose(pos=eye, lookat=_CAM_TARGET, up=_CAM_UP)

    def _apply_light_intensity(self) -> None:
        """Scale the default directional light's intensity.

        Private-API hop on ``scene.visualizer.rasterizer._context._scene.light_nodes[0].light``
        — Genesis 0.4.x does not expose a public post-build setter
        (exploration §4, RFC-008 §4.1). Pinned by
        ``genesis-world>=0.4,<0.5`` in ``pyproject.toml``. If 0.5
        rearranges the internals, this site raises ``AttributeError``
        loudly.
        """
        if not self._render_in_obs:
            return
        pyscene = self._scene.visualizer.rasterizer._context._scene
        # pyrender's scene exposes ``directional_light_nodes`` as an
        # iterable (a set on 0.4.x); Genesis's default ``VisOptions.lights``
        # seeds exactly one directional light — take it by position in
        # iteration order.
        light_node = next(iter(pyscene.directional_light_nodes))
        light_node.light.intensity = _BASELINE_LIGHT_INTENSITY * self._light_intensity

    def _apply_one_perturbation(self, name: str, value: float) -> None:
        if name == "lighting_intensity":
            self._light_intensity = float(value)
            if self._render_in_obs:
                self._apply_light_intensity()
        elif name == "camera_offset_x":
            self._cam_offset[0] = float(value)
            if self._render_in_obs:
                self._apply_camera_pose()
        elif name == "camera_offset_y":
            self._cam_offset[1] = float(value)
            if self._render_in_obs:
                self._apply_camera_pose()
        elif name == "object_texture":
            new_choice = 1 if round(float(value)) != 0 else 0
            if new_choice != self._texture_choice:
                # Capture the active cube's XY before the swap so the
                # new active cube inherits the seed-randomised (or
                # ``object_initial_pose_*``-overridden) position —
                # otherwise the axis would silently teleport the cube
                # to the origin, corrupting per-seed state (caught in
                # RFC-008 pre-implementation review).
                old_xy = self._cube_pos()
                self._cube.set_pos((0.0, 0.0, _DISTRACTOR_HIDDEN_Z))
                self._cube, self._cube_alt = self._cube_alt, self._cube
                self._cube.set_pos((float(old_xy[0]), float(old_xy[1]), _CUBE_REST_Z))
                # Identity quat on the new active cube — reset's quat
                # write earlier hit the old (now hidden) active cube,
                # so without this line the new active cube inherits
                # whatever quat it had last time it was active (likely
                # post-grasp from the previous episode it was live).
                self._cube.set_quat((1.0, 0.0, 0.0, 0.0))
            self._texture_choice = new_choice
        elif name == "object_initial_pose_x":
            # State-affecting: overrides the random cube X from the
            # seed-driven randomisation (matches MuJoCo qpos-write
            # semantics — RFC-007 §6.4).
            cur = self._cube_pos()
            self._cube.set_pos((float(value), float(cur[1]), _CUBE_REST_Z))
        elif name == "object_initial_pose_y":
            cur = self._cube_pos()
            self._cube.set_pos((float(cur[0]), float(value), _CUBE_REST_Z))
        elif name == "distractor_count":
            # State-affecting: teleport first ``count`` distractors to
            # their rest Z; rest stay at _DISTRACTOR_HIDDEN_Z
            # (restore_baseline already hid them all). RFC-007 §6.5
            # — teleport-away semantic, documented deviation from
            # MuJoCo's visibility+collision toggle.
            count = round(float(value))
            for i, d in enumerate(self._distractors):
                xy = _DISTRACTOR_BASELINE_XY[i]
                z = _DISTRACTOR_REST_Z if i < count else _DISTRACTOR_HIDDEN_Z
                d.set_pos((float(xy[0]), float(xy[1]), z))
        # Unknown axis names are blocked at set_perturbation; the else
        # branch is unreachable by contract (and mypy's exhaustive-if
        # check flags nothing to handle at this layer).

    def close(self) -> None:
        """Release Genesis scene resources. Idempotent.

        Genesis scenes are GC-managed at the Python level; dropping
        the reference frees the underlying compiled-kernel state.
        The global ``gs.init`` state is not cleaned up — it's a
        module-level singleton reused by any follow-up env instance
        in the same process (RFC-007 §4.6).
        """
        # Best-effort drop; Genesis doesn't currently expose a public
        # ``scene.destroy`` at 0.4.6, so we just null our handles so
        # the reference count drops.
        self._scene = None
        self._target = None
        self._cube = None
        self._ee = None
        self._distractors = []
