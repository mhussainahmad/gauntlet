"""PyBullet tabletop backend — state + image obs (RFC-005 + RFC-006).

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
* **Post-RFC-006 cosmetic-axis parity** (§3.5). ``lighting_intensity``,
  ``object_texture``, and ``camera_offset_{x,y}`` now mutate
  ``obs["image"]`` when ``render_in_obs=True``. :attr:`VISUAL_ONLY_AXES`
  is empty (matching MuJoCo's ``TabletopEnv``); the Suite loader's
  cosmetic-only rejection is a no-op on this backend. A user running a
  cosmetic-only sweep with ``render_in_obs=False`` will still see
  pairwise-identical state-only cells — documented, not a bug, same
  shape MuJoCo has always had.

Scene layout: per RFC §5.1, built from PyBullet primitives
(createMultiBody + loadTexture). Only ``plane.urdf`` from
``pybullet_data`` is loaded as a URDF — everything else is authored as
a box/cylinder shape so the repo ships no custom URDF.
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
from gauntlet.env.base import CameraSpec

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

_LARGE_CONSTRAINT_FORCE: float = 1000.0

# Camera + light constants for the headless render path (RFC-006 §3.4).
# Semantic (not pixel) parity with the MuJoCo ``main`` camera in
# ``assets/tabletop.xml`` — same eye position, same target region, same
# light direction / ambient coefficient. Cross-backend numerical pixel
# parity is explicitly NOT a goal (RFC-005 §7.4).
_CAM_EYE_BASELINE: tuple[float, float, float] = (0.6, -0.6, 0.8)
_CAM_TARGET: tuple[float, float, float] = (0.0, 0.0, _TABLE_TOP_Z)
_CAM_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)
_CAM_FOV: float = 45.0
_CAM_NEAR: float = 0.01
_CAM_FAR: float = 5.0
_CAM_LIGHT_AMBIENT: float = 0.3  # matches MuJoCo headlight ambient
_DEFAULT_RENDER_SIZE: tuple[int, int] = (224, 224)  # matches TabletopEnv


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
            # B-06 — declared so the suite linter / loader can route the
            # axis through ``VISUAL_ONLY_AXES`` and reject runs that try
            # to combine ``object_swap`` with this backend (anti-feature:
            # cross-backend asset parity is yak-shave; see backlog B-06).
            "object_swap",
            # B-42 — categorical render-camera extrinsics axis. Real
            # impl: the apply branch stores the index on a shadow
            # attribute, ``_render_obs_image`` and
            # ``_render_obs_images_multi`` consume it via
            # :func:`_apply_extrinsics_to_camera_basis`. RoboView-Bias
            # (arXiv 2509.22356) inspired.
            "camera_extrinsics",
        }
    )
    # Empty once image rendering exists (RFC-006 §3.5): every cosmetic axis
    # is now observable on ``obs["image"]`` when ``render_in_obs=True``.
    # Aligns PyBullet with MuJoCo's TabletopEnv, which has always declared
    # ``frozenset()`` for the same reason. A user running a cosmetic-only
    # sweep with ``render_in_obs=False`` will still see pairwise-identical
    # state-only cells — documented, not a bug, and matches MuJoCo.
    #
    # B-06 — ``object_swap`` is the lone non-empty entry: the backend
    # has no asset library for the alternate semantic objects (anti-
    # feature note in the backlog), so the loader / linter rejects any
    # suite naming this axis on PyBullet.
    #
    # B-42 — ``camera_extrinsics`` is intentionally NOT in this set:
    # the PyBullet renderer mutates the view matrix on the way to
    # ``getCameraImage``, so the axis is observable on
    # ``obs["image"]`` when ``render_in_obs=True``. State-only obs
    # remain pairwise-identical across cells, same caveat as the
    # other cosmetic axes.
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset({"object_swap"})

    MAX_LINEAR_STEP: float = 0.05
    MAX_ANGULAR_STEP: float = 0.1
    GRASP_RADIUS: float = 0.05
    TARGET_RADIUS: float = 0.05
    GRIPPER_OPEN: float = 1.0
    GRIPPER_CLOSED: float = -1.0

    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    # ``render_in_obs`` interaction with the Runner's trajectory recorder
    # (``trajectory_dir``, RFC-004): works, but the recorder currently casts
    # every obs value to float64 before stacking (see
    # ``src/gauntlet/runner/worker.py``). Images go into the NPZ as float64,
    # ~8x the uint8 footprint. Not a correctness issue; a memory-efficiency
    # follow-up (dtype-preserving recorder) is tracked outside RFC-006.

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
        render_in_obs: bool = False,
        render_size: tuple[int, int] = _DEFAULT_RENDER_SIZE,
        cameras: list[CameraSpec] | None = None,
    ) -> None:
        """Construct a headless PyBullet client and build the scene once.

        ``render_in_obs`` enables the cached TINY-renderer path with a
        constant projection matrix. ``cameras`` overrides it with one
        per-spec view matrix and projection; the first spec is aliased
        to ``obs["image"]`` for parity with the MuJoCo backend.
        """
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

        # Multi-cam validation runs before any pybullet client is opened.
        # Same contract as the MuJoCo backend (see CameraSpec docstring).
        # cameras=[] is treated as cameras=None per RFC §2.
        self._cameras: tuple[CameraSpec, ...] = tuple(cameras) if cameras else ()
        self._multi_cam_enabled: bool = bool(self._cameras)
        if self._multi_cam_enabled:
            _validate_pybullet_camera_specs(self._cameras)

        self._max_steps = max_steps
        self._n_substeps = n_substeps
        # Multi-cam takes precedence over the legacy render_in_obs flag
        # (RFC §2). Setting _render_in_obs=False here disables the
        # legacy single-camera codepath; the multi-cam codepath handles
        # all rendering and aliases obs['image'] to the first cam.
        self._render_in_obs = render_in_obs and not self._multi_cam_enabled
        self._render_size: tuple[int, int] = (int(render_size[0]), int(render_size[1]))

        # ---- spaces (identical to TabletopEnv; ``image`` only when rendering on) ----
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
            # uint8 RGB frame from the headless TINY renderer. Shape / dtype /
            # bounds match TabletopEnv(render_in_obs=True) byte-for-byte — VLA
            # adapters (RFC-001, RFC-002) read the same obs["image"] key.
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        if self._multi_cam_enabled:
            # Multi-cam advertises a Dict subspace with one Box per spec
            # plus a top-level 'image' alias for the first cam (matches
            # MuJoCo's contract — see RFC §2 / TabletopEnv equivalent).
            per_cam_p: dict[str, gym.spaces.Space[Any]] = {}
            for spec in self._cameras:
                ch, cw = spec.size
                per_cam_p[spec.name] = spaces.Box(
                    low=0, high=255, shape=(ch, cw, 3), dtype=np.uint8
                )
            obs_spaces["images"] = spaces.Dict(per_cam_p)
            first_h, first_w = self._cameras[0].size
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(first_h, first_w, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

        # Projection matrix is constant per instance — fixed intrinsics, no
        # perturbation axis touches it. Cache once so per-step render skips
        # the (trivial) compute. View matrix depends on _cam_eye_offset and
        # is rebuilt per render call in _render_obs_image.
        if self._render_in_obs:
            h, w = self._render_size
            self._proj_matrix: list[float] | None = p.computeProjectionMatrixFOV(
                fov=_CAM_FOV,
                aspect=float(w) / float(h),
                nearVal=_CAM_NEAR,
                farVal=_CAM_FAR,
            )
        else:
            self._proj_matrix = None

        # Per-spec projection matrices for the multi-cam path. Each spec
        # may carry its own aspect ratio so a single shared matrix
        # cannot satisfy them all. View matrices are computed per call
        # in _render_obs_images_multi (depend on spec.pose only — pose
        # is fixed at construction so we could cache them too, but the
        # compute is trivial and per-call keeps the call site clear).
        self._extra_proj_matrices: dict[str, list[float]] = {}
        if self._multi_cam_enabled:
            for spec in self._cameras:
                ch, cw = spec.size
                self._extra_proj_matrices[spec.name] = p.computeProjectionMatrixFOV(
                    fov=_CAM_FOV,
                    aspect=float(cw) / float(ch),
                    nearVal=_CAM_NEAR,
                    farVal=_CAM_FAR,
                )

        # ---- per-session PyBullet client ----
        self._client: int = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

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
        # These are *not* observable on state-only obs (§6.2); the follow-up
        # rendering RFC plumbs them into the getCameraImage path.
        self._light_intensity: float = 1.0
        self._cam_eye_offset: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._current_texture_id: int = -1  # populated by _build_scene

        # B-42 — active camera-extrinsics registry. Each entry is a
        # 6-tuple ``(dx, dy, dz, drx, dry, drz)`` (translation in
        # metres, rotation in radians, XYZ Euler) applied as a delta
        # to the render camera's baseline pose at render time. Index 0
        # defaults to the no-op baseline; the runner rebinds via
        # :meth:`set_camera_extrinsics_list`. ``_extrinsics_index``
        # tracks the active cell's index — read by the render path.
        _ExtrinsicsTuple = tuple[float, float, float, float, float, float]
        self._camera_extrinsics_list: tuple[_ExtrinsicsTuple, ...] = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self._extrinsics_index: int = 0

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
        p.changeDynamics(self._cube_id, -1, lateralFriction=1.0, physicsClientId=cid)
        p.changeVisualShape(
            self._cube_id,
            -1,
            textureUniqueId=self._tex_default_id,
            physicsClientId=cid,
        )
        self._current_texture_id = self._tex_default_id

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
                body_id,
                -1,
                collisionFilterGroup=0,
                collisionFilterMask=0,
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
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
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
        self._target_pos = np.array([target_xy[0], target_xy[1], _TABLE_TOP_Z], dtype=np.float64)
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
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        """Advance one control step (parity with MuJoCo TabletopEnv pipeline)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        self._apply_ee_command(a[0:3], a[3:6])
        self._gripper_state = self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
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
                    f"distractor_count must be in [0, {_N_DISTRACTOR_SLOTS}]; got {count}"
                )
        if name == "camera_extrinsics":
            # B-42 — index into the active camera-extrinsics registry.
            # Mirrors the MuJoCo ``set_perturbation`` validation path:
            # round float to int, reject out-of-range loud rather
            # than silently snapping to the baseline.
            idx = round(float(value))
            n_entries = len(self._camera_extrinsics_list)
            if idx < 0 or idx >= n_entries:
                raise ValueError(
                    f"camera_extrinsics: index {idx} (rounded from {value!r}) "
                    f"out of range [0, {n_entries}); registry currently has "
                    f"{n_entries} entry(ies). Set the registry via "
                    "set_camera_extrinsics_list before queueing."
                )
        self._pending_perturbations[name] = float(value)

    def set_camera_extrinsics_list(
        self,
        entries: tuple[tuple[float, float, float, float, float, float], ...],
    ) -> None:
        """Rebind the active camera-extrinsics registry (B-42).

        Each entry is a 6-tuple ``(dx, dy, dz, drx, dry, drz)`` where
        translation deltas are in metres and rotation deltas are
        radians (XYZ Euler, PyBullet camera convention — same axis
        order as the MuJoCo backend so a single suite YAML drives both
        backends consistently). Index 0 is the baseline by convention;
        the runner sets the per-cell index via :meth:`set_perturbation`
        and the apply branch stores it on a shadow attribute that the
        render path consumes.

        Calling this method does NOT touch the renderer — the index
        → 6-tuple dispatch happens at render time. Mirrors the MuJoCo
        :meth:`gauntlet.env.tabletop.TabletopEnv.set_camera_extrinsics_list`
        contract.

        Anti-feature note (spec): SO(3) rotations don't compose
        linearly so per-axis Sobol sensitivity indices on this axis
        are biased. The report module emits a :class:`UserWarning`
        when the axis is present in a Sobol report.

        Raises:
            ValueError: if ``entries`` is empty or any entry has a
                length other than 6.
        """
        if len(entries) == 0:
            raise ValueError("camera_extrinsics: registry must be non-empty")
        sanitised: list[tuple[float, float, float, float, float, float]] = []
        for i, e in enumerate(entries):
            if len(e) != 6:
                raise ValueError(
                    f"camera_extrinsics: entry {i} must be a 6-tuple "
                    f"(dx, dy, dz, drx, dry, drz); got length {len(e)}"
                )
            sanitised.append(
                (
                    float(e[0]),
                    float(e[1]),
                    float(e[2]),
                    float(e[3]),
                    float(e[4]),
                    float(e[5]),
                )
            )
        self._camera_extrinsics_list = tuple(sanitised)

    @property
    def camera_extrinsics_list(
        self,
    ) -> tuple[tuple[float, float, float, float, float, float], ...]:
        """Return the active camera-extrinsics registry (B-42)."""
        return self._camera_extrinsics_list

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
        self._cam_eye_offset = np.zeros(2, dtype=np.float64)
        # B-42 — extrinsics index resets to 0 (baseline). The
        # registry itself stays bound across resets — the runner
        # rebinds it once per suite, not once per cell.
        self._extrinsics_index = 0

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

        Walks RFC-005 §6 row by row. Pose overrides must run AFTER the
        random cube XY write in :meth:`reset`, which is exactly the
        order we already have — :meth:`reset` writes the random pose
        first and then calls this method.
        """
        for name, value in self._pending_perturbations.items():
            self._apply_one_perturbation(name, value)

    def _apply_one_perturbation(self, name: str, value: float) -> None:
        """Per-axis branch dispatcher (RFC-005 §6 table)."""
        cid = self._client

        if name == "object_initial_pose_x":
            # State-effecting. Override the random X write from reset.
            _, quat = p.getBasePositionAndOrientation(self._cube_id, physicsClientId=cid)
            cur_pos_tup, _ = p.getBasePositionAndOrientation(self._cube_id, physicsClientId=cid)
            new_pos = [float(value), float(cur_pos_tup[1]), _CUBE_REST_Z]
            p.resetBasePositionAndOrientation(self._cube_id, new_pos, quat, physicsClientId=cid)
            p.resetBaseVelocity(
                self._cube_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=cid,
            )
            return

        if name == "object_initial_pose_y":
            cur_pos_tup, quat = p.getBasePositionAndOrientation(self._cube_id, physicsClientId=cid)
            new_pos = [float(cur_pos_tup[0]), float(value), _CUBE_REST_Z]
            p.resetBasePositionAndOrientation(self._cube_id, new_pos, quat, physicsClientId=cid)
            p.resetBaseVelocity(
                self._cube_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=cid,
            )
            return

        if name == "distractor_count":
            # State-effecting on rollouts where a revealed distractor
            # blocks the grasp path. set_perturbation's validator
            # clamps to [0, 10], so round() is safe here.
            n = round(float(value))
            for i, body_id in enumerate(self._distractor_ids):
                if i < n:
                    # Reveal + enable collisions.
                    p.changeVisualShape(
                        body_id,
                        -1,
                        rgbaColor=[0.6, 0.6, 0.9, 1.0],
                        physicsClientId=cid,
                    )
                    p.setCollisionFilterGroupMask(
                        body_id,
                        -1,
                        collisionFilterGroup=1,
                        collisionFilterMask=1,
                        physicsClientId=cid,
                    )
                else:
                    # Return to hidden + ghosted baseline.
                    p.changeVisualShape(
                        body_id,
                        -1,
                        rgbaColor=[0.6, 0.6, 0.9, 0.0],
                        physicsClientId=cid,
                    )
                    p.setCollisionFilterGroupMask(
                        body_id,
                        -1,
                        collisionFilterGroup=0,
                        collisionFilterMask=0,
                        physicsClientId=cid,
                    )
            return

        if name == "object_texture":
            # Cosmetic on state-only obs (§6.2). Swap the bound texture
            # UID. value ~0 → default (red), value ~1 → alt (green).
            use_alt = float(value) >= 0.5
            tex_id = self._tex_alt_id if use_alt else self._tex_default_id
            p.changeVisualShape(
                self._cube_id,
                -1,
                textureUniqueId=tex_id,
                physicsClientId=cid,
            )
            self._current_texture_id = tex_id
            return

        if name == "lighting_intensity":
            # Cosmetic (§6.1). Headless DIRECT mode has no runtime light
            # API; the rendering follow-up RFC reads this in
            # getCameraImage(lightDiffuseCoeff=...).
            self._light_intensity = float(value)
            return

        if name == "camera_offset_x":
            # Cosmetic — camera pose is a rendering-time concept.
            self._cam_eye_offset = self._cam_eye_offset.copy()
            self._cam_eye_offset[0] = float(value)
            return

        if name == "camera_offset_y":
            self._cam_eye_offset = self._cam_eye_offset.copy()
            self._cam_eye_offset[1] = float(value)
            return

        if name == "camera_extrinsics":
            # B-42 — store the index on a shadow attribute; the render
            # path reads ``_camera_extrinsics_list[_extrinsics_index]``
            # and applies the structured 6-tuple as a delta on top of
            # the baseline (eye, target, up) basis. ``set_perturbation``
            # already validated the index range; the apply branch
            # re-checks defensively.
            idx = round(float(value))
            n_entries = len(self._camera_extrinsics_list)
            if idx < 0 or idx >= n_entries:
                raise ValueError(
                    f"camera_extrinsics apply: index {idx} out of range [0, {n_entries})"
                )
            self._extrinsics_index = idx
            return

        # Unreachable — set_perturbation validates before queueing.
        raise ValueError(f"internal: unknown perturbation axis {name!r}")

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        cube_pos_tup, cube_quat_xyzw = p.getBasePositionAndOrientation(
            self._cube_id, physicsClientId=self._client
        )
        cube_pos = np.asarray(cube_pos_tup, dtype=np.float64)
        cube_quat_wxyz = _xyzw_to_wxyz(cube_quat_xyzw)  # §7.3 conversion
        ee_pos = self._ee_pos.copy()
        gripper = np.array([self._gripper_state], dtype=np.float64)
        obs: dict[str, NDArray[Any]] = {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat_wxyz,
            "ee_pos": ee_pos,
            "gripper": gripper,
            "target_pos": self._target_pos.copy(),
        }
        if self._render_in_obs:
            obs["image"] = self._render_obs_image()
        if self._multi_cam_enabled:
            images = self._render_obs_images_multi()
            obs["images"] = images  # type: ignore[assignment]
            # Backwards-compat alias to the first cam's frame — RFC §2.
            # Defensive copy for the same reason the MuJoCo backend
            # makes one in TabletopEnv._build_obs: prevents an in-place
            # mutation of ``obs["image"]`` by a downstream consumer
            # from silently corrupting ``obs["images"][first]``.
            first_name = self._cameras[0].name
            obs["image"] = images[first_name].copy()
        return obs

    def _render_obs_image(self) -> NDArray[np.uint8]:
        """Render the main camera into a uint8 (H, W, 3) array.

        Headless ``ER_TINY_RENDERER`` (no GL context). Consumes the four
        cosmetic axes through live env state: ``lightDiffuseCoeff`` ←
        ``self._light_intensity`` (``lighting_intensity``); camera eye
        ← ``_CAM_EYE_BASELINE + self._cam_eye_offset`` on x, y
        (``camera_offset_x``, ``camera_offset_y``); cube texture is
        already bound on the cube via ``_apply_one_perturbation`` and
        picked up by the rasteriser automatically (``object_texture``).

        ``shadow=0`` is unconditional — shadow pass is non-deterministic
        on some wheels (RFC-006 §2.3). Segmentation pass is suppressed
        via ``ER_NO_SEGMENTATION_MASK`` to cut ~15% render time on the
        software path.
        """
        assert self._proj_matrix is not None  # gate: render_in_obs=True
        h, w = self._render_size
        eye0 = (
            _CAM_EYE_BASELINE[0] + float(self._cam_eye_offset[0]),
            _CAM_EYE_BASELINE[1] + float(self._cam_eye_offset[1]),
            _CAM_EYE_BASELINE[2],
        )
        # B-42 — apply the active extrinsics delta on top of the
        # camera_offset baseline. ``_apply_extrinsics_to_basis`` is a
        # no-op when the active index points at the (0, 0, 0, 0, 0, 0)
        # registry slot, so the legacy single-cam path is unchanged
        # for callers that do not opt into the axis.
        delta = self._camera_extrinsics_list[self._extrinsics_index]
        eye, target, up = _apply_extrinsics_to_basis(eye0, _CAM_TARGET, _CAM_UP, delta)
        view = p.computeViewMatrix(
            cameraEyePosition=list(eye),
            cameraTargetPosition=list(target),
            cameraUpVector=list(up),
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view,
            projectionMatrix=self._proj_matrix,
            lightDirection=[0.0, 0.0, 1.0],
            lightColor=[1.0, 1.0, 1.0],
            lightDiffuseCoeff=float(self._light_intensity),
            lightAmbientCoeff=_CAM_LIGHT_AMBIENT,
            lightSpecularCoeff=0.0,
            shadow=0,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        # getCameraImage returns (H, W, 4) RGBA uint8; drop alpha for MuJoCo
        # parity. np.ascontiguousarray guarantees the returned array is
        # C-contiguous regardless of how pybullet packed the buffer.
        arr = np.asarray(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return np.ascontiguousarray(arr)

    def _render_obs_images_multi(self) -> dict[str, NDArray[np.uint8]]:
        """Render every CameraSpec via per-spec view + projection matrices.

        Each spec's pose is converted to a (eye, target, up) triple via
        :func:`_camera_basis_from_pose` (matches the MuJoCo convention:
        camera looks along its local ``-Z`` axis, with local ``+Y`` as
        up, after applying XYZ Euler rotation). Light parameters mirror
        the single-cam path (``lightDiffuseCoeff`` ←
        ``self._light_intensity``) so the cosmetic axes still affect
        every cam consistently. ``object_texture`` is bound at the
        cube level so it shows up automatically.
        """
        out: dict[str, NDArray[np.uint8]] = {}
        for spec in self._cameras:
            ch, cw = spec.size
            eye, target, up = _camera_basis_from_pose(spec.pose)
            view = p.computeViewMatrix(
                cameraEyePosition=list(eye),
                cameraTargetPosition=list(target),
                cameraUpVector=list(up),
            )
            proj = self._extra_proj_matrices[spec.name]
            _, _, rgb, _, _ = p.getCameraImage(
                width=cw,
                height=ch,
                viewMatrix=view,
                projectionMatrix=proj,
                lightDirection=[0.0, 0.0, 1.0],
                lightColor=[1.0, 1.0, 1.0],
                lightDiffuseCoeff=float(self._light_intensity),
                lightAmbientCoeff=_CAM_LIGHT_AMBIENT,
                lightSpecularCoeff=0.0,
                shadow=0,
                flags=p.ER_NO_SEGMENTATION_MASK,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self._client,
            )
            arr = np.asarray(rgb, dtype=np.uint8).reshape(ch, cw, 4)[:, :, :3]
            out[spec.name] = np.ascontiguousarray(arr)
        return out

    def _build_info(self) -> dict[str, Any]:
        return {
            "success": self._success,
            "grasped": self._grasped,
            "step": self._step_count,
        }

    @staticmethod
    def _xy_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))


def _quat_mul_wxyz(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
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


# ---------------------------------------------------------- multi-cam helpers


def _validate_pybullet_camera_specs(specs: tuple[CameraSpec, ...]) -> None:
    """Reject invalid CameraSpec lists at construction time.

    Mirrors the MuJoCo backend's contract verbatim — same message
    formats, same checks. See
    ``docs/polish-exploration-multi-camera.md`` §2.
    """
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec.name, str) or spec.name == "":
            raise ValueError(f"camera name must be a non-empty string; got {spec.name!r}")
        if spec.name in seen:
            raise ValueError(f"duplicate camera name: {spec.name!r}")
        seen.add(spec.name)
        h, w = spec.size
        if int(h) <= 0 or int(w) <= 0:
            raise ValueError(
                f"camera {spec.name!r} size must be (height, width) of positive ints; "
                f"got {spec.size!r}"
            )
        if len(spec.pose) != 6:
            raise ValueError(
                f"camera {spec.name!r} pose must be (x, y, z, rx, ry, rz); got {spec.pose!r}"
            )


def _apply_extrinsics_to_basis(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
    delta: tuple[float, float, float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """B-42 — apply a 6-D extrinsics delta to a camera basis.

    ``delta`` is ``(dx, dy, dz, drx, dry, drz)``: translation in
    metres, rotation in radians (XYZ Euler, body-fixed). The camera
    is treated as a *rigid* mount: a pure translation shifts the
    eye in world frame, the look direction is unchanged, and the
    target re-anchors at ``new_eye + old_forward * dist`` so the
    rendered image matches "the camera is mounted slightly off" —
    not "the camera orbits a fixed point". This matches the MuJoCo
    backend's contract (mutating ``cam_pos`` alone leaves the look
    direction fixed via ``cam_quat``) and the
    RoboView-Bias / "Do You Know Where Your Camera Is?" paper
    framing (camera misalignment, not orbital perturbation).

    Rotation is applied as ``R = Rx @ Ry @ Rz`` to the camera's
    local basis: the new forward / up are ``R @ forward_unit`` /
    ``R @ up``, and the new target is reconstructed as
    ``new_eye + new_forward * |target - eye|`` so the look
    distance is preserved.

    A zero delta returns the input basis unchanged (numerically: the
    rotation matrix is the identity and translation adds zero).
    """
    dx, dy, dz, drx, dry, drz = delta
    eye_arr = np.array(eye, dtype=np.float64)
    tgt_arr = np.array(target, dtype=np.float64)
    up_arr = np.array(up, dtype=np.float64)

    # Translation in world frame: shift eye, keep target stationary.
    new_eye = eye_arr + np.array([dx, dy, dz], dtype=np.float64)

    # Rotation in the camera's local frame (right-multiply convention,
    # matching the MuJoCo backend): build R = Rx @ Ry @ Rz, then
    # apply to forward / up.
    cx, sx = float(np.cos(drx)), float(np.sin(drx))
    cy, sy = float(np.cos(dry)), float(np.sin(dry))
    cz, sz = float(np.cos(drz)), float(np.sin(drz))
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot = rot_x @ rot_y @ rot_z

    forward_world = tgt_arr - eye_arr
    look_dist = float(np.linalg.norm(forward_world))
    if look_dist > 0.0:
        forward_unit = forward_world / look_dist
    else:
        # Degenerate: eye and target coincide. Fall back to "+x" so
        # the math stays well-defined; in practice the env never
        # passes a zero-length basis.
        forward_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        look_dist = 1.0
    new_forward = rot @ forward_unit
    new_up = rot @ up_arr
    new_target = new_eye + new_forward * look_dist

    return (
        (float(new_eye[0]), float(new_eye[1]), float(new_eye[2])),
        (float(new_target[0]), float(new_target[1]), float(new_target[2])),
        (float(new_up[0]), float(new_up[1]), float(new_up[2])),
    )


def _camera_basis_from_pose(
    pose: tuple[float, float, float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Convert a CameraSpec pose to PyBullet ``(eye, target, up)`` triple.

    Pose convention (RFC §2): world-frame ``(x, y, z)`` in metres,
    XYZ Euler ``(rx, ry, rz)`` in radians. The camera looks along its
    local ``-Z`` axis with local ``+Y`` as up — same as MuJoCo's
    ``<camera>`` default. We build a 3x3 rotation matrix from XYZ
    Euler (R = Rx @ Ry @ Rz), then ``forward = R @ (0, 0, -1)``,
    ``up = R @ (0, 1, 0)``, and ``target = eye + forward``.
    """
    x, y, z, rx, ry, rz = pose
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot = rot_x @ rot_y @ rot_z
    forward = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up_local = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    eye = (float(x), float(y), float(z))
    target = (eye[0] + float(forward[0]), eye[1] + float(forward[1]), eye[2] + float(forward[2]))
    up = (float(up_local[0]), float(up_local[1]), float(up_local[2]))
    return eye, target, up
