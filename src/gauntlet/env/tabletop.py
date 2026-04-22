"""Deterministic MuJoCo pick-and-place tabletop environment.

See ``GAUNTLET_SPEC.md`` §3 (core abstractions) and §6 (design principles).

The scene is intentionally minimal:

* one cube resting on a table (MuJoCo freejoint),
* one target zone (visualised as a translucent cylinder site),
* a "floating" end-effector represented by a mocap body — no kinematic
  chain, no actuators. Phase 1 cares about a deterministic
  :class:`gymnasium.Env` that can be perturbed, not about a realistic arm.

Action layout (``shape=(7,), dtype=float64, bounds=[-1, 1]``) matches
:data:`gauntlet.policy.scripted.DEFAULT_PICK_AND_PLACE_TRAJECTORY`::

    [dx, dy, dz, drx, dry, drz, gripper]

The first six entries are per-step end-effector twist commands scaled by
:attr:`TabletopEnv.MAX_LINEAR_STEP` / :attr:`TabletopEnv.MAX_ANGULAR_STEP`.
The gripper command snaps to a binary ``open (+1) / closed (-1)`` state
(snap, not ramp — chosen for determinism and simplicity).

Grasp model
-----------
When the gripper is commanded closed *and* the end-effector is within
:attr:`TabletopEnv.GRASP_RADIUS` of the cube centre, :attr:`_grasped` is
set to ``True``. While grasped, the cube's freejoint qpos/qvel are
snapped to the end-effector pose *after* ``mj_step`` so physics cannot
yank it out of the gripper. Opening the gripper releases the cube.

Success
-------
Success is declared on the first step where the cube's XY position is
within :attr:`TabletopEnv.TARGET_RADIUS` of the target centre.
``terminated=True`` from that step on; ``info["success"]`` exposes the
same flag.

Reproducibility (spec §6 hard rule)
-----------------------------------
The *only* entropy source is :attr:`_rng`, built from the seed handed to
:meth:`reset`. No wall-clock, no ``os.urandom``, no ``np.random.seed``.
Fixed seed plus fixed action sequence yields bit-identical observations
across runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

__all__ = ["TabletopEnv"]


_ASSET_PATH: Path = Path(__file__).parent / "assets" / "tabletop.xml"

# Number of pre-allocated distractor slots in the MJCF.
# DistractorCount perturbation enables the first N (0 <= N <= 10).
N_DISTRACTOR_SLOTS: Final[int] = 10

# Material names for the object_texture perturbation. Both are defined in
# assets/tabletop.xml. The perturbation flips the cube_geom's matid
# between these two; geom_rgba stays at the MuJoCo default (0.5,0.5,0.5,1)
# so the material colour is what shows.
_CUBE_MATERIAL_DEFAULT: Final[str] = "cube_mat"
_CUBE_MATERIAL_ALT: Final[str] = "cube_alt_mat"

# Set of accepted perturbation axis names. The canonical ordered tuple
# lives in ``gauntlet.env.perturbation.AXIS_NAMES``; we duplicate the
# membership set here as a module-local constant to keep ``set_perturbation``
# free of any cross-module import (avoids a perturbation -> tabletop ->
# perturbation cycle).
_KNOWN_AXIS_NAMES: Final[frozenset[str]] = frozenset(
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

# Observation / action type aliases for the gym.Env generic parameters.
# We keep ``Any`` inside ``NDArray`` here only to match ``gauntlet.policy.base``;
# externally everything is float64.
_ObsType = dict[str, NDArray[Any]]
_ActType = NDArray[np.float64]


class TabletopEnv(gym.Env[_ObsType, _ActType]):
    """Minimal pick-and-place tabletop env with a mocap end-effector.

    Observation (``spaces.Dict``):
        ``cube_pos``     — ``Box(shape=(3,), float64)``
        ``cube_quat``    — ``Box(shape=(4,), float64)`` (MuJoCo wxyz order)
        ``ee_pos``       — ``Box(shape=(3,), float64)``
        ``gripper``      — ``Box(shape=(1,), float64)``, ``-1`` closed, ``+1`` open
        ``target_pos``   — ``Box(shape=(3,), float64)``

    Action (``spaces.Box``): ``shape=(7,), float64, bounds=[-1, 1]``.

    Attributes
    ----------
    MAX_LINEAR_STEP:
        Per-step translational scale (metres). An action of ``1.0`` moves
        the end-effector this far in one step.
    MAX_ANGULAR_STEP:
        Per-step rotational scale (radians).
    GRASP_RADIUS:
        Max EE-to-cube distance at which a close command grabs the cube.
    TARGET_RADIUS:
        Max cube-XY-to-target-XY distance that counts as success.
    GRIPPER_OPEN, GRIPPER_CLOSED:
        The two snap states of the gripper proxy.
    """

    # gymnasium declares `metadata` as an instance attribute on Env; subclasses
    # override at class-scope by convention, so the RUF012 warning does not apply.
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    # Per-step action scaling.
    MAX_LINEAR_STEP: float = 0.05  # metres per unit action
    MAX_ANGULAR_STEP: float = 0.1  # radians per unit action

    # Grasp / success tolerances.
    GRASP_RADIUS: float = 0.05
    TARGET_RADIUS: float = 0.05

    # Gripper snap states.
    GRIPPER_OPEN: float = 1.0
    GRIPPER_CLOSED: float = -1.0

    # Scene constants (must agree with assets/tabletop.xml).
    _TABLE_TOP_Z: float = 0.42  # = table.pos.z (0.4) + table_top.size.z (0.02)
    _CUBE_HALF: float = 0.025  # cube_geom.size along each axis
    _CUBE_REST_Z: float = _TABLE_TOP_Z + _CUBE_HALF
    _TABLE_HALF_X: float = 0.5
    _TABLE_HALF_Y: float = 0.5

    # Randomisation ranges (conservative — keep cube & target on the table).
    _CUBE_INIT_HALFRANGE: float = 0.15
    _TARGET_HALFRANGE: float = 0.2

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
    ) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {max_steps}")
        if n_substeps <= 0:
            raise ValueError(f"n_substeps must be positive; got {n_substeps}")

        self._max_steps = max_steps
        self._n_substeps = n_substeps

        # Load model once; reset uses mj_resetData + our own randomisation.
        self._model: Any = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
        self._data: Any = mujoco.MjData(self._model)

        # Cached mujoco indices (set in _cache_indices) — type: int | ndarray(int).
        self._cube_body_id: int = 0
        self._cube_geom_id: int = 0
        self._cube_qpos_adr: int = 0
        self._cube_qvel_adr: int = 0
        self._ee_body_id: int = 0
        self._ee_mocap_id: int = 0
        self._target_site_id: int = 0
        self._main_cam_id: int = 0
        self._cube_material_default_id: int = 0
        self._cube_material_alt_id: int = 0
        self._distractor_geom_ids: tuple[int, ...] = ()
        self._cache_indices()

        # Baseline snapshot of every model field a perturbation may touch.
        # Captured BEFORE any perturbation can run; restore_baseline() copies
        # these back into the live model. Stored as plain ndarray copies so
        # later mutations to model arrays cannot retroactively change them.
        self._baseline: dict[str, NDArray[np.float64]] = {}
        self._snapshot_baseline()

        # Pending per-episode perturbations. Set via set_perturbation BEFORE
        # the next reset(); reset() restores the baseline, re-randomises from
        # the seed, then applies these on top, then clears the dict so the
        # next episode starts fresh. See spec §5 task 4: "applies to the env
        # before reset()".
        self._pending_perturbations: dict[str, float] = {}

        # Per-episode state.
        self._rng: np.random.Generator = np.random.default_rng()
        self._step_count: int = 0
        self._grasped: bool = False
        self._gripper_state: float = self.GRIPPER_OPEN
        self._target_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._success: bool = False

        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
                "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            }
        )

    # ------------------------------------------------------------------ setup

    def _cache_indices(self) -> None:
        """Resolve name -> id lookups against the loaded model.

        Raises a clear error if any expected name (cube, ee, target,
        main camera, distractor_0..distractor_9) is missing from the MJCF.
        """
        m = self._model
        self._cube_body_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube"))
        self._cube_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"))
        if self._cube_geom_id < 0:
            raise RuntimeError("cube_geom not found in loaded MJCF")
        jnt_adr = int(m.body_jntadr[self._cube_body_id])
        self._cube_qpos_adr = int(m.jnt_qposadr[jnt_adr])
        self._cube_qvel_adr = int(m.jnt_dofadr[jnt_adr])
        self._ee_body_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee"))
        self._ee_mocap_id = int(m.body_mocapid[self._ee_body_id])
        if self._ee_mocap_id < 0:
            raise RuntimeError("ee body is not a mocap body in the loaded MJCF")
        self._target_site_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target"))
        self._main_cam_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "main"))
        if self._main_cam_id < 0:
            raise RuntimeError("camera 'main' not found in loaded MJCF")

        self._cube_material_default_id = int(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, _CUBE_MATERIAL_DEFAULT)
        )
        if self._cube_material_default_id < 0:
            raise RuntimeError(f"cube material '{_CUBE_MATERIAL_DEFAULT}' missing from MJCF")
        self._cube_material_alt_id = int(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, _CUBE_MATERIAL_ALT)
        )
        if self._cube_material_alt_id < 0:
            raise RuntimeError(f"cube material '{_CUBE_MATERIAL_ALT}' missing from MJCF")

        distractor_ids: list[int] = []
        for i in range(N_DISTRACTOR_SLOTS):
            geom_name = f"distractor_{i}_geom"
            gid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
            if gid < 0:
                raise RuntimeError(f"expected distractor slot '{geom_name}' missing from MJCF")
            distractor_ids.append(gid)
        self._distractor_geom_ids = tuple(distractor_ids)

    # -------------------------------------------------------------- baseline

    def _snapshot_baseline(self) -> None:
        """Cache the original value of every model field a perturbation can touch.

        Stored as ndarray copies (not views) so subsequent in-place writes
        to the live model can never retroactively modify the snapshot.
        Called once from ``__init__``, before any perturbation can run.
        """
        m = self._model
        cube_g = self._cube_geom_id
        d_ids = list(self._distractor_geom_ids)
        cam = self._main_cam_id
        snap: dict[str, NDArray[np.float64]] = {
            "light_diffuse_0": np.array(m.light_diffuse[0], dtype=np.float64).copy(),
            "cam_pos_main": np.array(m.cam_pos[cam], dtype=np.float64).copy(),
            # cube_geom_rgba is the per-geom override; with a material attached
            # MuJoCo loads it as 0.5,0.5,0.5,1 by default. We snapshot it so
            # restore_baseline puts it back if anyone ever overwrites it.
            "cube_geom_rgba": np.array(m.geom_rgba[cube_g], dtype=np.float64).copy(),
            # The actual colour switch for object_texture flips this matid.
            "cube_geom_matid": np.array([m.geom_matid[cube_g]], dtype=np.float64).copy(),
            "distractor_rgba": np.array([m.geom_rgba[g] for g in d_ids], dtype=np.float64).copy(),
            "distractor_contype": np.array(
                [m.geom_contype[g] for g in d_ids], dtype=np.float64
            ).copy(),
            "distractor_conaffinity": np.array(
                [m.geom_conaffinity[g] for g in d_ids], dtype=np.float64
            ).copy(),
        }
        self._baseline = snap

    def restore_baseline(self) -> None:
        """Copy every cached baseline value back into the live model.

        Called from :meth:`reset` *before* re-randomising, so per-episode
        state is clean and one episode's perturbation cannot leak into
        the next. Does NOT clear the pending-perturbation queue; that
        queue is the input to the next reset, not state to wipe.
        """
        m = self._model
        cube_g = self._cube_geom_id
        cam = self._main_cam_id
        d_ids = self._distractor_geom_ids
        m.light_diffuse[0] = self._baseline["light_diffuse_0"]
        m.cam_pos[cam] = self._baseline["cam_pos_main"]
        m.geom_rgba[cube_g] = self._baseline["cube_geom_rgba"]
        m.geom_matid[cube_g] = int(self._baseline["cube_geom_matid"][0])
        for i, gid in enumerate(d_ids):
            m.geom_rgba[gid] = self._baseline["distractor_rgba"][i]
            m.geom_contype[gid] = int(self._baseline["distractor_contype"][i])
            m.geom_conaffinity[gid] = int(self._baseline["distractor_conaffinity"][i])

    # ---------------------------------------------------------- perturbation

    def set_perturbation(self, name: str, value: float) -> None:
        """Queue a named scalar perturbation for the next :meth:`reset`.

        Per spec §5 task 4 ("applies to the env before reset()"), every
        axis is recorded into a pending dict and applied by ``reset`` AFTER
        ``restore_baseline`` and AFTER the seed-driven randomisation. This
        gives a single coherent contract for all 7 axes — pose axes
        override the random XY, model-state axes mutate the model.

        Raises:
            ValueError: if ``name`` is not a known axis or the value is out
                of range for an axis with a hard bound (currently only
                ``distractor_count``).
        """
        if name not in _KNOWN_AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        if name == "distractor_count":
            count = round(float(value))
            if count < 0 or count > N_DISTRACTOR_SLOTS:
                raise ValueError(
                    f"distractor_count must be in [0, {N_DISTRACTOR_SLOTS}]; got {count}"
                )
        self._pending_perturbations[name] = float(value)

    def _apply_pending_perturbations(self) -> None:
        """Apply every queued perturbation on top of the just-randomised state."""
        for name, value in self._pending_perturbations.items():
            self._apply_one_perturbation(name, value)

    def _apply_one_perturbation(self, name: str, value: float) -> None:
        """Mutate the model / cube qpos for a single axis. Internal."""
        m = self._model
        if name == "lighting_intensity":
            m.light_diffuse[0] = np.array([value, value, value], dtype=np.float64)
        elif name == "camera_offset_x":
            base = self._baseline["cam_pos_main"]
            m.cam_pos[self._main_cam_id] = np.array(
                [base[0] + value, base[1], base[2]], dtype=np.float64
            )
        elif name == "camera_offset_y":
            base = self._baseline["cam_pos_main"]
            m.cam_pos[self._main_cam_id] = np.array(
                [base[0], base[1] + value, base[2]], dtype=np.float64
            )
        elif name == "object_texture":
            choose_alt = round(float(value)) != 0
            mat_id = self._cube_material_alt_id if choose_alt else self._cube_material_default_id
            m.geom_matid[self._cube_geom_id] = mat_id
        elif name == "object_initial_pose_x":
            self._data.qpos[self._cube_qpos_adr + 0] = float(value)
        elif name == "object_initial_pose_y":
            self._data.qpos[self._cube_qpos_adr + 1] = float(value)
        elif name == "distractor_count":
            count = round(float(value))
            for i, gid in enumerate(self._distractor_geom_ids):
                if i < count:
                    base_rgba = self._baseline["distractor_rgba"][i].copy()
                    base_rgba[3] = 1.0
                    m.geom_rgba[gid] = base_rgba
                    m.geom_contype[gid] = 1
                    m.geom_conaffinity[gid] = 1
                else:
                    m.geom_rgba[gid] = self._baseline["distractor_rgba"][i]
                    m.geom_contype[gid] = int(self._baseline["distractor_contype"][i])
                    m.geom_conaffinity[gid] = int(self._baseline["distractor_conaffinity"][i])

    # --------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Deterministic reset.

        ``seed`` is the *only* entropy source. Calling ``reset(seed=s)``
        twice yields identical initial observations.
        """
        del options
        super().reset(seed=seed)
        # We manage our own RNG to avoid coupling to gymnasium's internal
        # stream ordering; the seed is the contract.
        self._rng = np.random.default_rng(seed)

        # Wipe any stale model perturbations from a prior episode BEFORE
        # we touch qpos / mocap / target. After this point the model is
        # bit-identical to construction time.
        self.restore_baseline()

        mujoco.mj_resetData(self._model, self._data)

        # Randomise cube initial XY (keep it on the table, above the surface).
        cube_xy = self._rng.uniform(
            low=-self._CUBE_INIT_HALFRANGE,
            high=self._CUBE_INIT_HALFRANGE,
            size=2,
        ).astype(np.float64)
        # Identity quaternion (wxyz = 1,0,0,0).
        qpos = self._data.qpos
        qpos[self._cube_qpos_adr + 0] = cube_xy[0]
        qpos[self._cube_qpos_adr + 1] = cube_xy[1]
        qpos[self._cube_qpos_adr + 2] = self._CUBE_REST_Z
        qpos[self._cube_qpos_adr + 3] = 1.0
        qpos[self._cube_qpos_adr + 4] = 0.0
        qpos[self._cube_qpos_adr + 5] = 0.0
        qpos[self._cube_qpos_adr + 6] = 0.0
        # Zero cube velocity.
        qvel = self._data.qvel
        for k in range(6):
            qvel[self._cube_qvel_adr + k] = 0.0

        # Randomise target XY (independent of cube).
        target_xy = self._rng.uniform(
            low=-self._TARGET_HALFRANGE,
            high=self._TARGET_HALFRANGE,
            size=2,
        ).astype(np.float64)
        self._target_pos = np.array(
            [target_xy[0], target_xy[1], self._TABLE_TOP_Z], dtype=np.float64
        )
        # The target is a site; move it by patching site_pos on the model
        # (sites don't have a qpos slot). This keeps renders / obs in sync.
        self._model.site_pos[self._target_site_id] = self._target_pos

        # Reset end-effector: hover above the cube start location.
        mocap_start = np.array(
            [cube_xy[0], cube_xy[1], self._CUBE_REST_Z + 0.15],
            dtype=np.float64,
        )
        self._data.mocap_pos[self._ee_mocap_id] = mocap_start
        self._data.mocap_quat[self._ee_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Apply any per-episode perturbations queued via set_perturbation.
        # Pose overrides clobber the random cube XY just written above.
        # All other axes mutate model fields. Clear the queue afterwards so
        # one episode's perturbation cannot silently re-apply on the next
        # reset.
        if self._pending_perturbations:
            self._apply_pending_perturbations()
            self._pending_perturbations = {}

        # Episode flags.
        self._step_count = 0
        self._grasped = False
        self._gripper_state = self.GRIPPER_OPEN
        self._success = False

        # Populate body_xpos / site_xpos before we read them for the obs.
        mujoco.mj_forward(self._model, self._data)

        return self._build_obs(), self._build_info()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Pipeline: clip → update mocap pose → update grasp state → ``mj_step``
        → if grasped, snap cube to EE → build obs/reward/flags.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"action must have shape (7,); got {a.shape}")
        a = np.clip(a, -1.0, 1.0).astype(np.float64, copy=False)

        # 1. Update mocap pose.
        self._apply_ee_command(a[0:3], a[3:6])

        # 2. Update gripper + grasp state (snap semantics).
        self._gripper_state = self.GRIPPER_OPEN if a[6] > 0.0 else self.GRIPPER_CLOSED
        self._update_grasp_state()

        # 3. Physics.
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # 4. If grasped, snap cube to EE pose (post-physics, overrides
        #    any contact-induced drift).
        if self._grasped:
            self._snap_cube_to_ee()
            # Keep site / xpos caches coherent.
            mujoco.mj_forward(self._model, self._data)

        # 5. Bookkeeping + success.
        self._step_count += 1
        cube_pos = np.array(self._data.xpos[self._cube_body_id], dtype=np.float64)
        if self._xy_distance(cube_pos, self._target_pos) <= self.TARGET_RADIUS:
            self._success = True

        terminated = self._success
        truncated = (not terminated) and self._step_count >= self._max_steps
        reward = -float(self._xy_distance(cube_pos, self._target_pos))
        return self._build_obs(), reward, terminated, truncated, self._build_info()

    def render(self) -> NDArray[np.uint8]:
        """Render an RGB frame (debug only — tests must not call this).

        ``mujoco.Renderer`` is imported lazily so module import stays
        headless-safe on CI workers without GL.
        """
        # Lazy import: constructing a Renderer opens a GL context.
        renderer = mujoco.Renderer(self._model)
        renderer.update_scene(self._data)
        pixels = renderer.render()
        renderer.close()
        return np.asarray(pixels, dtype=np.uint8)

    def close(self) -> None:
        """Drop references to the MuJoCo model / data.

        MuJoCo's Python bindings are GC-managed; dropping the last
        reference releases the underlying C structs.
        """
        self._data = None
        self._model = None

    # --------------------------------------------------------------- helpers

    def _apply_ee_command(
        self,
        linear: NDArray[np.float64],
        angular: NDArray[np.float64],
    ) -> None:
        """Translate + rotate the mocap body by small deltas."""
        cur_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        new_pos = cur_pos + linear * self.MAX_LINEAR_STEP
        self._data.mocap_pos[self._ee_mocap_id] = new_pos

        # Rotation: build a small axis-angle quat and multiply into the
        # current mocap_quat. Using MuJoCo's quat ops avoids Euler drift.
        axis_angle = angular * self.MAX_ANGULAR_STEP
        angle = float(np.linalg.norm(axis_angle))
        if angle > 0.0:
            axis = axis_angle / angle
            dq = np.zeros(4, dtype=np.float64)
            mujoco.mju_axisAngle2Quat(dq, axis, angle)
            cur_quat = np.array(self._data.mocap_quat[self._ee_mocap_id], dtype=np.float64)
            new_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(new_quat, dq, cur_quat)
            mujoco.mju_normalize4(new_quat)
            self._data.mocap_quat[self._ee_mocap_id] = new_quat

    def _update_grasp_state(self) -> None:
        """Snap grasp flag based on gripper command + EE-cube proximity."""
        if self._gripper_state == self.GRIPPER_OPEN:
            self._grasped = False
            return
        # Gripper commanded closed.
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        cube_pos = np.array(self._data.xpos[self._cube_body_id], dtype=np.float64)
        dist = float(np.linalg.norm(ee_pos - cube_pos))
        if dist <= self.GRASP_RADIUS:
            self._grasped = True
        # Already-grasped objects stay grasped while the gripper is closed,
        # even if physics briefly separates them; _snap_cube_to_ee then
        # pulls the cube back each step.

    def _snap_cube_to_ee(self) -> None:
        """Overwrite cube freejoint qpos/qvel to track the mocap body."""
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        ee_quat = np.array(self._data.mocap_quat[self._ee_mocap_id], dtype=np.float64)
        qpos = self._data.qpos
        qpos[self._cube_qpos_adr + 0] = ee_pos[0]
        qpos[self._cube_qpos_adr + 1] = ee_pos[1]
        qpos[self._cube_qpos_adr + 2] = ee_pos[2]
        qpos[self._cube_qpos_adr + 3] = ee_quat[0]
        qpos[self._cube_qpos_adr + 4] = ee_quat[1]
        qpos[self._cube_qpos_adr + 5] = ee_quat[2]
        qpos[self._cube_qpos_adr + 6] = ee_quat[3]
        qvel = self._data.qvel
        for k in range(6):
            qvel[self._cube_qvel_adr + k] = 0.0

    def _build_obs(self) -> dict[str, NDArray[np.float64]]:
        cube_pos = np.array(self._data.xpos[self._cube_body_id], dtype=np.float64)
        cube_quat = np.array(self._data.xquat[self._cube_body_id], dtype=np.float64)
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        gripper = np.array([self._gripper_state], dtype=np.float64)
        return {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
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
