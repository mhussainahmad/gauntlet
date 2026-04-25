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

import contextlib
from pathlib import Path
from typing import Any, ClassVar, Final

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gauntlet.env.base import CameraSpec

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

    # Phase 2 Task 5 step 4 — GauntletEnv Protocol introspection points.
    # AXIS_NAMES is the canonical membership set previously held in a module
    # local ``_KNOWN_AXIS_NAMES``; promoted to ClassVar so the Suite loader
    # can query backend-specific axis support without constructing the env.
    # VISUAL_ONLY_AXES is empty on MuJoCo because the renderer consumes the
    # cosmetic axes via ``render()`` (see RFC-005 §6.2); state-only backends
    # (first-cut PyBullet) declare the cosmetic subset instead.
    AXIS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "lighting_intensity",
            "camera_offset_x",
            "camera_offset_y",
            "object_texture",
            "object_initial_pose_x",
            "object_initial_pose_y",
            "distractor_count",
            # B-32 — initial-state OOD shift axis (LIBERO-PRO/Plus framing).
            # See ``set_initial_state_ood_prior`` for prior configuration.
            "initial_state_ood",
        }
    )
    VISUAL_ONLY_AXES: ClassVar[frozenset[str]] = frozenset()

    # B-32 — default OOD prior. The training-time cube initial-XY
    # distribution is uniform on ``[-_CUBE_INIT_HALFRANGE, _CUBE_INIT_HALFRANGE]``
    # for x and y (z fixed at the rest height), so the in-distribution
    # mean is the table centre and the per-axis std is
    # ``halfrange / sqrt(3)`` (variance of the uniform). Z std is 0 by
    # default — the OOD axis perturbs the table-plane pose, not height.
    # User-supplied priors override these via ``set_initial_state_ood_prior``
    # (or the YAML ``prior_mean`` / ``prior_std`` axis fields once wired).
    _OOD_DEFAULT_PRIOR_MEAN: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.0)

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

    # B-32 — std of the training initial-XY distribution. Uniform on
    # ``[-_CUBE_INIT_HALFRANGE, _CUBE_INIT_HALFRANGE]`` has variance
    # ``(2 * halfrange)^2 / 12`` so std = ``halfrange / sqrt(3)``. Z std
    # defaults to 0 because the rest height is constant during training.
    _OOD_DEFAULT_PRIOR_STD: ClassVar[tuple[float, float, float]] = (
        _CUBE_INIT_HALFRANGE / 1.7320508075688772,
        _CUBE_INIT_HALFRANGE / 1.7320508075688772,
        0.0,
    )

    def __init__(
        self,
        *,
        max_steps: int = 200,
        n_substeps: int = 5,
        render_in_obs: bool = False,
        render_size: tuple[int, int] = (224, 224),
        cameras: list[CameraSpec] | None = None,
    ) -> None:
        """Construct the env, optionally injecting multi-camera scene elements.

        ``render_in_obs`` adds the legacy single ``main`` camera frame to
        ``obs["image"]``. ``cameras`` overrides that path: each spec is
        injected into the MJCF as an extra ``<camera>`` and exposed under
        ``obs["images"][name]`` with the first spec aliased to ``obs["image"]``
        for downstream parity. ``cameras=[]`` is treated as ``None``.
        """
        super().__init__()
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

        # ``cameras=[]`` is treated identically to ``cameras=None`` per
        # RFC §2 — the single-cam default path is sacred. Validation +
        # the multi-cam scene-graph injection happen further down.
        self._cameras: tuple[CameraSpec, ...] = tuple(cameras) if cameras else ()
        self._multi_cam_enabled: bool = bool(self._cameras)
        if self._multi_cam_enabled:
            _validate_camera_specs(self._cameras)

        self._max_steps = max_steps
        self._n_substeps = n_substeps
        # When multi-cam is on, ``cameras`` takes precedence over the
        # legacy ``render_in_obs`` flag (RFC §2: legacy kwargs are
        # silently ignored). The ``_render_in_obs`` shadow stays False
        # so the legacy single-camera ``obs["image"]`` codepath is
        # disabled and the multi-cam codepath owns rendering.
        self._render_in_obs = render_in_obs and not self._multi_cam_enabled
        self._render_size: tuple[int, int] = (int(render_size[0]), int(render_size[1]))
        # Cached offscreen renderer, built lazily on first reset so import /
        # construction stays headless-safe. Reused for the env lifetime;
        # released in ``close``. See RFC Phase 2 Task 1 §5.
        self._obs_renderer: Any | None = None
        # Per-CameraSpec offscreen renderer cache. Each entry's GL context
        # is built lazily on first ``_render_obs_images`` call (matching
        # the single-cam laziness contract) and released in ``close``.
        # Keyed by ``CameraSpec.name`` so duplicate-name validation in
        # ``_validate_camera_specs`` is what protects this dict. Cam ids
        # for the injected ``<camera>`` elements live in ``_extra_cam_ids``.
        self._extra_renderers: dict[str, Any] = {}
        self._extra_cam_ids: dict[str, int] = {}

        # Load model once; reset uses mj_resetData + our own randomisation.
        # When multi-cam is enabled we inject one ``<camera>`` per spec
        # into the MJCF before parsing, then resolve their ids in
        # ``_cache_indices``. Reading the asset as a string + injecting
        # extra cameras is cheap (one regex sub) and keeps the legacy
        # ``main`` camera available so ``camera_offset_*`` perturbations
        # and the single-cam render path keep working.
        if self._multi_cam_enabled:
            xml_str = _inject_camera_elements(_ASSET_PATH.read_text(), self._cameras)
            self._model: Any = mujoco.MjModel.from_xml_string(xml_str)
        else:
            self._model = mujoco.MjModel.from_xml_path(str(_ASSET_PATH))
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

        # B-32 — OOD prior for the ``initial_state_ood`` axis. Tuples of
        # length 3 (x, y, z). Defaults to the env's training-time
        # initial-XY distribution; user overrides via
        # :meth:`set_initial_state_ood_prior` (or, post-wiring, via the
        # YAML ``prior_mean`` / ``prior_std`` axis fields).
        self._ood_prior_mean: tuple[float, float, float] = type(self)._OOD_DEFAULT_PRIOR_MEAN
        self._ood_prior_std: tuple[float, float, float] = type(self)._OOD_DEFAULT_PRIOR_STD
        # Snapshot of the seed last passed to :meth:`reset`. Used to
        # build a deterministic, isolated sub-stream for OOD sign draws
        # so that adding the ``initial_state_ood`` axis to a queued set
        # of perturbations cannot perturb the main ``self._rng`` stream
        # (and thus cannot shift the random target / cube draws other
        # axes depend on).
        self._last_reset_seed: int | None = None

        # Per-episode state.
        self._rng: np.random.Generator = np.random.default_rng()
        self._step_count: int = 0
        self._grasped: bool = False
        self._gripper_state: float = self.GRIPPER_OPEN
        self._target_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._success: bool = False

        # B-21 actuator telemetry: per-control-step scratch slots
        # populated inside :meth:`step` (one control step = one
        # ``n_substeps``-long ``mj_step`` block). Surfaced through
        # :meth:`_build_info` as ``actuator_energy_delta`` (joule-
        # equivalent of ``|tau . qvel| * dt`` summed over the substeps)
        # and ``actuator_torque_norm`` (L2 norm of ``actuator_force`` at
        # the end of the block). Both are ``None`` after :meth:`reset`
        # so the worker's accumulator only ticks on real control steps.
        self._last_step_actuator_energy: float | None = None
        self._last_step_torque_norm: float | None = None

        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        obs_spaces: dict[str, spaces.Space[Any]] = {
            "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "cube_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "gripper": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
        if self._render_in_obs:
            h, w = self._render_size
            # uint8 RGB image from the cached offscreen renderer. Only present
            # when render_in_obs=True so the default path stays byte-identical
            # to Phase 1.
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        if self._multi_cam_enabled:
            # Per-camera Box spaces under a Dict subspace. The legacy
            # ``image`` key is also advertised at the top level, sized to
            # the first spec, so consumers that only know about
            # ``obs["image"]`` (e.g. the runner's video recorder at
            # ``runner/worker.py:417``) keep working. RFC §2.
            per_cam: dict[str, spaces.Space[Any]] = {}
            for spec in self._cameras:
                ch, cw = spec.size
                per_cam[spec.name] = spaces.Box(low=0, high=255, shape=(ch, cw, 3), dtype=np.uint8)
            obs_spaces["images"] = spaces.Dict(per_cam)
            first_h, first_w = self._cameras[0].size
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(first_h, first_w, 3), dtype=np.uint8
            )
        self.observation_space: spaces.Dict = spaces.Dict(obs_spaces)

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

        # Resolve injected multi-cam ids (no-op when ``cameras=None``).
        # Each injected element is named ``cam_<spec.name>`` to avoid
        # colliding with the existing ``main`` camera.
        if self._multi_cam_enabled:
            self._extra_cam_ids = {}
            for spec in self._cameras:
                cam_name = f"cam_{spec.name}"
                cid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name))
                if cid < 0:
                    raise RuntimeError(
                        f"injected camera '{cam_name}' missing from MJCF "
                        f"— _inject_camera_elements bug"
                    )
                self._extra_cam_ids[spec.name] = cid

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
        gives a single coherent contract for all axes — pose axes
        override the random XY, model-state axes mutate the model. The
        ``initial_state_ood`` axis (B-32) reuses the pose-override path:
        the queued value is the unitless sigma multiplier ``V`` and the
        env converts it to an XY pose via the configured prior.

        Raises:
            ValueError: if ``name`` is not a known axis or the value is out
                of range for an axis with a hard bound (currently only
                ``distractor_count``).
        """
        if name not in type(self).AXIS_NAMES:
            raise ValueError(f"unknown perturbation axis: {name!r}")
        if name == "distractor_count":
            count = round(float(value))
            if count < 0 or count > N_DISTRACTOR_SLOTS:
                raise ValueError(
                    f"distractor_count must be in [0, {N_DISTRACTOR_SLOTS}]; got {count}"
                )
        if name == "initial_state_ood" and float(value) < 0:
            # Negative sigma multipliers are nonsense — the magnitude
            # encodes "how many sigma out", with the sign drawn per-dim
            # from the env seed at apply time.
            raise ValueError(f"initial_state_ood: sigma multiplier must be >= 0; got {value}")
        self._pending_perturbations[name] = float(value)

    def set_initial_state_ood_prior(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        """Configure the training-distribution prior for the OOD axis (B-32).

        ``mean`` and ``std`` are 3-tuples ``(x, y, z)`` describing the
        training initial-pose distribution. The ``initial_state_ood``
        axis interprets a queued sigma multiplier ``V`` as a pose of
        ``mean + V * std * sign_per_dim`` (per LIBERO-PRO / LIBERO-Plus),
        where ``sign_per_dim`` is drawn from a deterministic sub-stream
        seeded by the env reset seed (so every (seed, V) pair is
        reproducible). Each std entry must be ``>= 0``.

        Defaults match the env's uniform training XY distribution; call
        this method only when you have a different (e.g. auto-fit-from-
        reference-run) prior.
        """
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(
                f"initial_state_ood prior: mean and std must each be length 3; "
                f"got len(mean)={len(mean)}, len(std)={len(std)}"
            )
        if any(float(s) < 0 for s in std):
            raise ValueError(f"initial_state_ood prior: std entries must be >= 0; got std={std}")
        self._ood_prior_mean = (float(mean[0]), float(mean[1]), float(mean[2]))
        self._ood_prior_std = (float(std[0]), float(std[1]), float(std[2]))

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
        elif name == "initial_state_ood":
            # B-32 — sigma multiplier ``V`` mapped to a per-dim pose
            # ``prior_mean + V * prior_std * sign_per_dim``. Sign is
            # drawn from a deterministic sub-stream so the main
            # ``self._rng`` stream (which other axes / target draws
            # consume) is not perturbed by the presence of this axis.
            sigma = float(value)
            mean = self._ood_prior_mean
            std = self._ood_prior_std
            # Spawn an isolated child stream from the current seed via
            # SeedSequence; falls back to a constant seed (0) when the
            # caller passed seed=None to reset(). Either way the output
            # is reproducible from the env's contract (``reset(seed=s)``
            # is the only entropy source).
            base_seed = self._last_reset_seed if self._last_reset_seed is not None else 0
            sign_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 0xB32]))
            signs = np.where(sign_rng.random(3) < 0.5, -1.0, 1.0).astype(np.float64)
            pose = (
                mean[0] + sigma * std[0] * float(signs[0]),
                mean[1] + sigma * std[1] * float(signs[1]),
                mean[2] + sigma * std[2] * float(signs[2]),
            )
            qpos = self._data.qpos
            qpos[self._cube_qpos_adr + 0] = pose[0]
            qpos[self._cube_qpos_adr + 1] = pose[1]
            # The cube is a free joint with three positional DoF; z is
            # mutable too, but the rest height keeps the cube on the
            # table when the default std[z]=0 collapses the offset.
            qpos[self._cube_qpos_adr + 2] = self._CUBE_REST_Z + pose[2]
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
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Deterministic reset.

        ``seed`` is the *only* entropy source. Calling ``reset(seed=s)``
        twice yields identical initial observations.
        """
        del options
        super().reset(seed=seed)
        # We manage our own RNG to avoid coupling to gymnasium's internal
        # stream ordering; the seed is the contract.
        self._rng = np.random.default_rng(seed)
        # Snapshot the seed so the ``initial_state_ood`` apply branch
        # can spawn an isolated child stream for sign draws without
        # consuming entropy from ``self._rng``.
        self._last_reset_seed = seed

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

        # Wipe per-control-step actuator telemetry: a freshly reset env
        # has not driven the actuators yet, so :meth:`_build_info`
        # MUST NOT report stale energy / torque from a prior episode.
        # The ``None`` sentinel is also the worker's signal to skip
        # the accumulator on the very first reset-time obs.
        self._last_step_actuator_energy = None
        self._last_step_torque_norm = None

        # Populate body_xpos / site_xpos before we read them for the obs.
        mujoco.mj_forward(self._model, self._data)

        return self._build_obs(), self._build_info()

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
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

        # 3. Physics. We integrate B-21 actuator-energy telemetry over
        #    the substep loop (energy ~ |tau . qvel| * dt accumulated
        #    per substep, since both ``actuator_force`` and ``qvel``
        #    evolve inside the loop). The torque-norm sample is taken
        #    after the final substep — that's the value the policy's
        #    next observation reflects, and matches what a real
        #    safety monitor would log.
        #
        #    The current Tabletop MJCF is mocap-driven (``model.nu == 0``,
        #    no ``<actuator>`` elements), so ``qfrc_actuator`` /
        #    ``actuator_force`` are identically zero / empty. Reporting
        #    energy=0.0 in that case would be actively misleading next
        #    to a high-failure cluster row ("the policy commanded zero
        #    force") so we leave the per-step slots ``None``; the worker
        #    sees absent ``info["actuator_*"]`` keys and the Episode
        #    carries ``None``, same as a non-MuJoCo backend. A future
        #    joint-torque MJCF asset will surface real values without
        #    further wiring.
        if self._model.nu > 0:
            dt_substep = float(self._model.opt.timestep)
            energy_accum = 0.0
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
                # ``qfrc_actuator`` is the actuator-generated generalized
                # force in qvel space (same dimension as ``qvel``), so
                # the element-wise product and absolute-value sum is
                # the well-defined |power| over the DOFs without an
                # actuator-to-DOF index lookup.
                qfrc = np.asarray(self._data.qfrc_actuator, dtype=np.float64)
                qvel = np.asarray(self._data.qvel, dtype=np.float64)
                energy_accum += float(np.sum(np.abs(qfrc * qvel))) * dt_substep
            # Per-actuator force vector at the end of the substep block
            # — the operationally relevant "what torque did the policy
            # just command" sample. Always >= 0; the L2 norm is
            # invariant to actuator ordering, which keeps cross-run
            # comparisons stable even if the MJCF reorders <actuator>
            # elements.
            actuator_force = np.asarray(self._data.actuator_force, dtype=np.float64)
            self._last_step_actuator_energy = energy_accum
            self._last_step_torque_norm = float(np.linalg.norm(actuator_force))
        else:
            for _ in range(self._n_substeps):
                mujoco.mj_step(self._model, self._data)
            # Telemetry deliberately not populated — see comment above.
            self._last_step_actuator_energy = None
            self._last_step_torque_norm = None

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
        reference releases the underlying C structs. The cached obs
        renderer is explicitly closed first so its GL context is freed
        on the worker thread that built it (leaking would bite when the
        runner spins up ``n_workers >= 2``).
        """
        if self._obs_renderer is not None:
            # Renderer teardown is best-effort; GL contexts can whine on close
            # but we're tearing everything down anyway.
            with contextlib.suppress(Exception):
                self._obs_renderer.close()
            self._obs_renderer = None
        # Same best-effort teardown for the per-CameraSpec renderers; each
        # carries its own GL context that needs to be released on the
        # worker that built it (matters when n_workers >= 2).
        for renderer in self._extra_renderers.values():
            with contextlib.suppress(Exception):
                renderer.close()
        self._extra_renderers = {}
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

    def _build_obs(self) -> dict[str, NDArray[Any]]:
        cube_pos = np.array(self._data.xpos[self._cube_body_id], dtype=np.float64)
        cube_quat = np.array(self._data.xquat[self._cube_body_id], dtype=np.float64)
        ee_pos = np.array(self._data.mocap_pos[self._ee_mocap_id], dtype=np.float64)
        gripper = np.array([self._gripper_state], dtype=np.float64)
        obs: dict[str, NDArray[Any]] = {
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "ee_pos": ee_pos,
            "gripper": gripper,
            "target_pos": self._target_pos.copy(),
        }
        if self._render_in_obs:
            obs["image"] = self._render_obs_image()
        if self._multi_cam_enabled:
            images = self._render_obs_images()
            obs["images"] = images  # type: ignore[assignment]
            # Legacy single-camera alias — first spec wins. Defensive
            # copy: without ``.copy()`` ``obs["image"]`` and
            # ``obs["images"][first]`` would be the same ndarray, and
            # any consumer that in-place mutates ``obs["image"]`` (a
            # normalisation pass, an aliasing tensor conversion, etc.)
            # would silently corrupt the per-cam dict. The runner's
            # video buffer at runner/worker.py:417 already takes its
            # own .copy() but not every downstream is so disciplined.
            first_name = self._cameras[0].name
            obs["image"] = images[first_name].copy()
        return obs

    def _render_obs_image(self) -> NDArray[np.uint8]:
        """Render the main camera into a uint8 (H, W, 3) array.

        The offscreen ``mujoco.Renderer`` is constructed on first call and
        cached on the instance; opening a GL context is expensive and many
        callers will build the env purely to query observation shapes. Uses
        the existing ``"main"`` camera — no new camera is introduced.
        """
        if self._obs_renderer is None:
            h, w = self._render_size
            self._obs_renderer = mujoco.Renderer(self._model, height=h, width=w)
        self._obs_renderer.update_scene(self._data, camera=self._main_cam_id)
        pixels = self._obs_renderer.render()
        return np.asarray(pixels, dtype=np.uint8)

    def _render_obs_images(self) -> dict[str, NDArray[np.uint8]]:
        """Render every CameraSpec into ``{name: uint8 (H, W, 3)}``.

        One ``mujoco.Renderer`` per spec is cached lazily on first call —
        each spec carries its own ``size``, so a single shared renderer
        cannot satisfy them all. Cached renderers are reused for the env
        lifetime and torn down in ``close``. Output frames are
        defensively copied via ``np.asarray(..., dtype=np.uint8)`` to
        decouple from the renderer's internal pixel buffer (mirrors the
        single-cam path's contract).
        """
        out: dict[str, NDArray[np.uint8]] = {}
        for spec in self._cameras:
            renderer = self._extra_renderers.get(spec.name)
            if renderer is None:
                h, w = spec.size
                renderer = mujoco.Renderer(self._model, height=h, width=w)
                self._extra_renderers[spec.name] = renderer
            cam_id = self._extra_cam_ids[spec.name]
            renderer.update_scene(self._data, camera=cam_id)
            pixels = renderer.render()
            out[spec.name] = np.asarray(pixels, dtype=np.uint8)
        return out

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "success": self._success,
            "grasped": self._grasped,
            "step": self._step_count,
        }
        # B-21: only emit the actuator-telemetry keys when we actually
        # have a fresh sample (i.e. inside the post-:meth:`step` info,
        # never inside the post-:meth:`reset` info). The worker reads
        # these keys with ``info.get(...)`` and only ticks its
        # accumulator when both are present, so absence == "this is a
        # reset, not a control step".
        if self._last_step_actuator_energy is not None:
            info["actuator_energy_delta"] = self._last_step_actuator_energy
        if self._last_step_torque_norm is not None:
            info["actuator_torque_norm"] = self._last_step_torque_norm
        return info

    @staticmethod
    def _xy_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))


# ---------------------------------------------------------- multi-cam helpers


def _validate_camera_specs(specs: tuple[CameraSpec, ...]) -> None:
    """Reject invalid CameraSpec lists at construction time.

    See ``docs/polish-exploration-multi-camera.md`` §2 — this enforces
    the public contract pinned in the RFC: non-empty unique names,
    positive image dimensions. The pose tuple is structurally typed
    by NamedTuple; no range check applies.
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


def _inject_camera_elements(xml_str: str, specs: tuple[CameraSpec, ...]) -> str:
    """Inject one ``<camera>`` element per spec into the MJCF string.

    The injected elements live just before the closing ``</worldbody>``
    tag and are named ``cam_<spec.name>`` to avoid colliding with the
    existing ``main`` camera. Pose is written as MuJoCo XYZ Euler in
    radians (RFC §2). The legacy single-camera codepath continues to
    use the ``main`` camera so ``camera_offset_*`` perturbations are
    untouched.

    Raises:
        RuntimeError: if the input XML lacks a ``</worldbody>`` close tag.
    """
    close_tag = "</worldbody>"
    if close_tag not in xml_str:
        raise RuntimeError("MJCF asset is missing </worldbody>; cannot inject cameras")
    inserts: list[str] = []
    for spec in specs:
        x, y, z, rx, ry, rz = spec.pose
        inserts.append(
            f'    <camera name="cam_{spec.name}" pos="{x} {y} {z}" euler="{rx} {ry} {rz}"/>'
        )
    block = "\n".join(inserts)
    return xml_str.replace(close_tag, f"{block}\n  {close_tag}", 1)
