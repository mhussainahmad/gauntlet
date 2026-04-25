"""Per-worker rollout executor.

The Runner pushes :class:`WorkItem` records into a multiprocessing
:class:`multiprocessing.pool.Pool`; each worker turns one item into one
:class:`Episode` via :func:`run_work_item`.

Process lifecycle (matches Pin 3 in the task brief):

* ``_pool_initializer`` runs once per worker on pool startup. It builds
  the env (``env_factory()``) and stashes it in this module's
  ``_WORKER_STATE`` global. **MJCF load happens here, exactly once per
  worker, never per episode.** MuJoCo's :class:`MjModel` is not picklable,
  so the env must be constructed inside the worker, never passed in.
* ``run_work_item`` is the per-item entry point. It restores the
  baseline, applies perturbations, builds a fresh policy via
  ``policy_factory()`` (also stashed in the worker globals), seeds it,
  resets the env, then drives the rollout to completion.

The same :func:`execute_one` core is also called directly by the Runner's
in-process fast path (``n_workers == 1``) and by :mod:`gauntlet.replay`,
so every code path produces bit-identical Episode objects from the
same inputs.

Determinism contract
--------------------
* ``WorkItem.episode_seq`` is a :class:`numpy.random.SeedSequence` —
  picklable, deterministic, derived from the master seed via
  ``master.spawn(n_cells)[cell_idx].spawn(episodes_per_cell)[ep_idx]``.
* The env seed handed to :meth:`GauntletEnv.reset` is
  ``int(episode_seq.generate_state(1, dtype=np.uint32)[0])`` — a stable
  uint32 unique to (master, cell_idx, ep_idx). This is also recorded as
  :attr:`Episode.seed` so the rollout can be reproduced.
* The policy RNG is ``np.random.default_rng(episode_seq)`` — a separate
  deterministic stream from the same SeedSequence node, so policy
  randomness is decorrelated from env randomness but equally reproducible.

Why two streams from one node? The pin asks for
``policy_rng = np.random.default_rng(episode_seq)``; we keep that exact
call. The env, however, takes an int (gymnasium's ``reset(seed=...)``
contract), so we derive a uint32 from the same node. Empirically,
``SeedSequence.spawn(...)[i].entropy`` is shared across siblings — it
echoes the master seed — and would make every episode reset to the
same state if used as the env seed. ``generate_state`` is the canonical
way to extract a per-spawn-unique scalar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.policy.base import Policy, ResettablePolicy
from gauntlet.runner.episode import Episode
from gauntlet.runner.video import VideoWriter, video_path_for

__all__ = [
    "VideoConfig",
    "WorkItem",
    "WorkerInitArgs",
    "execute_one",
    "extract_env_seed",
    "pool_initializer",
    "run_work_item",
    "trajectory_path_for",
    "write_trajectory_npz",
]


# ----------------------------------------------------------------------------
# Video recording configuration — opt-in side-channel.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoConfig:
    """Per-run video recording configuration.

    All fields are picklable so the Runner can ship one ``VideoConfig``
    across the spawn boundary into every worker. ``video_dir`` is
    resolved by the Runner (relative paths are anchored to the run
    output dir before serialisation).

    Attributes:
        video_dir: Directory for per-episode MP4s. The Runner mkdirs it
            once on entry.
        fps: Output framerate handed to :class:`VideoWriter`.
        record_only_failures: When True, only ``success=False``
            episodes get an MP4. The frame buffer is built either way
            because success is unknown until the rollout ends; only the
            *write* is suppressed.
        relative_to: When set, ``Episode.video_path`` is the MP4 path
            relative to this anchor. Defaults to ``video_dir.parent``,
            which makes the embed-from-HTML path
            ``"videos/episode_*.mp4"`` work when the HTML lives next to
            the videos directory.
    """

    video_dir: Path
    fps: int = 30
    record_only_failures: bool = False
    relative_to: Path | None = None


# ----------------------------------------------------------------------------
# Work item — the unit of cross-process work.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One (cell, episode) rollout request.

    All fields are picklable. ``episode_seq`` is a
    :class:`numpy.random.SeedSequence`; numpy guarantees it round-trips
    through pickle (verified at the spec design stage).

    Attributes:
        suite_name: Echoed from :class:`gauntlet.suite.Suite.name`.
        cell_index: :attr:`gauntlet.suite.SuiteCell.index`.
        episode_index: Zero-based episode within the cell.
        perturbation_values: ``{axis_name: scalar}`` — frozen view of the
            cell's values; will be applied via
            :meth:`GauntletEnv.set_perturbation`.
        episode_seq: SeedSequence node for this episode. The worker
            derives both the env seed (uint32) and the policy RNG from
            this single node so both streams are reproducible.
        master_seed: Echo of the master seed (or the auto-generated
            entropy when no seed was supplied). Recorded in the resulting
            Episode's ``metadata["master_seed"]`` so even None-seeded
            runs can be reproduced from the report.
        n_cells: Total number of cells in the originating suite at run
            time. Recorded in :attr:`Episode.metadata["n_cells"]` so the
            replay tool can reconstruct the exact ``SeedSequence.spawn``
            tree the worker used, even if the suite YAML is edited
            between run and replay.
        episodes_per_cell: Echo of :attr:`gauntlet.suite.Suite.episodes_per_cell`
            at run time. Recorded in
            :attr:`Episode.metadata["episodes_per_cell"]` for the same
            reason as ``n_cells`` — the replay spawn tree is path-
            dependent on ``(n_cells, episodes_per_cell)`` and must be
            reconstructable from the Episode alone.
    """

    suite_name: str
    cell_index: int
    episode_index: int
    perturbation_values: dict[str, float]
    episode_seq: np.random.SeedSequence
    master_seed: int
    n_cells: int
    episodes_per_cell: int


# ----------------------------------------------------------------------------
# Worker process state.
#
# The pool initializer constructs the env + caches the policy factory
# in this module-level dict. Each worker process gets its own copy of
# the module (spawn start method = fresh interpreter), so concurrent
# workers do not share env state.
# ----------------------------------------------------------------------------


@dataclass
class WorkerInitArgs:
    """Init args for :func:`pool_initializer`.

    Both fields are zero-arg callables to side-step the pickle boundary:

    * ``env_factory`` — builds any :class:`GauntletEnv` implementation
      (MuJoCo ``TabletopEnv``, PyBullet ``PyBulletTabletopEnv``, or a
      third-party registered backend). Per-backend invariants (e.g.
      MuJoCo's non-picklable MjModel) still push construction into the
      worker; the Protocol does not constrain them further.
    * ``policy_factory`` — builds a fresh :class:`Policy` per episode.
      Stashed once and called per-item; future torch/GPU policies that
      cannot be pickled (e.g. HuggingFacePolicy) can therefore live
      entirely inside the worker.
    """

    env_factory: Callable[[], GauntletEnv]
    policy_factory: Callable[[], Policy]
    # Optional per-episode trajectory dump directory. When ``None`` (the
    # default) the worker writes zero bytes — the Runner's behaviour is
    # byte-identical to Phase 1. When a Path is supplied the worker emits
    # one NPZ per episode following the ``cell_NNNN_ep_NNNN.npz`` scheme.
    trajectory_dir: Path | None = None
    # Optional per-episode MP4 video config. ``None`` means "no video"
    # — the runner stays byte-identical to the pre-PR behaviour. When
    # set, every worker accumulates ``obs["image"]`` per step and
    # writes one MP4 per episode (or only failures when
    # ``record_only_failures=True``). Requires the env to expose
    # ``obs["image"]`` (i.e. ``render_in_obs=True``); enforced at the
    # first reset.
    video_config: VideoConfig | None = None


# Worker-local cache shape. ``total=False`` because pre-init reads
# (and tests that bypass the pool path) see an empty dict; the
# initializer sets all four keys atomically. The TypedDict replaces
# the previous ``dict[str, Any]`` so consumers get the precise type
# of each ``_WORKER_STATE.get(...)`` lookup; runtime semantics are
# unchanged (identical key set, identical ``dict.get(...)``-with-
# ``None``-fallback read pattern in ``run_work_item``).
class _WorkerState(TypedDict, total=False):
    env: GauntletEnv
    policy_factory: Callable[[], Policy]
    trajectory_dir: Path | None
    video_config: VideoConfig | None


# Worker-local cache. Populated by ``pool_initializer``; read by
# ``run_work_item``.
_WORKER_STATE: _WorkerState = {}


def pool_initializer(args: WorkerInitArgs) -> None:
    """Run once per worker process at pool startup.

    Constructs one env per worker (whatever concrete
    :class:`GauntletEnv` the ``env_factory`` returns) and caches the
    policy factory. Subsequent items reuse the env via
    :meth:`GauntletEnv.restore_baseline` — the Protocol contract
    guarantees the backend returns to a ``__init__``-equivalent state
    between episodes without reconstructing the scene.

    Also caches ``trajectory_dir``; when non-``None`` each per-item call
    writes one NPZ sidecar after Episode construction. The attribute is
    always present in ``_WORKER_STATE`` (set to ``None`` when disabled)
    so :func:`run_work_item` does not have to branch on ``KeyError``.
    """
    env = args.env_factory()
    _WORKER_STATE["env"] = env
    _WORKER_STATE["policy_factory"] = args.policy_factory
    _WORKER_STATE["trajectory_dir"] = args.trajectory_dir
    _WORKER_STATE["video_config"] = args.video_config


# ----------------------------------------------------------------------------
# Per-item work.
# ----------------------------------------------------------------------------


def extract_env_seed(seq: np.random.SeedSequence) -> int:
    """Derive a stable uint32 env seed from a :class:`SeedSequence` node.

    ``SeedSequence.entropy`` is shared across siblings spawned from the
    same parent (it echoes the parent's seed), so it is unsuitable as a
    per-episode env seed. ``generate_state`` produces a deterministic
    per-spawn-unique scalar — the canonical way to bridge from a
    SeedSequence node to gymnasium's ``int``-typed seed.
    """
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def trajectory_path_for(trajectory_dir: Path, cell_index: int, episode_index: int) -> Path:
    """Return the canonical trajectory NPZ path for a (cell, episode).

    Deterministic: the same ``(cell_index, episode_index)`` always maps
    to the same path, so reruns overwrite in place and ``monitor score``
    can match NPZs to Episode rows without a separate index. The cell /
    episode indices are themselves deterministic from ``(master_seed,
    perturbation_config)`` via the Suite's grid enumeration, so the
    filename encodes a stable identity.

    The ``:04d`` width is generous: a 10k-cell x 10k-episode suite still
    fits. Real suites are many orders of magnitude smaller.
    """
    return trajectory_dir / f"cell_{cell_index:04d}_ep_{episode_index:04d}.npz"


def write_trajectory_npz(
    path: Path,
    *,
    obs_arrays: dict[str, NDArray[np.float64]],
    actions: NDArray[np.float64],
    seed: int,
    cell_index: int,
    episode_index: int,
) -> None:
    """Write one episode's trajectory to an NPZ sidecar.

    Contents (see RFC §6):

    * ``obs_<key>`` — one array per :meth:`GauntletEnv._build_obs` key,
      stacked over timesteps. Shape ``(T, *obs_shape)`` where ``T`` is
      the number of env steps. Always present.
    * ``action`` — shape ``(T, 7)`` float64.
    * ``seed`` — scalar int64; echoes :attr:`Episode.seed`.
    * ``cell_index`` / ``episode_index`` — scalar int64; the
      filename-matching cross-check ``monitor score`` uses.

    ``savez_compressed`` trades ~2x CPU for ~3x smaller proprio files —
    disk is almost always the bottleneck on a long reference sweep.

    The parent directory is created if missing so callers don't need to
    pre-mkdir it per-episode (the Runner creates it once up front; this
    is belt-and-braces for callers that bypass the Runner path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, NDArray[Any]] = {f"obs_{k}": v for k, v in obs_arrays.items()}
    payload["action"] = actions
    # Scalar identity arrays — RFC §6 "defensive scalar-arrays
    # cross-check on read". int64 covers every realistic master_seed /
    # index; ``np.asarray(... dtype=int64)`` wraps Python ints as 0-D
    # ndarrays so ``int(npz["seed"])`` on the reader side round-trips.
    payload["seed"] = np.asarray(seed, dtype=np.int64)
    payload["cell_index"] = np.asarray(cell_index, dtype=np.int64)
    payload["episode_index"] = np.asarray(episode_index, dtype=np.int64)
    # numpy 2.x stubs model ``**kwds`` as the ``allow_pickle`` bool rather
    # than further arrays; at runtime the arrays are accepted as kwargs
    # verbatim (see ``np.savez_compressed`` docstring). We cast the
    # savez_compressed callable to sidestep the stub mismatch — the
    # runtime contract is stable and documented.
    _savez: Callable[..., None] = np.savez_compressed
    _savez(path, **payload)


def execute_one(
    env: GauntletEnv,
    policy_factory: Callable[[], Policy],
    item: WorkItem,
    *,
    trajectory_dir: Path | None = None,
    video_config: VideoConfig | None = None,
) -> Episode:
    """Drive one (cell, episode) rollout to completion.

    Shared by the worker entrypoint :func:`run_work_item`, by the
    Runner's in-process fast path, and by :mod:`gauntlet.replay` (which
    reconstructs a :class:`WorkItem` from an existing Episode to
    re-simulate it bit-identically). Every caller runs the *same* lines
    of code, so an Episode produced by ``n_workers=1``, one produced by
    ``n_workers=4``, and one produced by ``replay_one`` are all
    bit-identical for the same inputs.

    Pipeline (mirrors Pin 3):

    1. ``env.restore_baseline()`` wipes any model mutation from the
       previous episode handled by this worker.
    2. Apply every queued ``perturbation_value`` via
       :meth:`GauntletEnv.set_perturbation`.
    3. Build a fresh ``policy`` via ``policy_factory()``.
    4. Build the policy RNG from ``item.episode_seq`` (decorrelated from
       the env stream but reproducible from the same SeedSequence node).
    5. ``env.reset(seed=...)`` with the derived uint32 env seed.
    6. If the policy is :class:`ResettablePolicy`, hand it the RNG.
    7. Step until terminated / truncated, accumulating reward.

    Trajectory capture contract (Phase 2, additive):
    When ``trajectory_dir`` is a :class:`Path`, the worker buffers obs
    and actions per step and writes one NPZ per episode **after** the
    :class:`Episode` is built, so the in-memory Episode is byte-identical
    to the ``trajectory_dir=None`` path. The NPZ write is a pure
    side-effect outside the determinism contract.

    Video capture contract (Polish, additive):
    When ``video_config`` is a :class:`VideoConfig`, the worker captures
    ``obs["image"]`` per step (requires the env was constructed with
    ``render_in_obs=True`` — checked at the first reset) and writes one
    MP4 per episode after the :class:`Episode` is built. The
    :attr:`Episode.video_path` field is populated with the relative
    path. ``record_only_failures=True`` suppresses the *write* but the
    buffer still grows during the rollout. The default
    ``video_config=None`` keeps in-memory memory and disk I/O byte-
    identical to the pre-PR behaviour.
    """
    env.restore_baseline()
    for name, value in item.perturbation_values.items():
        env.set_perturbation(name, value)

    policy = policy_factory()
    policy_rng = np.random.default_rng(item.episode_seq)
    env_seed = extract_env_seed(item.episode_seq)

    obs, _ = env.reset(seed=env_seed)
    if isinstance(policy, ResettablePolicy):
        policy.reset(policy_rng)

    # Per-step buffers. We allocate only when asked — a ``None``
    # trajectory_dir keeps rollout memory usage identical to Phase 1.
    record_trajectory = trajectory_dir is not None
    obs_buffer: dict[str, list[NDArray[np.float64]]] = {}
    action_buffer: list[NDArray[np.float64]] = []
    if record_trajectory:
        # Initialise the obs buffer from the first-observation key set so
        # the NPZ schema is stable regardless of order of first append.
        for k in obs:
            obs_buffer[k] = []

    # Video frame buffer. Allocated only when ``video_config`` is set;
    # the env-side contract (``render_in_obs=True``) is checked here
    # using the public observation keys, never the backend's private
    # attribute, so MuJoCo / PyBullet / Genesis backends are all
    # supported uniformly. Record_only_failures still requires the
    # buffer because success is not known until the rollout ends.
    record_video = video_config is not None
    frame_buffer: list[NDArray[np.uint8]] = []
    if record_video and "image" not in obs:
        raise ValueError(
            "record_video=True requires the env to expose obs['image']; "
            "construct with render_in_obs=True"
        )

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    # B-21 actuator-cost telemetry. We accumulate from the env's
    # per-step ``info["actuator_energy_delta"]`` /
    # ``info["actuator_torque_norm"]`` keys. Backends that do not
    # publish these (PyBullet, Genesis, Isaac, fake test envs) leave
    # ``telemetry_observed`` False and the Episode carries
    # ``actuator_energy=None`` / ``mean_torque_norm=None`` /
    # ``peak_torque_norm=None``. The "MuJoCo first, others None"
    # contract from B-21 falls out of the env→worker info dict here
    # rather than an ``isinstance(env, TabletopEnv)`` switch.
    telemetry_observed = False
    energy_total = 0.0
    torque_norm_sum = 0.0
    torque_norm_peak = 0.0
    torque_norm_samples = 0
    while not (terminated or truncated):
        action = policy.act(obs)
        if record_trajectory:
            # Capture the obs that was fed into ``policy.act`` and the
            # action produced for it. Arrays are defensively copied so a
            # subsequent in-place mutation by the env step cannot change
            # what we've buffered.
            for k, v in obs.items():
                obs_buffer[k].append(np.asarray(v, dtype=np.float64).copy())
            action_buffer.append(np.asarray(action, dtype=np.float64).copy())
        if record_video:
            # Defensive copy so a subsequent env.step that recycles the
            # render buffer cannot mutate what we will encode later.
            frame_buffer.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        energy_delta = info.get("actuator_energy_delta")
        torque_norm = info.get("actuator_torque_norm")
        if energy_delta is not None and torque_norm is not None:
            telemetry_observed = True
            energy_total += float(energy_delta)
            t = float(torque_norm)
            torque_norm_sum += t
            torque_norm_samples += 1
            if t > torque_norm_peak:
                torque_norm_peak = t

    # Resolve the three Episode fields. ``telemetry_observed`` False
    # collapses all three to ``None`` so the report's failure-cluster
    # render shows a dash, not a misleading ``0.00``.
    actuator_energy: float | None = None
    mean_torque_norm: float | None = None
    peak_torque_norm: float | None = None
    if telemetry_observed and torque_norm_samples > 0:
        actuator_energy = energy_total
        mean_torque_norm = torque_norm_sum / torque_norm_samples
        peak_torque_norm = torque_norm_peak

    success = bool(info.get("success", False))

    # Compute the video path BEFORE Episode construction so the field
    # can be set in the same constructor call (keeping the schema
    # write-once). The path is None unless we will actually write a
    # file — i.e. video_config is set AND (record_only_failures is
    # False OR the rollout failed) AND the rollout produced at least
    # one frame. A T=0 rollout has nothing to encode; the Episode
    # honestly carries ``video_path=None`` so downstream HTML never
    # links a missing file.
    relative_video_path: str | None = None
    will_write_video = (
        video_config is not None
        and (not video_config.record_only_failures or not success)
        and bool(frame_buffer)
    )
    if video_config is not None and will_write_video:
        absolute = video_path_for(
            video_config.video_dir,
            cell_index=item.cell_index,
            episode_index=item.episode_index,
            seed=env_seed,
        )
        anchor = video_config.relative_to or video_config.video_dir.parent
        try:
            relative_video_path = str(absolute.relative_to(anchor))
        except ValueError:
            # ``video_dir`` is not under ``relative_to``; fall back to
            # the absolute path string. The HTML embed will still work
            # if the user opens the file from the same machine, and
            # the user was warned in the Runner docstring.
            relative_video_path = str(absolute)

    episode = Episode(
        suite_name=item.suite_name,
        cell_index=item.cell_index,
        episode_index=item.episode_index,
        seed=env_seed,
        perturbation_config=dict(item.perturbation_values),
        success=success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        step_count=int(step_count),
        total_reward=float(total_reward),
        metadata={
            "master_seed": int(item.master_seed),
            "n_cells": int(item.n_cells),
            "episodes_per_cell": int(item.episodes_per_cell),
        },
        video_path=relative_video_path,
        actuator_energy=actuator_energy,
        mean_torque_norm=mean_torque_norm,
        peak_torque_norm=peak_torque_norm,
    )

    if record_trajectory:
        assert trajectory_dir is not None  # type-narrowing for mypy.
        # An early-terminated rollout can have zero steps; write an
        # empty-but-well-formed NPZ so ``monitor score`` can still match
        # the file to the Episode row and decide how to handle T=0.
        obs_arrays: dict[str, NDArray[np.float64]] = {
            k: (
                np.stack(v, axis=0)
                if v
                else np.zeros((0, *np.asarray(obs[k]).shape), dtype=np.float64)
            )
            for k, v in obs_buffer.items()
        }
        actions_arr = (
            np.stack(action_buffer, axis=0) if action_buffer else np.zeros((0, 7), dtype=np.float64)
        )
        write_trajectory_npz(
            trajectory_path_for(trajectory_dir, item.cell_index, item.episode_index),
            obs_arrays=obs_arrays,
            actions=actions_arr,
            seed=env_seed,
            cell_index=item.cell_index,
            episode_index=item.episode_index,
        )

    if video_config is not None and will_write_video:
        # Re-derive the absolute path here so this branch does not have
        # to thread it through the Episode construction block above.
        # ``will_write_video`` already guards on a non-empty buffer +
        # the record_only_failures policy, so the encode is always
        # safe to call.
        absolute = video_path_for(
            video_config.video_dir,
            cell_index=item.cell_index,
            episode_index=item.episode_index,
            seed=env_seed,
        )
        VideoWriter(fps=video_config.fps).write(absolute, frame_buffer)

    return episode


def run_work_item(item: WorkItem) -> Episode:
    """Pool entrypoint — turn one :class:`WorkItem` into one :class:`Episode`.

    Reads the per-worker env + policy factory cached by
    :func:`pool_initializer` and delegates to :func:`execute_one`.
    Raises :class:`RuntimeError` if the initializer was skipped (which
    can only happen in tests that bypass the Pool path).
    """
    env_obj = _WORKER_STATE.get("env")
    policy_factory_obj = _WORKER_STATE.get("policy_factory")
    if env_obj is None or policy_factory_obj is None:
        raise RuntimeError(
            "Worker state not initialised; pool_initializer was not called. "
            "This is a Runner bug — file an issue with the reproduction."
        )
    # The pool initializer guarantees these types; cast for mypy.
    env: GauntletEnv = env_obj
    policy_factory: Callable[[], Policy] = policy_factory_obj
    # ``trajectory_dir`` may be absent when the initializer is the
    # pre-Phase-2 form; default to None so the key-missing path is the
    # byte-identical one.
    trajectory_dir: Path | None = _WORKER_STATE.get("trajectory_dir")
    # ``video_config`` is the Polish-task addition. Same key-missing
    # safety net as ``trajectory_dir``: a worker initialised by an
    # older code path runs without video recording.
    video_config: VideoConfig | None = _WORKER_STATE.get("video_config")
    return execute_one(
        env,
        policy_factory,
        item,
        trajectory_dir=trajectory_dir,
        video_config=video_config,
    )
