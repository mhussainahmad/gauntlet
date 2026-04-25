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

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from gauntlet.env.base import GauntletEnv
from gauntlet.policy.base import Policy, ResettablePolicy, SamplablePolicy
from gauntlet.runner.episode import Episode
from gauntlet.runner.parquet import TrajectoryDict, parquet_path_for, write_parquet
from gauntlet.runner.video import VideoWriter, video_path_for

# Public alias for the literal accepted by ``Runner.run`` and
# ``execute_one``'s ``trajectory_format`` knob (B-23). ``"npz"`` is the
# pre-B-23 default and is byte-identical to the previous behaviour;
# ``"parquet"`` and ``"both"`` opt into the [parquet] extra and emit
# one Parquet sidecar per episode (alongside the NPZ in ``"both"``
# mode, replacing it in ``"parquet"`` mode).
TrajectoryFormat = Literal["npz", "parquet", "both"]

__all__ = [
    "CONSISTENCY_STRIDE",
    "TrajectoryFormat",
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


# B-18 mode-collapse measurement cadence. Sampling at every step would
# multiply policy inference cost by N; striding 5 keeps the measurement
# bounded while still smoothing per-step jitter. Hoisted to module scope
# so tests and callers can reference the exact constant.
CONSISTENCY_STRIDE: int = 5

# Default N (action chunks per measured step) — matches the B-18 spec's
# "sample N action chunks per state" wording.
_CONSISTENCY_N: int = 8


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
    # B-23 trajectory output format. ``"npz"`` (the default) keeps the
    # pre-B-23 byte-identical behaviour: one ``np.savez_compressed``
    # per episode, no pyarrow import. ``"parquet"`` writes one Parquet
    # file per episode instead (requires the [parquet] extra).
    # ``"both"`` writes both side-by-side. Ignored when
    # ``trajectory_dir is None``.
    trajectory_format: TrajectoryFormat = "npz"
    # Optional per-episode MP4 video config. ``None`` means "no video"
    # — the runner stays byte-identical to the pre-PR behaviour. When
    # set, every worker accumulates ``obs["image"]`` per step and
    # writes one MP4 per episode (or only failures when
    # ``record_only_failures=True``). Requires the env to expose
    # ``obs["image"]`` (i.e. ``render_in_obs=True``); enforced at the
    # first reset.
    video_config: VideoConfig | None = None
    # B-18: opt into the mode-collapse measurement (calls
    # ``policy.act_n`` every ``CONSISTENCY_STRIDE``-th step on
    # :class:`SamplablePolicy` instances). Default ``False`` keeps the
    # rollout byte-identical to the pre-PR path. Greedy policies are
    # honestly skipped (Episode.action_variance is ``None``).
    measure_action_consistency: bool = False
    # B-30: optional per-rollout actuator-energy budget (joule-equivalent
    # units). When set AND the env publishes actuator telemetry, the
    # worker compares the rollout's accumulated ``actuator_energy``
    # against this threshold and writes the boolean into
    # :attr:`Episode.energy_over_budget`. ``None`` (the default) keeps
    # the field ``None`` on every Episode, byte-identical to pre-B-30.
    energy_budget: float | None = None
    # B-37: optional per-step inference-latency budget (milliseconds).
    # When set AND a rollout's measured p99 ``policy.act`` latency
    # exceeds this threshold, the worker stamps
    # ``Episode.metadata["inference_budget_violated"] = True``. Anti-
    # feature (deliberate): a soft flag, not a hard fail. The run
    # continues so the user sees every offender, not just the first.
    # ``None`` (the default) leaves ``metadata`` unchanged for the
    # latency-budget key — pre-B-37 byte-identical for the ``metadata``
    # contents. Latency itself is always measured (zero-config); only
    # the *budget compare* is gated on this knob.
    max_inference_ms: float | None = None


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
    trajectory_format: TrajectoryFormat
    video_config: VideoConfig | None
    measure_action_consistency: bool
    energy_budget: float | None
    max_inference_ms: float | None


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
    _WORKER_STATE["trajectory_format"] = args.trajectory_format
    _WORKER_STATE["video_config"] = args.video_config
    _WORKER_STATE["measure_action_consistency"] = args.measure_action_consistency
    _WORKER_STATE["energy_budget"] = args.energy_budget
    _WORKER_STATE["max_inference_ms"] = args.max_inference_ms


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
    trajectory_format: TrajectoryFormat = "npz",
    video_config: VideoConfig | None = None,
    measure_action_consistency: bool = False,
    energy_budget: float | None = None,
    max_inference_ms: float | None = None,
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

    Inference-latency contract (B-37, always-on):
    Every call to ``policy.act(obs)`` is bracketed by
    :func:`time.perf_counter` deltas; the per-step values are
    accumulated in milliseconds. At rollout end the worker computes
    ``p50`` / ``p99`` / ``max`` via :func:`numpy.percentile` and writes
    them to :attr:`Episode.inference_latency_ms_p50` etc. The B-18
    ``policy.act_n`` sampling call (when active) is **deliberately not
    timed**: it samples N=8 candidates and would inflate p99 by ~Nx on
    every measured step, biasing the report toward measurement
    overhead rather than the policy's deployed critical path. The
    overhead of ``perf_counter`` itself is sub-microsecond on modern
    Linux; the backlog's anti-feature concern about per-call overhead
    is mitigated by that fact, so the measurement is unconditional
    rather than gated behind a ``--track-latency`` flag.

    Inference-budget gate (B-37, soft):
    When ``max_inference_ms`` is set AND the rollout's p99 latency
    exceeds it, the worker writes
    ``Episode.metadata["inference_budget_violated"] = True``. The flag
    is **soft**: it never aborts the run, never flips ``success``, and
    is *absent* (rather than ``False``) when the budget was met or no
    budget was configured. The user wants to see every offender, not
    have the run die halfway and hide the rest.
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
    # ``reward_buffer`` / ``terminated_buffer`` / ``truncated_buffer``
    # are B-23 additions: the Parquet schema declares per-step columns
    # for those three signals, and the NPZ writer ignores them so
    # ``trajectory_format="npz"`` (the default) stays byte-identical.
    record_trajectory = trajectory_dir is not None
    obs_buffer: dict[str, list[NDArray[np.float64]]] = {}
    action_buffer: list[NDArray[np.float64]] = []
    reward_buffer: list[float] = []
    terminated_buffer: list[bool] = []
    truncated_buffer: list[bool] = []
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
    # B-18 mode-collapse measurement. Only active when
    # ``measure_action_consistency=True`` AND the policy implements
    # :class:`SamplablePolicy`. Greedy / open-loop policies legitimately
    # cannot expose this metric (no stochasticity to sample from); we
    # leave ``Episode.action_variance=None`` for them rather than
    # report a misleading ``0.0`` (see B-18 docstring on Episode).
    sample_policy = (
        policy if measure_action_consistency and isinstance(policy, SamplablePolicy) else None
    )
    variance_sum = 0.0
    variance_count = 0
    # B-30 safety-violation accumulators. ``safety_observed`` flips True
    # the first time the env publishes any of the three per-step safety
    # keys; backends that publish nothing leave it False and the
    # Episode carries ``None`` for all three counted fields. Mirrors
    # the actuator-telemetry "MuJoCo first, others None" contract.
    safety_observed = False
    n_collisions_acc = 0
    n_joint_excursions_acc = 0
    n_workspace_excursions_acc = 0
    # B-02 behavioural-metrics accumulators. ``behavior_observed`` flips
    # True the first time the env publishes any of the four per-step
    # ``behavior_*`` keys; backends that publish nothing leave it False
    # and the Episode carries ``None`` for all five derived fields.
    # Mirrors the actuator-telemetry / safety-violation contract. The
    # ``ee_pos_buffer`` is the single place we incur per-step memory
    # growth — at 24 bytes per sample it is negligible next to the
    # trajectory NPZ buffer (which is gated behind a separate opt-in).
    behavior_observed = False
    ee_pos_buffer: list[NDArray[np.float64]] = []
    near_collision_acc = 0
    peak_force_acc = 0.0
    last_control_dt: float | None = None
    success_step_count: int | None = None
    # B-37 per-step inference-latency buffer. Always allocated — see
    # the docstring's "always-on" contract above. One float per
    # ``policy.act(obs)`` call, in milliseconds. ``policy.act_n``
    # (B-18 measurement) is deliberately excluded — its N=8 sampling
    # would inflate every measured step by ~Nx and bias p99 toward
    # measurement overhead rather than the deployed critical path.
    inference_latency_buffer: list[float] = []
    while not (terminated or truncated):
        if sample_policy is not None and step_count % CONSISTENCY_STRIDE == 0:
            # Sample N actions for the current obs (state-preserving on
            # the SamplablePolicy contract), reduce per-axis variance to
            # a scalar via mean across action dims, accumulate.
            samples = sample_policy.act_n(obs, n=_CONSISTENCY_N)
            stacked = np.asarray(samples, dtype=np.float64)
            if stacked.ndim == 2 and stacked.shape[0] >= 2:
                per_axis_var = stacked.var(axis=0)
                variance_sum += float(per_axis_var.mean())
                variance_count += 1
        # B-37 per-step inference-latency measurement. Bracket
        # ``policy.act`` with :func:`time.perf_counter` deltas; the
        # ``policy.act_n`` call above is intentionally NOT timed. The
        # measurement is always-on (see the docstring's anti-feature
        # discussion); the overhead is sub-microsecond on modern Linux
        # and dwarfed by any realistic ``policy.act`` cost.
        _t0 = time.perf_counter()
        action = policy.act(obs)
        _t1 = time.perf_counter()
        inference_latency_buffer.append((_t1 - _t0) * 1000.0)
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
        if record_trajectory:
            # Per-step reward / terminated / truncated. Ignored by the
            # NPZ writer (schema unchanged), surfaced as columns by
            # the Parquet writer (B-23). The per-step terminal flags
            # are False until the final step; the Parquet schema
            # carries that signal verbatim so a DuckDB ``WHERE
            # terminated`` filter trivially picks the last step.
            reward_buffer.append(float(reward))
            terminated_buffer.append(bool(terminated))
            truncated_buffer.append(bool(truncated))
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
        # B-30 safety telemetry. Each key is independent — an env may
        # publish collisions but not workspace bounds, etc. ``safety_observed``
        # flips on the first of any present key so partial-coverage envs
        # still get tagged as "telemetry surfaced" rather than ``None``
        # across the board.
        col_delta = info.get("safety_n_collisions_delta")
        joint_violation = info.get("safety_joint_limit_violation")
        workspace_violation = info.get("safety_workspace_excursion")
        if col_delta is not None:
            safety_observed = True
            n_collisions_acc += int(col_delta)
        if joint_violation is not None:
            safety_observed = True
            if bool(joint_violation):
                n_joint_excursions_acc += 1
        if workspace_violation is not None:
            safety_observed = True
            if bool(workspace_violation):
                n_workspace_excursions_acc += 1
        # B-02 behavioural telemetry. The four ``behavior_*`` keys move
        # as a unit — backends that publish any of them must publish
        # all four, by the env-side contract. We still defensively gate
        # on the leading near-collision key so a partial-coverage env
        # surfaces honestly as "not measured" rather than silently
        # mis-aggregating.
        near_delta = info.get("behavior_near_collision_delta")
        if near_delta is not None:
            behavior_observed = True
            near_collision_acc += int(near_delta)
            peak_step = info.get("behavior_peak_contact_force")
            if peak_step is not None:
                peak_step_f = float(peak_step)
                if peak_step_f > peak_force_acc:
                    peak_force_acc = peak_step_f
            ee_step = info.get("behavior_ee_pos")
            if ee_step is not None:
                ee_pos_buffer.append(np.asarray(ee_step, dtype=np.float64).copy())
            dt_step = info.get("behavior_control_dt")
            if dt_step is not None:
                last_control_dt = float(dt_step)
        # Snapshot the step count at which ``info["success"]`` first
        # flipped True so ``time_to_success`` reflects the time-to-done
        # even when the env keeps running for a tail of post-success
        # steps (it does not in TabletopEnv, but the contract is
        # backend-agnostic).
        if success_step_count is None and bool(info.get("success", False)):
            success_step_count = step_count

    # Resolve the B-18 action-variance scalar. ``None`` covers three
    # honest cases: (a) the user did not opt in via
    # ``measure_action_consistency=True``, (b) the policy is greedy /
    # open-loop and does not implement :class:`SamplablePolicy`, or
    # (c) the rollout produced fewer than one measurable step (a T=0
    # episode that terminated at reset). The HTML report renders this
    # as a dash, never as ``0.0`` (which would mean "true mode
    # collapse" — distinct from "not measured here").
    action_variance: float | None = None
    if sample_policy is not None and variance_count > 0:
        action_variance = variance_sum / variance_count

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

    # B-30 safety-violation field resolution. ``safety_observed`` False
    # collapses every counted field to ``None`` (not ``0``) — distinct
    # backends that publish nothing must not look like backends that
    # publish "zero violations". ``energy_over_budget`` is a derived
    # bool: present only when both an ``energy_budget`` is configured
    # AND the env publishes torque telemetry (so we have a real
    # ``actuator_energy`` to compare against).
    n_collisions: int | None = None
    n_joint_limit_excursions: int | None = None
    n_workspace_excursions: int | None = None
    if safety_observed:
        n_collisions = int(n_collisions_acc)
        n_joint_limit_excursions = int(n_joint_excursions_acc)
        n_workspace_excursions = int(n_workspace_excursions_acc)
    energy_over_budget: bool | None = None
    if energy_budget is not None and actuator_energy is not None:
        energy_over_budget = bool(actuator_energy > energy_budget)

    success = bool(info.get("success", False))

    # B-02 behavioural-metrics resolution. ``behavior_observed`` False
    # collapses every derived field to ``None`` — distinct from any
    # observed-but-zero value (e.g. a stationary policy that never
    # moved still reports ``near_collision_count=0``, not ``None``).
    # The five fields divide along their data dependencies:
    #
    # * ``time_to_success`` needs ``control_dt`` AND ``success``. A
    #   failed rollout has nothing to time to (``None``).
    # * ``path_length_ratio`` needs >= 2 EE samples. Stationary policies
    #   (initial == final, distance < 1e-6 m) report ``None`` rather
    #   than ``inf`` — distance/0 is undefined, not "infinite detour".
    # * ``jerk_rms`` needs >= 4 EE samples (third finite difference)
    #   AND ``control_dt`` for the dt^3 denominator.
    # * ``near_collision_count`` and ``peak_force`` are direct
    #   accumulator reads; ``int`` and ``float`` respectively.
    time_to_success: float | None = None
    path_length_ratio: float | None = None
    jerk_rms: float | None = None
    near_collision_count: int | None = None
    peak_force: float | None = None
    if behavior_observed:
        near_collision_count = int(near_collision_acc)
        peak_force = float(peak_force_acc)
        if success and success_step_count is not None and last_control_dt is not None:
            time_to_success = float(success_step_count) * float(last_control_dt)
        if len(ee_pos_buffer) >= 2:
            ee_arr = np.stack(ee_pos_buffer, axis=0)
            # Path length — sum of consecutive-sample L2 distances.
            seg = np.linalg.norm(np.diff(ee_arr, axis=0), axis=1)
            path_length = float(seg.sum())
            straight = float(np.linalg.norm(ee_arr[-1] - ee_arr[0]))
            # 1e-6 m guard: any tighter and IEEE-754 round-off in the
            # MuJoCo integrator (typical 1e-9 noise on a "stationary"
            # mocap pose) would split a stationary policy into an
            # ``inf``-ratio outlier. Wider would falsely declare a
            # 1mm legitimate motion as "stationary".
            if straight > 1e-6:
                path_length_ratio = path_length / straight
            if len(ee_pos_buffer) >= 4 and last_control_dt is not None:
                # Third finite difference (forward, length T-3) — the
                # canonical numerical-jerk estimator. Higher-order
                # schemes would require T >= 5+; we keep the bar at
                # T >= 4 so even short rollouts are measurable.
                dt3 = float(last_control_dt) ** 3
                third = ee_arr[3:] - 3.0 * ee_arr[2:-1] + 3.0 * ee_arr[1:-2] - ee_arr[:-3]
                jerk = third / dt3
                jerk_mag_sq = np.sum(jerk * jerk, axis=1)
                jerk_rms = float(np.sqrt(jerk_mag_sq.mean()))

    # B-37 inference-latency aggregation. ``policy.act`` was timed on
    # every step (see the docstring's always-on contract); a T=0
    # rollout (env truncates at reset, never reaches ``act``) leaves
    # the buffer empty and the three Episode fields stay ``None``,
    # honouring the same partial-coverage convention as the B-21 / B-30
    # / B-02 telemetry trios. ``np.percentile`` with the default linear
    # interpolation is the canonical p50 / p99 estimator and matches
    # the wording in the B-37 spec / VLA-Perf framing.
    inference_latency_ms_p50: float | None = None
    inference_latency_ms_p99: float | None = None
    inference_latency_ms_max: float | None = None
    if inference_latency_buffer:
        latency_arr = np.asarray(inference_latency_buffer, dtype=np.float64)
        inference_latency_ms_p50 = float(np.percentile(latency_arr, 50))
        inference_latency_ms_p99 = float(np.percentile(latency_arr, 99))
        inference_latency_ms_max = float(latency_arr.max())

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

    # Build the metadata dict up-front so the B-37 budget flag, when
    # set, lands alongside the existing reproducibility echoes in a
    # single ``Episode`` constructor call (keeping the schema write-
    # once). The flag is *only* present when (a) a budget was
    # configured AND (b) the rollout's p99 exceeded it — the absent-
    # rather-than-False convention keeps the soft-flag semantics
    # honest. Values are typed as ``int | float | str | bool`` per
    # the Episode.metadata field annotation; the bool literal lands as
    # bool, not int, under pydantic's mode="json" round-trip.
    episode_metadata: dict[str, float | int | str | bool] = {
        "master_seed": int(item.master_seed),
        "n_cells": int(item.n_cells),
        "episodes_per_cell": int(item.episodes_per_cell),
    }
    if (
        max_inference_ms is not None
        and inference_latency_ms_p99 is not None
        and inference_latency_ms_p99 > max_inference_ms
    ):
        episode_metadata["inference_budget_violated"] = True

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
        metadata=episode_metadata,
        video_path=relative_video_path,
        actuator_energy=actuator_energy,
        mean_torque_norm=mean_torque_norm,
        peak_torque_norm=peak_torque_norm,
        action_variance=action_variance,
        n_collisions=n_collisions,
        n_joint_limit_excursions=n_joint_limit_excursions,
        energy_over_budget=energy_over_budget,
        n_workspace_excursions=n_workspace_excursions,
        time_to_success=time_to_success,
        path_length_ratio=path_length_ratio,
        jerk_rms=jerk_rms,
        near_collision_count=near_collision_count,
        peak_force=peak_force,
        inference_latency_ms_p50=inference_latency_ms_p50,
        inference_latency_ms_p99=inference_latency_ms_p99,
        inference_latency_ms_max=inference_latency_ms_max,
    )

    if record_trajectory:
        assert trajectory_dir is not None  # type-narrowing for mypy.
        # An early-terminated rollout can have zero steps; write an
        # empty-but-well-formed NPZ / Parquet so ``monitor score`` and
        # downstream DuckDB / pandas globs can still match the file to
        # the Episode row and decide how to handle T=0.
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
        # B-23: gate each format independently so ``"both"`` writes
        # both sidecars and ``"parquet"`` skips the NPZ entirely. The
        # NPZ schema is unchanged from Phase 1; the Parquet schema is
        # documented in :mod:`gauntlet.runner.parquet`.
        if trajectory_format in ("npz", "both"):
            write_trajectory_npz(
                trajectory_path_for(trajectory_dir, item.cell_index, item.episode_index),
                obs_arrays=obs_arrays,
                actions=actions_arr,
                seed=env_seed,
                cell_index=item.cell_index,
                episode_index=item.episode_index,
            )
        if trajectory_format in ("parquet", "both"):
            rewards_arr = np.asarray(reward_buffer, dtype=np.float64)
            terminated_arr = np.asarray(terminated_buffer, dtype=np.bool_)
            truncated_arr = np.asarray(truncated_buffer, dtype=np.bool_)
            write_parquet(
                parquet_path_for(trajectory_dir, item.cell_index, item.episode_index),
                TrajectoryDict(
                    observations=obs_arrays,
                    actions=actions_arr,
                    rewards=rewards_arr,
                    terminated=terminated_arr,
                    truncated=truncated_arr,
                ),
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
    # B-23 trajectory_format. Defaults to ``"npz"`` so a worker
    # initialised by a pre-B-23 code path keeps the byte-identical
    # behaviour.
    trajectory_format: TrajectoryFormat = _WORKER_STATE.get("trajectory_format", "npz")
    # ``video_config`` is the Polish-task addition. Same key-missing
    # safety net as ``trajectory_dir``: a worker initialised by an
    # older code path runs without video recording.
    video_config: VideoConfig | None = _WORKER_STATE.get("video_config")
    # B-18 mode-collapse opt-in. Default ``False`` keeps the rollout
    # byte-identical for workers initialised by pre-B-18 code paths.
    measure = bool(_WORKER_STATE.get("measure_action_consistency", False))
    # B-30 actuator-energy budget. ``None`` keeps Episode.energy_over_budget
    # ``None`` for workers initialised by pre-B-30 code paths.
    energy_budget: float | None = _WORKER_STATE.get("energy_budget")
    # B-37 inference-latency budget. ``None`` leaves
    # ``Episode.metadata["inference_budget_violated"]`` *absent* on
    # every Episode; the latency p50 / p99 / max fields themselves are
    # always populated regardless (always-on measurement).
    max_inference_ms: float | None = _WORKER_STATE.get("max_inference_ms")
    return execute_one(
        env,
        policy_factory,
        item,
        trajectory_dir=trajectory_dir,
        trajectory_format=trajectory_format,
        video_config=video_config,
        measure_action_consistency=measure,
        energy_budget=energy_budget,
        max_inference_ms=max_inference_ms,
    )
