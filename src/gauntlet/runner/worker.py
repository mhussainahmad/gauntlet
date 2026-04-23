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

The same :func:`_execute_one` core is also called directly by the Runner's
in-process fast path (``n_workers == 1``), so both code paths produce
bit-identical Episode objects from the same inputs.

Determinism contract
--------------------
* ``WorkItem.episode_seq`` is a :class:`numpy.random.SeedSequence` —
  picklable, deterministic, derived from the master seed via
  ``master.spawn(n_cells)[cell_idx].spawn(episodes_per_cell)[ep_idx]``.
* The env seed handed to :meth:`TabletopEnv.reset` is
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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.base import Policy, ResettablePolicy
from gauntlet.runner.episode import Episode

__all__ = [
    "WorkItem",
    "WorkerInitArgs",
    "extract_env_seed",
    "pool_initializer",
    "run_work_item",
    "trajectory_path_for",
    "write_trajectory_npz",
]


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
            :meth:`TabletopEnv.set_perturbation`.
        episode_seq: SeedSequence node for this episode. The worker
            derives both the env seed (uint32) and the policy RNG from
            this single node so both streams are reproducible.
        master_seed: Echo of the master seed (or the auto-generated
            entropy when no seed was supplied). Recorded in the resulting
            Episode's ``metadata["master_seed"]`` so even None-seeded
            runs can be reproduced from the report.
    """

    suite_name: str
    cell_index: int
    episode_index: int
    perturbation_values: dict[str, float]
    episode_seq: np.random.SeedSequence
    master_seed: int


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

    * ``env_factory`` — builds a :class:`TabletopEnv`. The MjModel inside
      is not picklable, so the env must be born inside the worker.
    * ``policy_factory`` — builds a fresh :class:`Policy` per episode.
      Stashed once and called per-item; future torch/GPU policies that
      cannot be pickled (e.g. HuggingFacePolicy) can therefore live
      entirely inside the worker.
    """

    env_factory: Callable[[], TabletopEnv]
    policy_factory: Callable[[], Policy]
    # Optional per-episode trajectory dump directory. When ``None`` (the
    # default) the worker writes zero bytes — the Runner's behaviour is
    # byte-identical to Phase 1. When a Path is supplied the worker emits
    # one NPZ per episode following the ``cell_NNNN_ep_NNNN.npz`` scheme.
    trajectory_dir: Path | None = None


# Worker-local cache. Populated by ``pool_initializer``; read by
# ``run_work_item``. Typed as ``Any`` because the values change shape
# across the lifecycle and mypy does not need to track per-worker
# globals across the pickle boundary.
_WORKER_STATE: dict[str, Any] = {}


def pool_initializer(args: WorkerInitArgs) -> None:
    """Run once per worker process at pool startup.

    Loads the MJCF (one MjModel per worker) and caches the policy
    factory. Subsequent items reuse the env via
    :meth:`TabletopEnv.restore_baseline`; the model is never re-loaded.

    Also caches ``trajectory_dir``; when non-``None`` each per-item call
    writes one NPZ sidecar after Episode construction. The attribute is
    always present in ``_WORKER_STATE`` (set to ``None`` when disabled)
    so :func:`run_work_item` does not have to branch on ``KeyError``.
    """
    env = args.env_factory()
    _WORKER_STATE["env"] = env
    _WORKER_STATE["policy_factory"] = args.policy_factory
    _WORKER_STATE["trajectory_dir"] = args.trajectory_dir


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

    * ``obs_<key>`` — one array per :meth:`TabletopEnv._build_obs` key,
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


def _execute_one(
    env: TabletopEnv,
    policy_factory: Callable[[], Policy],
    item: WorkItem,
    *,
    trajectory_dir: Path | None = None,
) -> Episode:
    """Drive one (cell, episode) rollout to completion.

    Shared by the worker entrypoint :func:`run_work_item` and by the
    Runner's in-process fast path. Both paths run the *same* lines of
    code, so an Episode produced by ``n_workers=1`` is bit-identical to
    one produced by ``n_workers=4`` for the same inputs.

    Pipeline (mirrors Pin 3):

    1. ``env.restore_baseline()`` wipes any model mutation from the
       previous episode handled by this worker.
    2. Apply every queued ``perturbation_value`` via
       :meth:`TabletopEnv.set_perturbation`.
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

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
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
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

    success = bool(info.get("success", False))
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
        metadata={"master_seed": int(item.master_seed)},
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

    return episode


def run_work_item(item: WorkItem) -> Episode:
    """Pool entrypoint — turn one :class:`WorkItem` into one :class:`Episode`.

    Reads the per-worker env + policy factory cached by
    :func:`pool_initializer` and delegates to :func:`_execute_one`.
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
    env: TabletopEnv = env_obj
    policy_factory: Callable[[], Policy] = policy_factory_obj
    # ``trajectory_dir`` may be absent when the initializer is the
    # pre-Phase-2 form; default to None so the key-missing path is the
    # byte-identical one.
    trajectory_dir: Path | None = _WORKER_STATE.get("trajectory_dir")
    return _execute_one(env, policy_factory, item, trajectory_dir=trajectory_dir)
