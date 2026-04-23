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
from typing import Any

import numpy as np

from gauntlet.env.base import GauntletEnv
from gauntlet.policy.base import Policy, ResettablePolicy
from gauntlet.runner.episode import Episode

__all__ = [
    "WorkItem",
    "WorkerInitArgs",
    "extract_env_seed",
    "pool_initializer",
    "run_work_item",
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
            :meth:`GauntletEnv.set_perturbation`.
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


# Worker-local cache. Populated by ``pool_initializer``; read by
# ``run_work_item``. Typed as ``Any`` because the values change shape
# across the lifecycle and mypy does not need to track per-worker
# globals across the pickle boundary.
_WORKER_STATE: dict[str, Any] = {}


def pool_initializer(args: WorkerInitArgs) -> None:
    """Run once per worker process at pool startup.

    Constructs one env per worker (whatever concrete
    :class:`GauntletEnv` the ``env_factory`` returns) and caches the
    policy factory. Subsequent items reuse the env via
    :meth:`GauntletEnv.restore_baseline` — Protocol contract guarantees
    the backend returns to a ``__init__``-equivalent state between
    episodes without reconstructing the scene.
    """
    env = args.env_factory()
    _WORKER_STATE["env"] = env
    _WORKER_STATE["policy_factory"] = args.policy_factory


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


def _execute_one(env: GauntletEnv, policy_factory: Callable[[], Policy], item: WorkItem) -> Episode:
    """Drive one (cell, episode) rollout to completion.

    Shared by the worker entrypoint :func:`run_work_item` and by the
    Runner's in-process fast path. Both paths run the *same* lines of
    code, so an Episode produced by ``n_workers=1`` is bit-identical to
    one produced by ``n_workers=4`` for the same inputs.

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

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

    success = bool(info.get("success", False))
    return Episode(
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
    env: GauntletEnv = env_obj
    policy_factory: Callable[[], Policy] = policy_factory_obj
    return _execute_one(env, policy_factory, item)
