"""Parallel rollout orchestrator.

See ``GAUNTLET_SPEC.md`` §3 (architecture) and §6 (reproducibility is
non-negotiable).

The :class:`Runner` takes a :class:`Suite` plus a zero-arg
``policy_factory`` and produces one :class:`Episode` per (cell, episode).
Cells are run in parallel across processes via a ``spawn``-context
:class:`multiprocessing.pool.Pool`; for ``n_workers == 1`` we skip the
pool and execute in-process to keep stack traces readable and to dodge
spawn overhead on small suites. Both paths share the same per-item
function (:func:`gauntlet.runner.worker.execute_one`), so the n=1 and
n=N outputs are bit-identical for the same inputs.

Seed derivation
---------------
The single source of entropy is ``suite.seed`` (or, when ``None``, an
OS-entropy :class:`numpy.random.SeedSequence` we create and record on
every Episode for later reproduction). From that root, per-episode
seeds are derived using :class:`numpy.random.SeedSequence.spawn`::

    master = np.random.SeedSequence(suite.seed)        # or fresh()
    cell_seqs = master.spawn(n_cells)                  # one per cell
    episode_seqs = cell_seqs[cell_idx].spawn(eps_per_cell)
    episode_seq = episode_seqs[ep_idx]                 # one per episode

The worker (:mod:`gauntlet.runner.worker`) then turns ``episode_seq``
into:

* ``env_seed: int`` — uint32 from ``episode_seq.generate_state(1, ...)``.
  Stored on :attr:`Episode.seed`. Reproduces the rollout end-to-end.
* ``policy_rng`` — ``np.random.default_rng(episode_seq)``. Decorrelated
  from the env stream but deterministic from the same node.

**Departure from the original Pin 2 wording.** The pin asked for
``env.reset(seed=episode_seq.entropy)``; empirically every spawned child
inherits its parent's ``entropy`` value, which would force every episode
to reset identically (and would break uniqueness of :attr:`Episode.seed`
across a run). ``generate_state`` is the canonical bridge from a
SeedSequence node to a per-spawn-unique scalar. The rest of the pin —
the ``master.spawn(n_cells)[i].spawn(eps_per_cell)[j]`` derivation tree
and ``np.random.default_rng(episode_seq)`` for the policy stream — is
preserved verbatim.

The master seed (literal int, or auto-generated entropy when
``suite.seed is None``) is echoed into every Episode's
``metadata["master_seed"]`` so a None-seeded run is also reproducible
end-to-end.

Output ordering
---------------
:meth:`run` returns Episodes sorted by ``(cell_index, episode_index)``,
regardless of the order in which workers complete. This is a stable
contract downstream code (Report, CLI) keys off.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory, registered_envs
from gauntlet.policy.base import Policy
from gauntlet.runner.episode import Episode
from gauntlet.runner.worker import (
    WorkerInitArgs,
    WorkItem,
    execute_one,
    pool_initializer,
    run_work_item,
)
from gauntlet.suite.schema import Suite

__all__ = ["Runner"]


class Runner:
    """Execute a :class:`Suite` against a :class:`Policy` in parallel.

    Construction parameters configure the *how* of execution; the
    suite + policy come in via :meth:`run`, mirroring the spec's
    "takes a Policy, a Suite, returns a list of Episode results" wording.
    """

    def __init__(
        self,
        *,
        n_workers: int = 1,
        env_factory: Callable[[], GauntletEnv] | None = None,
        start_method: str = "spawn",
        trajectory_dir: Path | None = None,
    ) -> None:
        """Configure the runner.

        Args:
            n_workers: Number of worker processes. ``1`` triggers the
                in-process fast path; ``>= 2`` uses a multiprocessing
                pool. Must be ``>= 1``.
            env_factory: Zero-arg factory returning a :class:`GauntletEnv`.
                When ``None`` (the default), :meth:`run` dispatches via
                :func:`gauntlet.env.registry.get_env_factory` on
                ``suite.env`` — so a ``tabletop-pybullet`` suite picks
                up :class:`PyBulletTabletopEnv`, not MuJoCo's
                :class:`TabletopEnv`. Supply an explicit factory to
                override the registry (tests use this to inject fakes
                or fast-config variants). Must be pickle-friendly
                (module-level fn or class) when ``n_workers >= 2``
                because the ``spawn`` start method pickles it across
                the process boundary.
            start_method: multiprocessing start method.
                ``"spawn"`` is the only safe choice with MuJoCo — fork
                is known to leak GL contexts. We refuse other values
                with a clear error.
            trajectory_dir: Optional directory to dump per-episode NPZ
                trajectories into. When ``None`` (the default) Phase 2
                behaviour is **byte-identical** to Phase 1 — no disk
                writes, no per-step buffering. When a :class:`Path` is
                supplied each rollout writes one NPZ named
                ``cell_<cell:04d>_ep_<ep:04d>.npz`` under this directory
                *after* the :class:`Episode` is built, so the Episode
                list returned by :meth:`run` is the same in both cases.
                The directory is created on :meth:`run` entry.

        Raises:
            ValueError: If ``n_workers < 1`` or ``start_method != "spawn"``.
        """
        if n_workers < 1:
            raise ValueError(
                f"n_workers must be >= 1; got {n_workers}. Use 1 for the in-process fast path."
            )
        if start_method != "spawn":
            # MuJoCo's renderer holds GL state that fork() will not
            # safely duplicate. We refuse non-spawn start methods to
            # protect the reproducibility contract.
            raise ValueError(
                f"start_method must be 'spawn'; got {start_method!r}. MuJoCo is not fork-safe."
            )
        self._n_workers = n_workers
        self._env_factory = env_factory
        self._start_method = start_method
        self._trajectory_dir = trajectory_dir

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        policy_factory: Callable[[], Policy],
        suite: Suite,
    ) -> list[Episode]:
        """Execute every (cell x episode) rollout.

        Args:
            policy_factory: Zero-arg callable returning a fresh
                :class:`Policy`. Each worker calls it exactly once per
                episode it handles. Must be picklable when
                ``n_workers >= 2`` (module-level functions and classes
                pickle cleanly under ``spawn``; lambdas defined inside
                tests do not).
            suite: The :class:`Suite` to evaluate.

        Returns:
            List of :class:`Episode` records sorted by
            ``(cell_index, episode_index)``. Length is exactly
            ``suite.num_cells() * suite.episodes_per_cell``.

        Raises:
            ValueError: If ``suite.env`` is not registered at dispatch time.
                The Suite loader imports backends on demand (RFC-005 §11.2);
                any env that reaches the Runner should already be registered.
        """
        # Defensive validation. Suite.env is already validated at
        # construction time, but the Runner is the boundary between
        # config and execution — re-check so a malformed Suite produced
        # by a future loader cannot silently launch a bad run.
        known = registered_envs()
        if suite.env not in known:
            listed = ", ".join(sorted(known))
            raise ValueError(
                f"Runner: unsupported env {suite.env!r}; registered envs: "
                f"{{{listed}}}. Load the Suite via gauntlet.suite.load_suite(...) "
                f"or import the backend subpackage before calling Runner.run()."
            )

        # Resolve the env factory: a caller-supplied override wins, else
        # dispatch via the registry on ``suite.env``. The registry path is
        # what keeps ``gauntlet run suite.yaml`` honest — a suite declaring
        # ``env: tabletop-pybullet`` must produce :class:`PyBulletTabletopEnv`,
        # not the MuJoCo built-in.
        env_factory = (
            self._env_factory if self._env_factory is not None else get_env_factory(suite.env)
        )

        # Ensure the trajectory dir exists exactly once on the main
        # process before any worker touches it, so a common parent (even
        # on a fresh ``tmp_path``) never races across multiple workers.
        if self._trajectory_dir is not None:
            self._trajectory_dir.mkdir(parents=True, exist_ok=True)

        work_items = self._build_work_items(suite)
        if self._n_workers == 1:
            episodes = self._run_in_process(env_factory, policy_factory, work_items)
        else:
            episodes = self._run_pool(env_factory, policy_factory, work_items)

        # Stable output order — independent of worker completion order.
        episodes.sort(key=lambda ep: (ep.cell_index, ep.episode_index))
        return episodes

    # ------------------------------------------------------------------
    # Work-item construction (the seed-derivation lives here).
    # ------------------------------------------------------------------

    def _build_work_items(self, suite: Suite) -> list[WorkItem]:
        """Enumerate (cell, episode) pairs and attach SeedSequence nodes.

        The derivation uses ``SeedSequence.spawn`` exactly as Pin 2
        prescribes: one spawn level for cells, a second for episodes
        within each cell. This keeps the per-episode streams independent
        across the (cell, episode) lattice.

        When ``suite.seed is None`` we instantiate ``SeedSequence()``
        (OS entropy) and record its ``entropy`` integer on every
        WorkItem so the resulting Episodes carry the bit needed to
        reproduce that exact run via a future ``Suite(seed=master_seed)``.
        """
        cells = list(suite.cells())
        n_cells = len(cells)
        eps_per_cell = suite.episodes_per_cell

        if suite.seed is None:
            master = np.random.SeedSequence()
        else:
            master = np.random.SeedSequence(suite.seed)
        # SeedSequence.entropy is typed as ``int | Sequence[int] | None`` in
        # the numpy stubs to cover the case where a user supplied a sequence
        # of ints. We always supply either ``None`` (-> auto-generated int)
        # or a plain ``int``, so the runtime type is always ``int`` here.
        entropy = master.entropy
        if not isinstance(entropy, int):
            raise RuntimeError(
                f"unexpected SeedSequence.entropy type {type(entropy).__name__}; "
                "Runner only accepts scalar int seeds."
            )
        master_seed_echo: int = entropy

        # spawn(0) is legal but we will never enter the loop below if
        # the suite has zero cells; pydantic validation already forbids
        # an empty axes mapping, so n_cells >= 1 in practice.
        cell_seqs = master.spawn(n_cells)

        items: list[WorkItem] = []
        for cell_idx, cell in enumerate(cells):
            episode_seqs = cell_seqs[cell_idx].spawn(eps_per_cell)
            for ep_idx in range(eps_per_cell):
                items.append(
                    WorkItem(
                        suite_name=suite.name,
                        cell_index=cell.index,
                        episode_index=ep_idx,
                        # Defensive copy: SuiteCell.values is a Mapping
                        # (often a dict). dict(...) flattens it to a
                        # mutable, picklable, equality-comparable form.
                        perturbation_values=dict(cell.values),
                        episode_seq=episode_seqs[ep_idx],
                        master_seed=master_seed_echo,
                        # Topology echo for trajectory replay (see
                        # docs/phase2-rfc-004-trajectory-replay.md §3).
                        # These two integers are the minimum needed to
                        # reconstruct the spawn tree from the Episode
                        # alone, even if the suite YAML is edited
                        # between run and replay.
                        n_cells=n_cells,
                        episodes_per_cell=eps_per_cell,
                    )
                )
        return items

    # ------------------------------------------------------------------
    # Execution paths.
    # ------------------------------------------------------------------

    def _run_in_process(
        self,
        env_factory: Callable[[], GauntletEnv],
        policy_factory: Callable[[], Policy],
        work_items: list[WorkItem],
    ) -> list[Episode]:
        """Single-process fast path. Loads the env once, reuses it.

        Used when ``n_workers == 1``. Produces Episodes that are
        bit-identical to the multi-worker path for the same inputs
        because ``execute_one`` is the same function in both cases.
        """
        env = env_factory()
        try:
            return [
                execute_one(env, policy_factory, item, trajectory_dir=self._trajectory_dir)
                for item in work_items
            ]
        finally:
            env.close()

    def _run_pool(
        self,
        env_factory: Callable[[], GauntletEnv],
        policy_factory: Callable[[], Policy],
        work_items: list[WorkItem],
    ) -> list[Episode]:
        """Multiprocessing path. ``spawn`` start method only.

        Each worker loads the MJCF once via ``pool_initializer`` and
        reuses that env across every WorkItem the pool routes to it.
        ``Pool.map`` preserves input order, but we still re-sort in
        :meth:`run` so callers never depend on pool internals.
        """
        ctx = mp.get_context(self._start_method)
        init_args = WorkerInitArgs(
            env_factory=env_factory,
            policy_factory=policy_factory,
            trajectory_dir=self._trajectory_dir,
        )
        with ctx.Pool(
            processes=self._n_workers,
            initializer=pool_initializer,
            initargs=(init_args,),
        ) as pool:
            return list(pool.map(run_work_item, work_items))
