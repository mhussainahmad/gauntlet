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
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory, registered_envs
from gauntlet.policy.base import Policy
from gauntlet.report.wilson import wilson_interval
from gauntlet.runner.cache import EpisodeCache
from gauntlet.runner.episode import Episode
from gauntlet.runner.provenance import (
    capture_gauntlet_version,
    capture_git_commit,
    compute_suite_hash,
)
from gauntlet.runner.worker import (
    TrajectoryFormat,
    VideoConfig,
    WorkerInitArgs,
    WorkItem,
    execute_one,
    extract_env_seed,
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
        record_video: bool = False,
        video_dir: Path | None = None,
        video_fps: int = 30,
        record_only_failures: bool = False,
        cache_dir: Path | None = None,
        policy_id: str | None = None,
        max_steps: int | None = None,
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
            record_video: When ``True`` the runner buffers ``obs["image"]``
                per step and writes one MP4 per episode via
                :class:`gauntlet.runner.video.VideoWriter`. The env MUST
                expose ``obs["image"]`` (i.e. constructed with
                ``render_in_obs=True``); otherwise a clear
                :class:`ValueError` is raised inside the worker on the
                first reset. Requires the optional ``[video]`` extra
                (``pip install "gauntlet[video]"``); the :class:`VideoWriter`
                lazy-imports ``imageio`` and surfaces a clear ``ImportError``
                with the install hint if the extra is missing. Default
                ``False`` keeps in-memory + on-disk behaviour byte-
                identical to the pre-PR runner.
            video_dir: Directory to write per-episode MP4s into. Ignored
                unless ``record_video=True``. Defaults to
                ``trajectory_dir / "videos"`` when ``trajectory_dir`` is
                set, else falls back to ``Path("videos")``. The
                :attr:`gauntlet.runner.Episode.video_path` field is the
                MP4 path *relative* to ``video_dir.parent`` so the HTML
                report can embed ``<video src="videos/...mp4">`` without
                a web server. For the embed to work the user must keep
                the HTML report alongside the videos directory.
            video_fps: Output framerate for the MP4 encoder. Must be a
                positive integer. Default 30.
            record_only_failures: Opt-in flag that suppresses MP4 writes
                for ``success=True`` episodes (saves disk on long
                sweeps). The frame buffer still grows during the
                rollout because success is unknown until
                ``info["success"]`` arrives; only the *write* is gated.
                Default ``False`` — every episode gets an MP4.
            cache_dir: Optional directory for the file-per-Episode
                rollout cache (see :class:`gauntlet.runner.cache.EpisodeCache`
                and ``docs/polish-exploration-incremental-cache.md``).
                When ``None`` (the default) no cache is constructed, no
                cache lookups happen, and the hot path is byte-identical
                to pre-PR behaviour. When set, the Runner computes a
                content-addressed key per (suite, axis_config, env_seed,
                policy_id, max_steps) cell, returns cached Episodes
                straight from disk on a hit, and writes fresh ones via
                ``cache.put`` on a miss. Trajectory NPZs and MP4 videos
                are NOT replayed on a hit — the cache returns Episode
                objects only.
            policy_id: Caller-supplied policy identifier baked into the
                cache key. When ``cache_dir`` is set and ``policy_id is
                None`` the Runner falls back to
                ``policy_factory().__class__.__name__`` (constructed
                once on the parent process at the start of
                :meth:`run`). Users who swap weights inside the same
                policy class (e.g. SmolVLA checkpoint A vs B) MUST
                pass an explicit ``policy_id`` or the cache will
                silently return stale rollouts.
            max_steps: Per-episode env step cap echoed into the cache
                key. Required when ``cache_dir`` is set because every
                backend bakes ``max_steps`` into its constructor and
                :class:`GauntletEnv` does not expose a public getter;
                an honest cache key needs the value as input. Ignored
                when ``cache_dir is None``.

        Raises:
            ValueError: If ``n_workers < 1``, ``start_method != "spawn"``,
                ``video_fps`` is not a positive int, or ``cache_dir`` is
                set without ``max_steps``.
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
        if record_video and (not isinstance(video_fps, int) or video_fps <= 0):
            raise ValueError(f"video_fps must be a positive int; got {video_fps!r}.")
        # Cache requires an explicit max_steps because the cache key
        # depends on it and GauntletEnv does not expose a public getter.
        # See ``docs/polish-exploration-incremental-cache.md`` §2.
        if cache_dir is not None and max_steps is None:
            raise ValueError(
                "cache_dir is set but max_steps is None; the cache key depends on "
                "max_steps and GauntletEnv does not expose it. Pass max_steps "
                "explicitly (matching the value baked into env_factory)."
            )
        self._n_workers = n_workers
        self._env_factory = env_factory
        self._start_method = start_method
        self._trajectory_dir = trajectory_dir
        self._record_video = record_video
        self._video_dir = video_dir
        self._video_fps = video_fps
        self._record_only_failures = record_only_failures
        self._cache_dir = cache_dir
        self._policy_id = policy_id
        self._max_steps = max_steps
        # Lazily constructed on the first ``run`` call when caching is
        # enabled. Held on the instance so :meth:`cache_stats` can be
        # called by callers (and by the CLI) after the run completes.
        self._cache: EpisodeCache | None = None
        # B-26: cell index at which the last :meth:`run` call pruned,
        # or ``None`` when the run completed every cell (the common
        # case — pruning is opt-in). Exposed via
        # :attr:`last_pruned_at_cell` so the CLI can stamp the value
        # onto the Report after :func:`gauntlet.report.analyze.build_report`
        # without :func:`build_report` itself needing to know about
        # pruning.
        self._last_pruned_at_cell: int | None = None

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        policy_factory: Callable[[], Policy],
        suite: Suite,
        prune_after_cells: int | None = None,
        prune_min_success: float | None = None,
        trajectory_format: TrajectoryFormat = "npz",
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
            prune_after_cells: Optuna-style early-stop knob (B-26).
                When set, the Runner checks the running overall success-
                rate Wilson 95% CI every ``prune_after_cells`` cells
                and aborts the run when the CI clearly straddles
                ``prune_min_success`` from above (lower bound greater)
                or below (upper bound less). Must be paired with
                ``prune_min_success``; passing one without the other
                raises :class:`ValueError`. Default ``None`` keeps the
                run byte-identical to the pre-B-26 path. **Pruning
                destroys the "fair sample of the perturbation space"
                reading by definition** — only use it as a quick
                pass/fail screen, never for a calibrated report.
                Force-disabled with a :class:`UserWarning` when
                ``suite.sampling == "sobol"`` (Sobol indices need full
                sample structure per B-19).
            prune_min_success: Pass/fail threshold for the prune
                decision (a value in ``[0, 1]``). See
                ``prune_after_cells``.
            trajectory_format: B-23 trajectory output format. ``"npz"``
                (the default) keeps the pre-B-23 byte-identical
                behaviour: one ``np.savez_compressed`` per episode,
                no ``pyarrow`` import. ``"parquet"`` writes one
                Parquet file per episode instead of NPZ; ``"both"``
                writes both side-by-side. Both non-default modes
                require the ``[parquet]`` extra (``pip install
                "gauntlet[parquet]"``); a missing extra raises a
                clean :class:`ImportError` from the worker on the
                first write. Ignored when ``trajectory_dir is
                None`` (no trajectory writes happen at all).

        Returns:
            List of :class:`Episode` records sorted by
            ``(cell_index, episode_index)``. Length is exactly
            ``suite.num_cells() * suite.episodes_per_cell`` for an
            un-pruned run; shorter when pruning kicks in (only the
            cells the Runner actually rolled appear).

        Raises:
            ValueError: If ``suite.env`` is not registered at dispatch time.
                The Suite loader imports backends on demand (RFC-005 §11.2);
                any env that reaches the Runner should already be registered.
                Also raised when only one of ``prune_after_cells`` /
                ``prune_min_success`` is supplied, when
                ``prune_after_cells <= 0``, or when
                ``prune_min_success`` is outside ``[0, 1]``.
        """
        # ----------------------------------------------------------------
        # B-26 prune-config validation. Both parameters travel together;
        # silently honouring one without the other would let a typo
        # ("--prune-after-cells 5" without "--prune-min-success ...")
        # produce a misleading no-op run. Reject loudly at the boundary.
        # ----------------------------------------------------------------
        prune_enabled = self._validate_prune_config(
            suite=suite,
            prune_after_cells=prune_after_cells,
            prune_min_success=prune_min_success,
        )
        # Reset the last-prune sticky state so a Runner reused across
        # multiple :meth:`run` calls never leaks the previous run's
        # value onto a fresh report.
        self._last_pruned_at_cell = None
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

        # Resolve the video config in the parent process so workers
        # never have to mkdir or pick a default path. ``None`` keeps
        # the byte-identical opt-out path through ``execute_one``.
        video_config = self._resolve_video_config()
        if video_config is not None:
            video_config.video_dir.mkdir(parents=True, exist_ok=True)

        work_items = self._build_work_items(suite)

        # Cache lookup happens BEFORE dispatch — only cache misses go to
        # the executor. ``cache_dir is None`` short-circuits the entire
        # block: no EpisodeCache is constructed, no per-cell key
        # computation runs, and the dispatch path is byte-identical to
        # the pre-PR behaviour.
        cached_episodes: list[Episode] = []
        items_to_run: list[WorkItem] = work_items
        cache_keys: dict[tuple[int, int], str] = {}
        if self._cache_dir is not None:
            self._cache = EpisodeCache(root=self._cache_dir)
            resolved_policy_id = self._resolve_policy_id(policy_factory)
            assert self._max_steps is not None  # narrowed by __init__ guard
            cached_episodes, items_to_run, cache_keys = self._partition_by_cache(
                self._cache,
                work_items,
                suite=suite,
                policy_id=resolved_policy_id,
                max_steps=self._max_steps,
            )

        if items_to_run:
            if prune_enabled:
                # Type narrowing for mypy --strict: the validator above
                # only returns True when both knobs are non-None.
                assert prune_after_cells is not None
                assert prune_min_success is not None
                fresh = self._run_with_pruning(
                    env_factory,
                    policy_factory,
                    items_to_run,
                    video_config=video_config,
                    trajectory_format=trajectory_format,
                    cached_episodes=cached_episodes,
                    suite=suite,
                    prune_after_cells=prune_after_cells,
                    prune_min_success=prune_min_success,
                )
            elif self._n_workers == 1:
                fresh = self._run_in_process(
                    env_factory,
                    policy_factory,
                    items_to_run,
                    video_config=video_config,
                    trajectory_format=trajectory_format,
                )
            else:
                fresh = self._run_pool(
                    env_factory,
                    policy_factory,
                    items_to_run,
                    video_config=video_config,
                    trajectory_format=trajectory_format,
                )
        else:
            fresh = []

        # Stamp B-22 provenance onto every freshly-rolled Episode BEFORE
        # the cache write so a future cache hit returns an Episode that
        # carries the original commit's provenance (the bit that lets a
        # repro from a different checkout detect "this cache entry is
        # from commit X, you are on commit Y"). Cached episodes already
        # carry their original provenance and are NOT mutated here.
        gauntlet_version = capture_gauntlet_version()
        suite_hash = compute_suite_hash(suite)
        git_commit = capture_git_commit()
        fresh = [
            ep.model_copy(
                update={
                    "gauntlet_version": gauntlet_version,
                    "suite_hash": suite_hash,
                    "git_commit": git_commit,
                }
            )
            for ep in fresh
        ]

        # Persist newly-rolled Episodes to the cache (a no-op when the
        # cache is disabled; cache_keys is empty in that case).
        if self._cache is not None:
            for episode in fresh:
                key = cache_keys.get((episode.cell_index, episode.episode_index))
                if key is not None:
                    self._cache.put(key, episode)

        # Combine cached + freshly-rolled episodes; the sort below
        # restores the canonical (cell_index, episode_index) ordering.
        episodes = cached_episodes + fresh

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
    # Cache helpers.
    # ------------------------------------------------------------------

    def _resolve_policy_id(self, policy_factory: Callable[[], Policy]) -> str:
        """Return the caller-supplied ``policy_id`` or a class-name fallback.

        When the user passed an explicit ``policy_id`` to
        :meth:`__init__`, return it verbatim — no factory call needed.
        Otherwise call ``policy_factory()`` once on the parent process
        and use the resulting policy's class name. The factory call is a
        side-effecting one (e.g. it may load a torch checkpoint), but
        this is the only way to derive a default ID; users with
        expensive factories who know the class is stable should pass an
        explicit ``policy_id`` to avoid the construction.
        """
        if self._policy_id is not None:
            return self._policy_id
        sample = policy_factory()
        return type(sample).__name__

    def _partition_by_cache(
        self,
        cache: EpisodeCache,
        work_items: list[WorkItem],
        *,
        suite: Suite,
        policy_id: str,
        max_steps: int,
    ) -> tuple[list[Episode], list[WorkItem], dict[tuple[int, int], str]]:
        """Split work items into (cache hits, cache misses, key lookup table).

        For each WorkItem, derives the env seed via :func:`extract_env_seed`
        (pure; no env construction), computes the cache key via
        :meth:`EpisodeCache.make_key`, and asks the cache for the
        Episode. Hits are returned in the first list (sorted by the
        Runner's outer ``run`` method); misses are returned as the
        items_to_run list along with a ``{(cell, ep): key}`` lookup so
        the post-dispatch ``cache.put`` loop knows which key to use.
        """
        hits: list[Episode] = []
        misses: list[WorkItem] = []
        keys: dict[tuple[int, int], str] = {}
        for item in work_items:
            env_seed = extract_env_seed(item.episode_seq)
            key = EpisodeCache.make_key(
                suite,
                axis_config=item.perturbation_values,
                seed=env_seed,
                episodes_per_cell=item.episodes_per_cell,
                max_steps=max_steps,
                env_name=suite.env,
                policy_id=policy_id,
            )
            cached = cache.get(key)
            if cached is not None:
                hits.append(cached)
            else:
                misses.append(item)
                keys[(item.cell_index, item.episode_index)] = key
        return hits, misses, keys

    def cache_stats(self) -> dict[str, int]:
        """Return the in-process cache hit / miss / put counters.

        When caching is disabled (``cache_dir=None``) the Runner has no
        cache to query; this returns zeros for parity with the active
        path.
        """
        if self._cache is None:
            return {"hits": 0, "misses": 0, "puts": 0}
        return self._cache.stats()

    # ------------------------------------------------------------------
    # B-26 early-stop pruning.
    # ------------------------------------------------------------------

    @property
    def last_pruned_at_cell(self) -> int | None:
        """Cell count at which the last :meth:`run` aborted, or ``None``.

        Mirrors :meth:`cache_stats` — exposed so the CLI can stamp
        :attr:`gauntlet.report.schema.Report.pruned_at_cell` on the
        report after :func:`gauntlet.report.analyze.build_report` runs,
        without ``build_report`` itself needing to know about pruning.
        Reset to ``None`` at the top of every :meth:`run` call so a
        Runner reused across runs never leaks the previous run's
        value.
        """
        return self._last_pruned_at_cell

    def _validate_prune_config(
        self,
        *,
        suite: Suite,
        prune_after_cells: int | None,
        prune_min_success: float | None,
    ) -> bool:
        """Validate the ``(prune_after_cells, prune_min_success)`` pair.

        Returns ``True`` when pruning is enabled and the run loop
        should take the chunked dispatch path; ``False`` when the
        knobs are absent (the byte-identical pre-B-26 path) or when
        ``suite.sampling == "sobol"`` (force-disabled with a warning).
        Raises :class:`ValueError` for half-set configurations or
        out-of-range values — those are user errors at the call
        boundary, not silent no-ops.
        """
        both_none = prune_after_cells is None and prune_min_success is None
        both_set = prune_after_cells is not None and prune_min_success is not None
        if both_none:
            return False
        if not both_set:
            raise ValueError(
                "prune_after_cells and prune_min_success must be supplied together; "
                f"got prune_after_cells={prune_after_cells!r}, "
                f"prune_min_success={prune_min_success!r}.",
            )
        # Both set — validate ranges.
        assert prune_after_cells is not None  # mypy narrowing
        assert prune_min_success is not None  # mypy narrowing
        if prune_after_cells <= 0:
            raise ValueError(
                f"prune_after_cells must be > 0; got {prune_after_cells}.",
            )
        if not (0.0 <= prune_min_success <= 1.0):
            raise ValueError(
                f"prune_min_success must be in [0, 1]; got {prune_min_success}.",
            )
        # Sobol indices need the full sample structure (B-19); pruning
        # would invalidate them. Warn and force-disable rather than
        # error — the caller may have set the prune knobs in a wrapper
        # script and a Sobol sweep landing there should still complete.
        if suite.sampling == "sobol":
            warnings.warn(
                "Early-stop pruning is force-disabled for sampling='sobol' "
                "(Sobol sensitivity indices need the full sample structure "
                "per B-19). Running the full sweep.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True

    def _run_with_pruning(
        self,
        env_factory: Callable[[], GauntletEnv],
        policy_factory: Callable[[], Policy],
        work_items: list[WorkItem],
        *,
        video_config: VideoConfig | None,
        trajectory_format: TrajectoryFormat,
        cached_episodes: list[Episode],
        suite: Suite,
        prune_after_cells: int,
        prune_min_success: float,
    ) -> list[Episode]:
        """Execute work items in cell-aligned chunks with a Wilson-CI prune check.

        Each chunk covers ``prune_after_cells`` *additional* cells'
        worth of work items (cells, not items — the prune knob is
        per-cell per the B-26 spec). After every chunk the running
        overall success rate is computed across BOTH cached episodes
        and freshly-rolled episodes; if the Wilson 95% CI clearly
        falls on one side of ``prune_min_success`` the run aborts and
        any not-yet-dispatched work items are dropped.

        We force ``n_workers == 1`` for the chunked path: a Pool's
        ``map`` blocks until every item completes, which would
        defeat the entire point of early-stop. Honest pool-based
        pruning would need an async dispatch refactor that is out of
        scope for B-26.
        """
        # Each cell contributes ``episodes_per_cell`` work items. The
        # work items are sorted (cell_index, episode_index) ascending
        # by :meth:`_build_work_items`, so a contiguous chunk of
        # ``chunk_items`` items corresponds to a contiguous range of
        # cells once we account for the cells already covered by
        # cached_episodes.
        eps_per_cell = suite.episodes_per_cell
        chunk_items = max(1, prune_after_cells * eps_per_cell)

        # ``completed_cells`` counts BOTH cached and freshly-rolled
        # cells — a cache hit is a real result that contributes to
        # the success-rate CI, so it should also count toward "cells
        # done" for the prune cadence.
        completed_cells = self._unique_cells_in(cached_episodes)
        running_success = sum(1 for ep in cached_episodes if ep.success)
        running_n = len(cached_episodes)

        env = env_factory()
        fresh: list[Episode] = []
        try:
            cursor = 0
            while cursor < len(work_items):
                chunk = work_items[cursor : cursor + chunk_items]
                cursor += len(chunk)
                for item in chunk:
                    episode = execute_one(
                        env,
                        policy_factory,
                        item,
                        trajectory_dir=self._trajectory_dir,
                        trajectory_format=trajectory_format,
                        video_config=video_config,
                    )
                    fresh.append(episode)
                    running_n += 1
                    if episode.success:
                        running_success += 1
                # Update completed_cells to reflect every cell touched
                # by this chunk (including its cached_episodes share).
                completed_cells = self._unique_cells_in(cached_episodes + fresh)
                if self._should_prune(
                    successes=running_success,
                    n=running_n,
                    threshold=prune_min_success,
                ):
                    self._last_pruned_at_cell = completed_cells
                    break
        finally:
            env.close()
        return fresh

    @staticmethod
    def _unique_cells_in(episodes: list[Episode]) -> int:
        """Count distinct ``cell_index`` values across an episode list."""
        return len({ep.cell_index for ep in episodes})

    @staticmethod
    def _should_prune(*, successes: int, n: int, threshold: float) -> bool:
        """Return True iff the Wilson 95% CI clearly straddles ``threshold``.

        ``ci_low > threshold`` -> policy clearly PASSES the bar; prune.
        ``ci_high < threshold`` -> policy clearly FAILS the bar; prune.
        Otherwise the CI brackets the threshold and we keep sampling.
        """
        ci_low, ci_high = wilson_interval(successes, n)
        if ci_low is None or ci_high is None:
            # n == 0 — no information yet, keep going.
            return False
        return ci_low > threshold or ci_high < threshold

    # ------------------------------------------------------------------
    # Execution paths.
    # ------------------------------------------------------------------

    def _resolve_video_config(self) -> VideoConfig | None:
        """Build the per-run video configuration, or ``None`` if disabled.

        Resolved once in the parent process so workers receive a
        ready-to-use config (no path-defaulting in the worker hot
        path). Default ``video_dir`` strategy mirrors the
        ``trajectory_dir / "videos"`` convention so a single
        ``Runner(record_video=True, trajectory_dir=...)`` Just Works.
        """
        if not self._record_video:
            return None
        if self._video_dir is not None:
            video_dir = self._video_dir
        elif self._trajectory_dir is not None:
            video_dir = self._trajectory_dir / "videos"
        else:
            video_dir = Path("videos")
        return VideoConfig(
            video_dir=video_dir,
            fps=self._video_fps,
            record_only_failures=self._record_only_failures,
        )

    def _run_in_process(
        self,
        env_factory: Callable[[], GauntletEnv],
        policy_factory: Callable[[], Policy],
        work_items: list[WorkItem],
        *,
        video_config: VideoConfig | None,
        trajectory_format: TrajectoryFormat,
    ) -> list[Episode]:
        """Single-process fast path. Loads the env once, reuses it.

        Used when ``n_workers == 1``. Produces Episodes that are
        bit-identical to the multi-worker path for the same inputs
        because ``execute_one`` is the same function in both cases.
        """
        env = env_factory()
        try:
            return [
                execute_one(
                    env,
                    policy_factory,
                    item,
                    trajectory_dir=self._trajectory_dir,
                    trajectory_format=trajectory_format,
                    video_config=video_config,
                )
                for item in work_items
            ]
        finally:
            env.close()

    def _run_pool(
        self,
        env_factory: Callable[[], GauntletEnv],
        policy_factory: Callable[[], Policy],
        work_items: list[WorkItem],
        *,
        video_config: VideoConfig | None,
        trajectory_format: TrajectoryFormat,
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
            trajectory_format=trajectory_format,
            video_config=video_config,
        )
        with ctx.Pool(
            processes=self._n_workers,
            initializer=pool_initializer,
            initargs=(init_args,),
        ) as pool:
            return list(pool.map(run_work_item, work_items))
