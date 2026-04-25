"""In-process trajectory replay — re-simulate one Episode, optionally off-grid.

See ``docs/phase2-rfc-004-trajectory-replay.md``. :func:`replay_one` is
the library primitive: given a target :class:`Episode`, the suite it
came from, a policy factory, and zero-or-more axis overrides, produce
a fresh Episode that is bit-identical to the original when no overrides
are passed, and differs only by the overridden axis value(s) when they
are.

The implementation reuses :func:`gauntlet.runner.execute_one` directly
— not :meth:`Runner.run` — because the RFC §7 argument is load-bearing:
a one-cell synthetic Suite handed to the Runner would spawn from a
one-cell tree, and ``master.spawn(1)[0]`` is a different node than
``master.spawn(n_cells)[cell_index]``. Reconstructing the original
spawn node directly from ``(master_seed, n_cells, episodes_per_cell,
cell_index, episode_index)`` is the only honest path to bit-identity.

Legacy fallback (RFC §9 test 11): Episodes produced before Phase 2
Task 4 step 1 may not carry ``n_cells`` / ``episodes_per_cell`` in
their metadata. In that case we fall back to the values declared on
the suite, emitting a warning to stderr. If the suite YAML has been
edited between the run and the replay, the fallback values will not
match the original run's topology and bit-identity will silently
break — the warning is the user's cue to check.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory
from gauntlet.policy.base import Policy
from gauntlet.replay.overrides import validate_overrides
from gauntlet.runner import Episode, execute_one
from gauntlet.runner.provenance import (
    capture_gauntlet_version,
    capture_git_commit,
    compute_suite_hash,
)
from gauntlet.runner.worker import WorkItem
from gauntlet.suite.schema import Suite

__all__ = [
    "replay_one",
]


def _reconstruct_episode_seq(
    *,
    master_seed: int,
    n_cells: int,
    episodes_per_cell: int,
    cell_index: int,
    episode_index: int,
) -> np.random.SeedSequence:
    """Rebuild the exact :class:`numpy.random.SeedSequence` node the
    original Runner used for this (cell, episode) pair.

    This mirrors :meth:`gauntlet.runner.Runner._build_work_items`
    byte-for-byte — see also the module docstring of
    :mod:`gauntlet.runner.runner` ("Seed derivation" section). The two
    spawn levels (cells, then episodes) are not interchangeable with a
    single flat spawn; flattening produces a different node, which
    would silently break bit-identity.
    """
    master = np.random.SeedSequence(master_seed)
    cell_seqs = master.spawn(n_cells)
    episode_seqs = cell_seqs[cell_index].spawn(episodes_per_cell)
    return episode_seqs[episode_index]


def _topology_from_metadata(target: Episode, suite: Suite) -> tuple[int, int]:
    """Extract ``(n_cells, episodes_per_cell)`` from the Episode metadata.

    Falls back to the suite's values with a stderr warning for legacy
    Episodes that predate Phase 2 Task 4 step 1. The warning is a soft
    signal only — bit-identity holds when the suite is unchanged and
    breaks quietly when it is edited. The user is expected to re-run
    the original suite to get a topology-echoing Episode if they need
    guaranteed bit-identity.
    """
    raw_n_cells = target.metadata.get("n_cells")
    raw_eps = target.metadata.get("episodes_per_cell")
    # `type(...) is int` (not isinstance) so a stray bool doesn't slip
    # past — bool is a subclass of int and would otherwise type-match
    # here, which would silently feed the spawn tree the wrong shape.
    if type(raw_n_cells) is int and type(raw_eps) is int:
        return raw_n_cells, raw_eps
    # Legacy Episode — topology fields absent.
    fallback_n_cells = suite.num_cells()
    fallback_eps = suite.episodes_per_cell
    print(
        f"warning: Episode {target.cell_index}:{target.episode_index} "
        f"predates topology echo (no metadata[n_cells]/[episodes_per_cell]); "
        f"falling back to suite values "
        f"(n_cells={fallback_n_cells}, episodes_per_cell={fallback_eps}). "
        "Bit-identity holds only if the suite has not been edited since "
        "the original run.",
        file=sys.stderr,
    )
    return fallback_n_cells, fallback_eps


def _build_perturbation_values(
    target: Episode,
    overrides: dict[str, float],
) -> dict[str, float]:
    """Apply *overrides* on top of the target's perturbation_config.

    Iteration order is ``target.perturbation_config.keys()`` — i.e.
    the order the original run applied the axes, inherited from the
    suite's axis-declaration order at run time. RFC §6 is the load-
    bearing argument for this: ``camera_offset_x`` and
    ``camera_offset_y`` are not orthogonal (each writes the full
    ``cam_pos``), so preserving application order is how we guarantee
    bit-identity. Overrides come in via the ``overrides`` dict but do
    NOT change iteration order — only the values.

    Override keys that reference axes absent from the target's
    perturbation_config are silently ignored here; name validation
    against ``suite.axes`` happens in :func:`validate_overrides`, and
    an axis in ``suite.axes`` but missing from
    ``target.perturbation_config`` cannot occur (the Runner's contract
    is that every declared axis appears in every Episode).
    """
    result: dict[str, float] = {}
    for name, original_value in target.perturbation_config.items():
        result[name] = overrides.get(name, original_value)
    return result


def replay_one(
    *,
    target: Episode,
    suite: Suite,
    policy_factory: Callable[[], Policy],
    overrides: dict[str, float] | None = None,
    env_factory: Callable[[], GauntletEnv] | None = None,
) -> Episode:
    """Re-simulate *target* with optional axis overrides.

    Args:
        target: The Episode to replay. ``target.cell_index``,
            ``target.episode_index``, ``target.perturbation_config``
            and ``target.metadata["master_seed"]`` carry the full
            identity tuple. ``target.metadata["n_cells"]`` and
            ``target.metadata["episodes_per_cell"]`` are consulted to
            reconstruct the SeedSequence tree; missing values fall back
            to the suite with a warning to stderr.
        suite: The Suite the Episode came from. Asserted to match
            ``target.suite_name``; mismatch raises :class:`ValueError`.
            Used for override name / envelope validation.
        policy_factory: Zero-arg callable returning a fresh
            :class:`Policy`. Not constrained to be picklable — replay
            runs in-process, so closures and non-pickleable policies
            (e.g. torch models with CUDA tensors) are fine here.
        overrides: Optional ``{axis_name: float}`` to replace values on
            the replayed rollout. Empty / ``None`` yields a zero-
            override replay, which RFC §9 pins as bit-identical to
            the original Episode.
        env_factory: Optional zero-arg env factory. When ``None`` (the
            default), dispatches via
            :func:`gauntlet.env.registry.get_env_factory` on
            ``suite.env``, matching the Runner's own dispatch contract.
            The env is closed in a ``finally`` after the rollout.

    Returns:
        A freshly produced :class:`Episode`. Identical in every
        Pydantic field to *target* when *overrides* is empty; the
        ``perturbation_config`` picks up overridden values otherwise,
        and ``success`` / ``step_count`` / ``total_reward`` reflect
        the outcome of the overridden rollout.

    Raises:
        ValueError: if the suite name disagrees with
            ``target.suite_name``, if ``metadata["master_seed"]`` is
            absent or non-integer, or if the override validation fails.
    """
    if suite.name != target.suite_name:
        raise ValueError(
            f"suite name mismatch: episode.suite_name={target.suite_name!r} vs "
            f"suite.name={suite.name!r}"
        )

    overrides = overrides if overrides is not None else {}
    # Raises OverrideError (a ValueError subclass) on any issue.
    validate_overrides(overrides, suite)

    raw_master = target.metadata.get("master_seed")
    if not isinstance(raw_master, int):
        raise ValueError(
            f"episode {target.cell_index}:{target.episode_index} is missing "
            "metadata[master_seed]; cannot reconstruct the spawn tree"
        )
    master_seed: int = raw_master

    n_cells, episodes_per_cell = _topology_from_metadata(target, suite)

    episode_seq = _reconstruct_episode_seq(
        master_seed=master_seed,
        n_cells=n_cells,
        episodes_per_cell=episodes_per_cell,
        cell_index=target.cell_index,
        episode_index=target.episode_index,
    )

    perturbation_values = _build_perturbation_values(target, overrides)

    item = WorkItem(
        suite_name=target.suite_name,
        cell_index=target.cell_index,
        episode_index=target.episode_index,
        perturbation_values=perturbation_values,
        episode_seq=episode_seq,
        master_seed=master_seed,
        n_cells=n_cells,
        episodes_per_cell=episodes_per_cell,
    )

    factory = env_factory if env_factory is not None else get_env_factory(suite.env)
    env = factory()
    try:
        replayed = execute_one(env, policy_factory, item)
    finally:
        env.close()

    # Echo the target's B-22 provenance trio onto the replayed Episode so
    # ``replayed.model_dump() == target.model_dump()`` holds for the
    # zero-override path. ``execute_one`` does not stamp these — only
    # :meth:`Runner.run` does — so without this post-population the
    # replayed Episode would carry ``None`` for all three fields and the
    # bit-identity contract from the module docstring would silently break
    # (regression introduced when B-22 / B-40 added the fields).
    #
    # Provenance is copied from *target* (preserving target identity)
    # rather than recomputed from the current checkout, because the
    # contract the tests pin is full-Episode equality against the original
    # — when run in the same checkout the two sources happen to agree, but
    # a replay across checkouts must still echo the *original* run's
    # commit / version / suite hash. For legacy Episodes that pre-date
    # B-22 (any field is ``None``), fall back to the same helpers
    # :meth:`Runner.run` uses so the replayed Episode at least carries the
    # current checkout's provenance instead of propagating ``None``.
    return replayed.model_copy(
        update={
            "gauntlet_version": (
                target.gauntlet_version
                if target.gauntlet_version is not None
                else capture_gauntlet_version()
            ),
            "suite_hash": (
                target.suite_hash if target.suite_hash is not None else compute_suite_hash(suite)
            ),
            "git_commit": (
                target.git_commit if target.git_commit is not None else capture_git_commit()
            ),
        }
    )
