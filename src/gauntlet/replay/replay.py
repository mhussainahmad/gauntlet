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

from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.base import Policy
from gauntlet.replay.overrides import validate_overrides
from gauntlet.runner import Episode, execute_one
from gauntlet.runner.worker import WorkItem
from gauntlet.suite.schema import Suite

__all__ = [
    "replay_one",
]


def _default_env_factory() -> TabletopEnv:
    """Module-level default env factory for the in-process driver."""
    return TabletopEnv()


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
    if isinstance(raw_n_cells, int) and isinstance(raw_eps, int):
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
    env_factory: Callable[[], TabletopEnv] | None = None,
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
        env_factory: Optional zero-arg env factory. Defaults to a
            stock :class:`TabletopEnv`. The env is closed in a
            ``finally`` after the rollout.

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

    factory = env_factory if env_factory is not None else _default_env_factory
    env = factory()
    try:
        return execute_one(env, policy_factory, item)
    finally:
        env.close()
