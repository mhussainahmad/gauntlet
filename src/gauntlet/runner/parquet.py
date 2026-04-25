"""Per-episode Parquet trajectory dumps (B-23) — shared, queryable format.

NPZ is the right format for one-off Python scripts (numpy round-trips,
fast load, no extra deps). Parquet is the right format for the *shared*
case: a fleet operator with a Tableau seat, a pandas notebook, a polars
script, or a DuckDB CLI session run from any language. Both formats
coexist via :class:`Runner.run`'s ``trajectory_format`` knob; the
default ("npz") is byte-identical to the pre-B-23 path and never
imports pyarrow.

DuckDB one-liner against an episode-shaped output dir::

    duckdb -c "SELECT step, action_0 \
        FROM 'out/trajectories/*.parquet' \
        WHERE reward < -0.5"

DuckDB's glob expansion + Arrow-native parquet reader make this a
zero-server, sub-second query over thousands of per-episode files —
the local-first analytics fit PRODUCT.md.

Schema
------
One row per env step. Columns:

* ``step`` — int64, 0..T-1.
* ``observation_<key>[_<i>]`` — float64. 1-D obs arrays are flattened
  to one column per dimension (``observation.qpos.0``,
  ``observation.qpos.1`` …); scalar obs become ``observation.<key>``.
  Higher-rank obs (e.g. ``obs["image"]`` of shape ``(H,W,3)``) are
  skipped — Parquet wide-column blow-up is the wrong tool for image
  data; capture frames via :class:`Runner(record_video=True)` instead.
* ``action_0`` … ``action_<n-1>`` — float64. The 7-DoF action vector,
  flattened.
* ``reward`` — float64, per-step reward.
* ``terminated`` — bool, per-step terminal flag (always False until
  the last step).
* ``truncated`` — bool, per-step truncation flag.

The ``[parquet]`` extra (``pip install "gauntlet[parquet]"``) is
mandatory; calling :func:`write_parquet` without it raises a clean
:class:`ImportError` with the install hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["TrajectoryDict", "parquet_path_for", "write_parquet"]


@dataclass(frozen=True)
class TrajectoryDict:
    """In-memory trajectory snapshot fed to :func:`write_parquet`.

    All arrays share the same leading dimension ``T`` (the env step
    count). A ``T == 0`` rollout is allowed and produces an empty-but-
    well-formed Parquet file (zero rows, full schema) so a downstream
    glob over the trajectory dir never hits a missing-file error.

    Attributes:
        observations: ``{key: array}`` mirroring the env's obs dict.
            Each array is shape ``(T, *obs_shape)``; the writer
            flattens 0-D / 1-D entries to scalar / per-dim columns and
            skips higher-rank entries (with no warning — this is a
            documented schema choice, see module docstring).
        actions: ``(T, action_dim)`` float64 actions.
        rewards: ``(T,)`` float64 per-step reward.
        terminated: ``(T,)`` bool per-step terminal flag.
        truncated: ``(T,)`` bool per-step truncation flag.
    """

    observations: dict[str, NDArray[np.float64]]
    actions: NDArray[np.float64]
    rewards: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]


def parquet_path_for(trajectory_dir: Path, cell_index: int, episode_index: int) -> Path:
    """Return the canonical Parquet path for a (cell, episode).

    Mirrors :func:`gauntlet.runner.worker.trajectory_path_for` exactly
    apart from the ``.parquet`` suffix — so a side-by-side ``npz`` /
    ``parquet`` directory layout pairs up by basename and a DuckDB /
    pandas glob is deterministic across reruns.
    """
    return trajectory_dir / f"cell_{cell_index:04d}_ep_{episode_index:04d}.parquet"


def write_parquet(path: Path, trajectory: TrajectoryDict) -> Path:
    """Write one episode's trajectory to a Parquet sidecar.

    Lazy-imports :mod:`pyarrow` inside the function so the default
    install path (``trajectory_format="npz"``, the byte-identical
    pre-B-23 default) never sees pyarrow at import time. The
    ``[parquet]`` extra is mandatory for this call; missing it raises
    a clean :class:`ImportError` with the ``pip install
    "gauntlet[parquet]"`` hint.

    Args:
        path: Output ``.parquet`` path; parent dirs are created if
            missing (belt-and-braces for callers that bypass the
            Runner-mkdirs path).
        trajectory: In-memory trajectory snapshot. All arrays share the
            same leading dimension ``T``; the writer does not validate
            shapes (the Runner is the only producer and never violates
            the contract).

    Returns:
        The same ``path`` written, for caller convenience (mirrors
        the ergonomics of :func:`pathlib.Path.write_bytes`).

    Raises:
        ImportError: if the ``[parquet]`` extra is missing. Message
            includes the ``pip install "gauntlet[parquet]"`` hint.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "Parquet trajectory dumps require the optional [parquet] extra. "
            'Install with `pip install "gauntlet[parquet]"` (pulls '
            "pyarrow>=15, ~80 MB wheel)."
        ) from exc

    n_steps = int(trajectory.rewards.shape[0])
    columns: dict[str, Any] = {"step": np.arange(n_steps, dtype=np.int64)}

    # Observations: scalar -> one column; 1-D -> per-dim columns;
    # higher-rank -> skipped (image obs blow up the Parquet schema and
    # are better captured via the [video] extra instead).
    for key, arr in trajectory.observations.items():
        if arr.ndim == 1:
            # Scalar obs (one value per step). Common shape: ``(T,)``.
            columns[f"observation.{key}"] = np.asarray(arr, dtype=np.float64)
        elif arr.ndim == 2:
            # Vector obs (e.g. qpos, qvel). Shape ``(T, D)`` -> D cols.
            for i in range(arr.shape[1]):
                columns[f"observation.{key}.{i}"] = np.asarray(arr[:, i], dtype=np.float64)
        # ndim >= 3 (images, point clouds): skipped silently — see module
        # docstring for the rationale (Parquet is the wrong tool for
        # image data; use ``Runner(record_video=True)`` instead).

    # Actions: ``(T, action_dim)`` -> one column per dim.
    actions = np.asarray(trajectory.actions, dtype=np.float64)
    if actions.ndim == 2:
        for i in range(actions.shape[1]):
            columns[f"action_{i}"] = actions[:, i]
    else:
        # Defensive: a scalar-action env (action_dim == 0 collapsed)
        # still produces a single ``action_0`` column.
        columns["action_0"] = actions.reshape(-1).astype(np.float64)

    columns["reward"] = np.asarray(trajectory.rewards, dtype=np.float64)
    columns["terminated"] = np.asarray(trajectory.terminated, dtype=np.bool_)
    columns["truncated"] = np.asarray(trajectory.truncated, dtype=np.bool_)

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, path)
    return path
