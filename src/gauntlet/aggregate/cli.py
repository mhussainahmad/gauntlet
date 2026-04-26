"""CLI helpers for ``gauntlet aggregate`` — Phase 3 Task 19.

Mirrors the layout of :func:`gauntlet.diff._build_diff` and the
``gauntlet.compare`` helpers: the @app.command shell stays in the top-
level :mod:`gauntlet.cli`, the per-subcommand business logic lives next
to its module surface so the test layer can drive it without spinning
up a Typer ``CliRunner``.

This module owns three things:

* :func:`build_cluster_payload` — pure function from a scan root +
  ``max_clusters`` cap to the JSON-friendly cluster payload that
  ``gauntlet aggregate --cluster-output`` writes. Wraps
  :func:`gauntlet.aggregate.cluster_fleet_failures` and the private
  ``_result_to_payload`` serialiser so the CLI stays a thin wrapper.
* :func:`write_cluster_json` — write the payload to disk. Lifted out so
  the test layer can pin the exact ``json.dumps`` shape (sorted keys,
  ``indent=2``, trailing newline) without re-implementing it inline.
* :func:`format_cluster_summary` — the one-line stderr summary the CLI
  echoes after the artefact is written. Pure / testable in isolation
  rather than buried inside the typer handler.

The actual ``@app.command("aggregate")`` Typer wiring stays in
``gauntlet/cli.py`` — that's where the ``--out`` / ``--html`` /
``--persistence-threshold`` flags live and the new clustering flags
slot in alongside them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

from gauntlet.aggregate.fleet_clustering import (
    FleetClusteringResult,
    _result_to_payload,
    cluster_fleet_failures,
)

__all__ = [
    "build_cluster_payload",
    "format_cluster_summary",
    "write_cluster_json",
]


# JSON-shape alias — recursive type for what ``json.dumps`` accepts.
# Identical to the alias in :mod:`gauntlet.aggregate.fleet_clustering`;
# kept local rather than imported so this CLI helper has no
# cross-module type dependency the linter would flag as unused.
_JsonValue: TypeAlias = (
    float
    | int
    | str
    | bool
    | None
    | dict[str, "_JsonValue"]
    | list["_JsonValue"]
    | tuple["_JsonValue", ...]
)


def build_cluster_payload(
    directory: Path,
    *,
    max_clusters: int,
) -> tuple[FleetClusteringResult, dict[str, _JsonValue]]:
    """Run the clustering pipeline and return both the result + JSON payload.

    Two-tuple return so the CLI can echo human-friendly counts (from
    the structured :class:`FleetClusteringResult`) AND serialise the
    JSON-friendly payload (from the dump dict) in one call.

    Args:
        directory: Scan root passed straight through to
            :func:`cluster_fleet_failures`. Path validation /
            sandboxing is the lower layer's responsibility — see
            :func:`gauntlet.security.safe_join`.
        max_clusters: Hard cap on the number of clusters to return.
            ``< 1`` is rejected by the lower layer with
            :class:`ValueError`.

    Returns:
        A ``(result, payload)`` tuple. ``payload`` is suitable for
        ``json.dumps`` directly; ``result`` is the full
        :class:`FleetClusteringResult` for callers that want to
        consume it programmatically.
    """
    result = cluster_fleet_failures(directory, max_clusters=max_clusters)
    payload = _result_to_payload(result)
    return result, payload


def write_cluster_json(payload: dict[str, _JsonValue], path: Path) -> None:
    """Write the cluster payload to ``path`` as pretty-printed JSON.

    Format matches the rest of the harness's ``--out`` artefact
    contract: ``indent=2``, sorted keys for byte-identical re-runs, and
    a trailing newline so the file plays nicely with text-mode tools
    that expect a final ``\\n``.

    The parent directory is created if it doesn't exist — the CLI
    handler accepts a path under a fresh ``--out`` directory and we
    don't want to crash because ``mkdir -p`` was implicit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(encoded + "\n", encoding="utf-8")


def format_cluster_summary(result: FleetClusteringResult) -> str:
    """Render the one-line summary the CLI echoes to stderr.

    Pure / testable. The format follows the rest of the aggregate
    subcommand's stderr style — leading two-space indent, lowercase
    label, parenthetical caveat for the silhouette score (which is
    informational and may legitimately be ``None``).
    """
    n_clusters = len(result.clusters)
    sil = "n/a" if result.silhouette is None else f"{result.silhouette:+.3f}"
    return (
        f"  failure-mode clusters: {n_clusters} "
        f"({result.n_unique_failures} unique signature(s); silhouette={sil})"
    )
