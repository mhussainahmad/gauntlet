"""Shared helpers for the Phase 2.5 T12 performance benchmark suite.

Lives under ``scripts/perf/`` next to the four ``bench_*.py`` entry
points so each script imports the same provenance / percentile /
sidecar primitives. Pure stdlib (no ``Any``, no extras) so
``uv run mypy --strict scripts/perf/_common.py`` passes cleanly.

Why a separate package directory rather than enhancing the existing
``scripts/bench_*.py`` family in place: the existing scripts (added by
earlier Polish tasks) deliberately bypass the Runner to isolate
env-primitive regressions; the T12 family runs *through* the Runner
to capture the full rollout stack. The two families are
complementary, so the new scripts live in their own subdirectory and
the existing ones stay untouched. See ``docs/benchmarks.md`` for the
side-by-side description.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import gauntlet

__all__ = [
    "capture_git_commit",
    "emit_sidecar",
    "percentile",
    "provenance_fields",
    "utc_iso8601_now",
]


def capture_git_commit() -> str | None:
    """Return ``git rev-parse HEAD`` or ``None`` on any failure.

    Mirrors :func:`gauntlet.runner.provenance.capture_git_commit` shape
    but inlined so a bench script standing on its own (for ``python
    scripts/perf/bench_rollout.py`` outside a uv-managed checkout) does
    not have to pull in the full runner provenance module just for the
    SHA. Fail-soft: any subprocess error path returns ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def utc_iso8601_now() -> str:
    """Return the current UTC time as a stable ISO 8601 string.

    Format ``YYYY-MM-DDTHH:MM:SSZ`` — the same shape ``datetime.fromisoformat``
    round-trips. Second-resolution is enough for bench-result correlation;
    sub-second precision would just add noise to the JSON diff.
    """
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def provenance_fields() -> dict[str, str | None]:
    """Bundle of ``version``, ``timestamp``, ``git_commit`` provenance fields.

    Spec'd by Phase 2.5 T12 to land in every benchmark JSON sidecar
    so a downstream regression-monitor can correlate a number with the
    exact tree it came from.
    """
    return {
        "version": gauntlet.__version__,
        "timestamp": utc_iso8601_now(),
        "git_commit": capture_git_commit(),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Returns ``0.0`` on an empty input so the JSON output stays well-formed
    even when a smoke run captures zero samples. Nearest-rank (no
    interpolation) — matches the existing ``scripts/bench_*.py`` family.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, round(pct / 100.0 * n)))
    return sorted_values[rank - 1]


def emit_sidecar(summary: dict[str, object], out_path: Path) -> None:
    """Write ``summary`` to ``out_path`` as pretty-printed JSON.

    Creates parent directories if they do not exist. The summary dict
    uses ``object`` (not ``Any``) values because every leaf the four
    bench scripts emit is JSON-encodable: ``str | int | float | bool |
    None | list | dict``. Strict mypy is happy; ``json.dumps`` only
    asks for that shape at runtime.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote sidecar: {out_path}")
