"""Provenance capture for ``repro.json`` (B-22).

Every :class:`gauntlet.runner.Episode` produced by :class:`Runner` is
stamped with three provenance fields so the run can be re-executed
bit-identically on a fresh checkout:

* ``gauntlet_version`` — the installed distribution version.
* ``suite_hash`` — sha256 of the canonical Pydantic-serialised Suite.
* ``git_commit`` — ``git rev-parse HEAD`` if the working copy is in a
  git checkout, else ``None``.

The capture is intentionally fail-soft: a missing distribution, a
non-git checkout, or a misconfigured ``git`` binary all degrade to
``None`` so a non-installed test runner or a packaged tarball can still
emit a valid ``repro.json`` (with the matching field empty).
"""

from __future__ import annotations

import hashlib
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gauntlet.suite.schema import Suite

__all__ = [
    "capture_gauntlet_version",
    "capture_git_commit",
    "compute_suite_hash",
]


def capture_gauntlet_version() -> str | None:
    """Return the installed ``gauntlet`` distribution version, or ``None``.

    ``None`` is returned when the package is not importable as an
    installed distribution (e.g. a sys.path checkout without
    ``pip install -e .``). Wrapped in a try/except so a partially-set-up
    test environment never breaks ``Runner.run``.
    """
    try:
        return version("gauntlet")
    except PackageNotFoundError:
        return None


def compute_suite_hash(suite: Suite) -> str:
    """SHA-256 of the canonical Suite payload.

    Hashes :meth:`Suite.model_dump_json` with ``sort_keys`` semantics
    (Pydantic emits the declared field order; ``sort_keys=True`` on the
    encoded JSON makes the result invariant to future field-order
    changes). Stable across YAML reformatting because the input is the
    validated Pydantic model, not the raw file bytes.
    """
    # ``model_dump_json`` already produces a deterministic byte string
    # for a given model (Pydantic emits keys in declaration order).
    payload = suite.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_git_commit(cwd: Path | None = None) -> str | None:
    """Return ``git rev-parse HEAD`` in *cwd*, or ``None``.

    Fail-soft: returns ``None`` when ``git`` is missing, when ``cwd`` is
    not a git checkout, when the subprocess times out, or when the
    return code is non-zero. Never raises.

    A 2-second timeout is intentionally short — a healthy ``git
    rev-parse`` returns in milliseconds; anything slower means a
    locked index or a network-mounted ``.git`` and is not worth
    blocking the rollout for.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
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
