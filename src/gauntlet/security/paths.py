"""Path-traversal safe join — Phase 2.5 Task 16.

The harness writes Parquet trajectory dumps, MP4 videos, cache files,
and report artefacts under a user-supplied output root. When the
filename component is itself derived from user input (an axis name
echoed into a filename, a cluster id, etc.), naive
``base / user_part`` is a CWE-22 vector — a value of ``../etc/passwd``
or an absolute path string would escape the sandbox.

This module exposes one helper, :func:`safe_join`, which performs the
join and then verifies the resolved path is still under the resolved
``base``. Any escape raises :class:`PathTraversalError` (a
:class:`ValueError` subclass) so call-sites can translate to a clean
CLI error without an unhandled traceback.

The threat model is documented in :doc:`docs/security`. The honest
scope: gauntlet is a developer tool, the user is the trust root, and
``safe_join`` is a defence-in-depth layer applied at the few internal
boundaries where the filename component is *not* fully user-controlled.
Boundaries that intentionally accept cross-dir paths (e.g. a
``pilot_report`` resolved relative to a suite YAML's parent) are
called out explicitly in the PR body and not hardened with this
helper.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "PathTraversalError",
    "safe_join",
]


class PathTraversalError(ValueError):
    """Raised by :func:`safe_join` when the resolved path escapes ``base``.

    Subclasses :class:`ValueError` so existing CLI-error envelopes that
    catch ``ValueError`` keep working without an explicit branch.
    """


def safe_join(
    base: Path | str,
    *parts: str | Path,
    follow_symlinks: bool = True,
) -> Path:
    """Join ``base`` with ``parts`` and reject any escape from ``base``.

    Behaviour:

    1. ``base`` is resolved to an absolute path. If it does not exist,
       it is still resolved positionally (``Path.resolve(strict=False)``)
       so the helper can be used to construct paths *before* the output
       directory exists.
    2. ``parts`` are joined onto ``base`` and the result is resolved.
    3. The resolved result MUST be ``base`` itself or a descendant of
       it. Otherwise :class:`PathTraversalError` is raised.

    The check catches:

    * ``..`` segments that escape via parent-traversal
      (``safe_join(base, "../etc/passwd")``).
    * Absolute-path injection where a component happens to be an
      absolute path; ``Path("/a") / "/b" == Path("/b")``, so naive joins
      silently drop ``base``.
    * Symlink escapes when ``follow_symlinks=False`` — in that mode the
      helper rejects any path whose components include a symlink that
      points outside ``base``.

    Args:
        base: The trusted root. Resolved to an absolute path.
        *parts: One or more path components joined onto ``base``. May
            be ``str`` or :class:`Path`.
        follow_symlinks: If True (default), the resolved path is allowed
            to traverse symlinks as long as the final resolved path is
            still under ``base``. If False, a stricter check rejects any
            symlink whose target leaves ``base``. Default ``True`` keeps
            backward compatibility for sinks that legitimately accept a
            symlinked output dir; pass ``False`` for hardened call-sites
            (e.g. opening a user-supplied file from a shared CI runner).

    Returns:
        The resolved absolute path inside ``base``.

    Raises:
        PathTraversalError: If the resolved path would escape ``base``
            (parent-traversal, absolute-path injection, or — when
            ``follow_symlinks=False`` — a symlink escape).
    """
    base_resolved = Path(base).resolve(strict=False)

    if not parts:
        # Degenerate case: ``safe_join(base)`` → ``base``. Still useful
        # for callers that want to canonicalise ``base`` through the
        # same resolver path the joined form uses.
        return base_resolved

    # Build the joined path positionally. ``Path.__truediv__`` happily
    # discards ``base`` if a part is absolute, so we explicitly reject
    # any absolute-path component before constructing the joined form.
    # That makes the failure mode explicit rather than silently mapping
    # ``safe_join("/srv/out", "/etc/passwd")`` → ``/etc/passwd``.
    joined: Path = base_resolved
    for part in parts:
        part_path = Path(part)
        if part_path.is_absolute():
            raise PathTraversalError(
                f"absolute-path injection rejected: base={base_resolved!s} part={part_path!s}",
            )
        joined = joined / part_path

    # ``Path.resolve(strict=False)`` collapses ``..`` segments and
    # follows symlinks if they exist on disk. For non-existent paths it
    # returns the lexically-resolved form. Either way, the post-resolve
    # ancestry check is the security gate.
    resolved = joined.resolve(strict=False)

    # is_relative_to was added in Python 3.9 and is the canonical
    # ancestry check. Equivalent to the
    # ``str(resolved).startswith(str(base) + os.sep)`` pattern but
    # without the trailing-slash and case-sensitivity gotchas.
    if resolved != base_resolved and not resolved.is_relative_to(base_resolved):
        raise PathTraversalError(
            f"path-traversal rejected: resolved={resolved!s} escapes base={base_resolved!s}",
        )

    if not follow_symlinks:
        # Strict mode — walk every existing component of the joined
        # (pre-resolve) path. If any component is a symlink whose
        # target's resolved form leaves ``base``, reject. We do this
        # rather than ``Path.is_symlink()`` on the final path because
        # the escape can be triggered by a symlink at any depth.
        cursor = base_resolved
        for part in parts:
            for segment in Path(part).parts:
                cursor = cursor / segment
                if cursor.is_symlink():
                    target = cursor.resolve(strict=False)
                    if target != base_resolved and not target.is_relative_to(base_resolved):
                        raise PathTraversalError(
                            f"symlink-escape rejected: {cursor!s} -> {target!s} "
                            f"escapes base={base_resolved!s}",
                        )

    return resolved
