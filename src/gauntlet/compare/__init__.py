"""Compare-side polish modules — see backlog B-24.

Public surface (kept minimal; the core compare logic still lives in
:mod:`gauntlet.cli` ``_build_compare`` for now):

* :func:`to_github_summary` — render a ``compare.json`` payload as a
  GitHub Actions ``$GITHUB_STEP_SUMMARY``-compatible markdown blob.
"""

from __future__ import annotations

from gauntlet.compare.github_summary import to_github_summary as to_github_summary

__all__ = ["to_github_summary"]
