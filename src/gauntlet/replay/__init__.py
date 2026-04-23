"""Trajectory replay — see ``docs/phase2-rfc-004-trajectory-replay.md``.

Public surface:

* :func:`replay_one` — the library primitive. Given a target
  :class:`gauntlet.runner.Episode`, the suite it came from, a policy
  factory, and zero-or-more axis overrides, re-simulate the rollout
  in-process and return a fresh Episode.
* :func:`parse_override` / :func:`validate_overrides` — the
  ``--override`` CLI helpers, exported for advanced users that want
  to drive replay from a notebook without going through the CLI.
* :class:`OverrideError` — the :class:`ValueError` subclass raised by
  override parsing / validation.

Replay is fundamentally a single-Episode, interactive operation. See
the RFC §2 Non-goals for why it is not a sweep, not multi-process,
and does not emit an HTML artefact.
"""

from __future__ import annotations

from gauntlet.replay.overrides import OverrideError as OverrideError
from gauntlet.replay.overrides import parse_override as parse_override
from gauntlet.replay.overrides import validate_overrides as validate_overrides
from gauntlet.replay.replay import replay_one as replay_one

__all__ = [
    "OverrideError",
    "parse_override",
    "replay_one",
    "validate_overrides",
]
