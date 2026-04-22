"""Parallel rollout orchestration — see ``GAUNTLET_SPEC.md`` §5 task 6.

Public surface:

* :class:`Runner` — execute a :class:`gauntlet.suite.Suite` against a
  :class:`gauntlet.policy.Policy`, in parallel across worker processes,
  with full seed control.
* :class:`Episode` — Pydantic model for one rollout's result. Phase 1
  task 7 (:mod:`gauntlet.report`) reads from this; never rename or drop
  fields.
* :class:`WorkItem` — the unit of cross-process work. Exposed for
  advanced users that want to plug their own executor; ordinary callers
  should not need it.
"""

from __future__ import annotations

from gauntlet.runner.episode import Episode as Episode
from gauntlet.runner.runner import Runner as Runner
from gauntlet.runner.worker import WorkItem as WorkItem

__all__ = [
    "Episode",
    "Runner",
    "WorkItem",
]
