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
* :func:`execute_one` — the per-episode primitive shared by the
  in-process and pool execution paths. Phase 2's :mod:`gauntlet.replay`
  is its second public caller; ordinary callers should continue to go
  through :meth:`Runner.run`.
"""

from __future__ import annotations

from gauntlet.runner.episode import Episode as Episode
from gauntlet.runner.provenance import (
    capture_gauntlet_version as capture_gauntlet_version,
)
from gauntlet.runner.provenance import capture_git_commit as capture_git_commit
from gauntlet.runner.provenance import compute_suite_hash as compute_suite_hash
from gauntlet.runner.runner import Runner as Runner
from gauntlet.runner.worker import WorkItem as WorkItem
from gauntlet.runner.worker import execute_one as execute_one

__all__ = [
    "Episode",
    "Runner",
    "WorkItem",
    "capture_gauntlet_version",
    "capture_git_commit",
    "compute_suite_hash",
    "execute_one",
]
