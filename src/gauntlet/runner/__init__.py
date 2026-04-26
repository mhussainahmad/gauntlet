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

from gauntlet.runner.determinism import IMAGE_OBS_KEYS as IMAGE_OBS_KEYS
from gauntlet.runner.determinism import STATE_OBS_KEYS as STATE_OBS_KEYS
from gauntlet.runner.determinism import assert_byte_identical as assert_byte_identical
from gauntlet.runner.determinism import episode_hash as episode_hash
from gauntlet.runner.determinism import obs_state_hash as obs_state_hash
from gauntlet.runner.determinism import rollout_hash as rollout_hash
from gauntlet.runner.episode import Episode as Episode
from gauntlet.runner.provenance import (
    SUITE_PROVENANCE_HASH_VERSION as SUITE_PROVENANCE_HASH_VERSION,
)
from gauntlet.runner.provenance import (
    capture_gauntlet_version as capture_gauntlet_version,
)
from gauntlet.runner.provenance import capture_git_commit as capture_git_commit
from gauntlet.runner.provenance import compute_env_asset_shas as compute_env_asset_shas
from gauntlet.runner.provenance import compute_suite_hash as compute_suite_hash
from gauntlet.runner.provenance import (
    compute_suite_provenance_hash as compute_suite_provenance_hash,
)
from gauntlet.runner.runner import Runner as Runner
from gauntlet.runner.worker import TrajectoryFormat as TrajectoryFormat
from gauntlet.runner.worker import WorkItem as WorkItem
from gauntlet.runner.worker import execute_one as execute_one

__all__ = [
    "IMAGE_OBS_KEYS",
    "STATE_OBS_KEYS",
    "SUITE_PROVENANCE_HASH_VERSION",
    "Episode",
    "Runner",
    "TrajectoryFormat",
    "WorkItem",
    "assert_byte_identical",
    "capture_gauntlet_version",
    "capture_git_commit",
    "compute_env_asset_shas",
    "compute_suite_hash",
    "compute_suite_provenance_hash",
    "episode_hash",
    "execute_one",
    "obs_state_hash",
    "rollout_hash",
]
