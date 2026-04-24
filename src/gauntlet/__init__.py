"""Gauntlet — an evaluation harness for learned robot policies.

Why this exists
---------------
Learned manipulation policies (open-ended VLAs, behaviour-cloning baselines,
diffusion policies, RL agents) routinely advertise headline success rates
that quietly average across many easy initial conditions and a handful of
catastrophic failures. Gauntlet is the harness that takes a policy plus a
declarative perturbation grid (a *suite*) and produces the breakdown that
hides behind the mean: per-axis marginals, per-cell aggregates, failure
clusters, 2D heatmaps, and an HTML artifact you can hand to a colleague.

Top-level layout
----------------
The public API is grouped into single-purpose subpackages:

* :mod:`gauntlet.env` — environment Protocol and built-in MuJoCo backend.
  The ``tabletop-pybullet`` / ``tabletop-genesis`` / ``tabletop-isaac``
  backends live in nested subpackages and register on demand.
* :mod:`gauntlet.policy` — :class:`~gauntlet.policy.Policy` Protocol plus
  :class:`~gauntlet.policy.RandomPolicy`, :class:`~gauntlet.policy.ScriptedPolicy`,
  and the lazy VLA adapters (:class:`~gauntlet.policy.HuggingFacePolicy`,
  :class:`~gauntlet.policy.LeRobotPolicy`).
* :mod:`gauntlet.suite` — declarative YAML grid (:class:`~gauntlet.suite.Suite`)
  and its loader.
* :mod:`gauntlet.runner` — parallel rollout orchestrator
  (:class:`~gauntlet.runner.Runner`) producing
  :class:`~gauntlet.runner.Episode` records.
* :mod:`gauntlet.report` — failure-analysis schema + HTML renderer
  (:func:`~gauntlet.report.build_report`, :func:`~gauntlet.report.write_html`).
* :mod:`gauntlet.replay` — single-episode re-simulation with optional
  axis overrides.
* :mod:`gauntlet.monitor` — runtime drift detector (torch-backed, opt-in
  via the ``[monitor]`` extra).
* :mod:`gauntlet.ros2` — ROS 2 publisher / recorder bridges (rclpy-backed,
  opt-in via the ``[ros2]`` extra).
* :mod:`gauntlet.cli` — ``gauntlet`` command-line entry point that wires
  the subpackages above into ``run`` / ``report`` / ``compare`` / ``replay``
  / ``monitor`` / ``ros2`` subcommands.

See ``GAUNTLET_SPEC.md`` for the canonical design vocabulary, the
seven-axis perturbation surface, and the §6 hard rules (reproducibility,
small deps, never aggregate away failures) that every subpackage is
written against.
"""

from __future__ import annotations

from gauntlet.env.gym_registration import register_envs

__version__ = "0.1.0"

# Register the four shipped backends with gymnasium's global registry on
# package import — the standard gymnasium-ecosystem convention. Heavy
# backends use string ``entry_point``s so this call does NOT pull in
# pybullet / genesis / isaacsim. Idempotent: safe to call again from
# tests, multiprocessing workers, or user code.
register_envs()

__all__ = ["__version__", "register_envs"]
