"""Episode result schema.

See :mod:`gauntlet.runner.runner` for the producer side.

The schema is deliberately minimal — Phase 1 task 7 (``Report``) extends it
with reporting helpers but must NOT rename or drop any existing field.
``ConfigDict(extra="forbid")`` is the contract that lets downstream code
key off the exact field names without worrying about silent additions.

Field invariants:

* ``suite_name`` — ``Suite.name`` of the originating suite.
* ``cell_index`` — :attr:`gauntlet.suite.SuiteCell.index` of the cell that
  produced this episode (zero-based ordinal in
  :meth:`gauntlet.suite.Suite.cells` enumeration order).
* ``episode_index`` — zero-based index within ``cell_index``; in
  ``[0, episodes_per_cell)``.
* ``seed`` — the integer seed handed to :meth:`TabletopEnv.reset`. This is
  the bit that bit-reproduces the rollout. Derived deterministically from
  the master seed (see :class:`gauntlet.runner.runner.Runner` docstring).
* ``perturbation_config`` — a copy of :attr:`SuiteCell.values`.

Extensibility:

* ``metadata`` accepts arbitrary scalar values (the pydantic ``int`` type
  is unbounded so it round-trips Python's ``SeedSequence.entropy``, which
  can exceed 64 bits). The Runner uses it to record ``master_seed`` for
  reproducibility of None-seed runs.
* ``video_path`` (Polish "rollout MP4 video recording") is an optional
  string carrying the relative path of an MP4 dumped by the runner when
  the user opts in via ``Runner(record_video=True)``. The default is
  ``None`` so old JSONs (pre-PR Episode dicts) load cleanly under the
  field default. A *new* Episode always emits ``"video_path": null``
  when no video is recorded, which is semantically inert; an older
  ``gauntlet`` reader that lacks the field WILL reject the JSON because
  of ``extra="forbid"`` — that is the standard schema-addition trade-off
  documented here so it does not surprise downstream consumers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Episode"]


class Episode(BaseModel):
    """Result of one rollout. Pure data; no behaviour."""

    # ``ser_json_inf_nan="strings"`` round-trips ``float('nan')`` /
    # ``float('inf')`` through JSON as ``"NaN"`` / ``"Infinity"`` /
    # ``"-Infinity"`` instead of pydantic's default ``null`` (which then
    # fails revalidation as ``float``). Live Runner-emitted Episodes do
    # not currently carry non-finite floats, but a third-party env that
    # returns NaN reward (broken policy, NaN obs from a learned model)
    # would otherwise produce an unreplayable episodes.json.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

    # Identity / reproducibility.
    suite_name: str
    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]

    # Outcome.
    success: bool
    terminated: bool
    truncated: bool
    step_count: int
    total_reward: float

    # Extensible bag for future fields (trajectory summary, timing, master
    # seed echo, etc.). Task 7 will add reporting helpers that read keys
    # from here without changing the schema above.
    metadata: dict[str, float | int | str | bool] = Field(default_factory=dict)

    # Optional path to an MP4 recording of this episode. Populated only
    # when the user opts in via ``Runner(record_video=True)``. Stored as
    # a string (not :class:`pathlib.Path`) so the schema round-trips
    # losslessly through pydantic's JSON mode. The path is always
    # *relative* — typically ``"videos/episode_cell{NNNN}_ep{NNNN}_seed
    # {S}.mp4"`` relative to the run output directory — so the HTML
    # report can embed it via ``<video src="{video_path}">`` without a
    # web server. Default ``None`` keeps the field semantically inert
    # for the byte-identical opt-out path.
    video_path: str | None = None

    # ------------------------------------------------------------------
    # Provenance trio (B-22) — populated by :class:`Runner` so every
    # Episode carries enough context to be re-rolled on a fresh checkout.
    # All three default to ``None`` for backwards compatibility with
    # episodes.json files emitted before B-22; an old reader that lacks
    # these fields would still reject the JSON because of
    # ``extra="forbid"`` (see the ``video_path`` note above), but a new
    # reader on an old file loads cleanly under the field default.
    # ------------------------------------------------------------------

    # Installed ``gauntlet`` distribution version captured at run time
    # via :func:`importlib.metadata.version`. ``None`` when the package
    # is not importable as an installed distribution (e.g. a checkout
    # used without ``pip install -e .``).
    gauntlet_version: str | None = None

    # SHA-256 hex digest of the canonical Suite payload that produced
    # this Episode (``Suite.model_dump_json(...)`` bytes; see
    # :func:`gauntlet.runner.runner.compute_suite_hash`). Stable under
    # YAML formatting differences because the hash is computed over the
    # validated Pydantic model, not the raw file bytes.
    suite_hash: str | None = None

    # Output of ``git rev-parse HEAD`` captured in the working copy at
    # run time, or ``None`` when the run happened outside a git checkout
    # (e.g. a packaged tarball or a CI shallow without ``.git``).
    git_commit: str | None = None

    # ------------------------------------------------------------------
    # Actuation-cost telemetry (B-21) — per-rollout summary of how much
    # the policy works the actuators. Surfaced in the failure-clusters
    # table so a "succeeds at 5x the energy cost" policy is visible
    # before it ships to real hardware (Safe-control-gym, arxiv
    # 2109.06325; Safety-Gymnasium, NeurIPS 2023).
    #
    # Cross-backend caveat (anti-feature, deliberately documented):
    # torque semantics differ across simulators. MuJoCo envs that ship
    # joint actuators populate these from ``mj_data.actuator_force``
    # and ``mj_data.qvel`` (joint-space torque & velocity). PyBullet,
    # Genesis, and Isaac envs do not surface comparable per-step
    # torque to the worker yet, so they leave all three fields
    # ``None`` and the report renders blanks (not zero).
    #
    # The shipped :class:`gauntlet.env.tabletop.TabletopEnv` is itself
    # mocap-driven (``model.nu == 0`` — no ``<actuator>`` elements) so
    # it ALSO leaves these fields ``None`` rather than emit a
    # misleading ``0.0``. Wiring the metric to PyBullet / Genesis /
    # Isaac, and shipping a joint-torque tabletop MJCF asset, are
    # B-21 follow-ups.
    # ------------------------------------------------------------------

    # Cumulative actuator energy over the rollout, in joule-equivalent
    # units (``sum(|tau . qvel| * dt)`` over all control steps).
    # ``None`` means "this backend doesn't capture torque telemetry"
    # — distinct from ``0.0`` ("backend captured, the policy did
    # nothing"). The report renders ``None`` as a dash.
    actuator_energy: float | None = None

    # Mean L2 norm of the actuator-force vector across all control
    # steps (per-step ``np.linalg.norm(actuator_force)``, averaged).
    # Same ``None`` semantics as ``actuator_energy``.
    mean_torque_norm: float | None = None

    # Peak (max) L2 norm of the actuator-force vector observed in any
    # single step. Same ``None`` semantics as ``actuator_energy``;
    # always ``>= mean_torque_norm`` when both are populated.
    peak_torque_norm: float | None = None

    # ------------------------------------------------------------------
    # Action mode-collapse telemetry (B-18) — measures policy-action
    # consistency by sampling N candidate actions per (sub-sampled)
    # rollout step from a stochastic / chunked policy and reporting the
    # mean per-step per-axis variance, averaged over the measured steps.
    #
    # Computation contract (see :func:`gauntlet.runner.worker.execute_one`):
    # at every ``_CONSISTENCY_STRIDE``-th rollout step the worker calls
    # :meth:`gauntlet.policy.SamplablePolicy.act_n` (default ``n=8``),
    # computes per-axis variance ``np.var(actions, axis=0)`` (length =
    # action_dim), reduces to a scalar via mean across axes, and
    # averages those scalars across the measured steps. The per-axis
    # framing matches the B-18 spec ("per-axis 'action consistency'
    # column"), and the cross-axis mean keeps the field a single
    # float rather than a variable-length vector.
    #
    # ``None`` semantics — same shape as the actuator-cost trio above:
    # the policy did not implement :class:`SamplablePolicy` (greedy
    # baselines, :class:`gauntlet.policy.ScriptedPolicy`, OpenVLA-greedy)
    # OR the user did not opt in via ``Runner(measure_action_consistency=True)``.
    # ``0.0`` would be ambiguous: a sampleable policy that has truly
    # collapsed onto a single mode reports ``0.0``; an un-measured
    # rollout reports ``None``. The HTML report renders ``None`` as a
    # dash, never as zero.
    #
    # Anti-feature (documented in ``docs/backlog.md`` B-18): the
    # column is asymmetric across the policy zoo; greedy policies
    # cannot expose it without stochasticity to sample from.
    # ------------------------------------------------------------------

    action_variance: float | None = None

    # ------------------------------------------------------------------
    # Safety-violation telemetry (B-30) — first-class per-rollout
    # accounting of how badly the policy abused the world to score its
    # success. The "failures over averages" rule (PRODUCT.md, GAUNTLET
    # spec §6) extends to *unsafe* successes: a policy that dings the
    # table on every successful pick is not actually a successful
    # policy. Refs: Safety-Gymnasium NeurIPS 2023, safe-control-gym
    # arXiv 2109.06325.
    #
    # Schema is four FLAT fields rather than a nested
    # ``safety_violations`` dict so JSON / Parquet stays tractable and
    # the HTML report can render directly without un-nesting.
    #
    # Cross-backend caveat (anti-feature, deliberately documented):
    # collision-detection semantics differ wildly between simulators.
    # MuJoCo's ``mj_data.ncon`` delta is the portable definition we
    # adopt; PyBullet, Genesis, and Isaac approximate it differently
    # (or not at all). The shipped MuJoCo :class:`TabletopEnv`
    # populates ``info["safety_n_collisions_delta"]`` /
    # ``info["safety_joint_limit_violation"]`` /
    # ``info["safety_workspace_excursion"]`` per step; other backends
    # emit nothing and the worker leaves all four fields ``None``.
    # ``None`` is distinct from ``0`` — ``0`` means "backend captured,
    # zero violations observed", ``None`` means "this backend doesn't
    # capture safety telemetry". The HTML report renders ``None`` as
    # a dash, never as zero.
    #
    # ``energy_over_budget`` is derived in the worker (not the env) by
    # comparing the accumulated ``actuator_energy`` to an optional
    # per-run ``energy_budget`` threshold (see
    # :class:`gauntlet.runner.worker.WorkerInitArgs.energy_budget` /
    # :func:`gauntlet.runner.worker.execute_one`'s
    # ``energy_budget`` kwarg). Stays ``None`` whenever
    # ``actuator_energy is None`` (no torque telemetry to compare
    # against) or no budget was configured.
    # ------------------------------------------------------------------

    # Total contact count delta (sum of per-step ``max(ncon - ncon_prev,
    # 0)``) over the rollout. New contacts only — steady-state contacts
    # (e.g. cube resting on the table) are not counted.
    n_collisions: int | None = None

    # Number of control steps where any joint exceeded its
    # ``model.jnt_range`` bound. Cumulative count, not a per-step flag;
    # a single excursion that lasted three steps counts as 3.
    n_joint_limit_excursions: int | None = None

    # True when the rollout's accumulated ``actuator_energy`` exceeded
    # the per-run ``energy_budget`` threshold; False when measured and
    # within budget; ``None`` when no budget was configured OR the
    # backend did not surface ``actuator_energy`` (cannot evaluate the
    # threshold). Distinct from ``False`` (which means "we checked,
    # the policy stayed within budget").
    energy_over_budget: bool | None = None

    # Number of control steps where the end-effector left the env's
    # advertised workspace bounds. Same cumulative count semantics as
    # ``n_joint_limit_excursions``.
    n_workspace_excursions: int | None = None

    # ------------------------------------------------------------------
    # Sim-vs-real provenance tag (B-28) — explicit "this episode came
    # from a sim rollout" / "real-robot rollout" marker. Default
    # ``None`` keeps the field semantically inert for the legacy
    # sim-only flow (a None reader on a new JSON would still reject
    # under ``extra="forbid"`` per the ``video_path`` note above; the
    # reverse direction loads cleanly under the field default).
    #
    # Real-robot consumers (B-28: ``gauntlet aggregate-sim-real``) tag
    # the real-side episodes explicitly; the sim-side flow leaves the
    # field ``None`` rather than retroactively backfilling ``"sim"``
    # because pre-B-28 episodes.json files would otherwise need a
    # migration. Downstream pairing keys off
    # ``(suite_hash, cell_index, episode_index)`` — the directory
    # layout is the real signal — and uses ``source`` only as a
    # post-hoc sanity check.
    # ------------------------------------------------------------------

    source: Literal["sim", "real"] | None = None
