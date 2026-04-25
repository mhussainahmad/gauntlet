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
    # Conformal failure-prediction telemetry (B-01) — split-conformal
    # detector calibrated on a held-out successful-rollout set; see
    # :mod:`gauntlet.monitor.conformal`. Both fields default ``None``
    # for backwards compatibility (Episodes from before the detector
    # ran, OR from a greedy policy whose ``action_variance`` itself is
    # ``None``).
    #
    # Schema:
    # * ``failure_score`` — the candidate's ``action_variance`` divided
    #   by the calibration threshold. Greater than 1.0 means "more
    #   uncertain than the calibration set at this confidence level";
    #   less than 1.0 means "in distribution". ``None`` matches the
    #   B-18 asymmetry — greedy policies cannot expose a score.
    # * ``failure_alarm`` — convenience boolean, ``True`` iff
    #   ``failure_score > 1.0``. Pre-populated by ``gauntlet monitor
    #   conformal score`` so the report renderer does not need to
    #   re-thread the threshold to colour the row.
    #
    # Refs: FIPER (arxiv 2510.09459, NeurIPS 2025), FAIL-Detect (arxiv
    # 2503.08558).
    # ------------------------------------------------------------------

    failure_score: float | None = None
    failure_alarm: bool | None = None

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
    # Behavioural metrics beyond binary success (B-02) — per-rollout
    # numbers that distinguish policies tied on success rate. Refs:
    # RoboEval (RSS 2025, robo-eval.github.io) — "policies with identical
    # success rates have meaningfully different behavioural profiles, and
    # binary success becomes uninformative at the easy/hard tails". The
    # report surfaces these in the failure-clusters table so a "succeeds
    # but jerks the gripper around" policy is visible before it ships.
    #
    # Cross-backend caveat (anti-feature, deliberately documented):
    # collision and contact-force semantics differ across simulators.
    # MuJoCo's ``mj_data.contact[i].dist`` and :func:`mujoco.mj_contactForce`
    # are the portable definitions we adopt; PyBullet, Genesis, and Isaac
    # approximate them differently (or not at all). The shipped MuJoCo
    # :class:`TabletopEnv` populates ``info["behavior_*"]`` per step;
    # other backends emit nothing and the worker leaves all five fields
    # ``None``. ``None`` is distinct from ``0`` / ``0.0`` — those mean
    # "backend captured, the policy moved nothing / never came close to
    # contact"; ``None`` means "this backend doesn't capture behaviour
    # telemetry at all". The HTML report renders ``None`` as a dash,
    # never as zero.
    # ------------------------------------------------------------------

    # Wall-clock seconds from the start of the rollout to the step at
    # which ``info["success"]`` first flipped True, computed from the
    # env's per-step ``info["behavior_control_dt"]`` (typically
    # ``model.opt.timestep * n_substeps``). ``None`` for unsuccessful
    # rollouts (no ``done`` step to time to) AND for backends that do
    # not publish ``control_dt``. Distinct from ``0.0`` (which would
    # mean "succeeded at reset" — impossible in the current envs).
    time_to_success: float | None = None

    # End-effector path length divided by the straight-line distance
    # from the initial to the final EE position. ``1.0`` means the
    # policy moved in a perfectly straight line; ``> 1.0`` quantifies
    # detour. ``None`` when (a) the backend does not publish
    # ``info["behavior_ee_pos"]``, OR (b) the rollout produced fewer
    # than 2 EE samples, OR (c) the straight-line distance is below
    # 1e-6 m (a stationary policy — the ratio is undefined, distinct
    # from ``inf``). The HTML report renders these as dashes.
    path_length_ratio: float | None = None

    # RMS of the third-time-derivative of EE position over the rollout
    # (``sqrt(mean(||jerk_t||^2))``), in m / s^3. Computed via finite
    # differences on the buffered ``info["behavior_ee_pos"]`` samples;
    # ``jerk_t = (ee[t+3] - 3*ee[t+2] + 3*ee[t+1] - ee[t]) / dt^3`` with
    # ``dt`` from ``info["behavior_control_dt"]``. ``None`` when (a) the
    # backend does not publish EE position / control_dt, OR (b) the
    # rollout produced fewer than 4 EE samples (third differences are
    # undefined). Lower is smoother — high jerk often correlates with
    # near-collisions in real hardware.
    jerk_rms: float | None = None

    # Cumulative count of control steps where the closest pairwise
    # object distance fell below the env's near-collision threshold
    # (1cm in :class:`TabletopEnv`). Sum over the rollout, NOT a
    # per-step flag. Steady-state contacts (cube on table) are excluded
    # by the env-side filter so a trivial pickup does not run up the
    # counter. ``None`` for backends that do not publish
    # ``info["behavior_near_collision_delta"]``.
    near_collision_count: int | None = None

    # Maximum L2 norm of any contact force (3-component force vector,
    # torques excluded) observed in any single step, in newtons.
    # MuJoCo derivation: :func:`mujoco.mj_contactForce` per active
    # contact, take L2 norm of the first 3 components, max across
    # contacts and across steps. ``None`` for backends that do not
    # publish ``info["behavior_peak_contact_force"]``.
    peak_force: float | None = None

    # ------------------------------------------------------------------
    # Per-step inference-latency telemetry (B-37) — wall-clock cost of
    # the policy's ``act(obs)`` call, summarised across the rollout.
    # Surfaces the deployment-blocking question VLA-Perf (arxiv
    # 2602.18397) calls out: a policy can hit a high success rate and
    # still be undeployable because its 99th-percentile per-step latency
    # blows past the target hardware's control-loop budget.
    #
    # Fields are populated by :func:`gauntlet.runner.worker.execute_one`
    # which wraps each ``policy.act(obs)`` call with a
    # :func:`time.perf_counter` delta. ``policy.act_n`` (B-18 mode-collapse
    # measurement) is intentionally NOT timed — it samples N candidates
    # and would spike p99 by ~Nx on every measured step, biasing the
    # report toward measurement overhead rather than the critical path.
    #
    # Backend-agnostic by construction: the timing happens around the
    # policy interface, not inside any env. Every backend (MuJoCo,
    # PyBullet, Genesis, Isaac, fakes) populates these fields uniformly.
    #
    # ``None`` semantics (mirrors B-21 / B-30 / B-02): the worker did
    # not measure (legacy ``Episode.model_validate`` path on a pre-B-37
    # JSON, or a T=0 rollout that produced zero ``act`` calls). A
    # measured rollout that observed zero latency would still be
    # >= 0.0, never ``None`` — so the HTML report renders ``None`` as a
    # dash, never as ``0.00``.
    #
    # The budget gate lives on :attr:`metadata` rather than as a flat
    # field — see :class:`gauntlet.runner.runner.Runner.max_inference_ms`
    # / the ``--max-inference-ms`` CLI flag. Anti-feature (deliberate):
    # ``inference_budget_violated`` is a SOFT flag — it tags the
    # offending episode for sortable post-hoc inspection but never
    # aborts the run, never flips ``success``, and is absent from
    # ``metadata`` (rather than ``False``) when the budget was met or
    # not configured. The user wants to see *every* slow rollout, not
    # have the run die halfway through and hide the rest.
    # ------------------------------------------------------------------

    # 50th-percentile per-step ``policy.act`` latency in milliseconds.
    inference_latency_ms_p50: float | None = None

    # 99th-percentile per-step ``policy.act`` latency in milliseconds.
    # The deployment-blocking percentile — VLA-Perf (arxiv 2602.18397)
    # recommends 10-100 ms as the operating envelope; the ``--max-
    # inference-ms`` flag compares against this field.
    inference_latency_ms_p99: float | None = None

    # Maximum per-step ``policy.act`` latency observed in any single
    # step, in milliseconds. Useful when the operator cares about the
    # worst-case rather than a percentile (e.g. real-time control with
    # no jitter budget).
    inference_latency_ms_max: float | None = None

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
