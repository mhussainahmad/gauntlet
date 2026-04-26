# Gauntlet Backlog

Open candidates for future work, sourced from the 2024–2026 robot-policy-evaluation literature plus an internal dependency-drift scan as of 2026-04-25. Each item is independent, has a clear paper or open-source precedent, and is sized in one of three buckets (S ≤ 1 day, M ≤ 3 days, L ≤ 1 week of focused work).

> All 36 original items shipped 2026-04-25. New candidates B-37+ added 2026-04-25.

## How to use

Pick the next backlog item by topology, not by ID. Items in the same category may share files; items across categories generally don't. Each entry's **Disjoint with** line names the modules it touches and any other backlog items that conflict.

Suggested ordering for the next continuous-polish loop (smallest, highest-rigour-payoff first):

1. **B-40** Suite-level provenance hash + result cache key — S, extends `runner/provenance.py`, pure DX.
2. **B-41** Statistical power calculator on `gauntlet suite plan` — S, extends `report/wilson.py` + B-08; tells the user how many samples they actually need.
3. **B-42** Camera-extrinsics perturbation axis — S, viewpoint is the highest-yield robustness axis per RoboView-Bias.
4. **B-43** Color / saturation visual-bias axis — S, backend-agnostic post-render, complements B-31.
5. **B-37** Inference-latency / wall-clock budget tracking — S, fills the `runner/` gap VLA-Perf flagged.
6. **B-38** Inference-delay jitter perturbation axis — S, builds on B-37; tests RTC-style real-time chunking.

After those S-class items the medium items become viable in pairs.

---

## Eval Primitives (new test types)

### B-01: Conformal-calibrated failure prediction signal
- **What:** Per-episode `failure_alarm: bool | None` + `failure_score: float` computed by FIPER-style action-chunk entropy + RND-on-observation, calibrated on a held-out successful-rollout set via split conformal prediction.
- **Why:** FIPER (arxiv 2510.09459, NeurIPS 2025) and FAIL-Detect (arxiv 2503.08558) both show the policy's own internal signals predict failure 0.5–3 s before it happens, with conformal prediction giving statistical false-positive bounds. Drop-in for the existing `monitor/` module — orthogonal to the AE that already lives there.
- **Scope:** M
- **Disjoint with:** `gauntlet.monitor` (extends), Episode schema (adds two fields). Conflicts mildly with B-04 (both add per-episode reliability columns — keep both, FIPER is online, B-04 is post-hoc).
- **Anti-feature?** Action-chunk entropy is only meaningful for stochastic / chunked policies (diffusion, π0, SmolVLA). Scripted and OpenVLA-greedy emit a constant zero — risks looking like a feature only the cool kids can use.

### B-02: Behavioural metrics beyond binary success
- **What:** Add `time_to_success`, `path_length_ratio` (vs. straight-line lower bound), `jerk_rms`, `near_collision_count`, `peak_force` to the Episode schema and surface them as sortable columns in the failure-cluster table.
- **Why:** RoboEval (RSS 2025, robo-eval.github.io) explicitly demonstrates that policies with identical success rates have meaningfully different behavioural profiles, and that binary success becomes uninformative at the easy/hard tails. Direct fit for the "failures over averages" rule — lets the report rank policies tied on the bit.
- **Scope:** M
- **Disjoint with:** Episode schema, report HTML template. No conflict.
- **Anti-feature?** Some metrics (peak_force) require contact-sensor wiring per backend; we'd ship MuJoCo first and `None` the rest, which sets a "feature works on default backend only" pattern that could metastasise.

### B-03: Wilson / Clopper-Pearson confidence intervals on per-cell success
- **What:** Replace `success_rate: 0.5` with `success_rate: 0.5, ci_low: 0.19, ci_high: 0.81` (Wilson 95% by default) on every CellBreakdown / AxisBreakdown. Render as `0.50 [0.19, 0.81]` in tabular numerals.
- **Why:** Reporting `5/10 → 50%` without a CI is statistical malpractice and the fleet operator is the one who suffers. Embarrassingly cheap (`scipy.stats.binom`), no new heavy deps, exactly the "rigorous, exact" personality.
- **Scope:** S
- **Disjoint with:** `report/` module. Touches diff thresholds (a flip whose CIs overlap is no longer a flip) — coordinate with B-20.
- **Anti-feature?** Adds visual width to every cell in the heatmap; could fight the dashboard's typographic density. Solvable but real.

### B-04: Calibration-aware abstention scoring
- **What:** New report section: "policy knew it was failing" — for each failed episode, did `failure_score` (B-01) cross the conformal threshold *before* the actual failure step? Report selective-prediction metrics: AURC (area under risk-coverage curve) and abstention-corrected success rate.
- **Why:** Implicit in FIPER / FAIL-Detect; LangForce (arxiv 2601.15197) and the surgical UQ paper (arxiv 2501.10561) all push abstention as a first-class capability. Tells the user "this checkpoint is wrong but at least it knows it" — which is the difference between a deployable policy and a dangerous one.
- **Scope:** M
- **Disjoint with:** Builds on B-01. New subsection of `report/`.
- **Anti-feature?** Only meaningful if B-01 ships first and you trust its calibration set; double-failure surface area.

### B-05: Instruction-paraphrase perturbation axis (language)
- **What:** New `PerturbationAxis` of categorical type `instruction_paraphrase` carrying a list of natural-language phrasings of the task. Suite YAML: `instruction_paraphrase: {values: ["pick up the red cube", "grab the crimson block", ...]}`.
- **Why:** LIBERO-PRO (arxiv 2510.03827) and LIBERO-Plus (arxiv 2510.13626) showed VLAs that score 90% on the canonical instruction collapse to 0% on paraphrases. LADEV (arxiv 2410.05191) automates this exact axis. Gauntlet has spatial axes but zero language axes — major gap for VLA users (who are now the majority adopters).
- **Scope:** S
- **Disjoint with:** Adds an axis name; only consumed by policies whose `act()` reads `obs["instruction"]`. Need to thread instruction through `TabletopEnv.set_perturbation`.
- **Anti-feature?** Random/Scripted policies don't read instructions, so this axis is dead-code on baselines — risks asymmetric coverage in cross-policy comparisons.

### B-06: Object-swap / semantic-distractor perturbation axis
- **What:** New `object_swap` axis: replace the target cube with a categorically-different object (mug, banana, screwdriver) drawn from a small object library shipped with gauntlet, tagged with which semantic class. Combined with `instruction_paraphrase` measures grounding.
- **Why:** LIBERO-PRO showed models persist in grasping when the target is replaced with an irrelevant item — a grounding failure invisible under standard eval. RoboCasa's procedural object library is the precedent.
- **Scope:** M
- **Disjoint with:** Requires a tiny MJCF object library in `env/assets/`. Touches all four backends (or declares the axis backend-restricted, like `VISUAL_ONLY_AXES`).
- **Anti-feature?** Cross-backend asset parity is the real cost; getting a banana mesh to look "the same" in MuJoCo / PyBullet / Genesis / Isaac is the kind of yak-shave that ate weeks of the rendering RFC.

### B-07: Adversarial / failure-seeking sampler
- **What:** New `sampling: adversarial` mode that, given a pilot run's `report.json`, fits a small Gaussian-process / Thompson-sampling bandit over the perturbation hypercube and concentrates new samples in high-failure regions.
- **Why:** RoboMD (arxiv 2412.02818) treats failure-mode discovery as a sequential search problem and finds 23% more unique vulnerabilities than uninformed sampling. Cheap because gauntlet already has the Sobol/LHS sampler interface — this is just another `Sampler` class.
- **Scope:** M
- **Disjoint with:** `suite/sampling.py`. Must take a prior `Report` as input — extends `Suite` schema with optional `pilot_report` field.
- **Anti-feature?** Adversarial sampling biases coverage and breaks the "this report is a fair sample of the perturbation space" reading — easy to misuse for cherry-picking. Document loudly.

### B-08: Common-random-numbers paired comparison
- **What:** When `gauntlet compare` / `diff` runs against two policies on the same suite, force both to share the per-episode env seed stream (already deterministic) AND share initial-state randomisation, so the only varying factor is the policy. Variance-reduced delta.
- **Why:** "Beyond Binary Success" (arxiv 2603.13616) and SureSim (arxiv 2510.04354) both lean on paired evaluation as the standard variance-reduction tool. Reduces required `episodes_per_cell` by 2-4× for the same compare-confidence. Pure stats win, no new deps.
- **Scope:** S
- **Disjoint with:** `runner/` (seed derivation already supports this), `compare/`, `diff/`. Coordinate CI updates with B-03.
- **Anti-feature?** Requires re-running both policies if you've only got one already-saved `episodes.json` — partial backwards-compat penalty.

### B-09: Long-horizon / skill-chaining task suites
- **What:** Add a `tabletop-stack` and `tabletop-pour` env (3-5 step rollouts) plus `subtask_completion: list[bool]` on Episode so partial credit shows up in the report.
- **Why:** LAMBDA (arxiv 2412.05313) and RoboCerebra (arxiv 2506.06677) explicitly call out that single-step pick-and-place misses the entire long-horizon failure surface (System-2 planning errors, mid-rollout drift). Gauntlet currently only ships one task family.
- **Scope:** L
- **Disjoint with:** New env(s) in `env/`, Episode schema (`subtask_completion`).
- **Anti-feature?** Subtask milestones are env-specific; either we centralise a milestone Protocol (good) or sprinkle ad-hoc lists everywhere (bad). The Protocol design is the real work.
- **Partial 2026-04-26**: `tabletop-stack` 3-cube stub env shipped (`gauntlet.env.tabletop_stack.TabletopStackEnv`, registered as `tabletop-stack`) along with the `SubtaskMilestone` Protocol seam (`gauntlet.env.base.SubtaskMilestone`) and an optional, default-`None` `Episode.subtask_completion: list[bool] | None` field. The env publishes `info["subtask_completion"]` per step; runner wiring to forward that into the Episode is deferred. Remaining for full B-09: (a) `tabletop-pour` env, (b) runner-side worker plumbing that reads `info["subtask_completion"]` into the Episode field, (c) report-side per-subtask credit rendering, (d) per-subtask reward decomposition + names registry as the schema-evolution follow-up. Stub env declares `AXIS_NAMES = frozenset()` — perturbation axes for the multi-cube scene are a follow-up too.

### B-10: Bimanual / multi-arm task family
- **What:** A `tabletop-bimanual` env (two floating end-effectors, hand-off task) plus the action-space dim doubling.
- **Why:** PerAct2, BiCoord (arxiv 2604.05831), RDT, and π0 are all explicitly bimanual; the field has clearly shifted. Gauntlet's single-arm-only env precludes evaluating any modern bimanual policy.
- **Scope:** L
- **Disjoint with:** New env, action-space contract (variable-width).
- **Anti-feature?** Variable action-dim breaks the "one shared action/obs space" elegance gauntlet currently sells. Probably needs a `tabletop-2` family kept separate.

### B-44: Worst-case continuous-perturbation search (Eva-VLA-style)
- **What:** New `sampling: worst_case_continuous` mode that, for axes with continuous parameters (light intensity, camera pose offset, object SE(3) jitter), runs a small gradient-free optimiser (CMA-ES via `cma`, or finite-difference inside the existing scipy stack) against a per-episode failure-score surrogate to find adversarial points. Surfaces a "worst observed configuration" cell next to the heatmap.
- **Why:** Eva-VLA (arxiv 2509.18953) demonstrates that worst-case search over continuous physical variations exposes failure modes invisible to grid/Sobol sampling — OpenVLA fails >90% on LIBERO-Long under their found worst-cases vs. ~20% on uniform sampling. Gauntlet's existing samplers all sample *configurations*; this samples *attacks*.
- **Scope:** M
- **Disjoint with:** `suite/sampling.py`, new `suite/worst_case.py`. **Distinct from B-07**: B-07 is a discrete-bandit search over the existing axis grid (Thompson sampling); B-44 is gradient-free continuous optimisation over the underlying physical parameters. Same user-facing pitch (find failures), different mechanism. Document the contrast.
- **Anti-feature?** Continuous worst-case is the strongest possible cherry-pick — a "this policy fails 100% on the adversarial cell" headline reads as alarmist when the cell is measure-zero. Must report alongside the unbiased baseline rate, never standalone.

## Backends (new sim backends)

### B-11: SoftGym/DaxBench-style deformable backend
- **What:** `tabletop-cloth` env wrapping SoftGym (or DaxBench's MPM) for cloth-folding / rope-routing tasks. Same `GauntletEnv` Protocol surface.
- **Why:** Cloth/rope manipulation is a major commercial use case (laundry, garment folding) and the entire VLA literature is rigid-body biased. SoftGym (CoRL 2020), DaxBench (ICLR 2023), GarmentLab (NeurIPS 2024) provide the precedent; none ship as a uniform-API harness.
- **Scope:** L
- **Disjoint with:** New `env/softgym.py`, new `[softgym]` extra. Touches the Protocol: cloth has no "single object pose" so axis names diverge.
- **Anti-feature?** SoftGym is barely maintained; DaxBench needs JAX. Picking the wrong dependency strands the backend.

### B-12: MuJoCo MJX / GPU-parallel backend
- **What:** A `tabletop-mjx` backend using MuJoCo MJX so 1000s of cells run in parallel on one GPU.
- **Why:** ManiSkill3 (arxiv 2410.00425, RSS 2025) shows 30,000+ FPS via GPU-parallel sim — gauntlet's `multiprocessing` Pool is fine for laptops but caps out around 16 workers. Even a 50× wall-clock cut on the canonical 1440-rollout suite changes the iteration loop fundamentally.
- **Scope:** L
- **Disjoint with:** New `env/mjx.py`, new `[mjx]` extra. Runner needs a "batched env" branch — non-trivial.
- **Anti-feature?** GPU dependency reintroduces what gauntlet has carefully kept optional. Could end up as a third-tier backend nobody uses outside of CI smoke.

### B-13: Mobile-base / navigation env
- **What:** `tabletop-mobile` env with a wheeled base and a 2-step nav-then-pick task.
- **Why:** RoboCasa, ManiSkill-HAB, BEHAVIOR-1K all showcase mobile manipulation as the next frontier. Gauntlet has zero navigation surface area today.
- **Scope:** M
- **Disjoint with:** New env, observation space gains `pose: SE2`, action space gains base-velocity dims.
- **Anti-feature?** Stretches the "tabletop" conceit; risks turning gauntlet into a mini-RoboCasa, which is not the lane.

## Policies (new adapter types)

### B-14: π0 / π0-fast adapter
- **What:** `Pi0Policy` adapter behind a `[pi0]` extra. Same shape as `HuggingFacePolicy` but wires the PI flow-matching action head.
- **Why:** π0 (Physical Intelligence) is the highest-profile open VLA after OpenVLA / SmolVLA and is what production teams are actually deploying. Sensor-attack-robustness paper (DOI 10.1145/3733800.3763262) showed π0 is dramatically more robust than OpenVLA — direct cross-policy comparison is exactly what gauntlet exists for.
- **Scope:** M
- **Disjoint with:** `policy/`, new `[pi0]` extra. Mirrors existing OpenVLA pattern.
- **Anti-feature?** PI's licensing on weights is restrictive; users may not be able to download what the adapter expects. Adapter-as-shelfware risk.

### B-15: GR00T-N1 / RDT adapter
- **What:** Adapters for NVIDIA GR00T-N1 (humanoid foundation model) and RDT (Robotics Diffusion Transformer).
- **Why:** Both are the most cited bimanual / humanoid foundation models of 2025. Same precedent as the OpenVLA adapter pattern — every adapter ships in <300 LoC.
- **Scope:** M (each)
- **Disjoint with:** `policy/`. GR00T-N1 needs the bimanual env from B-10; without it, embodiment-mismatch warning is the only signal.
- **Anti-feature?** GR00T-N1 inference needs serious GPU; `[groot]` extra risks being green-CI-only.

### B-16: Decision-Transformer / offline-RL baseline policy
- **What:** A `DecisionTransformerPolicy` reference adapter that loads a small DT checkpoint trained on a gauntlet trajectory dump.
- **Why:** Currently the only "trained" baselines users can compare against are the SOTA VLAs. A small DT trained from `Runner(trajectory_dir=...)` is the canonical "what does a tiny model trained on your own data do?" baseline — closes the loop on the trajectory-dump feature.
- **Scope:** M
- **Disjoint with:** `policy/`, new `[dt]` extra. Synergises with trajectory dumps.
- **Anti-feature?** Implies a training story, which gauntlet has explicitly stayed out of. Feature creep into the wrong domain.

## Reports (new analysis surfaces)

### B-17: Failure-mode taxonomy via trajectory clustering — Shipped 2026-04-26
- **Status:** Shipped 2026-04-26. New `gauntlet.report.trajectory_taxonomy` module + a "Failure-Mode Taxonomy" subsection rendered beneath the existing axis-config failure-clusters table (orthogonal — config clusters say *which* perturbations failed; trajectory clusters say *how* the rollouts unfolded). Loads each failed episode's `cell_NNNN_ep_NNNN.npz` action stream, builds a pairwise distance matrix (DTW via the new `[trajectory-taxonomy]` extra, falls back to euclidean truncate-to-`min(T)` when the extra is absent), runs deterministic average-linkage agglomerative clustering, and silhouette-searches `k ∈ [2, min(8, n_failed - 1)]` when `n_clusters` is unset. Anti-feature framing preserved verbatim: clusters are labelled by the medoid's `cell_NNNN_ep_NNNN` index, NOT by auto-generated prose; the public dataclasses carry no prose-label field and the test suite asserts a future "helpful" addition trips CI.

### B-18: Policy-action consistency / mode-collapse metric
- **What:** For chunked policies (diffusion, π0), sample N action chunks per state and compute action variance. Aggregate as a per-axis "action consistency" column in the report.
- **Why:** Diffusion Policy (RSS 2023, IJRR 2025) and Hybrid Consistency Policy (arxiv 2510.26670) both call out mode collapse as the dominant failure mode of stochastic policies. Currently invisible to gauntlet.
- **Scope:** M
- **Disjoint with:** Requires `Policy` Protocol extension: optional `act_n(obs, n) -> Action[]` for samplable policies. Backwards-compatible default.
- **Anti-feature?** Only sampleable policies expose this; the column is `None` for greedy ones — same asymmetry problem as B-01.

### B-19: Per-axis sensitivity rank (Sobol indices)
- **What:** Compute first-order and total Sobol sensitivity indices per axis from the existing rollout data and surface as a "which axis matters most" bar at the top of the report.
- **Why:** Gauntlet already ships a Sobol *sampler*; computing Sobol *indices* from the same data is the natural payoff (Saltelli, "Global Sensitivity Analysis"). Tells the user "lighting matters 4× more than camera offset for this checkpoint" — single-glance signal that survives the failure-clusters table.
- **Scope:** S
- **Disjoint with:** `report/`, scipy quasi-MC indices.
- **Anti-feature?** Sobol indices need quasi-MC sample structure to be unbiased; if the user runs the report on a cartesian-sampled suite, indices are noisy. Need to gate on `suite.sampling`.

### B-20: Regression-direction vs. variance attribution in `diff`
- **What:** For each cell-flip in `gauntlet diff`, decompose the delta into "policy got worse" (direction) vs. "noise" (CI overlap) using the Wilson CIs from B-03 and the paired-CRN reduction from B-08. Tag each flip as `regressed | improved | within_noise`.
- **Why:** Today `diff` cries wolf on every cell that crossed the threshold even if both rates have ±0.3 CIs that overlap completely. The fleet operator stops trusting the diff after the second false alarm.
- **Scope:** S
- **Disjoint with:** Builds on B-03 + B-08. `diff/` module.
- **Anti-feature?** None substantive — pure correctness fix.

### B-21: Energy / actuation-cost columns
- **What:** Cumulative `actuator_energy`, `mean_torque_norm`, and `peak_torque_norm` per Episode, surfaced in the failure-clusters table.
- **Why:** Safe-control-gym (arxiv 2109.06325), Safety-Gymnasium (NeurIPS 2023). A policy that succeeds at 5× the energy cost is a policy that breaks the actuator on real hardware. Gauntlet has the actuator data but throws it away.
- **Scope:** S
- **Disjoint with:** Episode schema, runner per-step bookkeeping.
- **Anti-feature?** Cross-backend torque semantics differ (mocap vs. joint-torque envs); risks `None`-soup again.

## Workflow / DX

### B-22: Episode-level seed manifest + `gauntlet repro <episode-id>`
- **What:** Ship a single `repro.json` per run that contains the full provenance for every Episode (commit hash, gauntlet version, suite hash, env seed, policy seed, axis config). New CLI subcommand `gauntlet repro <episode-id>` re-runs exactly that episode in the current checkout and asserts bit-identical output.
- **Why:** Reproducibility is a hard rule per spec §6 but currently demands the user piece together suite + seed + episode index by hand. Tools like DVC and Hydra ship this for free in adjacent ML domains.
- **Scope:** S
- **Disjoint with:** `runner/`, new CLI subcommand. Episode schema picks up `gauntlet_version: str`.
- **Anti-feature?** None — pure correctness-and-DX win.

### B-23: Parquet trajectory dumps + DuckDB query example
- **What:** `Runner(trajectory_dir=...)` emits Parquet (in addition to / instead of NPZ). Document a one-liner DuckDB query against the parquet dir for ad-hoc trajectory analysis.
- **Why:** NPZ is fine for one-off Python scripts; Parquet is the *shared* format that lets a non-Python user (a fleet operator with a Tableau seat) actually query trajectory data. DuckDB is the local-first, zero-server query engine that fits PRODUCT.md.
- **Scope:** S
- **Disjoint with:** `runner/`. New `[parquet]` extra (`pyarrow`).
- **Anti-feature?** `pyarrow` is heavy (~80 MB wheel); has to be optional. Two storage formats means two test surfaces.

### B-24: JUnit XML / GitHub-Actions-friendly machine output
- **What:** `gauntlet run --junit out/junit.xml` emits a JUnit-style XML where each cell × episode is a `<testcase>`. Also `gauntlet compare --github-summary` writes a stepSummary-compatible markdown.
- **Why:** Lets gauntlet drop into any CI without writing custom glue. The "robotics analogue of pytest" framing in spec §1 only fully delivers when the CI integration is one flag.
- **Scope:** S
- **Disjoint with:** `report/`, CLI.
- **Anti-feature?** XML is ugly and JUnit's schema doesn't quite fit per-axis structured results — emitted XML is a bit Procrustean.

### B-25: Suite linter / "did you actually exercise the axis?" warning
- **What:** `gauntlet suite check <suite.yaml>` lints for: unused axes (declared but every value identical), redundant Cartesian explosion (>10k cells), insufficient `episodes_per_cell` for the requested CI width (B-03), VISUAL_ONLY_AXES used on Isaac.
- **Why:** Common authoring footgun. Pytest / hypothesis have analogous lints. Cheap, deeply DX-positive.
- **Scope:** S
- **Disjoint with:** `suite/`, new CLI subcommand.
- **Anti-feature?** None — it's a linter.

### B-26: Optuna-style early-stop pruning for sweeps
- **What:** Optional `--prune-after-cells N --prune-min-success S` flag that aborts the run as soon as overall success-rate CI lower bound exceeds (or falls below) S after N cells, saving rollout cost on policies that clearly pass / clearly fail.
- **Why:** Optuna and Hyperband have made early-stop the default in adjacent ML. On a 1440-rollout sweep where the policy fails the first 100, running the other 1340 is pure waste.
- **Scope:** S
- **Disjoint with:** `runner/`. Conflicts with B-19 (Sobol indices need full sample structure) — must disable pruning when `sampling: sobol`.
- **Anti-feature?** Pruning destroys the "fair sample of the perturbation space" reading exactly when a regression is the most surprising. Default off, document loudly.

### B-27: W&B / MLflow optional sink
- **What:** `--wandb` / `--mlflow` flags that mirror per-Episode results to the chosen backend.
- **Why:** Many users already live in W&B/MLflow; gauntlet should not force a local-only workflow on them. Most adjacent eval frameworks (LeRobot, Octo) wire W&B by default.
- **Scope:** S
- **Disjoint with:** `runner/`, new optional extras.
- **Anti-feature?** **Directly contradicts PRODUCT.md "no cloud, no telemetry, no internet round-trip required at view time."** This is a sink that *may* be cloud and the auditor for that line is the user's network egress logs, not gauntlet. Only ship if it's purely opt-in, off by default, and the docs say "this leaves your machine."

### B-37: Per-step inference-latency tracking + budget gate
- **What:** New `runner/latency.py`. Wraps `policy.act()` with `time.perf_counter_ns()`, stores `inference_latency_ms: list[float]` on Episode (per-step), and surfaces `p50_ms`, `p95_ms`, `max_ms`, `over_budget_steps` as Episode columns. New CLI flag `--latency-budget-ms 100` marks an episode `latency_violated: True` when p95 exceeds the budget. Report HTML grows a "latency profile" subsection.
- **Why:** VLA-Perf (arxiv 2602.18397) is explicit that real-time deployability is a model-level property invisible to success-rate eval, and recommends 10–100 ms as the operating envelope. The Efficient-VLA survey (arxiv 2510.17111) reinforces that latency drives deployability decisions more than accuracy. Gauntlet measures success but not whether the policy can run on the target hardware — a critical gap for the fleet-operator persona in PRODUCT.md.
- **Scope:** S
- **Disjoint with:** `runner/runner.py`, `runner/episode.py` schema, `report/html.py`. No conflict.
- **Anti-feature?** Per-step `perf_counter` adds a tiny but non-zero overhead to every act call; on a 200-Hz control loop it's measurable. Default-on flag risks polluting the very thing it measures — gate behind `--track-latency`.

### B-38: Inference-delay jitter perturbation axis
- **What:** New `PerturbationAxis` of categorical type `inference_delay_jitter` with values like `0ms, 50ms, 200ms, 500ms`. Wraps `policy.act()` with `time.sleep(delay)` injected per step; the env continues forward-simulating during the sleep so the action arrives stale. Pairs with B-37's latency machinery.
- **Why:** Real-Time Chunking (arxiv 2506.07339, Black et al.) and the LeRobot RTC integration (huggingface.co/docs/lerobot/rtc) demonstrate that inference delay is a first-class cause of policy failure independent of model accuracy — "Leave No Observation Behind" (arxiv 2509.23224) shows even 100 ms of staleness collapses naive chunked policies. Gauntlet currently assumes zero-latency action delivery, masking this failure mode entirely.
- **Scope:** S
- **Disjoint with:** New axis class in `env/perturbation/axes.py`, hook in the runner's act-call path. Backend-agnostic (operates on the policy interface, not the sim).
- **Anti-feature?** Sleep-based delay is a sim of a sim — real-world inference delay co-occurs with thermal throttling, batch-pipeline contention, and network jitter that a flat `time.sleep` doesn't model. Useful as a coarse axis, dishonest as an exact predictor.

### B-39: Cross-checkpoint regression bisection (`gauntlet bisect`) — Shipped 2026-04-26
- **Status:** Shipped 2026-04-26. New `gauntlet.bisect` package + `gauntlet bisect` Typer subcommand. Binary-searches an ordered checkpoint list `[good, *intermediates, bad]` against a single target cell, paired against the cached good baseline via the B-08 `compute_paired_cells` engine. Decision rule collapses the search interval whenever the target cell's Newcombe / Tango paired-CI upper bound falls strictly below zero. Anti-feature framing preserved verbatim: no weight-space interpolation; the user supplies discrete checkpoint ids and a resolver (any `--policy` spec) that knows how to load each one.

### B-40: Suite-level provenance hash + result cache key
- **What:** Extend `runner/provenance.py` so each Suite emits a deterministic 16-char hash of `(suite YAML AST, axis values sorted, episodes_per_cell, gauntlet version, env asset SHAs)`. Use this hash as the on-disk cache key in `runner/cache.py`. New CLI `gauntlet suite hash <suite.yaml>` prints it. Two suites with identical semantics produce identical hashes regardless of YAML key ordering.
- **Why:** vla-eval (arxiv 2603.13966) explicitly identifies the "is this result from the same eval config?" provenance question as the dominant source of cross-paper irreproducibility, and recommends content-addressed config hashes as the fix. Gauntlet has per-episode provenance (B-22) but no suite-level fingerprint, so two suites that differ in formatting alone re-run instead of cache-hitting.
- **Scope:** S
- **Disjoint with:** `runner/provenance.py`, `runner/cache.py`, `suite/loader.py`. No conflicts.
- **Anti-feature?** Hash stability across gauntlet versions is brittle — a bug-fix to the YAML loader's normalisation invalidates every cached run. Need an explicit hash-version field and a graceful "stale cache, re-run" path.

### B-41: Statistical-power calculator on `gauntlet suite plan`
- **What:** Extend `report/wilson.py` with `required_episodes(p1, p2, alpha=0.05, power=0.8) -> int` (two-proportion z-test, paired-CRN-aware via B-08). New CLI `gauntlet suite plan <suite.yaml> --detect-delta 0.1 --power 0.8` reads each cell's existing rate (or a user-supplied baseline) and prints how many `episodes_per_cell` are needed to detect the requested effect size. Surfaces in B-25's suite-linter as a warning when the configured count is too low.
- **Why:** "Beyond Binary Success" (arxiv 2603.13616) and RoboEval (arxiv 2507.00435) both call out that 90% of published robot-policy comparisons run with statistically insufficient sample sizes — the operator infers "policy A beats B" from a difference smaller than the standard error. The fix is the standard pre-eval power calculation that every adjacent ML field does and gauntlet doesn't.
- **Scope:** S
- **Disjoint with:** `report/wilson.py`, `suite/linter.py`, new CLI subcommand. Synergises with B-03 + B-08.
- **Anti-feature?** A power-calc that says "you need 400 episodes per cell" gives the user license to *run only 400* and never look at the long tail of failure modes that only show up at 10× that count. Power-calc optimises for the binary-success summary statistic; it's exactly the rule that "failures over averages" §1 warns against.

## Real-world / Sim-to-real

### B-28: Sim-vs-real correlation report (SureSim-style)
- **What:** Given a directory of paired `(sim_episode, real_episode)` results matching on suite-cell-and-seed, compute the per-axis sim-real correlation and emit a "this axis matters in sim but not in real" / "vice versa" table.
- **Why:** SIMPLER (arxiv 2405.05941, CoRL 2024) and SureSim (arxiv 2510.04354, Oct 2025) both demonstrate that paired sim-real correlation is the gold-standard sim-to-real validity signal. Doesn't require gauntlet to *do* real-robot eval — just to *consume* it.
- **Scope:** M
- **Disjoint with:** New `aggregate/sim_real.py`. Episode schema needs an optional `source: "sim" | "real"` tag.
- **Anti-feature?** Requires the user to have real-robot data, which most won't. Niche, but the niche is the most demanding users.

### B-29: Embodiment-transfer scoring across backends
- **What:** Existing `gauntlet compare --allow-cross-backend` returns a yes/no. Extend to a structured "drift map" — per-axis-value, how far does the same policy-and-seed differ across sim backends?
- **Why:** Cross-embodiment eval (Open-X-Embodiment, RT-X) is the dominant transfer story. Gauntlet's 4-backend parity is the foundation for this analysis but the analysis itself isn't built. SIMPLER's "control and visual disparities" framing is the precedent.
- **Scope:** M
- **Disjoint with:** `compare/`, `diff/`. Builds on the existing cross-backend gate.
- **Anti-feature?** "Sim-vs-sim drift" is interesting only insofar as one of the sims is a stand-in for real — the analysis is one step removed from what the user actually wants (sim-vs-real).

## Safety / Robustness

### B-30: Safety constraint columns (collisions, joint limits, energy budget)
- **What:** New first-class `safety_violations` field on Episode: `{collisions: int, joint_limit_excursions: int, energy_over_budget: bool, workspace_excursions: int}`. A "successful" episode with any violation is tagged `success_unsafe` in the report.
- **Why:** Safety-Gymnasium (NeurIPS 2023), Safety-Gym, safe-control-gym (arxiv 2109.06325). Direct fit for "failures over averages" — a policy that dings the table on every successful pick is not actually a successful policy.
- **Scope:** M
- **Disjoint with:** Episode schema, all four backends (each must report violations). MuJoCo first; PyBullet/Genesis/Isaac follow.
- **Anti-feature?** Cross-backend collision-detection semantics are wildly different; "collision count" as a portable concept is a fiction. Honest answer: define the metric to be MuJoCo's `ncon`-delta and accept everyone else's number is approximate.

### B-31: Sensor-attack / adversarial-image perturbation axis
- **What:** New axis `image_attack` with values like `gaussian_noise`, `jpeg_compression_q10`, `random_patch_8x8`, `dropout_camera_2_of_3`.
- **Why:** "Exploring the Robustness of VLAs against Sensor Attacks" (DOI 10.1145/3733800.3763262) showed OpenVLA collapses under moderate visual attacks and π0 doesn't — exactly the kind of structured robustness story gauntlet should make legible.
- **Scope:** S
- **Disjoint with:** New `PerturbationAxis` that wraps `obs["image"]` post-render. Backend-agnostic (operates on the image, not the sim).
- **Anti-feature?** "Sensor attack" framing has security-research baggage that the engineer-shipping-checkpoints user doesn't care about — risks naming-by-academia rather than naming-by-use-case.

### B-32: Initial-state out-of-distribution shift axis
- **What:** New axis `initial_state_ood` that perturbs the cube's starting pose by N standard deviations *outside* the training initial-pose distribution (where the user provides the training prior or it's auto-fit from a reference run).
- **Why:** LIBERO-PRO, LIBERO-Plus and the OOD-detection literature (Task-Driven OOD Detection, IEEE T-RO 2024) all converge on initial-state shift as the highest-yield OOD axis. Gauntlet has `ObjectInitialPose` but nothing that explicitly probes the OOD tail.
- **Scope:** S
- **Disjoint with:** Extends existing `ObjectInitialPose` axis with OOD framing; one new axis class.
- **Anti-feature?** "OOD" is only meaningful relative to a training distribution the user must declare; if they don't, the axis collapses to "more aggressive ObjectInitialPose" and the framing is hollow.

### B-42: Camera-extrinsics perturbation axis
- **What:** New axis `camera_extrinsics` with structured values `{translation: [dx, dy, dz], rotation: [drx, dry, drz]}` applied to the rendering camera at episode start. Suite YAML supports both an enumerated list and a Sobol-friendly continuous range. Operates on the env's render camera, not the policy.
- **Why:** RoboView-Bias (arxiv 2509.22356) found viewpoint to be the *single most influential* visual factor across every evaluated agent — success rates fluctuate sharply between near-identical viewpoints — and "Do You Know Where Your Camera Is?" (arxiv 2510.02268) shows ACT / Diffusion Policy / SmolVLA all rely on background visual shortcuts when extrinsics aren't conditioned, so a tiny camera shift collapses them. LIBERO-Plus (already-cited 2510.13626) names camera-view-shift as one of seven canonical robustness dimensions. Gauntlet has `CameraNoise` but nothing that perturbs *pose*.
- **Scope:** S
- **Disjoint with:** New axis class in `env/perturbation/axes.py`; touches `TabletopEnv.set_perturbation` to wire the camera-pose offset. MuJoCo / PyBullet first; Genesis / Isaac follow the existing visual-axis backend-restriction pattern.
- **Anti-feature?** Camera-extrinsics as a numeric axis tempts users to over-fit Sobol indices on a parameterisation that's actually a 6-DoF non-Euclidean manifold (SO(3) rotations don't compose linearly). Disclaimer in the report or risk false-precision rankings.

### B-43: Color / saturation visual-bias perturbation axis
- **What:** New axis `color_shift` with values like `hue_+30, hue_-30, saturation_0.5, saturation_1.5, achromatic` applied as an HSV transform on `obs["image"]` post-render. Backend-agnostic, layered on top of the existing post-render `image_attack` infrastructure (B-31).
- **Why:** RoboView-Bias (arxiv 2509.22356) demonstrated all evaluated VLAs have a strong, asymmetric performance bias toward high-saturation hues over achromatic / low-saturation scenes — a grounding failure orthogonal to viewpoint and lighting. LIBERO-Plus's "image noise" dimension partially covers this; the explicit color/saturation cut is what RoboView-Bias adds and gauntlet currently lacks.
- **Scope:** S
- **Disjoint with:** Reuses the post-render hook from B-31. New small `env/color_attack.py` next to `env/image_attack.py`. No backend changes.
- **Anti-feature?** HSV-shift on the rendered RGB does not faithfully simulate real-world illumination changes (which alter materials' specular response too) — it's a stand-in. Honest path: name the axis `color_shift_synthetic` and warn that it's the post-render proxy, not the real-world thing.

## Dependency Drift (internal scan, 2026-04-25)

Surfaced by `uv pip list --outdated` against the current `pyproject.toml` ceilings. Each requires a major-version test pass + ceiling bump.

### B-33: Bump pandas to <4 (currently `<3`)
- **What:** Lift `pyproject.toml` ceiling on `pandas` from `<3` to `<4`. Test under pandas 3.x. Likely API touchpoints: aggregate / report dataframe construction, `to_dict("records")` semantics, NA-handling defaults.
- **Why:** pandas 3.0.x released; `<3` ceiling now blocks dependency resolution alongside other modern deps. Pandas 3.0 hardened nullable dtypes — may catch real `None` vs. `NaN` bugs in our analyse path.
- **Scope:** S
- **Disjoint with:** `pyproject.toml`, possibly `report/`, `aggregate/`. Tests should catch dtype regressions.
- **Anti-feature?** Pandas 3 changed several deprecation warnings to errors; some refactor possible.

### B-34: Bump pytest to <10 (currently `<9`)
- **What:** Lift `[dependency-groups.dev]` (and per-extra dev groups) ceiling on `pytest` from `<9` to `<10`. Re-pin `pytest-mock` if it lags. Test the existing suite under pytest 9.x.
- **Why:** pytest 9.x released; `<9` ceiling locks the dev environment. pytest 9 dropped Python <3.10 support and tightened a few fixture-scoping semantics — both are no-ops for this project but worth verifying.
- **Scope:** S
- **Disjoint with:** `pyproject.toml`. Pure dev-dep bump.
- **Anti-feature?** pytest-mock and pytest-cov plugin compat lag historically — may need to wait for the plugin ecosystem.

### B-35: Bump mujoco to <5 (currently `<4`)
- **What:** Lift `pyproject.toml` ceiling on `mujoco` from `<4` to `<5`. Verify the offscreen render path and the headless-no-DISPLAY CI path still work under MuJoCo 4.x.
- **Why:** MuJoCo 3.8.0 is current; 4.x will drop. The renderer C-API has had stability promises across 3.x — likely a clean bump but worth a CI validation.
- **Scope:** S
- **Disjoint with:** `pyproject.toml`. Render path tests under the `render` marker.
- **Anti-feature?** MuJoCo 4.x is on the horizon, not shipped — premature.

### B-36: Bump rich to <17 (currently `<16`)
- **What:** Lift `pyproject.toml` ceiling on `rich` from `<16` to `<17`. CLI Console / Text / traceback surface is small and stable across rich major versions.
- **Why:** rich 15.x released; the existing `<16` ceiling note in `pyproject.toml` already anticipated this trajectory.
- **Scope:** S
- **Disjoint with:** `pyproject.toml`. CLI rendering surface.
- **Anti-feature?** None substantive.

---

## Sources

Primary papers and benchmarks cited by the items above:

- LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models Beyond Memorization (arXiv 2510.03827)
- LIBERO-Plus: In-depth Robustness Analysis of VLA Models (arXiv 2510.13626)
- Failure Prediction at Runtime for Generative Robot Policies / FIPER (arXiv 2510.09459, NeurIPS 2025)
- Can We Detect Failures Without Failure Data? / FAIL-Detect (arXiv 2503.08558)
- RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields (arXiv 2412.02818)
- RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation (arXiv 2507.00435, RSS 2025)
- SureSim: Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators (arXiv 2510.04354)
- SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation (arXiv 2405.05941, CoRL 2024)
- RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies (arXiv 2506.18123)
- AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World (arXiv 2503.24278)
- Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison (arXiv 2603.13616)
- LADEV: Language-Driven Testing and Evaluation Platform for VLA Models (arXiv 2410.05191)
- Adversarial Data Collection for Imitation Learning (arXiv 2503.11646)
- Unsupervised Discovery of Failure Taxonomies from Deployment Logs (arXiv 2506.06570)
- LAMBDA: Benchmark for Data-Efficiency in Long-Horizon Indoor Mobile Manipulation (arXiv 2412.05313)
- RoboCerebra: Long-horizon Manipulation Benchmark (arXiv 2506.06677)
- VLA-Perf: Demystifying VLA Inference Performance (arXiv 2602.18397)
- ManiSkill3: GPU Parallelized Robotics Simulation (arXiv 2410.00425, RSS 2025)
- VLABench: Large-Scale Language-Conditioned Robotics Benchmark (ICCV 2025)
- BiCoord: Bimanual Manipulation Benchmark for Long-Horizon Spatial-Temporal Coordination (arXiv 2604.05831)
- Safety-Gymnasium: Unified Safe RL Benchmark (NeurIPS 2023)
- safe-control-gym: Unified Benchmark Suite for Safe Learning-based Control (arXiv 2109.06325)
- SoftGym, DaXBench, GarmentLab (NeurIPS 2024) — deformable backends
- Hybrid Consistency Policy: Decoupling Multi-Modal Diversity (arXiv 2510.26670)
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023 / IJRR 2025)
- π0: A Vision-Language-Action Flow Model for General Robot Control (Physical Intelligence)
- Open X-Embodiment / DROID (arXiv 2310.08864)
- Task-Driven Detection of Distribution Shifts With Statistical Guarantees for Robot Learning (IEEE T-RO 2024)
- Exploring the Robustness of VLAs against Sensor Attacks (ACM 10.1145/3733800.3763262)

Added 2026-04-25 (B-37+ scan):

- How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf (arXiv 2602.18397)
- Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey (arXiv 2510.17111)
- Real-Time Execution of Action Chunking Flow Policies (arXiv 2506.07339)
- Leave No Observation Behind: Real-time Correction for VLA Action Chunks (arXiv 2509.23224)
- vla-eval: A Unified Evaluation Harness for Vision-Language-Action Models (arXiv 2603.13966)
- WorldEval: World Model as Real-World Robot Policies Evaluator (arXiv 2505.19017)
- Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations (arXiv 2509.18953)
- RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation (arXiv 2509.22356)
- Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning (arXiv 2510.02268)
