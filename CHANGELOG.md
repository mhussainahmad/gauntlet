# Changelog

All notable changes to **Gauntlet** are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project commits to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from `0.2.0` onward (see `docs/stability.md`).

## [0.2.0] — 2026-05-28

First public release on PyPI. Phase 1 (MVP) and Phase 2 (real-policy
adapters, runtime observability) are complete; Phase 3 (fleet-scale
tooling) is partially shipped — see "Phase 3 (partial)" below.

### Added

#### Core (Phase 1)
- `Policy` adapter protocol (`gauntlet.policy.Policy`) with reference
  `RandomPolicy` and `ScriptedPolicy` implementations.
- MuJoCo tabletop env (`gauntlet.env.tabletop.TabletopEnv`) with seven
  perturbation axes: `lighting_intensity`, `camera_offset_x/y`,
  `object_texture`, `object_initial_pose`, `distractor_count`, plus
  the polish-added `camera_extrinsics`, `color_shift`,
  `inference_delay`, `instruction_paraphrase`, `object_swap`.
- Suite YAML schema and loader (`gauntlet.suite.Suite`) with
  Cartesian / Latin-Hypercube / Sobol / worst-case / adversarial
  samplers.
- Parallel `Runner` (`gauntlet.runner.Runner`) with fully-seeded,
  reproducible episodes and incremental rollout caching.
- Episode + Report schemas (`pydantic` v2) with Wilson confidence
  intervals on every cell / axis breakdown.
- HTML report generator with failure-cluster-first layout (jinja2 +
  Chart.js) and self-contained output.
- CLI: `gauntlet run`, `gauntlet report`, `gauntlet compare`,
  `gauntlet diff`, `gauntlet replay`, `gauntlet bisect`,
  `gauntlet suite plan` (statistical-power calculator).

#### Phase 2 — real-policy adapters + runtime observability
- HuggingFace policy adapter (`HuggingFacePolicy`) wrapping OpenVLA-style
  checkpoints; opt-in via `pip install 'gauntlet[hf]'`.
- LeRobot policy adapter (`LeRobotPolicy`) wrapping SmolVLA / π0 /
  diffusion-policy checkpoints; opt-in via
  `pip install 'gauntlet[lerobot]'`.
- π0, RDT, GR00T, Decision-Transformer policy adapters.
- PyBullet, Genesis, Isaac Sim backends — all four backends share
  byte-identical action/observation spaces and the canonical seven
  perturbation axes. Cross-backend `gauntlet compare` is gated by
  `--allow-cross-backend`.
- Runtime drift detector (`gauntlet monitor`) — small observation
  autoencoder + action-std OOD scoring; opt-in via
  `pip install 'gauntlet[monitor]'`.
- ROS 2 publisher / recorder bridges (`gauntlet ros2 publish/record`).
- Multi-camera observation support via `CameraSpec`.
- Conformal-calibrated failure prediction (FIPER + FAIL-Detect signals).
- Behavioural metrics beyond binary success: `time_to_success`,
  `path_length_ratio`, `jerk_rms`, `near_collision_count`, `peak_force`.
- Failure-mode taxonomy via DTW-clustered trajectory analysis.
- Inference-latency tracking + `--max-inference-ms` budget enforcement.
- Common-random-numbers paired comparison for variance-reduced
  `gauntlet compare`.

#### Phase 3 (partial)
- Fleet-wide failure-mode clustering (`gauntlet aggregate <runs-dir>`).
- Self-contained web dashboard.
- Real-to-sim scene reconstruction *input pipeline* — `RealSimRenderer`
  ships as a `typing.Protocol`; an external gaussian-splatting (or
  other) renderer plugin can register without touching the schema.

#### Tooling
- `gauntlet.plugins` entry-point system for third-party policies,
  envs, axes, samplers, sinks, and CLI commands.
- Gymnasium global-registry registration on package import.
- `pip install gauntlet` now resolves from PyPI.

### Changed
- Development Status classifier promoted from `2 - Pre-Alpha` to
  `3 - Alpha`.
- Public API is now covered by the semver policy in
  `docs/stability.md`. Anything outside `gauntlet.__all__` (or the
  documented public surface of subpackages) is private and may
  change without notice.

### Stability
- Public surface stable: `gauntlet.policy.Policy`,
  `gauntlet.policy.{RandomPolicy, ScriptedPolicy, HuggingFacePolicy,
  LeRobotPolicy}`, `gauntlet.suite.Suite`,
  `gauntlet.runner.{Runner, Episode}`,
  `gauntlet.report.{build_report, write_html, Report}`,
  `gauntlet.env.tabletop.TabletopEnv`, `gauntlet.register_envs`,
  `gauntlet.__version__`, and the `gauntlet run / report / compare /
  diff / replay / bisect / aggregate / monitor / ros2 / suite plan`
  CLI commands.
- Provisional: `gauntlet.realsim` (Phase 3 renderer protocol);
  `gauntlet.bisect`; the multi-camera `CameraSpec` API.

[0.2.0]: https://github.com/gauntlet-eval/gauntlet/releases/tag/v0.2.0
