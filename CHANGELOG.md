# Changelog

All notable changes to **Gauntlet** are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project commits to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from `0.2.0` onward (see `docs/stability.md`).

## [Unreleased]

### Added
- `env: crop-row` (`CropRowEnv`, gym id `gauntlet/CropRow-v0`, B-47):
  camera-guided crop-row following on a procedurally generated field,
  rendered in numpy (no simulator). The observation is the image only;
  ground-truth lateral / heading error are in `info`.
- Perturbation axes `dust_density`, `glare_intensity`, `motion_blur`,
  `weed_density` and `row_curvature` (crop-row only).
- `--policy crop-row-classical` (`CropRowClassicalPolicy`): excess-green
  / Otsu / calibrated line-fit row detector.
- `examples/crop_row/`: a CNN trained on clean scenes (`train_cnn.py`,
  torch) with numpy inference (`cnn_policy.py`, weights included).
- `examples/suites/crop-row-field.yaml` and `docs/crop-row.md`: the two
  models compared on 480 rollouts each.

## [0.4.0] — 2026-10-03

### Added
- `Episode.observation_invalid` / `Episode.action_invalid`. A NaN or
  ±Inf in an observation or in the policy's action now ends the rollout
  as a failure with the matching flag set, instead of flowing into the
  next `policy.act` / `env.step`. One diverged checkpoint no longer
  corrupts results silently or crashes a sweep; `gauntlet run` prints
  how many episodes were affected. Both fields default to `False`, so
  older `episodes.json` files still load, and `episode_hash` is
  unchanged.
- `gauntlet.runner.worker.validate_observation(obs)`.

### Changed
- Every backend's `step` (MuJoCo tabletop / push / stack / mobile,
  PyBullet, Genesis, Isaac) raises `ValueError` on a non-finite action.
  Previously `np.clip` passed NaN through to the simulator.
- README: 55-second video overview; absolute links so they also work on
  PyPI; feature table lists the stacking and mobile-base tasks and
  states each backend's verification level.
- `docs/api.md` documents every public symbol again (14 were missing).
- CI's torch-free job now runs `slow`-marked tests too, including the
  API-docs freshness check.

### Fixed
- MuJoCo `CameraSpec` angles were applied as degrees, not the
  documented radians: the asset has no `<compiler angle="radian"/>`, so
  `rx=1.2` compiled to a 1.2° tilt and custom cameras pointed almost
  straight down. Angles are now converted on injection. The multi-camera
  example in `docs/guide.md` uses poses checked against the fix.
- `gauntlet compare --help` no longer says the HTML companion is
  "deferred"; it points at `--github-summary` and `gauntlet diff`.

## [0.3.0] — 2026-10-03

### Added
- Experimental image observations on the Isaac Sim backend:
  `IsaacSimTabletopEnv(render_in_obs=True, render_size=(H, W))` adds a
  camera, key light and cube materials, and maps `lighting_intensity`,
  `camera_offset_x/y` and `object_texture` onto them. Written against
  the Isaac Sim 5.0 sources and tested only against a fake `isaacsim`
  namespace — **not run on real hardware**; the constructor warns.
  Each observation renders after all scene writes (including the
  grasped-cube snap), so frames are never one step or one episode
  stale.
- `gsplat` real-to-sim renderer now renders (B-46). The first call per
  scene fits a small set of 3D gaussians to the scene's frames; any
  viewpoint is then rasterised. Uses gsplat's CUDA rasterizer when it
  works and falls back to a pure-PyTorch rasterizer of the same model
  (CPU or GPU) otherwise. The default fit is a smoke-test reconstructor
  and says so. Poses are read as OpenGL camera-to-world by default;
  `camera_convention="opencv"` for COLMAP-style poses. The
  `[realsim-gsplat]` extra now includes Pillow.
- `env: tabletop-push` (B-45): planar pushing on the tabletop scene.
  No grasp, colliding end-effector, success only once the cube settles
  inside the target. Shares every tabletop axis. Ships with a
  closed-loop reference controller (`--policy scripted-push`) and
  `examples/suites/tabletop-push-smoke.yaml`.
- `image_attack`, `color_shift_synthetic` and `instruction_paraphrase`
  now run from a suite through `gauntlet run` / `Runner`. Previously the
  wrappers existed but the runner rejected the axes ("unknown
  perturbation axis"). Registry envs are built with `render_in_obs=True`
  when an image axis is present. Example:
  `examples/suites/tabletop-sensor-language.yaml`.

### Fixed
- `gauntlet replay` failed with "unknown perturbation axis" on episodes
  from suites using `image_attack`, `color_shift_synthetic` or
  `instruction_paraphrase`; it now builds the env through the same
  wrapper wiring as `gauntlet run`.
- `gauntlet replay --override axis=NaN` (or `inf`) was accepted; non-finite
  override values are now rejected.
- Stacking two of the observation wrappers dropped the backend's own
  axes: each wrapper read `type(env).AXIS_NAMES`, which is empty on a
  wrapper class.
- An active image attack or colour shift on an env that renders no
  image now raises instead of silently passing frames through.

## [0.2.1] — 2026-10-03

### Fixed
- HTML report: per-axis bar charts dropped every bucket whose value is
  a whole number (`0`, `100`, `4`, ...). The chart looked keys up as
  `"0"` while `report.json` stores `"0.0"`.
- MuJoCo tabletop: contacts already present at reset (the cube resting
  on the table) were counted as collisions on the first step, so every
  episode reported `n_collisions >= 4` and every success was tagged
  unsafe (`success_safe_rate == 0`). Near-collision counts and peak
  contact force had the same problem (the cube's weight on the table
  showed up as ~500 near-collisions per episode). All three now skip
  geom pairs that were already touching at reset.
- MuJoCo tabletop: distractors enabled by `distractor_count` were
  visible but never collided. The env flipped `geom_contype` at
  runtime, but MuJoCo's broadphase filters on the compiled
  `body_contype` / `body_conaffinity` masks first. The body masks are
  now updated with the geom's.
- Sobol total-order indices credited within-cell seed noise to every
  axis, so an axis with no effect read about 0.5 on a stochastic
  policy. They are now computed from the between-cell variance. Results
  are unchanged when each cell holds one episode.
- `--policy module:attr` now finds a module in the working directory
  when run through the `gauntlet` console script (it previously only
  worked under `python -m gauntlet.cli`).

### Added
- `examples/suites/tabletop-field-conditions.yaml` and
  `docs/field-robustness.md`: a worked baseline-vs-regressed example
  over control latency, placement variance, clutter and lighting.
  The reference benchmark attached to each release now includes it.
- `docs/guide.md`: the long-form feature documentation that used to
  live in the README.

### Changed
- README rewritten as a short front page; design notes moved to
  `docs/design/`.
- Internal backlog ids removed from report column headers.

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
  checkpoints; opt-in via `pip install 'gauntlet-robotics[hf]'`.
- LeRobot policy adapter (`LeRobotPolicy`) wrapping SmolVLA / π0 /
  diffusion-policy checkpoints; opt-in via
  `pip install 'gauntlet-robotics[lerobot]'`.
- π0, RDT, GR00T, Decision-Transformer policy adapters.
- PyBullet, Genesis, Isaac Sim backends — all four backends share
  byte-identical action/observation spaces and the canonical seven
  perturbation axes. Cross-backend `gauntlet compare` is gated by
  `--allow-cross-backend`.
- Runtime drift detector (`gauntlet monitor`) — small observation
  autoencoder + action-std OOD scoring; opt-in via
  `pip install 'gauntlet-robotics[monitor]'`.
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
- Published on PyPI as `gauntlet-robotics` (`pip install gauntlet-robotics`);
  the import package and `gauntlet` CLI command are unchanged.

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

[0.4.0]: https://github.com/mhussainahmad/gauntlet/releases/tag/v0.4.0
[0.3.0]: https://github.com/mhussainahmad/gauntlet/releases/tag/v0.3.0
[0.2.1]: https://github.com/mhussainahmad/gauntlet/releases/tag/v0.2.1
[0.2.0]: https://pypi.org/project/gauntlet-robotics/0.2.0/
