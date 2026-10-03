# Stability & Versioning Policy

From `0.2.0` onward, Gauntlet commits to **[Semantic
Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)** on its public
surface.

> A user pinning `gauntlet-robotics>=0.2,<0.3` in `pyproject.toml` will never
> have a passing CI break because Gauntlet renamed a public symbol,
> changed a CLI flag, or changed the on-disk schema. Breakages happen
> only on a minor (`0.x.0`) bump while we are pre-1.0, and only on a
> major (`x.0.0`) bump after 1.0.

## The public surface

Everything in the lists below is **public** and follows semver.
Everything else is **private** and may change without notice — even
inside a patch release.

### Top-level package
- `gauntlet.__version__` — the running version string.
- `gauntlet.register_envs()` — gymnasium global-registry registration.

### `gauntlet.policy`
Anything exported via `gauntlet.policy.__all__`. Notably:
`Policy`, `Action`, `Observation`, `ResettablePolicy`,
`SamplablePolicy`, `RandomPolicy`, `ScriptedPolicy`,
`HuggingFacePolicy` (opt-in via `[hf]` extra),
`LeRobotPolicy` (opt-in via `[lerobot]` extra),
`resolve_policy_factory`, `PolicySpecError`.

### `gauntlet.env`
Anything exported via `gauntlet.env.__all__`. Notably:
`GauntletEnv`, `TabletopEnv`, `TabletopStackEnv`,
`MobileTabletopEnv`, `CameraSpec`, `SubtaskMilestone`,
`PerturbationAxis`, `AXIS_NAMES`, `axis_for`, `register_env`,
`N_DISTRACTOR_SLOTS`.

### `gauntlet.suite`
Anything exported via `gauntlet.suite.__all__`. Notably:
`Suite`, `SuiteCell`, `AxisSpec`, `SamplingMode`,
`SAMPLING_MODES`, `BUILTIN_BACKEND_IMPORTS`, `WorstCaseConfig`,
`load_suite`, `load_suite_from_string`, `lint_suite`,
`LintFinding`, `LintSeverity`.

### `gauntlet.runner`
Anything exported via `gauntlet.runner.__all__`. Notably:
`Runner`, `Episode`, `episode_hash`, `rollout_hash`,
`obs_state_hash`, `assert_byte_identical`, `IMAGE_OBS_KEYS`,
`STATE_OBS_KEYS`, `compute_suite_hash`,
`compute_suite_provenance_hash`, `compute_env_asset_shas`,
`capture_git_commit`, `capture_gauntlet_version`.

### `gauntlet.report`
Anything exported via `gauntlet.report.__all__`. Notably:
`build_report`, `render_html`, `write_html`,
`Report`, `AxisBreakdown`, `CellBreakdown`, `FailureCluster`,
`Heatmap2D`, `SensitivityIndex`, `AbstentionMetrics`,
`compute_abstention_metrics`, plus the trajectory-taxonomy
symbols.

### `gauntlet.replay`
`replay_one`, `parse_override`, `validate_overrides`,
`OverrideError`.

### `gauntlet.monitor`
`ConformalFailureDetector`, `ActionEntropyStats`,
`action_entropy`, `DriftReport`, `PerEpisodeDrift` and the
trained-AE schema. Torch is an opt-in dep via `[monitor]`.

### `gauntlet.realsim` (provisional)
`RealSimRenderer` (Protocol), `RendererFactory`,
`register_renderer`, `get_renderer`, `list_renderers`,
`RendererRegistryError`, the scene-input I/O surface
(`load_scene`, `save_scene`, `ingest_frames`, `IngestionError`,
`SceneIOError`, `INTRINSICS_REQUIRED_KEYS`,
`IMAGE_MAGIC_BYTES`, `MANIFEST_FILENAME`).

The renderer-Protocol shape is **provisional** — it is committed to
semver, but third-party renderer plugins should expect non-breaking
additions (new optional kwargs) on minor releases until 1.0.

### `gauntlet.plugins`
The entry-point names listed in `pyproject.toml` under
`[project.entry-points.gauntlet.*]` — `gauntlet.policies`,
`gauntlet.envs`, `gauntlet.axes`, `gauntlet.samplers`,
`gauntlet.sinks`, `gauntlet.cli`. Adding new groups is a minor
bump; removing or renaming one is a major bump (pre-1.0: minor).

### CLI
Every `gauntlet <subcommand>` documented in `README.md`:
`run`, `report`, `compare`, `diff`, `replay`, `bisect`,
`aggregate`, `monitor train`, `monitor score`,
`ros2 publish`, `ros2 record`, `suite plan`, `suite lint`.

- Removing a flag, removing a subcommand, or changing a default
  in a way that flips success → failure is a breaking change.
- Adding a new flag with a backward-compatible default, or a new
  subcommand, is a non-breaking change.

### On-disk schemas
The following JSON / YAML schemas follow semver:
- `Suite` YAML (`gauntlet.suite.Suite`).
- `Episode` records (`episodes.json`).
- `Report` records (`report.json`).
- `DriftReport` records (`drift.json`).
- The fleet meta-report (`fleet_report.json`).
- Suite-provenance hashes (intentional invalidator: a Gauntlet
  version bump *does* invalidate the cache by design — that is
  documented behaviour, not a regression).

Additive fields are non-breaking. Removing or renaming a field,
or changing its semantics, is breaking.

## What is **not** public

Anything not listed above is private, including:

- Module paths not re-exported from a subpackage `__init__.py`.
- Helper functions whose name starts with `_`.
- Internal pydantic field names not surfaced in the schemas above.
- The on-disk layout of `out/` beyond the three files
  (`episodes.json`, `report.json`, `report.html`).
- The contents of the `out/cache/` rollout cache.
- The HTML report's CSS, DOM structure, and chart configuration.
- The contents of `gauntlet.env.assets/`.
- Test utilities under `tests/`.

## Deprecation policy

When a public symbol or CLI flag is on the path to removal:

1. **One minor release of warning.** The symbol is annotated with
   `DeprecationWarning` (or, for CLI, a stderr warning) at first
   use, with a pointer to the replacement.
2. **Removal on the next minor bump (pre-1.0)** or the next major
   bump (post-1.0).
3. **Documented in `CHANGELOG.md`** under a `Deprecated` section
   on introduction and under a `Removed` section on removal.

If a deprecation cannot meet the one-release-of-warning bar (e.g.
a security advisory), it is called out explicitly in the changelog
and a patch release is issued.

## Pre-1.0 caveats

Until `1.0.0`:

- A minor version bump (`0.x.0`) may break the public surface
  with the deprecation policy above. We aim not to.
- The provisional surfaces (`gauntlet.realsim` renderer
  Protocol; `gauntlet.bisect`; multi-camera `CameraSpec`)
  are still expected to evolve. Breakages there will still be
  called out in `CHANGELOG.md` but may skip the one-release
  warning window if the API was clearly experimental.

After `1.0.0` (target: once an external team has shipped a
production policy gated by a `gauntlet compare` run for ≥1
quarter), the policy tightens: breaking changes only on a major
bump, with at minimum one full minor release of `DeprecationWarning`.
