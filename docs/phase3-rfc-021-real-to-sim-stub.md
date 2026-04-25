# Phase 3 RFC 021 — Real-to-sim scene reconstruction stub

Status: Accepted
Phase: 3
Task: 18
Spec reference: `GAUNTLET_SPEC.md` §7 ("Real-to-sim scene reconstruction
(gaussian splatting from robot cameras into an eval environment)").

## 1. Why this exists

A team that wants to evaluate a policy against an environment that
mirrors a *real* customer scene currently has to hand-build a MuJoCo
XML or PyBullet URDF that approximates the geometry, lighting, and
texture observed by the deployed robot. That hand-building is the slow
path; it does not scale across customer scenes; and the resulting eval
environment drifts from reality every time the customer rearranges
their workspace.

The endgame the spec calls out is gaussian-splatting reconstruction of
those scenes from robot camera dumps directly into a renderable eval
backend. Shipping the full endgame at once is not viable — gaussian
splatting needs `torch` + CUDA + a multi-gigabyte training pipeline,
all of which violate spec §6 ("small deps", "no torch in the core").

This RFC therefore lands the *input pipeline* and the *renderer
extension point*. A future contributor (or a third-party plugin) can
implement an actual gaussian-splatting renderer against the
:class:`RealSimRenderer` Protocol without touching the schema or the
CLI surface.

## 2. Non-goals

* No renderer. No `torch`, no CUDA, no training loop, no dataset
  download. The whole point of this RFC is to defer those.
* No camera calibration helper. Users supply intrinsics + extrinsics
  via JSON; we do not run COLMAP / OpenCV / etc. on the frames.
* No image preprocessing. We do *not* resize, crop, undistort, or
  re-encode frames during ingest. The pipeline is a manifest builder
  on top of files the user already has.
* No new external Python dependency. Reuses `pydantic`, `typer`,
  `pathlib`, `numpy`. **Pillow is intentionally not required** — it
  lives only in optional extras (`[hf]`, `[lerobot]`, `[monitor]`)
  and we cannot pull it into the core just to validate frame
  headers. Header validation uses raw magic bytes (§4.4).
* No changes to existing schemas. :class:`Episode`, :class:`Report`,
  :class:`Suite`, etc. are untouched — backwards-compat is sacred,
  same rule the fleet aggregator (RFC 019) and dashboard (RFC 020)
  followed.

## 3. Surface this RFC adds

### 3.1 Library

* `gauntlet.realsim.Scene` — a serialised reconstruction input set.
* `gauntlet.realsim.CameraFrame` — one row of the manifest.
* `gauntlet.realsim.CameraIntrinsics` — pinhole intrinsics + optional
  distortion.
* `gauntlet.realsim.Pose` — a 4x4 row-major rigid transform.
* `gauntlet.realsim.RealSimRenderer` — `typing.Protocol` for a
  renderer implementation.
* `gauntlet.realsim.ingest_frames(frames_dir, calib, *, source=None)`
  — validate a directory of frames + a calibration spec and produce
  a :class:`Scene`.
* `gauntlet.realsim.save_scene(scene, dir)` /
  `gauntlet.realsim.load_scene(dir)` — round-trip on-disk persistence.
* `gauntlet.realsim.register_renderer(name, factory)` /
  `gauntlet.realsim.get_renderer(name)` — local renderer registry.

### 3.2 CLI

```
gauntlet realsim ingest <frames-dir> --calib <calib.json> --out <scene-dir>
                                    [--source <freeform-tag>]
                                    [--symlink]
gauntlet realsim info <scene-dir>
```

`info` dumps a one-screen summary of the manifest (frame count,
intrinsics ids, time range, source tag).

`ingest` writes `<out>/manifest.json` plus a `<out>/frames/` directory
of frame files. Default copies frames via `shutil.copy2`. Pass
`--symlink` to symlink instead — useful when the frames are large and
the user is iterating locally.

The subcommand group is wired the same way `monitor` and `ros2` are:
`realsim_app = typer.Typer(...); app.add_typer(realsim_app)`.

## 4. Key decisions

### 4.1 Pose representation: 4x4 row-major

Picked over quaternion + translation for three reasons:

1. **NeRFStudio / COLMAP `transforms.json` precedent.** The de-facto
   input format for the gaussian-splatting ecosystem stores per-frame
   `transform_matrix` as a 4x4 row-major list-of-lists. A future
   renderer plugin therefore gets near-zero adapter work — read the
   manifest, pull `pose`, hand it to the trainer.
2. **Single round-trip artefact.** Quaternion + translation requires
   two fields plus a re-hydration step on load; a 4x4 matrix is one
   field, validates structurally, and survives JSON without
   special handling.
3. **Validation surface is finite.** A `Pose` is valid if (a) it is
   4x4, (b) all 16 entries are finite floats, (c) the bottom row is
   `[0, 0, 0, 1]` within `1e-6` absolute tolerance. The rotation
   sub-block is *not* validated for orthonormality — robotics
   pipelines routinely emit slightly-off poses (numerical drift,
   per-frame state estimation jitter) and the renderer is responsible
   for renormalising. Documented in the docstring.

Trade-off: a 4x4 matrix is slightly larger on disk than a 7-vector
(quaternion + translation). At our target scale (typical robot dump
≤ 10k frames) the manifest is still ≤ 5 MB. Acceptable.

### 4.2 Intrinsics shared by id, not embedded per-frame

A robot recording typically has 1-4 cameras (wrist + scene + an
optional second wrist). Embedding the same `CameraIntrinsics` block
on every `CameraFrame` is wasteful and gives the schema two ways to
go inconsistent (per-frame copies that drift).

Instead `Scene.intrinsics` is a `dict[str, CameraIntrinsics]` and
each `CameraFrame.intrinsics_id` keys into it. The pipeline rejects
a manifest that references an unknown id at validation time.

### 4.3 Manifest format: JSON, not on-disk binary

Same rationale as `gauntlet.report.json` and
`gauntlet.aggregate.fleet_report.json` — text-readable, diffable,
trivially loadable from a notebook without our package, round-trips
through `tar` cleanly. The frame *images* stay binary; the manifest
just refs them by relative path.

`manifest.json` is the only file the schema knows about. Renderers
that produce derived artefacts (pre-computed point clouds, training
checkpoints) are expected to drop those into sibling directories
under the scene dir; the schema deliberately does not own that
namespace.

### 4.4 Image validation via magic bytes

Pillow is not a core dependency (§2). The pipeline therefore
recognises three frame container formats by magic bytes:

* **PNG**: `b"\x89PNG\r\n\x1a\n"` (8 bytes, RFC 2083 §3).
* **JPEG**: `b"\xff\xd8\xff"` (3 bytes — covers JFIF, EXIF, raw JPEG).
* **PPM**: `b"P6\n"` or `b"P3\n"` (the binary / ascii NetPBM
  variants — most useful for hand-rolled fixtures and for headless
  rendering pipelines that dump uncompressed frames).

Anything else is rejected with the file path and the first 8 bytes
in the error message. `Pose` / intrinsics validation runs first so
malformed metadata fails before we touch the image bytes.

### 4.5 Round-trip equivalence is structural, not byte-identical

`load_scene(save_scene(s))` returns a :class:`Scene` that is
`model_equal` to `s` at the pydantic level (same fields, same values,
same float normalisation). It does *not* guarantee the bytes of
`manifest.json` are identical across `save_scene` calls — Python
dict iteration is insertion-ordered but pydantic v2's
`model_dump_json` adds whitespace + key ordering decisions we do
not pin. Tests assert structural equivalence via
`Scene.model_dump(mode="json") == ...` rather than byte equality.

### 4.6 Renderer registry stays local

A separate `register_renderer` / `get_renderer` pair lives in
`gauntlet.realsim.renderer`, *not* in `gauntlet.plugins`. Two reasons:

1. The plugin module's public surface is pinned by an existing test
   (`tests/test_plugins.py::test_module_public_surface`). Bumping
   that surface for an extension point that has zero in-tree
   consumers is the wrong trade-off.
2. The realsim renderer sits at a different abstraction level from
   policy / env plugins (it owns a *scene*, not a sweep/episode), so
   bundling them in one entry-point group conflates concerns.

When a real gaussian-splatting renderer ships, the follow-up RFC
will decide whether to promote the registry into
`gauntlet.plugins` (one new entry-point group) or keep it local.
Either way, the :class:`RealSimRenderer` Protocol stays put.

## 5. Schema sketch

```python
class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float] = Field(default_factory=list)  # OpenCV order

class Pose(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")
    matrix: list[list[float]]  # 4x4 row-major; bottom row [0,0,0,1] +/- 1e-6

class CameraFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")
    path: str           # relative to Scene.frames_root, POSIX
    timestamp: float    # seconds, float
    intrinsics_id: str
    pose: Pose

class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="strings")
    version: int                     # bump on any incompatible schema move
    source: str | None               # freeform tag (robot id, log id, etc.)
    intrinsics: dict[str, CameraIntrinsics]
    frames: list[CameraFrame]
```

`Scene.version` starts at `1`. Any incompatible schema change bumps
this — the loader rejects unknown future versions with a clear
message rather than silently misparsing. Extensible newer-minor
fields can be added with defaults without bumping `version`.

## 6. Renderer Protocol

```python
@runtime_checkable
class RealSimRenderer(Protocol):
    def render(
        self,
        scene: Scene,
        viewpoint: Pose,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """Return an HxWx3 uint8 RGB image rendered from `viewpoint`."""
```

Returning `np.ndarray` (not a torch tensor) keeps the contract
backend-agnostic. A torch-backed renderer converts at its own
boundary; a CPU rasteriser (e.g. for tests) does not pay the torch
import cost.

The Protocol is `runtime_checkable` so a `Runner`-style adapter can
isinstance-test a candidate object before calling `render`.

`register_renderer(name, factory)` records a zero-argument factory
returning an object that satisfies the Protocol. `get_renderer(name)`
materialises one. The registry is a single module-level
`dict[str, Callable[[], RealSimRenderer]]` — fits this stub.
Re-registering the same name with the same factory is a no-op
(matches `gymnasium.envs.registration` precedent); re-registering
with a *different* factory raises.

## 7. CLI

```
gauntlet realsim ingest FRAMES_DIR --calib CALIB --out OUT [--source S] [--symlink]
gauntlet realsim info  SCENE_DIR
```

* `FRAMES_DIR` — directory holding the raw frames + a per-frame pose
  manifest. Layout requirement is documented at the schema level
  (§5): the `--calib` JSON enumerates frames by relative path, ids,
  and pose. Discovery does *not* glob the frames dir — the calib
  JSON is the source of truth, so a stray `.DS_Store` does not
  contaminate the scene.
* `--calib` — JSON file shaped like:
  ```json
  {
    "intrinsics": {"wrist": {...}, "scene": {...}},
    "frames": [
      {"path": "0001.png", "timestamp": 0.0,
       "intrinsics_id": "wrist", "pose": [[...], ...]}
    ]
  }
  ```
  Identical key vocabulary to the on-disk manifest, minus
  `version` / `source` (those come from CLI flags / defaults).
* `--out` — output scene directory. Created if missing.
* `--source` — freeform tag; default `None`.
* `--symlink` — symlink frames instead of copying. Default OFF.

`info` dumps to stderr (matching the rest of the CLI surface):

```
scene: <path>
  version: 1
  source: customer-A-log-2026-04-23
  intrinsics: 2 (wrist, scene)
  frames: 312
  time range: 0.000s -> 25.917s
```

## 8. Test plan

* `tests/test_realsim_schema.py` — pydantic round-trip; `Pose`
  validation (4x4 finite, bottom-row tolerance, off-tolerance
  rejection); `CameraIntrinsics` validation (positive width/height,
  positive fx/fy); `Scene.version` round-trips through JSON unchanged.
* `tests/test_realsim_pipeline.py` — `tmp_path` fixture writes a
  synthetic 1x1 PNG (hardcoded magic-byte blob), a calib JSON, and
  exercises:
  - happy-path ingest produces a `Scene` matching the calib;
  - missing frame file → clear error including the frame path;
  - calib refs unknown intrinsics_id → clear error naming the id;
  - non-image bytes (zeroed) → clear error including first 8 bytes;
  - `save_scene` / `load_scene` round-trip preserves the model;
  - `--symlink` mode produces symlinks (resolve target equals input).
* `tests/test_realsim_cli.py` — `typer.testing.CliRunner` drives
  `gauntlet realsim ingest` and `gauntlet realsim info`. Verifies
  manifest is emitted, exit code is 0 on success and non-zero on
  malformed inputs, and `info` stderr summary lists the expected
  counts.
* Renderer registry is exercised with a tiny in-test fake renderer
  to prove `register_renderer` / `get_renderer` work without
  shipping any real renderer in src/.

## 9. Open questions / deferred

* **Multi-camera time alignment.** Real robot dumps interleave wrist
  and scene cameras. The schema accepts this (each `CameraFrame`
  carries its own `timestamp` + `intrinsics_id`), but we do not
  *enforce* sorted-by-timestamp ordering. A future "scene
  validator" CLI could lint that ordering and call out gaps.
* **Depth frames.** `CameraFrame` does not carry a depth path yet.
  Adding `depth_path: str | None = None` later is backwards-compat
  (default ensures existing manifests still parse) and does not bump
  `version`. Deferred until a renderer needs it.
* **Foreground/background masks.** Same shape as the depth question
  — additive field, deferred until a renderer needs it.
* **Plugin entry-point promotion.** When the first real renderer
  lands, decide whether `RealSimRenderer` factories should be
  discoverable via `[project.entry-points."gauntlet.renderers"]`.
  See §4.6.
* **Renderer in-loop integration.** `gauntlet run` does not yet take
  a `--scene` flag. Integrating a registered renderer into the
  evaluation loop (so a perturbation axis can swap rendered scenes)
  is its own RFC; the schema we land here is the input contract that
  RFC will consume.
