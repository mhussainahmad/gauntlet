# Real-to-sim (Phase 3 T18) — input pipeline

This document covers the **input pipeline** half of the real-to-sim
scene-reconstruction story. The renderer half is intentionally
deferred; see "Out of scope" below.

Companion docs: `docs/phase3-rfc-021-real-to-sim-stub.md` (the RFC
itself), `docs/api.md` (the Real-to-sim section of the API
reference).

## Phased scope

The endgame the spec calls out is gaussian-splatting reconstruction of
real customer scenes from robot camera dumps directly into a
renderable eval backend. Shipping the full endgame at once is not
viable — gaussian splatting needs `torch` + CUDA + a multi-gigabyte
training pipeline, all of which violate the project's "small deps, no
torch in the core" rule.

Phase 3 Task 18 therefore lands the **input pipeline** plus the
**renderer extension point**:

| Status   | Surface                                                    |
| -------- | ---------------------------------------------------------- |
| Shipped  | `gauntlet.realsim.Scene` + manifest persistence (RFC 021). |
| Shipped  | `gauntlet.realsim.RealSceneInput` (raw capture-dir parse). |
| Shipped  | `gauntlet.realsim.scene_to_camera_extrinsics` (B-42 axis bridge). |
| Shipped  | `gauntlet.realsim.RealSimRenderer` Protocol + registry.    |
| Deferred | Gaussian-splatting / NeRF / mesh renderer.                 |

The renderer is tracked as a follow-up RFC. The seam is the named
exception `gauntlet.realsim.RendererNotImplementedError` — any future
helper that pretends to render must raise it until the renderer
lands. A meta-test (`tests/test_realsim_scene_to_axis.py::test_no_accidental_renderer_landed`)
walks `gauntlet.realsim.__all__` on every CI run and fails if a
public symbol accidentally grows a working `render` method.

## Two layers of input

The package now carries two complementary input surfaces:

1. **`Scene` (curated, manifest-backed).** `gauntlet.realsim.Scene` is
   the validated, canonical artefact a renderer will consume. Built
   by `ingest_frames(frames_dir, calib_json)`, persisted via
   `save_scene` / `load_scene`. Per-camera-id intrinsics, POSIX-relative
   frame paths, round-trips through `manifest.json`. See
   `docs/api.md#real-to-sim-scene-reconstruction` for the full
   reference.

2. **`RealSceneInput` (raw capture-dir parse).** A frozen dataclass
   that holds the *direct* parse of a COLMAP / Polycam / NeRFStudio
   export — flat `dict[str, float]` intrinsics, `list[np.ndarray]`
   extrinsics, optional `depth_maps` / `point_cloud` sidecars,
   freeform `metadata`. Built by
   `gauntlet.realsim.load_real_scene(capture_dir)`. Soft validation
   findings (e.g. depth-map files missing on disk) are surfaced via
   `validate_scene_input(scene) -> list[str]` rather than raised so
   the caller can decide whether to proceed.

The two layers exist because the curated `Scene` is what a renderer
*wants*, while the raw `RealSceneInput` is what a capture pipeline
*emits*. A future renderer can either consume the raw input directly
(notebook / one-off rendering) or go through `Scene` (suite-driven
fleet eval). The bridge between the layers is
`scene_to_camera_extrinsics` — see "Bridge to the camera_extrinsics
axis" below.

## Capture-dir layout

`load_real_scene(capture_dir)` expects a structured export:

```
<capture_dir>/
    intrinsics.json      # required
    extrinsics.json      # required: list[list[list[float]]] (N x 4 x 4)
    manifest.json        # optional: declares depth_maps / point_cloud / metadata
    depth_maps/          # optional: per-frame depth-map sidecars
        0.exr
        1.exr
        ...
    points.ply           # optional: aggregate point cloud
```

`intrinsics.json` carries the canonical pinhole keys
(`fx`, `fy`, `cx`, `cy`, `width`, `height`); distortion coefficients
(`k1`, `k2`, `p1`, `p2`, `k3`) are optional but a partial set raises
a soft warning from `validate_scene_input`.

`extrinsics.json` is a JSON-encoded list of 4x4 row-major matrices
(camera-to-world). The NeRFStudio `transforms.json` wrapper shape
(`{"frames": [...]}`) is also accepted — the parser unwraps the outer
dict if it has a `"frames"` key.

`manifest.json` (if present) declares optional sidecars and freeform
metadata:

```json
{
  "depth_maps": ["depth_maps/0.exr", "depth_maps/1.exr"],
  "point_cloud": "points.ply",
  "metadata": {
    "capture_device": "polycam-iPad-Pro",
    "software": "polycam-1.2",
    "captured_at": "2026-04-01T12:00:00Z"
  }
}
```

Every path-typed field — `point_cloud`, each entry of `depth_maps` —
flows through `gauntlet.security.safe_join` before being resolved.
A traversal value (`"../etc/passwd"`) or an absolute path raises
`RealSceneInputError`.

## Converting a COLMAP / Polycam export

A pre-existing COLMAP / Polycam export typically already carries a
`transforms.json` (NeRFStudio convention). Convert to the gauntlet
layout by:

1. Extracting the per-frame `transform_matrix` 4x4 lists into an
   `extrinsics.json` (either a flat `list[...]` or a
   `{"frames": [...]}` wrapper — both are accepted).
2. Pulling the camera intrinsics into a flat `intrinsics.json`
   (`fx`/`fy`/`cx`/`cy`/`width`/`height`).
3. (Optional) Writing a `manifest.json` pointing at depth-map and
   point-cloud sidecars when the export ships them.

The reverse direction — turning a `RealSceneInput` into a curated
`Scene` for `save_scene` — is left to the caller because it requires
the user-supplied frame files (rgb PNGs / JPEGs) which the raw
parser does *not* claim.

## Bridge to the `camera_extrinsics` axis (B-42)

`gauntlet.realsim.scene_to_camera_extrinsics(scene, n_samples=16)`
converts a `RealSceneInput` into a list of structured 6-D pose
deltas matching the on-disk schema for the
`camera_extrinsics` axis. The output dict shape is exactly
`gauntlet.suite.schema.ExtrinsicsValue`:

```python
[
    {"translation": [dx, dy, dz], "rotation": [drx, dry, drz]},
    ...
]
```

`translation` is the homogeneous translation column of each 4x4 pose
in metres; `rotation` is the XYZ-Euler decomposition of the rotation
sub-block in radians (MuJoCo / PyBullet camera convention).

When `len(scene.extrinsics_per_frame) > n_samples`, the bridge
sub-samples to `n_samples` evenly-spaced indices (always including the
first and last frame). When the capture has fewer frames than
`n_samples`, the bridge returns one entry per frame without
duplication or interpolation.

The result is a value list that a suite YAML's `extrinsics_values`
block can ingest verbatim — see B-42's
`extrinsics_values` shape in `docs/api.md` and the unit
test `test_scene_to_camera_extrinsics_output_matches_extrinsics_value_schema`.

## Out of scope (future work)

The following are deliberately deferred:

* **Renderer.** No `torch`, no CUDA, no training loop, no dataset
  download. The seam is `RendererNotImplementedError`. A follow-up
  RFC will land the first concrete renderer.
* **Camera-calibration helper.** Users supply intrinsics + extrinsics
  via JSON; the gauntlet pipeline does not run COLMAP / OpenCV /
  Polycam internally.
* **Image preprocessing.** `load_real_scene` does not resize, crop,
  undistort, or re-encode frames. The pipeline is a manifest builder
  on top of files the user already has.

These are tracked alongside the renderer in the phase-3 backlog and
will land as separate PRs.
