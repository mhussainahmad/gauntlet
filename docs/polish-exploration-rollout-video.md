# Polish exploration: rollout MP4 video recording

Status: exploration / pre-implementation
Owner: polish/rollout-video-recording branch

## 1. Why this matters (the domain win)

A learned-policy failure report that lists "8 failures in cell
(lighting=0.3, texture=1.0)" is far less actionable than the same cell
with a clickable thumbnail that plays the failed rollout. Manufacturing-
grade reliability work hinges on humans visually inspecting *where* the
policy went wrong: did the gripper miss; did the cube slip; did the arm
collide.

The harness already has every prerequisite:

* `TabletopEnv(render_in_obs=True)` exposes a uint8 RGB image on
  `obs["image"]` per step (`src/gauntlet/env/tabletop.py:636`).
* `PyBulletTabletopEnv` and `TabletopGenesisEnv` both follow the same
  contract (`src/gauntlet/env/pybullet/tabletop_pybullet.py`,
  `src/gauntlet/env/genesis/tabletop_genesis.py`).
* `Runner` already buffers per-step state in the worker when
  `trajectory_dir` is set (`src/gauntlet/runner/worker.py:303-388`),
  so an additional per-step buffer of `obs["image"]` is a parallel
  side-channel.
* The HTML report
  (`src/gauntlet/report/templates/report.html.jinja`) ships a per-cell
  failure-cluster table that already lists the offending axis-value
  combinations — the perfect anchor for an inline `<video>` element.

Wiring an opt-in `record_video=True` flag through the runner, dumping
per-episode MP4s, and embedding `<video>` thumbnails in the HTML report
turns gauntlet's failure analytics from text-on-a-page into a film
strip of failures.

## 2. Public API decision

`Runner.__init__` gains four optional kwargs. Defaults preserve byte-
identical Phase 1 behaviour:

```python
Runner(
    n_workers=1,
    env_factory=...,
    trajectory_dir=None,
    # NEW:
    record_video=False,
    video_dir=None,            # default: <trajectory_dir>/../videos when both unset, else explicit
    video_fps=30,
    record_only_failures=False,
)
```

`Runner.run` is unchanged in signature — the new kwargs configure the
*how* of execution, mirroring the existing `n_workers` /
`env_factory` / `trajectory_dir` shape.

When `record_video=True`:

* The Runner asserts that the env exposes `obs["image"]` after the
  first reset. We check `"image" in obs` on the public observation
  contract — never `env._render_in_obs`. This works across MuJoCo,
  PyBullet, and Genesis backends.
* Each worker accumulates `obs["image"]` arrays per step into a list.
  Memory bound: `H * W * 3 * max_steps` bytes per in-flight episode
  per worker (e.g. 224x224x3x200 = ~30 MB peak).
* After the rollout completes, the worker writes the frames to
  `<video_dir>/episode_cell<NNNN>_ep<NNNN>_seed<S>.mp4` via the
  `VideoWriter` wrapper.
* `Episode.video_path` is set to the relative path from the run
  output dir (so the HTML `<video src=...>` works without a server).

`record_only_failures=True` only suppresses the *write* step — the
buffer still grows during the rollout because success is unknown
until the final `info["success"]` arrives. The docstring says so.

## 3. Library decision: `imageio[ffmpeg]>=2.34,<3`

Considered:

| option | pros | cons |
|---|---|---|
| `imageio[ffmpeg]` | bundles `imageio-ffmpeg` (static ffmpeg, no system dep), tiny API surface, MIT/BSD, py3.11+ wheels | re-encodes frame-by-frame in pure-Python loop |
| `av` (PyAV) | direct libav bindings, fastest | requires system ffmpeg headers at install on some Linux SKUs, larger wheels, GPL/LGPL licence concerns |
| OpenCV `VideoWriter` | also widely used | huge wheel (50 MB+), pulls a whole vision stack |
| MuJoCo's `render` + manual ffmpeg subprocess | zero new deps | re-implements what imageio already does, fragile across OSes |

`imageio[ffmpeg]` wins on three axes that matter for this project:

1. *Optional extras-free CI must stay green.* The wheel resolves on
   ubuntu-latest with a single `pip install`, no apt step.
2. *Lazy import.* The wrapper imports `imageio.v3 as iio` inside
   `VideoWriter.write()`, so a torch-/extras-free import of
   `gauntlet.runner` never touches imageio.
3. *Re-encode speed is not the bottleneck.* A typical 200-step
   rollout writes ~30 MB of uint8 frames; libx264 from a Python loop
   handles that in well under one second. The actual rollout takes
   minutes on CPU MuJoCo, not the encode.

Pin: `imageio[ffmpeg]>=2.34,<3`. 2.34 is the first release with the
`v3` API stabilised; <3 catches any surprise major bump.

## 4. Memory model

Per-worker, per-in-flight-episode peak buffer:

```
H * W * 3 * max_steps  bytes
   224 * 224 * 3 * 200 ≈  30 MB         (default tabletop)
   224 * 224 * 3 *  20 ≈   3 MB         (test fixtures)
```

n_workers in flight at once: `min(n_workers, len(work_items))`. Every
worker holds at most one buffer at a time (frame buffer is freed
after the MP4 write). Total peak: `n_workers * H * W * 3 *
max_steps`. For a typical 8-worker run on a 200-step suite this is
~240 MB — comfortable on every laptop the harness has been sized for.

Documented in the `VideoWriter` docstring so a user opting in does
not get a surprise OOM on a 4k-step suite.

## 5. Schema-extension decision: `Episode.video_path` direct field

Two options:

A. Add `video_path: str | None = None` directly to `Episode`.
B. Write the path into `Episode.metadata["video_path"]`.

We pick (A). Reasoning:

* `Episode` is the contract surface downstream tools (`Report`,
  `gauntlet replay`, `gauntlet ros2 publish`) key off. A typed,
  optional field is greppable, autocompletes in IDEs, and is stable
  across pydantic dump/load round-trips.
* `metadata` is a `dict[str, float | int | str | bool]`. `str` is
  allowed — but the HTML template would have to know that one
  metadata key is special, which is exactly the leak we want to
  avoid.
* `model_config(extra="forbid")` already governs the schema. A
  default of `None` keeps **load** of older JSONs working — a
  pre-PR Episode dict simply fills in `None`. **Dump** of new
  Episodes always emits the field; downstream pydantic readers on
  an older `gauntlet` version would reject it via `extra="forbid"`.
  This is true of any schema addition and is called out so reviewers
  see it considered. Mitigation: the `[video]` extra is opt-in;
  `record_video=False` (default) keeps `video_path=None` which
  serialises as JSON `null` — semantically inert.

The replacement schema field also lets `gauntlet.report.html`
detect "this report has videos" by a single `any(ep.video_path for
ep in episodes)` check, no metadata-key sniffing.

## 6. Backwards-compatibility strategy

Three protections, in priority order:

1. *Default-off opt-in.* `record_video=False` is the default. Every
   existing code path through `Runner.__init__` and
   `Runner.run` is byte-identical to pre-PR behaviour.
2. *Lazy library import.* `imageio` is imported inside
   `VideoWriter.write()`, never at module scope. A torch-free /
   extras-free install can `import gauntlet.runner.video` without
   pulling imageio.
3. *Regression test.* `tests/test_runner_video_backcompat.py`
   (default-job, unmarked) runs a fixed-seed `Runner(record_video=
   False)` and asserts every `Episode` field — including
   `video_path is None` — matches the expected shape pre-PR. The
   schema-addition forward-compat caveat (older `gauntlet` versions
   reject the new field) is documented in the Episode docstring.

## 7. CI integration

A new `video-tests` job mirrors the existing `monitor-tests` shape:

```yaml
video-tests:
  name: video-tests (imageio[ffmpeg], py${{ matrix.python-version }})
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      python-version: ["3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with: { enable-cache: true, cache-dependency-glob: "uv.lock" }
    - name: Set up Python
      run: uv python install ${{ matrix.python-version }}
    - name: Sync dependencies (with [video] extra + video-dev group)
      run: uv sync --extra video --group video-dev --python ${{ matrix.python-version }}
    - name: Run video-marked tests
      run: uv run pytest -m video -q
```

Default job's marker exclusion list (`.github/workflows/ci.yml:52`)
gains `and not video` so the torch-/extras-free job does not collect
video-marker tests. The unmarked
`tests/test_runner_video_backcompat.py` remains in the default
collection.

## 8. Open questions (answered as part of this PR)

1. *Should the failure-clusters HTML row or the per-cell row carry the
   video?*
   The failure-cluster row, because that is where a human looks first.
   The per-cell `<details>` table also gets a column, gated on the
   row having an associated video.

2. *What happens if the worker crashes mid-rollout?*
   The frame buffer is local to the worker; on crash it is freed
   along with the worker. No partial MP4 is written — we only call
   `VideoWriter.write` after a full rollout completes.

3. *What if `imageio` is not installed but `record_video=True`?*
   `VideoWriter.__init__` (lazy) raises `ImportError` with the exact
   install hint: `pip install "gauntlet[video]"`. Pinned by a
   marker-gated test.

4. *Multi-worker (`n_workers >= 2`) MP4 path conflict?*
   The filename includes `cell_index` and `episode_index` (and seed
   for safety), which are unique across the (cell, episode) lattice.
   Workers cannot collide.

5. *Video file naming convention for HTML embed?*
   `videos/episode_cell{NNNN}_ep{NNNN}_seed{S}.mp4`, stored in
   `video_dir`. `Episode.video_path` is the path relative to
   `video_dir.parent` — typically `videos/episode_cell0001_ep0000_
   seed12345.mp4`. The HTML report embeds `<video src="{video_path}">`
   directly; if the user puts `video_dir` outside the HTML output
   directory the embed will break and we document the constraint.

6. *Why not also write a thumbnail PNG?*
   Out of scope for this PR. `<video preload="metadata">` lets the
   browser render the first frame as a poster automatically; that is
   sufficient for the failure-cluster table.
