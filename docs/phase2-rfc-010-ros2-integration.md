# Phase 2 RFC 010 — ROS 2 integration (publisher + recorder)

- **Status**: Draft
- **Phase**: 2, Task 10 (`GAUNTLET_SPEC.md` §7: "ROS 2 integration for running on real robots with logging.")
- **Author**: integration agent
- **Date**: 2026-04-24
- **Supersedes**: n/a
- **References**:
  - `docs/phase2-rfc-003-drift-detector.md` — closest precedent for a non-simulator subpackage with an extras-gated heavy dep + lazy import guard + torch-free public schema.
  - `docs/phase2-rfc-004-trajectory-replay.md` — second precedent for an integration / cross-cutting subpackage that reuses Runner outputs without modifying the Runner.
  - `docs/phase2-rfc-007-genesis-adapter.md` — extras / dev-group / pytest-marker / CI-job pattern, including a heavy dep that is awkward to install via PyPI alone.
  - `docs/phase2-exploration-task10-ros2-integration.md` — the measurement / install-landscape pass that constrains the scope decisions below.

---

## 1. Summary

`GAUNTLET_SPEC.md` §7 names "ROS 2 integration for running on real robots with logging" as a Phase 2 candidate. This RFC adds `gauntlet.ros2`, a torch-free subpackage that publishes `Episode` outcomes to a configurable ROS 2 topic and records subscribed messages from a real robot's topic to disk during a rollout. The two pieces are independent and each is wholly opt-in:

- `Ros2EpisodePublisher` — given a sequence of `Episode` objects (from an existing `gauntlet run`'s `episodes.json`, or live from a `Runner.run(...)` call), serialise each via JSON-in-`std_msgs.msg.String` and publish to a configurable topic. Use case: fleet-wide failure-mode aggregation across many real robots running gauntlet evaluations on a shared topic.
- `Ros2RolloutRecorder` — context manager that, while open, subscribes to a configurable observation topic on a real robot and dumps each received message to a JSON-lines file on disk. Use case: the "real robots with logging" half of §7 — capture a real robot's joint trajectory during an evaluation, replay it offline, or feed it into a future `Ros2HardwareEnv`.

The hard environmental constraint is that `rclpy` is not pip-installable (exploration §Q2). The `[ros2]` extra is therefore empty — `uv sync --extra ros2` is a no-op beyond pulling the dev tooling, and users separately `apt install ros-<distro>-rclpy` (or `docker run osrf/ros:<distro>-desktop`) and source the relevant `setup.bash` before `gauntlet ros2 publish` works. The lazy-import guard at `gauntlet.ros2.publisher` and `.recorder` raises a clean `ImportError` pointing at the apt/Docker install path when `rclpy` is absent — same shape RFC-003 used for torch.

The `Env` Protocol, the `Runner`, the `Episode` schema, and the `Report` schema are all unchanged. CLI surface grows by one subcommand group: `gauntlet ros2 publish` and `gauntlet ros2 record`. The default torch-free CI job stays green because (a) `gauntlet.ros2.schema` is pure pydantic and importable everywhere, (b) `gauntlet.ros2.publisher` / `.recorder` are accessed only via lazy `__getattr__` from `gauntlet.ros2.__init__`, and (c) the unit tests live behind a new `ros2` pytest marker and seed `sys.modules["rclpy"] = MagicMock()` in `tests/ros2/conftest.py` so collection never hits a real `import rclpy`.

## 2. Goals / non-goals

### Goals

- Ship `gauntlet.ros2` — torch-free, runtime-agnostic, gated behind a `[ros2]` extra purely for documentation / dev-group composition. Two halves:
  - **`Ros2EpisodePublisher`** — Episode-sequence publisher to a configurable topic, message contract = JSON in `std_msgs.msg.String` (§5).
  - **`Ros2RolloutRecorder`** — context-manager subscriber that dumps received messages to a JSON-lines file (§6).
- Keep `gauntlet.core` ROS-2-free. Importing `gauntlet`, `gauntlet.policy`, `gauntlet.env`, `gauntlet.runner`, `gauntlet.report`, `gauntlet.replay`, `gauntlet.monitor.schema` must not transitively import `rclpy`. Importing `gauntlet.ros2.schema` must also be `rclpy`-free (so the schema works in the default torch-free CI job).
- Keep the public `Policy` / `Env` / `Episode` / `Report` / `Suite` surfaces unchanged.
- Keep `mypy --strict` passing regardless of whether the `[ros2]` extra is installed (needs `[[tool.mypy.overrides]]` for `rclpy.*`, `std_msgs.*`, `sensor_msgs.*`, `geometry_msgs.*` — same pattern RFC-005 used for `pybullet*`).
- `uv sync --extra ros2 --group ros2-dev` enables the dev tooling needed to test the package; composes cleanly with every other extra.
- Reference workflow (publisher half — fleet aggregation):
  1. Many real robots run `gauntlet run` and produce `episodes.json` files.
  2. Each robot runs `gauntlet ros2 publish episodes.json --topic /gauntlet/episodes` to broadcast results onto a shared topic.
  3. A separate ROS 2 subscriber (Foxglove dashboard, log-aggregator, etc.) collates the stream.
- Reference workflow (recorder half — real-robot logging):
  1. A real robot is performing some task; its joint state is published on `/robot/joint_states`.
  2. An evaluator runs `gauntlet ros2 record --topic /robot/joint_states --out trajectories.jsonl --duration 30` to dump 30 seconds of messages to disk.
  3. The JSON-lines file is fed into a downstream analysis pipeline (or, in a future RFC, into `gauntlet replay` to re-simulate the captured trajectory in a sim env).

### Non-goals

- **`Ros2HardwareEnv(GauntletEnv)` over a real robot.** Building this honestly would require a real arm connected for any meaningful test — a mocked-rclpy unit test cannot confirm a real UR5e or Franka actually moved. Defer to a future task that pairs the env with a hardware-in-the-loop CI workflow_dispatch job.
- **Custom `.msg` packages (`gauntlet_ros_msgs`).** Custom messages need a colcon workspace + `package.xml` + `CMakeLists.txt`, which is out of scope for a Python-only PR. JSON-in-`std_msgs.msg.String` is the v1 contract (§5); a future RFC may introduce an ament package as an upgrade path.
- **`rclpy` declared in `[project.optional-dependencies] ros2`.** Not pip-installable in the official channel (exploration §Q2). Users install via apt or Docker.
- **ROS 1 (`rospy`).** End-of-life as of May 2025. Not supported.
- **Live HTML / Foxglove dashboard.** Out of scope; `Ros2EpisodePublisher` writes the topic, downstream tooling consumes it.
- **QoS configuration knobs.** v1 uses default reliability + history (depth=10); a future RFC may add `--qos` flags.
- **rosbag2 output.** v1 dumps JSON-lines for parity with the rest of the codebase. A future RFC may add a `--format rosbag2` option.
- **Network-bridging non-ROS hosts (zenoh, foxglove-bridge).** Not this RFC.
- **Real ROS 2 integration tests in CI.** All v1 tests mock `rclpy` at the module boundary. A future workflow_dispatch job (`docker run osrf/ros:humble-desktop`) is a clean follow-up.

## 3. Dependency placement decision

### Choice: **new `[ros2]` extra — empty (no PyPI deps).**

### Options considered

- **(A) `ros2 = []` — empty extra.** `uv sync --extra ros2` is a no-op beyond triggering group resolution. Documents the capability; users `apt install ros-<distro>-rclpy` separately. **Chosen.**
- **(B) `ros2 = ["rclpy-pip>=...]"`.** `rclpy-pip` is an unofficial PyPI shim, version-pinned to one ROS 2 distro at a time, not maintained by Open Robotics. Rejected — fragile, distro-pinned, and breaks when users have a real ROS 2 install on PATH.
- **(C) Fold into `[hf]` or `[monitor]`.** Wrong axis — `gauntlet.ros2` is about message-bus integration, not a torch-backed analyser or a VLA. Same separation argument RFC-002 / RFC-003 used for their independent extras.
- **(D) `ros2 = ["pyyaml>=6.0,<7"]` for an alt YAML-based message schema.** Rejected — JSON-in-`std_msgs.msg.String` (§5) is sufficient and uses only stdlib.

### Why (A)

Three facts drove this:

1. **`rclpy` is not on PyPI.** Exploration §Q2 measured this against PyPI, the index, and the official ROS 2 install docs. The only valid install paths are `apt install ros-<distro>-rclpy` or `docker run osrf/ros:<distro>-desktop`. Declaring `rclpy` in the extra would make `uv sync --extra ros2` fail to resolve on every machine.
2. **An empty extra still has documentation value.** `uv sync --extra ros2` succeeding (as a no-op) signals "this is a known-supported optional capability", and the dev-group composition (`--group ros2-dev` pulls `pytest-mock`) gives the test workflow somewhere to live. Same shape genesis took with its torch-explicit add-on (RFC-007 §4.1) — except here the missing dep is system-level, not pip-level.
3. **The lazy-import guard surfaces the install hint.** `gauntlet.ros2.publisher` does the `try: import rclpy except ImportError: raise ImportError(_INSTALL_HINT)` at module scope; the hint names both `apt install ros-humble-rclpy` and the `osrf/ros:humble-desktop` Docker image. Same UX as the `[monitor]` install hint — error fires on first symbol use, not on `import gauntlet`.

### `pyproject.toml` diff (fragments only — applied in commit 3)

```toml
# NEW — empty extra. rclpy is not pip-installable; users install via
# `apt install ros-<distro>-rclpy` or `docker run osrf/ros:<distro>-desktop`.
# The extra exists so `uv sync --extra ros2` is a recognised invocation
# and so the [ros2-dev] dev-group has a partner extra; the install-hint
# ImportError in `gauntlet.ros2.publisher` / `.recorder` points users at
# the apt / Docker install path on first use.
ros2 = []

# NEW — dev group analogue of monitor-dev / pybullet-dev / genesis-dev.
[dependency-groups]
ros2-dev = [
    {include-group = "dev"},
    "pytest-mock>=3.12,<4",
]

# NEW mypy overrides — let mypy import-check `src/gauntlet/ros2/*.py`
# even when rclpy/std_msgs/sensor_msgs/geometry_msgs are not installed
# (default CI job stays ros-free).
[[tool.mypy.overrides]]
module = [
    "rclpy", "rclpy.*",
    "std_msgs", "std_msgs.*",
    "sensor_msgs", "sensor_msgs.*",
    "geometry_msgs", "geometry_msgs.*",
]
ignore_missing_imports = true

# NEW pytest marker.
[tool.pytest.ini_options]
markers = [
    # ... existing markers ...
    "ros2: tests that require the [ros2] extra (rclpy mocked at the module boundary)",
]
```

### CI structure

One new job, mirroring `monitor-tests`:

- **Default job (unchanged shape, exclusion list grows)**: `pytest -m "not hf and not lerobot and not monitor and not pybullet and not genesis and not ros2"`. Stays the continuous enforcement of "no rclpy in the core".
- **NEW `ros2-tests`**: `uv sync --extra ros2 --group ros2-dev` → `pytest -m ros2 -q`. Standard `ubuntu-latest`. Notably this job does **NOT** install ROS 2 — the `tests/ros2/conftest.py` fixture seeds `sys.modules["rclpy"] = MagicMock()` so the unit tests run against the mock. A real-rclpy integration job is left to a follow-up workflow_dispatch.

All five jobs (lint-typecheck-test, hf, lerobot, monitor, pybullet, genesis, ros2 — seven once isaac lands on a sibling branch) block merges.

## 4. Public surface

```
src/gauntlet/ros2/
├── __init__.py            # Re-exports schema eagerly; routes Ros2EpisodePublisher
│                          # / Ros2RolloutRecorder through __getattr__ with the
│                          # install-hint ImportError. Mirrors the monitor pattern.
├── schema.py              # ROS-2-FREE. Pydantic Ros2EpisodePayload — the JSON
│                          # message contract (§5).
├── publisher.py           # ROS-2 only. Module-scope try: import rclpy; raises
│                          # ImportError with the install hint when missing.
│                          # Defines Ros2EpisodePublisher.
└── recorder.py            # ROS-2 only. Same module-scope import guard. Defines
                           # Ros2RolloutRecorder context manager.
```

Re-exported symbols on `gauntlet.ros2`:

```python
from gauntlet.ros2 import (
    Ros2EpisodePayload,        # eager — torch/rclpy-free, default-job safe
    Ros2EpisodePublisher,      # lazy via __getattr__ — raises ImportError without rclpy
    Ros2RolloutRecorder,       # lazy via __getattr__ — same
)
```

CLI surface (in `src/gauntlet/cli.py`):

```
gauntlet ros2 publish EPISODES_JSON --topic TOPIC [--node-name NAME] [--dry-run]
gauntlet ros2 record --topic TOPIC --out OUT.jsonl [--duration SECONDS] [--node-name NAME]
                                                  [--message-type std_msgs/msg/String]
```

## 5. Message schema (`Ros2EpisodePayload`)

JSON serialised inside `std_msgs.msg.String` — the only message type that ships in every ROS 2 distro by default. Pydantic schema, lives in `src/gauntlet/ros2/schema.py`, torch-free.

```python
# src/gauntlet/ros2/schema.py (excerpt — full source in commit 4)

class Ros2EpisodePayload(BaseModel):
    """JSON payload for /gauntlet/episodes — one Episode summary per message."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["v1"]                    # bumps drive future-RFC migrations
    suite_name: str
    cell_index: int
    episode_index: int
    seed: int
    perturbation_config: dict[str, float]

    success: bool
    terminated: bool
    truncated: bool
    step_count: int
    total_reward: float

    # Echoed from Episode.metadata. Empty {} when the producing Episode
    # had no metadata. Kept as a JSON-safe primitive bag so downstream
    # consumers don't have to negotiate types.
    metadata: dict[str, float | int | str | bool]
```

Construction from an `Episode`:

```python
Ros2EpisodePayload.from_episode(ep: Episode) -> Ros2EpisodePayload
```

Implementation: a pure mapping. No new contract; just a JSON-safe re-shape of the existing Episode fields. `schema_version="v1"` is hard-coded; bumps require a follow-up RFC.

### Why JSON, not custom .msg

Three reasons:

1. **Custom `.msg` packages need colcon.** A `gauntlet_ros_msgs` ament package would force every consumer to clone gauntlet's repo and build the package via `colcon build` — a non-trivial install ask for a Python-only library. JSON in `std_msgs/msg/String` works against any ROS 2 distro out of the box.
2. **Pydantic round-trip is the existing serialisation contract.** `Episode`, `Report`, `DriftReport` all use `model_dump_json()` / `model_validate_json()`. The same path applies here.
3. **JSON-in-String is a stable upgrade path.** If a future RFC ships `gauntlet_ros_msgs`, the JSON path remains supported as a fallback for users who can't / won't build a colcon workspace. The pydantic schema we land here becomes that future package's authoritative source.

Size: a typical Episode payload (canonical 7 axes + outcome fields + master_seed metadata) serialises to ~600 bytes JSON — well under ROS 2's default 1 MB message size limit.

## 6. `Ros2RolloutRecorder` — context-manager subscriber

```python
class Ros2RolloutRecorder:
    """Context-manager subscriber that dumps received messages to a JSONL file.

    Usage:

        with Ros2RolloutRecorder(
            topic="/robot/joint_states",
            out_path=Path("trajectories.jsonl"),
            duration_s=30.0,
        ) as recorder:
            # block until duration elapses or context exits
            recorder.spin_until_done()

    On context exit the subscription is destroyed and the JSONL file is
    flushed. Each line is a JSON object: {"timestamp": <float>, "topic":
    <str>, "data": <str>}, where ``data`` is the message's stringified
    payload (``str(msg)`` for arbitrary message types — sufficient for v1).
    """
```

### Spinning policy

`rclpy.spin_once(node, timeout_sec=0.1)` inside a polling loop in `spin_until_done`. Easy to test (the mock returns immediately), no background threads.

### Output format: JSON-lines

One JSON object per line: `{"timestamp": float, "topic": str, "data": str}`. Rationale: matches the rest of the codebase's JSON conventions, no rosbag2 dep, trivially parseable by every tool. A future RFC may add `--format rosbag2` for users who need the binary format.

### Type-erasure for the subscribed message type

v1 supports any message type via `str(msg)` of the received object. The CLI's `--message-type` flag dynamically imports the message class via `rclpy.get_message_class` at runtime; the recorder stores `str(msg)` in the JSON line. Lossy but generic. A future RFC may add a typed `--message-type sensor_msgs/msg/JointState` mode that introspects fields and emits structured JSON.

## 7. `Ros2EpisodePublisher` — Episode-sequence publisher

```python
class Ros2EpisodePublisher:
    """Publish Episode summaries to a ROS 2 topic as JSON in std_msgs/msg/String.

    Usage:

        publisher = Ros2EpisodePublisher(topic="/gauntlet/episodes")
        try:
            for ep in episodes:
                publisher.publish_episode(ep)
        finally:
            publisher.close()

    The publisher initialises ``rclpy``, creates a node, and creates a
    publisher of ``std_msgs/msg/String`` on the requested topic. Each
    ``publish_episode`` call serialises the Episode through
    :meth:`Ros2EpisodePayload.from_episode().model_dump_json()` and
    publishes the result.
    """
```

### Init / shutdown

`rclpy.init()` is called at constructor entry if `rclpy.ok()` is False; `rclpy.shutdown()` is called at `close()` (and via `__del__` as a safety net). The constructor takes a `node_name` kwarg (default `"gauntlet_episode_publisher"`) for ROS 2 graph hygiene.

### Composition over inheritance

`Ros2EpisodePublisher` does **NOT** subclass `rclpy.node.Node`. It composes — holds `self._node: Any` internally, created via `rclpy.create_node(node_name)`. Reasoning: with `rclpy.*` in `ignore_missing_imports`, `Node` types as `Any`; subclassing `Any` trips mypy `--strict`'s `disallow_subclassing_any` unless we add a per-module override (which we'd rather avoid for the ros2 modules — keeps the strict rule applying everywhere).

## 8. CLI design

### New subcommand group: `gauntlet ros2 publish` and `gauntlet ros2 record`

Mirror the `gauntlet monitor train` / `gauntlet monitor score` pattern (RFC-003 §9). The subcommand group is registered as `app.add_typer(ros2_app, name="ros2")` in `src/gauntlet/cli.py`.

```
gauntlet ros2 publish EPISODES_JSON
    --topic TEXT                        # default: /gauntlet/episodes
    [--node-name TEXT]                  # default: gauntlet_episode_publisher
    [--dry-run]                         # don't actually publish; print payloads to stderr

gauntlet ros2 record
    --topic TEXT                        # required
    --out PATH                          # required, JSONL output
    [--duration FLOAT]                  # default: 30.0 seconds; 0 = run forever
    [--node-name TEXT]                  # default: gauntlet_rollout_recorder
    [--message-type TEXT]               # default: std_msgs/msg/String
```

Both commands surface the same `_ROS2_INSTALL_HINT` ImportError as a clean CLI error when `rclpy` is missing. `--dry-run` on `publish` short-circuits the rclpy import entirely — useful for users who want to preview the JSON payloads without installing ROS 2.

## 9. Test plan

All tests live under `tests/ros2/`. The schema tests are NOT marked `@pytest.mark.ros2` (torch-free, runs in default job); the publisher / recorder / CLI / import-guard tests ARE marked.

### `tests/ros2/conftest.py` (the load-bearing piece)

Seeds `sys.modules` with `MagicMock()` instances for `rclpy`, `rclpy.node`, `std_msgs`, `std_msgs.msg`, `sensor_msgs`, `sensor_msgs.msg`, `geometry_msgs`, `geometry_msgs.msg` at module import time. This lets the `tests/ros2/test_publisher.py` and `test_recorder.py` modules use ordinary top-level `from gauntlet.ros2.publisher import Ros2EpisodePublisher` imports — collection itself does not blow up, because the module-scope `try: import rclpy` resolves against the mock.

Without this conftest, `pytest -m ros2` would fail at collection time on every machine that doesn't have `rclpy` installed (i.e. every CI machine, and most developer laptops).

### Unit tests (default torch-free job — NO `ros2` marker)

1. **`test_schema_round_trip`** — Construct `Ros2EpisodePayload` from a hand-built Episode, dump to JSON, validate-back, assert round-trip equality. Covers `from_episode` mapping.
2. **`test_schema_extra_forbid`** — Adding an unknown field to the JSON raises `ValidationError`. Pins the `ConfigDict(extra="forbid")` contract.
3. **`test_schema_metadata_round_trips_primitive_mix`** — Episode with `metadata={"master_seed": 42, "label": "smoke", "ok": True}` round-trips to/from the payload without coercion.

### `ros2`-marked tests (run in `ros2-tests` job — `@pytest.mark.ros2`)

4. **`test_import_guard_raises_install_hint_when_rclpy_missing`** — `monkeypatch.setitem(sys.modules, "rclpy", None)` + `importlib.reload(publisher_module)` → expect `ImportError` mentioning `apt install ros-humble-rclpy` and the Docker image. Mirrors the `torch_absent` fixture from `tests/test_import_guards.py`.
5. **`test_publisher_publishes_jsonized_payload`** — Construct `Ros2EpisodePublisher`; assert `rclpy.init` was called; assert `node.create_publisher` was called with `std_msgs/msg/String` and the topic name; call `publisher.publish_episode(ep)`; assert the published message's `.data` field deserialises to the expected `Ros2EpisodePayload`.
6. **`test_publisher_close_calls_rclpy_shutdown`** — `publisher.close()` → assert `rclpy.shutdown` called once; calling `close` twice is idempotent (does NOT call shutdown again because `rclpy.ok()` is False after the first call).
7. **`test_publisher_skips_init_when_rclpy_already_initialised`** — When `rclpy.ok()` is True at construction, `rclpy.init` is NOT called (avoids "rclpy already initialised" errors when nested under another node).
8. **`test_recorder_subscribes_and_writes_jsonl`** — Mock `node.create_subscription` to capture the callback; manually invoke the callback with a fake message; on context exit the JSONL file contains one line per invocation.
9. **`test_recorder_close_destroys_subscription_and_shuts_down_rclpy`** — Verify subscription teardown + `rclpy.shutdown` called.
10. **`test_recorder_duration_terminates_spin`** — `Ros2RolloutRecorder(..., duration_s=0.05).spin_until_done()` returns after ~0.05 s without hanging.
11. **`test_cli_ros2_publish_dry_run`** — `CliRunner().invoke(app, ["ros2", "publish", "ep.json", "--topic", "/g/e", "--dry-run"])` exits 0; the rclpy mock's `init` is NOT called (dry-run short-circuits before import); stderr contains the rendered payloads.
12. **`test_cli_ros2_publish_calls_publisher`** — `CliRunner().invoke(app, ["ros2", "publish", "ep.json", "--topic", "/g/e"])` exits 0; the publisher's `publish_episode` is called once per episode in the file.
13. **`test_cli_ros2_record_writes_output`** — `CliRunner().invoke(app, ["ros2", "record", "--topic", "/foo", "--out", "out.jsonl", "--duration", "0.05"])` exits 0; `out.jsonl` exists.
14. **`test_cli_ros2_publish_invalid_topic_errors`** — empty `--topic ""` exits non-zero with the appropriate error.

### Explicitly out of scope for this task

- Tests against a real ROS 2 install. Belongs to a follow-up workflow_dispatch job that runs inside `osrf/ros:humble-desktop`.
- Tests against a real robot. Hardware-in-the-loop.
- Performance benchmarks. Pub/sub of 600-byte messages is bound by `rclpy` itself; no perf budget to defend in this RFC.

## 10. CI footprint

New job `ros2-tests` in `.github/workflows/ci.yml`, byte-pattern-copied from `monitor-tests`:

```yaml
ros2-tests:
  name: ros2-tests (rclpy mocked, py${{ matrix.python-version }})
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      python-version: ["3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with:
        enable-cache: true
        cache-dependency-glob: "uv.lock"
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    - name: Sync dependencies (with [ros2] extra + ros2-dev group)
      run: >-
        uv sync
        --extra ros2
        --group ros2-dev
        --python ${{ matrix.python-version }}
    - name: Run ros2-marked tests (rclpy mocked at module boundary)
      run: uv run pytest -m ros2 -q
```

Default `lint-typecheck-test` job's `pytest` `-m` string extends to exclude `ros2`. No other job changes.

Notably the job does NOT install ROS 2 — `tests/ros2/conftest.py` seeds `sys.modules["rclpy"] = MagicMock()` so the unit tests run entirely against mocks. A real-rclpy integration job (in a `docker run osrf/ros:humble-desktop` container) is a clean follow-up that doesn't need to land in this PR.

## 11. Open questions

Each has a default in parentheses so the implementation agent is not blocked.

- **Default topic name for the publisher.** (`/gauntlet/episodes`. Documented in `--topic`'s help text. Users override per-fleet.)
- **Recorder output format.** JSON-lines vs rosbag2 vs NPZ. (**Default: JSON-lines**. Matches the rest of the codebase's serialisation conventions; no rosbag2 dep; trivially parseable. A future RFC may add `--format rosbag2`.)
- **QoS defaults.** Reliable, depth=10. (Default. A future RFC may add `--qos reliable|best-effort` flags for low-bandwidth deployments.)
- **Distro support claim.** (Document Humble + Jazzy as supported; other distros best-effort. The unit tests are distro-agnostic since rclpy is mocked.)
- **`Ros2EpisodePublisher` accepts a `Runner` callback hook.** Should we expose a "stream-as-you-rollout" mode where `Runner.run` calls the publisher directly per Episode? (**Default: no**. The Runner stays publisher-unaware. Streaming would couple Runner determinism to ROS 2's network state, which is the wrong direction. A future user wanting this can wrap `Runner.run`'s output in a for-loop and call `publish_episode` themselves; that pattern is the reference example shipped in `examples/publish_episodes_to_ros2.py`.)
- **Multiple publishers in one process.** (Supported — `rclpy.init` is guarded by `rclpy.ok()`. Documented in the publisher's docstring.)
- **Recorder catches `KeyboardInterrupt` mid-spin.** (Yes — `try/except KeyboardInterrupt` inside `spin_until_done` flushes the JSONL file and re-raises. Tested.)
- **Schema-version stamping on `Ros2EpisodePayload`.** (`schema_version: Literal["v1"]` — present from day one so future-RFC schema changes don't silently break old consumers. Same shape `Episode.metadata["master_seed"]` evolution showed: paying this cost up-front is cheaper than retrofitting.)

## 12. Future work

- **Custom `.msg` package (`gauntlet_ros_msgs`).** Ament package with proper message definitions; downstream consumers can subscribe with native typing instead of JSON parsing. JSON-in-String stays as a fallback.
- **`Ros2HardwareEnv(GauntletEnv)`.** Wraps a real robot under the env Protocol. Requires a hardware-in-the-loop CI workflow_dispatch job; the message contracts settled in this RFC are reusable.
- **Real-rclpy integration test job.** `docker run osrf/ros:humble-desktop`, source `/opt/ros/humble/setup.bash`, install gauntlet, run `pytest -m ros2_integration` (a new marker for tests that need a real rclpy). Optional, low-frequency.
- **Foxglove dashboard schema.** Author a `.foxglove-layout` file that visualises the published `Ros2EpisodePayload` stream. Pure config, no code.
- **QoS configuration knobs.** `--qos reliable|best-effort`, `--depth N`. Trivial pass-through to `rclpy.qos.QoSProfile`.
- **rosbag2 output for the recorder.** `--format rosbag2`. Wraps `rosbag2_py.SequentialWriter`.

## 13. Implementation checklist (target ~10 atomic commits)

Each row → one commit named `Phase 2 Task 10 step N: <subject>`. Each leaves `ruff check`, `ruff format --check`, `mypy --strict`, the narrow `pytest -m "not hf and not lerobot and not monitor and not pybullet and not genesis and not ros2" -q` pass, and (where applicable) `pytest tests/ros2/ -q` pass.

1. Exploration doc (`docs/phase2-exploration-task10-ros2-integration.md`). **Landed.**
2. RFC (this document).
3. `pyproject.toml`: add `ros2 = []` extra, `ros2-dev` dev group, `ros2` pytest marker, `[[tool.mypy.overrides]]` for `rclpy.*` / `std_msgs.*` / `sensor_msgs.*` / `geometry_msgs.*`, and `[tool.ruff.lint.per-file-ignores]` E402 for `src/gauntlet/ros2/publisher.py` + `recorder.py`.
4. `src/gauntlet/ros2/schema.py` (rclpy-free) + `src/gauntlet/ros2/__init__.py` with eager schema re-exports + lazy `__getattr__` for the publisher / recorder. Tests in `tests/ros2/test_schema.py` — no `ros2` marker, runs in default job.
5. `src/gauntlet/ros2/publisher.py` skeleton + module-scope `try: import rclpy` raising the install hint. `tests/ros2/conftest.py` seeding the rclpy MagicMock. `tests/ros2/test_import_guards.py` — `ros2`-marked, asserts the install hint when `sys.modules["rclpy"] = None`.
6. `Ros2EpisodePublisher` body. Tests `tests/ros2/test_publisher.py` mock `rclpy.init`, `rclpy.create_node`, `Node.create_publisher`, and assert the published JSON shape. All `ros2` marker.
7. `src/gauntlet/ros2/recorder.py` body — `Ros2RolloutRecorder` context manager. Tests `tests/ros2/test_recorder.py` mock `node.create_subscription`, manually invoke the captured callback, assert JSONL output. All `ros2` marker.
8. CLI subcommand `gauntlet ros2 publish` + `gauntlet ros2 record` wired into `src/gauntlet/cli.py`. Tests `tests/ros2/test_cli_ros2.py` — `typer.testing.CliRunner` + mock patching. All `ros2` marker.
9. CI job `ros2-tests` appended to `.github/workflows/ci.yml`. Default `lint-typecheck-test` job's `-m` exclusion grows to include `not ros2`.
10. README "ROS 2 integration" section + `examples/publish_episodes_to_ros2.py` (mocks rclpy if absent so it imports cleanly in the default torch-free job).

Each commit atomic. One PR at the end (branch: `phase-2/ros2-integration`; base: `main`).

---

## Appendix A — External facts anchoring this RFC (as of April 2026)

- `rclpy` is **not** distributed via PyPI in its official form. Install paths: `apt install ros-humble-rclpy` (Ubuntu 22.04 / Python 3.10), `apt install ros-jazzy-rclpy` (Ubuntu 24.04 / Python 3.12), or `docker run osrf/ros:<distro>-desktop`. Verified against ROS 2 official docs and PyPI's index.
- `std_msgs/msg/String` is part of every ROS 2 distro by default (ships in `ros-<distro>-std-msgs` which is a hard dep of `ros-<distro>-rclpy`). Single-field `data: string` with no size constraint up to ROS 2's default ~1 MB message size limit.
- `rclpy.init()` is idempotent only when guarded by `rclpy.ok()`. Calling `init` twice without an intervening `shutdown` raises `RCLError: rcl_init called more than once`.
- `rclpy.spin_once(node, timeout_sec=t)` returns either when one callback has been processed or when `t` seconds have elapsed, whichever is first. Standard primitive for poll-loop subscribers.
- `rclpy.node.Node.create_publisher(msg_type, topic, qos_profile)` and `Node.create_subscription(msg_type, topic, callback, qos_profile)` are stable APIs since ROS 2 Foxy (2020).
- `numpy`, `pydantic`, `typer` versions in the gauntlet core all support 3.11/3.12 and have no constraint that interacts with the ROS 2 install path. The `[ros2]` extra is empty, so `uv sync --extra ros2` produces the same lockfile state as `uv sync`.
- Pydantic v2's `Literal["v1"]` field is the standard pattern for schema-version stamping; serialises as a JSON string and validates exactly.
- `rclpy-pip` exists on PyPI but is unofficial, distro-pinned, and not maintained by Open Robotics. We do not use it.
