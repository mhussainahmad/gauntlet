# Phase 2 Task 10 Exploration: ROS 2 integration

**Date:** 2026-04-24
**Branch:** phase-2/ros2-integration
**Base:** cea727b (main, post-Task-8)
**Target middleware:** [ROS 2](https://docs.ros.org/) — `rclpy` Python bindings, distros Humble (LTS, 22.04) and Jazzy (LTS, 24.04).

---

## Q1: Goal

`GAUNTLET_SPEC.md` §7 lists "ROS 2 integration for running on real robots with logging" as a Phase 2 candidate. Two halves are named: the publisher half (broadcast structured episode outcomes onto a ROS 2 topic so a fleet of real robots running gauntlet evaluations can be aggregated) and the logger half (subscribe to a real robot's joint-state topic during a rollout and dump the trajectory to disk for later replay via `gauntlet replay`).

This exploration scopes what is buildable in a Python-only PR, what must be deferred to follow-up work (custom `.msg` packages, a `Ros2HardwareEnv(GauntletEnv)` over a real arm), and how to keep the default torch-free CI job green when `rclpy` is not pip-installable.

---

## Q2: ROS 2 install landscape

`rclpy` is **not** distributed via PyPI in its official form. It ships via:

| Distribution | Channel | Python | OS | Notes |
|---|---|---|---|---|
| Humble Hawksbill | `apt install ros-humble-rclpy` | 3.10 | Ubuntu 22.04 | LTS — supported until May 2027 |
| Jazzy Jalisco | `apt install ros-jazzy-rclpy` | 3.12 | Ubuntu 24.04 | LTS — supported until May 2029 |
| Rolling | `apt install ros-rolling-rclpy` | 3.12 | Ubuntu 24.04 | tracks main; not recommended for downstream |
| (any) | Docker `osrf/ros:humble-desktop` (etc.) | distro-pinned | container | recommended for CI / cross-platform |

There is an unofficial PyPI shim called `rclpy-pip` but it is fragile, version-pinned to one ROS 2 distro at a time, and not maintained by Open Robotics. We will **not** rely on it.

**Consequence for the `[ros2]` extra**: it cannot declare `rclpy` (or `std_msgs`, `sensor_msgs`, `geometry_msgs`) as a PyPI dependency — `uv sync --extra ros2` would fail to resolve. Two viable shapes:

- **(A) Empty `[ros2]` extra.** The extra exists for documentation purposes only; users `apt install ros-<distro>-rclpy` (or `docker run osrf/ros:<distro>-desktop`) and source the relevant `setup.bash` before `gauntlet ros2 publish` works. This RFC chooses (A).
- **(B) `[ros2]` extra carries pure-Python helpers** (e.g. `pyyaml` if we authored a YAML-based message schema). Rejected — the JSON-in-`std_msgs.msg.String` shape this RFC settles on (Q5) needs only stdlib `json`.

Both `(A)` and `(B)` produce the same user-facing error when `rclpy` is missing: a clean `ImportError` from `gauntlet.ros2.publisher` / `gauntlet.ros2.recorder` pointing at the apt/Docker install path.

---

## Q3: Valuable shape decision

Three candidate scopes for "ROS 2 integration":

- **(i) `Ros2HardwareEnv(GauntletEnv)` — wrap a real arm under the env Protocol.** Would require a real robot connected for any meaningful test. Out of scope: a mocked-rclpy unit test cannot confirm that a real UR5e or Franka actually moved, and CI without hardware can't catch IK / safety-limit bugs. Defer.
- **(ii) `Ros2EpisodePublisher` — broadcast `Episode` outcomes to a topic.** Useful for fleet-wide failure-mode aggregation: many real robots running gauntlet evaluations write to one shared topic; a downstream subscriber (a separate Foxglove dashboard, a log-aggregator, etc.) collates failures across the fleet. Wholly mockable in CI. **Chosen.**
- **(iii) `Ros2RolloutRecorder` — subscribe to a topic during a rollout and dump the messages to disk.** Useful for the "real robots with logging" half of the §7 charter: when a gauntlet evaluation ends, the recorder has a bag of joint-state samples that can be replayed via `gauntlet replay` (or analysed offline). Wholly mockable in CI. **Chosen.**

Picking (ii) + (iii) gives us both halves of the §7 brief without requiring hardware in CI. (i) is named in §11 future work — when a future RFC adds it, the message contracts (ii) and (iii) settle here are reusable.

---

## Q4: Mocking strategy for tests

The hard fact: CI cannot install `rclpy`. Therefore tests must mock at the import boundary.

Pattern (lifted from `tests/test_import_guards.py`'s torch handling and adapted):

- A `tests/ros2/conftest.py` seeds `sys.modules["rclpy"]`, `sys.modules["rclpy.node"]`, `sys.modules["std_msgs"]`, `sys.modules["std_msgs.msg"]`, `sys.modules["sensor_msgs"]`, `sys.modules["sensor_msgs.msg"]`, `sys.modules["geometry_msgs"]`, `sys.modules["geometry_msgs.msg"]` to `MagicMock()` instances **before any test module imports `gauntlet.ros2.publisher` / `.recorder`**. This lets the module-scope `try: import rclpy` succeed (against the mock), so test collection itself does not blow up.
- The unit tests then patch the specific seam they exercise: `mock_node = MagicMock()`; `mock_publisher = MagicMock()`; `mock_node.create_publisher.return_value = mock_publisher`. Construct a `Ros2EpisodePublisher` against the mocked rclpy; call `publisher.publish_episode(ep)`; assert `mock_publisher.publish.call_args.args[0].data == expected_json`.
- The import-guard test (`tests/ros2/test_import_guards.py`) re-runs the rclpy-absent contract: `monkeypatch.setitem(sys.modules, "rclpy", None)`, `importlib.reload(gauntlet.ros2.publisher)`, expect `ImportError` mentioning the apt/Docker install path. This mirrors the `torch_absent` / monitor pattern (`tests/test_import_guards.py:115-134`).
- The schema tests (`tests/ros2/test_schema.py`) need NO mocking — `gauntlet.ros2.schema` is pure pydantic and runs in the default torch-free job (no `ros2` marker).

**Why this works for the `ros2-tests` CI job that does NOT install ROS 2**: the conftest mock is the same `MagicMock` shape on every machine. `pytest -m ros2` runs against MagicMocks, not against rclpy. There is no "real ROS 2 integration test" in this PR — that's deferred to a future workflow_dispatch job that would `docker run osrf/ros:humble` and exercise real pub/sub.

---

## Q5: JSON-in-`std_msgs.msg.String` message contract

ROS 2 has a custom message system (`.msg` files compiled into Python/C++ bindings via `colcon build`). For a Python-only PR we cannot ship custom messages — that requires a `package.xml` + `CMakeLists.txt` + a colcon workspace, none of which is in the gauntlet pip-install path.

**Choice: serialise our payloads as JSON inside `std_msgs.msg.String`.** Rationale:

- `std_msgs` is part of every ROS 2 distro by default — no extra apt package.
- Pure-text payloads round-trip cleanly through `ros2 bag` / `ros2 topic echo` for debugging.
- `pydantic.BaseModel.model_dump_json()` is the existing serialisation contract used everywhere else in the codebase (`Episode`, `Report`, `DriftReport`).
- A future RFC can introduce a `gauntlet_ros_msgs` ament package with proper `.msg` files; the JSON path remains supported as a fallback for users who can't / won't build a colcon workspace. The pydantic schema we land here becomes that future package's authoritative source.

Schema: `Ros2EpisodePayload` mirrors `Episode` field-by-field (suite name, cell index, success, perturbation_config, reward, step count) plus a `gauntlet_version` field for downstream-consumer compatibility checks. Stays under 1 KB per episode in JSON form — well within ROS 2's default `1024 * 1024` byte message size limit.

---

## Q6: CI gating decision

New CI job `ros2-tests` mirrors `monitor-tests` / `pybullet-tests` / `genesis-tests`:

- Sync: `uv sync --extra ros2 --group ros2-dev` (the `[ros2]` extra is empty, so this only pulls dev tooling — fast, ~10s).
- Run: `uv run pytest -m ros2 -q`.
- The conftest seeds `rclpy` MagicMocks; no real ROS 2 install needed.

Default `lint-typecheck-test` job's `-m` exclusion grows to `"... and not ros2"`. Same shape Task 7 / Task 8 used to add `genesis`.

A second future-only `ros2-integration-tests` job would `docker run osrf/ros:humble-desktop`, source `/opt/ros/humble/setup.bash`, install gauntlet inside the container, and exercise real pub/sub against a real `rclpy`. This is left out of the present PR — the unit tests with mocked rclpy are the contract; a containerised integration test belongs to a follow-up that adds `Ros2HardwareEnv`.

---

## Q7: Risks

- **rclpy mock fidelity drift.** `rclpy.create_node` / `Node.create_publisher` / `Node.create_subscription` are stable APIs since ROS 2 Foxy (2020). The MagicMock surface only exercises method names + return-shape; if upstream renames a method, our tests miss it. Mitigation: pin an asserted method-name allowlist in `conftest.py` (`assert hasattr(rclpy, "init")` etc.) so a rename in a future rclpy release that breaks our shape produces a single clear failure rather than a mystery green-but-wrong test.
- **Spinning policy in `Ros2RolloutRecorder`.** `rclpy.spin_once(node)` is the clean primitive but blocks; `MultiThreadedExecutor` is heavier but composable. We choose `spin_once(node, timeout_sec=0.1)` inside a polling loop in the recorder's context-manager `__exit__`, with a configurable max-wait. Easy to test (the mock returns immediately), and avoids a background thread the user can't kill cleanly.
- **`Ctrl-C` during a CLI publish.** `rclpy.shutdown()` must run in a `finally`. The CLI subcommands wrap the publisher / recorder in `try/finally` to guarantee shutdown.
- **No QoS configuration in v1.** Default-reliability default-history. A future RFC adds a `--qos reliable|best-effort` flag; not blocking for v1.
- **`Episode.metadata` may carry non-JSON-serialisable values** (e.g. numpy scalars after a future RFC). We round-trip via `model_dump(mode="json")` which already coerces to JSON-safe primitives — same path the rest of the codebase uses.

---

## Q8: Module layout (preview — refined in the RFC)

```
src/gauntlet/ros2/
├── __init__.py            # re-exports schema; lazy __getattr__ for publisher/recorder
├── schema.py              # pydantic Ros2EpisodePayload (torch-free, rclpy-free)
├── publisher.py           # Ros2EpisodePublisher — module-scope try: import rclpy
└── recorder.py            # Ros2RolloutRecorder context manager — same guard

tests/ros2/
├── conftest.py            # seeds sys.modules with MagicMock for rclpy + std_msgs etc.
├── test_schema.py         # NOT marked ros2 — runs in default job, schema is pure pydantic
├── test_import_guards.py  # ros2 marker — re-patches rclpy to None, checks install-hint message
├── test_publisher.py      # ros2 marker — fully mocked rclpy seam
├── test_recorder.py       # ros2 marker — fully mocked subscription seam
└── test_cli_ros2.py       # ros2 marker — typer CliRunner + the same mock patching

examples/
└── publish_episodes_to_ros2.py   # imports cleanly in default torch-free job (mocks rclpy if absent)
```

---

## Q9: What this exploration does NOT cover

- A `Ros2HardwareEnv(GauntletEnv)` over a real robot — explicit non-goal (Q3 (i)).
- Custom `.msg` packages requiring colcon — deferred to a future RFC (Q5).
- ROS 1 (rospy) — out of scope. ROS 1 is end-of-life as of May 2025.
- Bridging to ROS 2 over the network from a non-ROS host (e.g. zenoh) — interesting, not this RFC.
- A live HTML / Foxglove dashboard for the published topic — out of scope; `Ros2EpisodePublisher` writes the topic, downstream tooling consumes it.

---

## Q10: Open decisions parked for the RFC

- **Topic name defaults.** Publisher: `/gauntlet/episodes`. Recorder: no default — user must pass `--topic`. Documented in the RFC's Open Questions with these as the chosen defaults.
- **Recorder output format.** NPZ per recorded message vs JSON-lines per message vs a single rosbag2 file. RFC Q-block — default is JSON-lines for parity with the rest of the codebase's serialisation conventions (no rosbag2 dep).
- **QoS defaults.** Reliable, depth=10. RFC Q-block.
- **Distro support claim.** RFC declares "Humble and Jazzy supported; other distros best-effort". Documented; no per-distro test matrix.
