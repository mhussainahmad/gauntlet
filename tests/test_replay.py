"""Trajectory replay tests — see ``docs/phase2-rfc-004-trajectory-replay.md`` §9.

The bit-identity tests are the load-bearing ones. They drive:

* :func:`test_zero_override_bit_identity` — pairs every field of a
  replayed Episode with its original, no exclusions.
* :func:`test_zero_override_every_cell` — parametrised over every
  ``(cell_index, episode_index)`` pair in a small grid so off-by-ones
  in the spawn-tree reconstruction are impossible to hide.
* :func:`test_axis_application_order_preserved` — pins the RFC §6
  invariant that ``camera_offset_x`` / ``camera_offset_y`` are applied
  in the order the original run applied them, not the user's CLI order.

All tests use the fast env factory (``max_steps=20``) and live in the
default test gate — no ``hf``/``lerobot``/``monitor`` markers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gauntlet.policy.scripted import ScriptedPolicy
from gauntlet.replay import (
    OverrideError,
    parse_override,
    replay_one,
    validate_overrides,
)
from gauntlet.runner import Episode, Runner
from gauntlet.suite.schema import AxisSpec, Suite

# ----------------------------------------------------------------------------
# Module-level factories. Kept here so the suite builder and the
# in-process Runner share exactly one entry point into the env.
# ----------------------------------------------------------------------------


def _make_fast_env() -> Any:
    """Small-budget env factory; keeps each rollout well under a second."""
    from gauntlet.env.tabletop import TabletopEnv

    return TabletopEnv(max_steps=20)


def _make_scripted_policy() -> ScriptedPolicy:
    return ScriptedPolicy()


# ----------------------------------------------------------------------------
# Suite helpers.
# ----------------------------------------------------------------------------


def _two_by_three_suite() -> Suite:
    """2 (lighting) x 3 (camera_x) grid, 2 episodes per cell. 12 episodes total.

    The lighting envelope is intentionally wider than the two grid
    points it enumerates so off-grid override tests have room to
    move without tripping the envelope check.
    """
    return Suite(
        name="replay-2x3",
        env="tabletop",
        seed=2024,
        episodes_per_cell=2,
        axes={
            "lighting_intensity": AxisSpec(low=0.3, high=1.5, steps=2),
            "camera_offset_x": AxisSpec(low=-0.05, high=0.05, steps=3),
        },
    )


def _single_axis_suite() -> Suite:
    """One axis, one value. The smallest suite that still exercises
    the spawn-tree reconstruction non-trivially when episodes_per_cell > 1."""
    return Suite(
        name="replay-1x1x3",
        env="tabletop",
        seed=77,
        episodes_per_cell=3,
        axes={
            "lighting_intensity": AxisSpec(low=0.8, high=0.8, steps=1),
        },
    )


def _camera_xy_suite() -> Suite:
    """camera_offset_x then camera_offset_y — pins the non-commutative
    application-order invariant of RFC §6."""
    return Suite(
        name="replay-camxy",
        env="tabletop",
        seed=99,
        episodes_per_cell=1,
        axes={
            "camera_offset_x": AxisSpec(low=-0.02, high=0.02, steps=2),
            "camera_offset_y": AxisSpec(low=-0.01, high=0.01, steps=2),
        },
    )


def _distractor_suite() -> Suite:
    """A suite whose lone axis is ``distractor_count`` with an
    envelope wide enough to let us flip from 0 distractors to 10 via
    an override and watch the rollout diverge."""
    return Suite(
        name="replay-distractors",
        env="tabletop",
        seed=13,
        episodes_per_cell=1,
        axes={
            "distractor_count": AxisSpec(low=0.0, high=10.0, steps=2),
        },
    )


def _run_suite(suite: Suite) -> list[Episode]:
    """Run *suite* end-to-end under the fast env and return every Episode.

    Callers pick the target by ``_find(episodes, cell_index,
    episode_index)``; keeping the two calls separate avoids re-running
    the suite once per parametrised test case.
    """
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    return runner.run(policy_factory=_make_scripted_policy, suite=suite)


def _find(episodes: list[Episode], cell_index: int, episode_index: int) -> Episode:
    for ep in episodes:
        if ep.cell_index == cell_index and ep.episode_index == episode_index:
            return ep
    raise AssertionError(f"no episode at ({cell_index}, {episode_index})")


# ----------------------------------------------------------------------------
# Bit-identity: the load-bearing contract.
# ----------------------------------------------------------------------------


def test_zero_override_bit_identity() -> None:
    """Replay with no overrides must be bit-identical to the original."""
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 2, 1)

    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        env_factory=_make_fast_env,
    )

    # Full-model equality — no field exclusions. The topology echo in
    # step 1 makes the metadata dict identical too.
    assert replayed.model_dump() == target.model_dump()


@pytest.mark.parametrize(
    ("cell_index", "episode_index"),
    [(c, e) for c in range(4) for e in range(2)],
)
def test_zero_override_every_cell(cell_index: int, episode_index: int) -> None:
    """Every (cell, episode) must round-trip bit-identically.

    Guards against off-by-one errors in the spawn-tree reconstruction
    — a flat single-level spawn, or swapped indices, would produce
    divergent seeds on some but not all nodes.
    """
    # 2x2 x 2 eps -> 4 cells, 2 eps each.
    suite = Suite(
        name="replay-2x2",
        env="tabletop",
        seed=4242,
        episodes_per_cell=2,
        axes={
            "lighting_intensity": AxisSpec(low=0.6, high=0.9, steps=2),
            "camera_offset_x": AxisSpec(low=-0.01, high=0.01, steps=2),
        },
    )
    episodes = _run_suite(suite)
    target = _find(episodes, cell_index, episode_index)

    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        env_factory=_make_fast_env,
    )
    assert replayed.model_dump() == target.model_dump()


def test_axis_application_order_preserved() -> None:
    """Replay preserves the runner's non-commutative axis application order.

    ``camera_offset_x`` and ``camera_offset_y`` both write the full
    ``cam_pos``; whichever is applied last wins. Replay must follow
    the original run's order, not the caller's ``--override`` order.
    """
    suite = _camera_xy_suite()
    # Pick a cell that exercises both axes with non-zero values.
    episodes = _run_suite(suite)
    target = _find(episodes, 3, 0)

    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        env_factory=_make_fast_env,
    )
    # Bit identity: same seed, same reward, same outcome.
    assert replayed.seed == target.seed
    assert replayed.total_reward == target.total_reward
    assert replayed.success == target.success
    # And the perturbation_config comes back in insertion order, not
    # alphabetical / hash order.
    assert list(replayed.perturbation_config.keys()) == ["camera_offset_x", "camera_offset_y"]


# ----------------------------------------------------------------------------
# Override semantics.
# ----------------------------------------------------------------------------


def test_override_keeps_non_overridden_axes_from_original() -> None:
    """Overriding one axis leaves the others at their original values."""
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 1, 0)

    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        overrides={"lighting_intensity": 1.2},
        env_factory=_make_fast_env,
    )
    assert replayed.perturbation_config["lighting_intensity"] == 1.2
    assert (
        replayed.perturbation_config["camera_offset_x"]
        == target.perturbation_config["camera_offset_x"]
    )


def test_single_override_changes_outcome() -> None:
    """A materially different override must at least tickle the env.

    The replayed seed matches (same reset entropy) but the rollout
    interacts with a different model state. We override
    ``object_initial_pose_x`` with a value far from the original —
    the cube starts on the opposite side of the table, which
    shifts the reward trajectory even under the scripted policy.
    At minimum one of (reward, success, step_count) must differ.
    """
    pose_suite = Suite(
        name="replay-pose",
        env="tabletop",
        seed=101,
        episodes_per_cell=1,
        axes={
            "object_initial_pose_x": AxisSpec(low=-0.1, high=0.1, steps=2),
        },
    )
    runner = Runner(n_workers=1, env_factory=_make_fast_env)
    episodes = runner.run(policy_factory=_make_scripted_policy, suite=pose_suite)
    target = _find(episodes, 0, 0)

    replayed = replay_one(
        target=target,
        suite=pose_suite,
        policy_factory=_make_scripted_policy,
        overrides={"object_initial_pose_x": 0.1},
        env_factory=_make_fast_env,
    )
    # Same reset entropy.
    assert replayed.seed == target.seed
    # The overridden value is echoed.
    assert replayed.perturbation_config["object_initial_pose_x"] == 0.1
    # Something about the rollout must differ from the baseline.
    assert (
        replayed.total_reward != target.total_reward
        or replayed.success != target.success
        or replayed.step_count != target.step_count
    )


def test_override_value_echoed_into_replayed_config() -> None:
    """The replayed Episode's perturbation_config carries the override value."""
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 0, 0)

    replayed = replay_one(
        target=target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        overrides={"lighting_intensity": 1.1},
        env_factory=_make_fast_env,
    )
    assert replayed.perturbation_config["lighting_intensity"] == 1.1


# ----------------------------------------------------------------------------
# Validation failures.
# ----------------------------------------------------------------------------


def test_replay_rejects_unknown_axis() -> None:
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 0, 0)

    with pytest.raises(OverrideError, match="not_a_real_axis"):
        replay_one(
            target=target,
            suite=suite,
            policy_factory=_make_scripted_policy,
            overrides={"not_a_real_axis": 1.0},
            env_factory=_make_fast_env,
        )


def test_replay_rejects_out_of_envelope_value() -> None:
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 0, 0)

    # lighting_intensity envelope is [0.5, 1.0] on this suite.
    with pytest.raises(OverrideError, match="outside the suite's declared envelope"):
        replay_one(
            target=target,
            suite=suite,
            policy_factory=_make_scripted_policy,
            overrides={"lighting_intensity": 99.9},
            env_factory=_make_fast_env,
        )


def test_replay_rejects_suite_name_mismatch() -> None:
    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 0, 0)
    other_suite = suite.model_copy(update={"name": "different-suite"})

    with pytest.raises(ValueError, match="suite name mismatch"):
        replay_one(
            target=target,
            suite=other_suite,
            policy_factory=_make_scripted_policy,
            env_factory=_make_fast_env,
        )


def test_replay_rejects_distractor_count_out_of_env_bound() -> None:
    """The env-level hard bound on distractor_count is re-asserted at
    the CLI boundary so the user sees the error before env construction.

    We build a deliberately-loose suite whose envelope admits values
    the env itself would reject, to prove the env-bound check fires
    independently of the suite envelope check.
    """
    loose_suite = Suite(
        name="replay-distractors-loose",
        env="tabletop",
        seed=13,
        episodes_per_cell=1,
        axes={
            # Envelope up to 20, but TabletopEnv clamps at 10.
            "distractor_count": AxisSpec(low=0.0, high=20.0, steps=2),
        },
    )
    with pytest.raises(OverrideError, match=r"env bound is \[0, 10\]"):
        validate_overrides({"distractor_count": 15.0}, loose_suite)


# ----------------------------------------------------------------------------
# Override parser / validator unit tests (no env).
# ----------------------------------------------------------------------------


def test_parse_override_happy_path() -> None:
    assert parse_override("lighting_intensity=1.2") == ("lighting_intensity", 1.2)
    # Whitespace is tolerated.
    assert parse_override("  camera_offset_x = -0.05 ") == ("camera_offset_x", -0.05)


def test_parse_override_rejects_missing_equals() -> None:
    with pytest.raises(OverrideError, match="exactly one"):
        parse_override("lighting_intensity")


def test_parse_override_rejects_double_equals() -> None:
    with pytest.raises(OverrideError, match="exactly one"):
        parse_override("lighting_intensity=1.0=2.0")


def test_parse_override_rejects_empty_axis_name() -> None:
    with pytest.raises(OverrideError, match="axis name on the left"):
        parse_override("=1.0")


def test_parse_override_rejects_non_numeric_value() -> None:
    with pytest.raises(OverrideError, match="not a valid float"):
        parse_override("lighting_intensity=not_a_number")


def test_parse_override_rejects_empty_value() -> None:
    with pytest.raises(OverrideError, match="value on the right"):
        parse_override("lighting_intensity=")


def test_validate_overrides_accepts_in_envelope() -> None:
    suite = _two_by_three_suite()
    validate_overrides(
        {"lighting_intensity": 0.75, "camera_offset_x": 0.0},
        suite,
    )  # no exception


def test_validate_overrides_categorical_tolerance() -> None:
    """A categorical axis accepts values within tolerance of one of its
    declared members; off-set values are rejected."""
    categorical_suite = Suite(
        name="replay-categorical",
        env="tabletop",
        seed=1,
        episodes_per_cell=1,
        axes={
            "object_texture": AxisSpec(values=[0.0, 1.0]),
        },
    )
    # Exactly-equal categorical value.
    validate_overrides({"object_texture": 1.0}, categorical_suite)
    # Within-tolerance (covers YAML round-trip float noise).
    validate_overrides({"object_texture": 1.0 + 1e-12}, categorical_suite)
    # Off-set value is rejected.
    with pytest.raises(OverrideError, match="outside the suite's declared envelope"):
        validate_overrides({"object_texture": 0.5}, categorical_suite)


# ----------------------------------------------------------------------------
# Legacy-Episode fallback.
# ----------------------------------------------------------------------------


def test_replay_legacy_episode_without_topology_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Episodes produced before Task 4 step 1 may lack n_cells /
    episodes_per_cell in metadata. Replay must fall back to the suite
    and warn to stderr, and bit-identity still holds when the suite
    is unchanged.
    """
    suite = _single_axis_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 0, 1)

    # Strip the topology metadata to simulate a legacy Episode.
    legacy_metadata = {
        k: v for k, v in target.metadata.items() if k not in {"n_cells", "episodes_per_cell"}
    }
    legacy_target = target.model_copy(update={"metadata": legacy_metadata})
    assert "n_cells" not in legacy_target.metadata
    assert "episodes_per_cell" not in legacy_target.metadata

    replayed = replay_one(
        target=legacy_target,
        suite=suite,
        policy_factory=_make_scripted_policy,
        env_factory=_make_fast_env,
    )

    captured = capsys.readouterr()
    assert "predates topology echo" in captured.err

    # With the suite unchanged, bit identity holds modulo the missing
    # topology metadata (the replayed Episode records them fresh).
    assert replayed.seed == target.seed
    assert replayed.total_reward == target.total_reward
    assert replayed.success == target.success
    assert replayed.step_count == target.step_count


# ----------------------------------------------------------------------------
# Spawn-tree sanity — cross-check against the RFC-specified derivation.
# ----------------------------------------------------------------------------


def _write_episodes_json(path: Any, episodes: list[Episode]) -> None:
    """Serialise *episodes* to *path* mirroring the CLI's own emission."""
    import json

    payload = [ep.model_dump(mode="json") for ep in episodes]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_two_by_three_yaml(path: Any) -> None:
    """Write a YAML suite that matches :func:`_two_by_three_suite` exactly."""
    import textwrap

    yaml_text = textwrap.dedent(
        """\
        name: replay-2x3
        env: tabletop
        seed: 2024
        episodes_per_cell: 2
        axes:
          lighting_intensity:
            low: 0.3
            high: 1.5
            steps: 2
          camera_offset_x:
            low: -0.05
            high: 0.05
            steps: 3
        """
    )
    path.write_text(yaml_text, encoding="utf-8")


# ----------------------------------------------------------------------------
# CLI tests. Drive the replay subcommand via typer's CliRunner. The
# Runner / env / policy layers are NOT mocked — these are end-to-end
# checks that the subcommand wires the library primitive correctly.
# ----------------------------------------------------------------------------


def test_replay_cli_happy_path(tmp_path: Any) -> None:
    """Run a tiny suite, then replay episode 3:1 with a lighting override.

    The emitted replay.json parses cleanly, carries both original and
    replayed Episodes as full objects, and echoes the override.
    """
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)

    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    out_path = tmp_path / "replay.json"
    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "3:1",
            "--override",
            "lighting_intensity=1.1",
            "--out",
            str(out_path),
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stderr

    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["episode_id"] == "3:1"
    assert payload["suite_name"] == "replay-2x3"
    assert payload["policy"] == "scripted"
    assert payload["overrides"] == {"lighting_intensity": 1.1}
    # Both original and replayed are full Episodes (model_validate works).
    Episode.model_validate(payload["original"])
    replayed = Episode.model_validate(payload["replayed"])
    assert replayed.perturbation_config["lighting_intensity"] == 1.1


def test_replay_cli_zero_override_bit_identical(tmp_path: Any) -> None:
    """CLI round-trip with no overrides emits a replayed Episode that
    matches the original field-for-field."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 1, 0)

    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)
    out_path = tmp_path / "replay.json"

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "1:0",
            "--out",
            str(out_path),
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stderr

    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    replayed = Episode.model_validate(payload["replayed"])
    assert replayed.model_dump() == target.model_dump()


def test_replay_cli_rejects_unknown_episode_id(tmp_path: Any) -> None:
    """Unknown ``--episode-id`` exits non-zero with a preview of the
    available pairs."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "99:99",
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code != 0
    assert "99:99" in result.stderr
    assert "available:" in result.stderr


def test_replay_cli_rejects_unknown_axis(tmp_path: Any) -> None:
    """A typoed ``--override AXIS=VALUE`` exits non-zero with a list of
    the declared axes."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--override",
            "not_an_axis=1.0",
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code != 0
    assert "not_an_axis" in result.stderr
    assert "legal axes:" in result.stderr


def test_replay_cli_rejects_out_of_envelope(tmp_path: Any) -> None:
    """An out-of-envelope override exits non-zero with the envelope."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--override",
            "lighting_intensity=99.9",
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code != 0
    assert "outside the suite's declared envelope" in result.stderr


def test_replay_cli_rejects_suite_name_mismatch(tmp_path: Any) -> None:
    """If the suite YAML's name disagrees with the Episode's
    suite_name, the CLI exits before touching the env."""
    import textwrap

    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)

    # Suite YAML with a different name.
    mismatch_yaml = tmp_path / "mismatch.yaml"
    mismatch_yaml.write_text(
        textwrap.dedent(
            """\
            name: not-the-same-suite
            env: tabletop
            seed: 2024
            episodes_per_cell: 2
            axes:
              lighting_intensity:
                low: 0.3
                high: 1.5
                steps: 2
              camera_offset_x:
                low: -0.05
                high: 0.05
                steps: 3
            """
        ),
        encoding="utf-8",
    )

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(mismatch_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code != 0
    assert "suite name mismatch" in result.stderr


def test_replay_cli_rejects_malformed_episode_id(tmp_path: Any) -> None:
    """An ``--episode-id`` without a colon exits with a helpful message."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    _write_two_by_three_yaml(suite_yaml)

    cli = CliRunner()
    result = cli.invoke(
        app,
        [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "not-a-pair",
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code != 0
    assert "CELL:EPISODE" in result.stderr


def test_replay_cli_override_order_independent_of_cli_order(tmp_path: Any) -> None:
    """CLI ``--override`` order must NOT influence axis application order.

    RFC §6: ``camera_offset_x`` and ``camera_offset_y`` are non-commutative
    in the env (each writes the full cam_pos), so the replayed Episode
    must apply them in the original suite's declared order regardless
    of the order the user typed the overrides. Two CLI invocations that
    differ only in ``--override`` order must produce byte-identical
    replayed Episodes.
    """
    import json

    from typer.testing import CliRunner

    from gauntlet.cli import app

    suite = _camera_xy_suite()
    episodes = _run_suite(suite)

    episodes_path = tmp_path / "episodes.json"
    _write_episodes_json(episodes_path, episodes)
    suite_yaml = tmp_path / "suite.yaml"
    import textwrap

    suite_yaml.write_text(
        textwrap.dedent(
            """\
            name: replay-camxy
            env: tabletop
            seed: 99
            episodes_per_cell: 1
            axes:
              camera_offset_x:
                low: -0.02
                high: 0.02
                steps: 2
              camera_offset_y:
                low: -0.01
                high: 0.01
                steps: 2
            """
        ),
        encoding="utf-8",
    )

    def _run_cli(overrides_in_order: list[str], out_name: str) -> Episode:
        out_path = tmp_path / out_name
        cli = CliRunner()
        args = [
            "replay",
            str(episodes_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--out",
            str(out_path),
            "--env-max-steps",
            "20",
        ]
        for ov in overrides_in_order:
            args.extend(["--override", ov])
        result = cli.invoke(app, args)
        assert result.exit_code == 0, result.stderr
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        return Episode.model_validate(payload["replayed"])

    xy_first = _run_cli(
        ["camera_offset_x=0.01", "camera_offset_y=0.005"],
        "xy_first.json",
    )
    yx_first = _run_cli(
        ["camera_offset_y=0.005", "camera_offset_x=0.01"],
        "yx_first.json",
    )
    assert xy_first.model_dump() == yx_first.model_dump()


def test_replay_cli_help_mentions_override(tmp_path: Any) -> None:
    """``gauntlet replay --help`` renders and documents --override."""
    from typer.testing import CliRunner

    from gauntlet.cli import app

    cli = CliRunner()
    result = cli.invoke(app, ["replay", "--help"])
    assert result.exit_code == 0
    for token in ("--suite", "--policy", "--episode-id", "--override", "--out"):
        assert token in result.stdout


def test_reconstructed_seed_matches_runner_seed() -> None:
    """End-to-end check of the two-level spawn reconstruction.

    Runs a small suite, extracts the seed of a specific Episode, and
    reconstructs the same uint32 from ``(master_seed, n_cells,
    episodes_per_cell, cell_index, episode_index)`` manually. If this
    test ever fails, the replay spawn-tree reconstruction has drifted
    from the Runner's and every downstream bit-identity test is a
    false-positive waiting to happen.
    """
    from gauntlet.runner.worker import extract_env_seed

    suite = _two_by_three_suite()
    episodes = _run_suite(suite)
    target = _find(episodes, 4, 1)

    master_seed = target.metadata["master_seed"]
    n_cells = target.metadata["n_cells"]
    eps_per_cell = target.metadata["episodes_per_cell"]
    assert isinstance(master_seed, int)
    assert isinstance(n_cells, int)
    assert isinstance(eps_per_cell, int)

    master = np.random.SeedSequence(master_seed)
    node = master.spawn(n_cells)[target.cell_index].spawn(eps_per_cell)[target.episode_index]
    assert extract_env_seed(node) == target.seed
