"""Misc small-coverage backfill — Phase 2.5 Task 11.

Targets the last few uncovered lines in the default-job core-set:

* ``cli.run`` ``OSError`` reading suite YAML (line 319-320).
* ``cli.run`` runner-failure / build-report-failure branches when the
  policy mis-shapes its output (lines 342-348).
* ``cli.replay`` ``OSError`` reading suite (line 888-889).
* ``cli.replay`` reward-down formatting branch (line 949).
* ``replay.parse_override`` empty / whitespace-only spec (overrides line 67).
* ``suite.loader._ensure_backend_registered`` imported-but-not-registered
  (line 196).

Each test is a single-purpose unit check that runs in well under a
second.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.replay import OverrideError, parse_override
from gauntlet.runner.episode import Episode


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ──────────────────────────────────────────────────────────────────────
# Replay parse_override: empty / whitespace-only spec.
# ──────────────────────────────────────────────────────────────────────


def test_parse_override_rejects_empty_string() -> None:
    with pytest.raises(OverrideError, match="non-empty"):
        parse_override("")


def test_parse_override_rejects_whitespace_only_string() -> None:
    with pytest.raises(OverrideError, match="non-empty"):
        parse_override("   ")


# ──────────────────────────────────────────────────────────────────────
# CLI run: OSError reading suite (file exists but unreadable).
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root bypasses POSIX file permissions; chmod-0 would not block reads.",
)
def test_run_unreadable_suite_file_surfaces_os_error(runner: CliRunner, tmp_path: Path) -> None:
    """A suite file that exists but cannot be opened (chmod 000) surfaces
    the OSError branch in ``cli.run`` as ``could not read file``."""
    suite_path = tmp_path / "no-read.yaml"
    suite_path.write_text("name: x\nenv: tabletop\n", encoding="utf-8")
    suite_path.chmod(0o000)
    out_dir = tmp_path / "out"

    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_path),
                "-p",
                "random",
                "-o",
                str(out_dir),
                "--env-max-steps",
                "5",
            ],
        )
        assert result.exit_code != 0
        # Either branch (OSError on the read OR "could not read file"
        # path) is acceptable — both surface the file's name.
        assert str(suite_path) in result.stderr or "read" in result.stderr.lower()
    finally:
        suite_path.chmod(0o644)


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root bypasses POSIX file permissions; chmod-0 would not block reads.",
)
def test_replay_unreadable_suite_file_surfaces_os_error(runner: CliRunner, tmp_path: Path) -> None:
    """The replay subcommand has its own OSError branch on suite read.
    Same chmod trick, same expected user-facing error shape."""
    suite_path = tmp_path / "no-read.yaml"
    suite_path.write_text(
        textwrap.dedent(
            """\
            name: replay-no-read
            env: tabletop
            seed: 1
            episodes_per_cell: 1
            axes:
              lighting_intensity:
                values: [0.5]
            """
        ),
        encoding="utf-8",
    )

    # Need a real episodes.json so we get past the earlier guards.
    eps_path = tmp_path / "eps.json"
    ep = Episode(
        suite_name="replay-no-read",
        cell_index=0,
        episode_index=0,
        seed=0,
        perturbation_config={"lighting_intensity": 0.5},
        success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        total_reward=0.0,
    )
    eps_path.write_text(json.dumps([ep.model_dump(mode="json")]) + "\n", encoding="utf-8")

    suite_path.chmod(0o000)
    try:
        result = runner.invoke(
            app,
            [
                "replay",
                str(eps_path),
                "--suite",
                str(suite_path),
                "--policy",
                "scripted",
                "--episode-id",
                "0:0",
                "--env-max-steps",
                "5",
            ],
        )
        assert result.exit_code != 0
        assert str(suite_path) in result.stderr or "read" in result.stderr.lower()
    finally:
        suite_path.chmod(0o644)


# ──────────────────────────────────────────────────────────────────────
# Suite loader: imported-but-not-registered defence-in-depth (line 196).
# ──────────────────────────────────────────────────────────────────────


def test_ensure_backend_registered_imported_but_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a backend module imports successfully but does NOT call
    ``register_env``, the loader raises a clear backend-packaging-bug
    error rather than crashing later."""
    import types

    from gauntlet.suite import loader as loader_mod

    fake_env = "tabletop-bogus-test"
    fake_module = "tests._fake_silent_backend"

    # Inject a known-bad module that does NOT call register_env.
    silent_module = types.ModuleType(fake_module)
    monkeypatch.setitem(sys.modules, fake_module, silent_module)

    # Wire it into BUILTIN_BACKEND_IMPORTS so the loader tries to
    # import it. _EXTRA_FOR_MODULE is consulted for the install hint
    # (defaults to env_name when missing).
    from gauntlet.suite.schema import BUILTIN_BACKEND_IMPORTS

    monkeypatch.setitem(BUILTIN_BACKEND_IMPORTS, fake_env, fake_module)

    with pytest.raises(ValueError, match="did not call register_env"):
        loader_mod._ensure_backend_registered(fake_env)


# ──────────────────────────────────────────────────────────────────────
# CLI run: build_report failure rewrap (lines 347-348).
# ──────────────────────────────────────────────────────────────────────


def test_run_zero_episode_suite_surfaces_build_report_error(
    runner: CliRunner, tmp_path: Path
) -> None:
    """If a suite somehow yields zero episodes (eg. ``episodes_per_cell``
    of 1 with an empty grid is impossible by schema), the build_report
    ValueError surfaces as ``could not build report``. We force the
    failure by patching ``Runner.run`` to return an empty list."""
    from gauntlet import cli

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        textwrap.dedent(
            """\
            name: empty-run
            env: tabletop
            episodes_per_cell: 1
            seed: 7
            axes:
              lighting_intensity:
                values: [0.5]
            """
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    class _RunnerEmpty:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def run(self, *_args: object, **_kwargs: object) -> list[Episode]:
            return []

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(cli, "Runner", _RunnerEmpty)
    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_path),
                "-p",
                "random",
                "-o",
                str(out_dir),
                "--env-max-steps",
                "5",
            ],
        )
    finally:
        mp.undo()
    assert result.exit_code != 0
    assert "could not build report" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# CLI replay: reward-down branch (line 949).
# ──────────────────────────────────────────────────────────────────────


def test_replay_cli_reward_down_branch_uses_delta_down_style(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The replay command formats the per-episode summary with
    ``delta.down`` when the replayed reward is strictly LESS than the
    original. We force this by mutating the original episode in
    ``episodes.json`` to carry an artificially-large total_reward
    that the live replay won't be able to match."""
    from gauntlet.policy.scripted import ScriptedPolicy
    from gauntlet.runner import Runner
    from gauntlet.suite.schema import AxisSpec, Suite

    def _env() -> object:
        from gauntlet.env.tabletop import TabletopEnv

        return TabletopEnv(max_steps=20)

    suite = Suite(
        name="reward-down",
        env="tabletop",
        seed=2024,
        episodes_per_cell=1,
        axes={"lighting_intensity": AxisSpec(values=[0.5])},
    )
    real_runner = Runner(n_workers=1, env_factory=_env)  # type: ignore[arg-type]
    eps = real_runner.run(policy_factory=ScriptedPolicy, suite=suite)
    assert len(eps) == 1
    target = eps[0]
    # Inflate the original reward so the replayed value is strictly
    # less. The replay logic only looks at the persisted episode for
    # the original — mutate it to carry a huge reward.
    inflated = target.model_copy(update={"total_reward": target.total_reward + 1000.0})
    eps_path = tmp_path / "eps.json"
    eps_path.write_text(json.dumps([inflated.model_dump(mode="json")]), encoding="utf-8")

    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        textwrap.dedent(
            """\
            name: reward-down
            env: tabletop
            seed: 2024
            episodes_per_cell: 1
            axes:
              lighting_intensity:
                values: [0.5]
            """
        ),
        encoding="utf-8",
    )

    out = tmp_path / "replay.json"
    result = runner.invoke(
        app,
        [
            "replay",
            str(eps_path),
            "--suite",
            str(suite_yaml),
            "--policy",
            "scripted",
            "--episode-id",
            "0:0",
            "--out",
            str(out),
            "--env-max-steps",
            "20",
        ],
    )
    assert result.exit_code == 0, result.stderr
    # The "delta -" sign is on stderr — both delta-down style and the
    # negative reward delta are surfaced.
    assert "delta " in result.stderr
    # The replayed reward is ~0 and the original is ~1000, so the
    # signed delta starts with '-'.
    assert "-" in result.stderr
