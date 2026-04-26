"""Unit tests for the four polish-task extension-point groups.

Covers:

* :func:`gauntlet.plugins.discover_axis_plugins` — ``gauntlet.axes`` group.
* :func:`gauntlet.plugins.discover_sampler_plugins` — ``gauntlet.samplers`` group.
* :func:`gauntlet.plugins.discover_sink_plugins` — ``gauntlet.sinks`` group.
* :func:`gauntlet.plugins.discover_cli_plugins` — ``gauntlet.cli`` group.

The discovery-mechanic tests (failure swallow, duplicate-name dedupe,
``lru_cache`` semantics, identity-collision warning) live next to the
two original groups in :mod:`tests.test_plugins`; this module focuses
on the *integration* surfaces each new group plumbs into:

* axis_for / Suite schema axis-name resolver,
* build_sampler / Suite schema sampling-mode resolver,
* CLI ``--sink`` plugin loader,
* ``gauntlet ext`` command registration.

Pattern (mirrors ``tests/test_plugins.py``):

* :class:`_FakeEntryPoint` duck-types
  :class:`importlib.metadata.EntryPoint` so we never need a real install.
* :func:`_patch_entry_points` swaps in our fakes for the duration of
  one test.
* The autouse :func:`_clear_caches` fixture invalidates every
  ``discover_*`` ``lru_cache`` so tests do not leak through each other.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, ClassVar
from unittest.mock import patch

import click
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from gauntlet.env.perturbation.axes import axis_for
from gauntlet.env.perturbation.base import (
    AXIS_KIND_CONTINUOUS,
    PerturbationAxis,
    make_continuous_sampler,
)
from gauntlet.plugins import (
    AXIS_ENTRY_POINT_GROUP,
    CLI_ENTRY_POINT_GROUP,
    SAMPLER_ENTRY_POINT_GROUP,
    SINK_ENTRY_POINT_GROUP,
    discover_axis_plugins,
    discover_cli_plugins,
    discover_sampler_plugins,
    discover_sink_plugins,
)
from gauntlet.runner.episode import Episode
from gauntlet.suite.sampling import build_sampler
from gauntlet.suite.schema import AxisSpec, Suite

# ──────────────────────────────────────────────────────────────────────
# Test plumbing — duck-typed EntryPoint + group dispatcher.
# ──────────────────────────────────────────────────────────────────────


class _FakeDist:
    """Minimal stand-in for ``EntryPoint.dist`` used in tests."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeEntryPoint:
    """Duck-typed :class:`importlib.metadata.EntryPoint`.

    ``gauntlet.plugins`` only touches ``.name``, ``.dist``, and
    ``.load()`` so a duck is enough.
    """

    def __init__(
        self,
        name: str,
        target: object,
        dist_name: str = "fake-pkg",
    ) -> None:
        self.name = name
        self._target = target
        self.dist = _FakeDist(dist_name)

    def load(self) -> object:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _patch_entry_points(group_to_eps: dict[str, list[_FakeEntryPoint]]) -> Any:
    """Patch ``gauntlet.plugins.entry_points`` to the supplied per-group dict.

    Unknown groups return an empty list (matches the real behaviour of
    :func:`importlib.metadata.entry_points` on an empty group).
    """

    def _fake(*, group: str) -> list[_FakeEntryPoint]:
        return group_to_eps.get(group, [])

    return patch("gauntlet.plugins.entry_points", side_effect=_fake)


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """Clear every ``discover_*`` lru_cache before AND after each test.

    Without this a mocked entry-point table from one test would leak
    into the next via the cached return.
    """
    discover_axis_plugins.cache_clear()
    discover_sampler_plugins.cache_clear()
    discover_sink_plugins.cache_clear()
    discover_cli_plugins.cache_clear()
    yield
    discover_axis_plugins.cache_clear()
    discover_sampler_plugins.cache_clear()
    discover_sink_plugins.cache_clear()
    discover_cli_plugins.cache_clear()


# ──────────────────────────────────────────────────────────────────────
# gauntlet.axes — discovery + axis_for + suite-schema resolver.
# ──────────────────────────────────────────────────────────────────────


def _make_plugin_axis() -> PerturbationAxis:
    """Trivial plugin axis factory used across the axis tests."""
    return PerturbationAxis(
        name="my_plugin_axis",
        kind=AXIS_KIND_CONTINUOUS,
        sampler=make_continuous_sampler(0.0, 1.0),
        low=0.0,
        high=1.0,
    )


def test_discover_axis_plugins_returns_loaded_factories() -> None:
    eps = [_FakeEntryPoint("my_plugin_axis", _make_plugin_axis)]
    with _patch_entry_points({AXIS_ENTRY_POINT_GROUP: eps}):
        result = discover_axis_plugins()
    assert set(result) == {"my_plugin_axis"}
    assert result["my_plugin_axis"] is _make_plugin_axis


def test_axis_for_falls_through_to_plugin() -> None:
    """A plugin axis name resolves through :func:`axis_for` even though
    it isn't in the canonical AXIS_NAMES tuple."""
    eps = [_FakeEntryPoint("my_plugin_axis", _make_plugin_axis)]
    with _patch_entry_points({AXIS_ENTRY_POINT_GROUP: eps}):
        axis = axis_for("my_plugin_axis")
    assert axis.name == "my_plugin_axis"
    assert axis.kind == AXIS_KIND_CONTINUOUS


def test_axis_for_builtin_wins_on_collision_with_warning() -> None:
    """A plugin re-registering a built-in axis name warns and the
    built-in factory is used."""

    def rogue_lighting() -> PerturbationAxis:  # pragma: no cover - never invoked
        return PerturbationAxis(
            name="lighting_intensity",
            kind=AXIS_KIND_CONTINUOUS,
            sampler=make_continuous_sampler(99.0, 100.0),
            low=99.0,
            high=100.0,
        )

    eps = [_FakeEntryPoint("lighting_intensity", rogue_lighting, dist_name="rogue-pkg")]
    with (
        _patch_entry_points({AXIS_ENTRY_POINT_GROUP: eps}),
        pytest.warns(RuntimeWarning, match=r"shadows the built-in 'lighting_intensity'"),
    ):
        axis = axis_for("lighting_intensity")
    # Built-in low/high (0.3 / 1.5) — NOT the rogue's 99/100.
    assert axis.low == 0.3
    assert axis.high == 1.5


def test_axis_for_plugin_name_mismatch_raises() -> None:
    """A plugin whose factory returns a PerturbationAxis with a different
    .name than the entry-point key must raise — this is a packaging bug
    in the plugin distribution and silent acceptance would misroute the
    runner's perturbation dispatch."""

    def misnamed_factory() -> PerturbationAxis:
        return PerturbationAxis(
            name="actually_other_name",
            kind=AXIS_KIND_CONTINUOUS,
            sampler=make_continuous_sampler(0.0, 1.0),
            low=0.0,
            high=1.0,
        )

    eps = [_FakeEntryPoint("declared_name", misnamed_factory)]
    with (
        _patch_entry_points({AXIS_ENTRY_POINT_GROUP: eps}),
        pytest.raises(ValueError, match=r"axis plugin 'declared_name'"),
    ):
        axis_for("declared_name")


def test_suite_schema_accepts_plugin_axis_name() -> None:
    """A YAML referencing a plugin axis validates clean — the schema
    consults the plugin registry as a fallback after AXIS_NAMES."""
    eps = [_FakeEntryPoint("my_plugin_axis", _make_plugin_axis)]
    with _patch_entry_points({AXIS_ENTRY_POINT_GROUP: eps}):
        suite = Suite.model_validate(
            {
                "name": "tiny",
                "env": "tabletop",
                "episodes_per_cell": 1,
                "axes": {
                    "my_plugin_axis": AxisSpec.model_validate(
                        {"low": 0.0, "high": 1.0, "steps": 2}
                    ).model_dump(),
                },
            }
        )
    assert "my_plugin_axis" in suite.axes


def test_suite_schema_rejects_unknown_axis_with_plugin_hint() -> None:
    """An unknown name surfaces a clean error mentioning both the
    built-in tuple and the (currently empty) plugin set."""
    with (
        _patch_entry_points({AXIS_ENTRY_POINT_GROUP: []}),
        pytest.raises(ValueError, match=r"legal plugin names are: <none installed>"),
    ):
        Suite.model_validate(
            {
                "name": "tiny",
                "env": "tabletop",
                "episodes_per_cell": 1,
                "axes": {
                    "totally_made_up": AxisSpec.model_validate(
                        {"low": 0.0, "high": 1.0, "steps": 2}
                    ).model_dump(),
                },
            }
        )


# ──────────────────────────────────────────────────────────────────────
# gauntlet.samplers — discovery + build_sampler + Suite schema.
# ──────────────────────────────────────────────────────────────────────


class _FakePluginSampler:
    """Minimal sampler that emits a single empty cell."""

    def sample(self, suite: Suite, rng: np.random.Generator) -> list[Any]:
        del rng
        from gauntlet.suite.schema import SuiteCell

        return [SuiteCell(index=0, values=dict.fromkeys(suite.axes, 0.0))]


def test_discover_sampler_plugins_returns_loaded_classes() -> None:
    eps = [_FakeEntryPoint("my_sampler", _FakePluginSampler)]
    with _patch_entry_points({SAMPLER_ENTRY_POINT_GROUP: eps}):
        result = discover_sampler_plugins()
    assert set(result) == {"my_sampler"}
    assert result["my_sampler"] is _FakePluginSampler


def test_build_sampler_falls_through_to_plugin() -> None:
    eps = [_FakeEntryPoint("my_sampler", _FakePluginSampler)]
    with _patch_entry_points({SAMPLER_ENTRY_POINT_GROUP: eps}):
        sampler = build_sampler("my_sampler")
    assert isinstance(sampler, _FakePluginSampler)
    # Iteration smoke — task spec asks for "a plugin sampler iterates".
    # The fake emits a single all-zero cell; the contract is that the
    # returned list satisfies SuiteCell so the runner can dispatch
    # ``index`` / ``values`` without further introspection.
    suite = Suite.model_validate(
        {
            "name": "tiny",
            "env": "tabletop",
            "episodes_per_cell": 1,
            "axes": {
                "lighting_intensity": AxisSpec.model_validate(
                    {"low": 0.0, "high": 1.0, "steps": 1}
                ).model_dump(),
            },
        }
    )
    cells = sampler.sample(suite, np.random.default_rng(0))
    assert len(cells) == 1
    assert cells[0].index == 0
    assert set(cells[0].values) == {"lighting_intensity"}


def test_build_sampler_builtin_wins_on_collision_with_warning() -> None:
    """A plugin re-registering a built-in sampling mode warns and the
    built-in is used. Mirrors the policies / envs / axes precedent so
    samplers do not silently fall behind the rest of the registry."""
    eps = [_FakeEntryPoint("cartesian", _FakePluginSampler, dist_name="rogue-pkg")]
    with (
        _patch_entry_points({SAMPLER_ENTRY_POINT_GROUP: eps}),
        pytest.warns(RuntimeWarning, match=r"shadows the built-in 'cartesian'"),
    ):
        sampler = build_sampler("cartesian")
    # Built-in CartesianSampler — NOT the rogue.
    from gauntlet.suite.sampling import CartesianSampler

    assert isinstance(sampler, CartesianSampler)


def test_suite_schema_accepts_plugin_sampling_mode() -> None:
    """A YAML naming a plugin sampler validates clean — the field
    validator consults the plugin registry as a fallback after
    SAMPLING_MODES."""
    eps = [_FakeEntryPoint("my_sampler", _FakePluginSampler)]
    with _patch_entry_points({SAMPLER_ENTRY_POINT_GROUP: eps}):
        suite = Suite.model_validate(
            {
                "name": "tiny",
                "env": "tabletop",
                "episodes_per_cell": 1,
                "sampling": "my_sampler",
                "n_samples": 4,
                "axes": {
                    "lighting_intensity": AxisSpec.model_validate(
                        {"low": 0.5, "high": 1.0}
                    ).model_dump(),
                },
            }
        )
    assert suite.sampling == "my_sampler"
    assert suite.n_samples == 4


def test_suite_schema_rejects_unknown_sampling_with_plugin_hint() -> None:
    with (
        _patch_entry_points({SAMPLER_ENTRY_POINT_GROUP: []}),
        pytest.raises(ValueError, match=r"legal plugin modes are: <none installed>"),
    ):
        Suite.model_validate(
            {
                "name": "tiny",
                "env": "tabletop",
                "episodes_per_cell": 1,
                "sampling": "made_up_sampler",
                "axes": {
                    "lighting_intensity": AxisSpec.model_validate(
                        {"low": 0.5, "high": 1.0, "steps": 2}
                    ).model_dump(),
                },
            }
        )


# ──────────────────────────────────────────────────────────────────────
# gauntlet.sinks — discovery + CLI --sink integration.
# ──────────────────────────────────────────────────────────────────────


class _RecordingSink:
    """Collects Episodes in-memory; used to verify --sink wiring."""

    instances: ClassVar[list[_RecordingSink]] = []

    def __init__(self, *, run_name: str, suite_name: str, config: object) -> None:
        self.run_name = run_name
        self.suite_name = suite_name
        self.config = config
        self.episodes: list[Episode] = []
        self.closed = False
        type(self).instances.append(self)

    def log_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def close(self) -> None:
        self.closed = True


def test_discover_sink_plugins_returns_loaded_classes() -> None:
    eps = [_FakeEntryPoint("my_sink", _RecordingSink)]
    with _patch_entry_points({SINK_ENTRY_POINT_GROUP: eps}):
        result = discover_sink_plugins()
    assert set(result) == {"my_sink"}
    assert result["my_sink"] is _RecordingSink


def test_cli_sink_flag_loads_plugin_and_receives_episodes(tmp_path: Any) -> None:
    """End-to-end: ``gauntlet run --sink my_sink`` instantiates the
    plugin sink, hands it every Episode, and closes it."""
    from gauntlet.cli import app

    _RecordingSink.instances.clear()

    eps = [_FakeEntryPoint("my_sink", _RecordingSink)]
    out = tmp_path / "out"
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "\n".join(
            [
                "name: tiny",
                "env: tabletop",
                "episodes_per_cell: 1",
                "axes:",
                "  lighting_intensity:",
                "    low: 0.5",
                "    high: 1.5",
                "    steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    with _patch_entry_points({SINK_ENTRY_POINT_GROUP: eps}):
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_yaml),
                "--policy",
                "random",
                "--out",
                str(out),
                "--sink",
                "my_sink",
                "--env-max-steps",
                "1",
            ],
        )
    assert result.exit_code == 0, result.stderr
    assert (out / "episodes.json").exists()
    assert len(_RecordingSink.instances) == 1
    sink = _RecordingSink.instances[0]
    # One axis x one step x one episode-per-cell == 1 episode.
    assert len(sink.episodes) == 1
    assert sink.closed is True


def test_cli_sink_flag_unknown_name_errors_cleanly(tmp_path: Any) -> None:
    """An unknown ``--sink <name>`` surfaces a clean CLI error rather
    than diving into the runner."""
    from gauntlet.cli import app

    out = tmp_path / "out"
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "\n".join(
            [
                "name: tiny",
                "env: tabletop",
                "episodes_per_cell: 1",
                "axes:",
                "  lighting_intensity:",
                "    low: 0.5",
                "    high: 1.5",
                "    steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    with _patch_entry_points({SINK_ENTRY_POINT_GROUP: []}):
        result = runner.invoke(
            app,
            [
                "run",
                str(suite_yaml),
                "--policy",
                "random",
                "--out",
                str(out),
                "--sink",
                "not_a_real_sink",
                "--env-max-steps",
                "1",
            ],
        )
    assert result.exit_code != 0
    # stderr is where the harness pipes _fail / _echo_err output.
    assert "not_a_real_sink" in result.stderr


# ──────────────────────────────────────────────────────────────────────
# gauntlet.cli — discovery + ext-app registration (Typer + Click).
# ──────────────────────────────────────────────────────────────────────


def _make_typer_plugin() -> typer.Typer:
    """Trivial Typer plugin used across the CLI tests."""
    plug = typer.Typer(name="my_typer_plugin", help="A test Typer plugin.")

    @plug.command("ping")
    def _ping() -> None:
        """Print 'typer-plugin-pong'."""
        typer.echo("typer-plugin-pong")

    return plug


@click.command("my_click_plugin")
def _click_plugin() -> None:
    """A test Click plugin."""
    click.echo("click-plugin-pong")


def test_discover_cli_plugins_returns_loaded_objects() -> None:
    plug = _make_typer_plugin()
    eps = [_FakeEntryPoint("my_typer_plugin", plug)]
    with _patch_entry_points({CLI_ENTRY_POINT_GROUP: eps}):
        result = discover_cli_plugins()
    assert set(result) == {"my_typer_plugin"}
    assert result["my_typer_plugin"] is plug


def test_ext_subapp_invokes_typer_plugin() -> None:
    """End-to-end: register a Typer plugin and invoke it under
    ``gauntlet ext my_typer_plugin ping``."""
    from gauntlet import cli

    plug = _make_typer_plugin()
    eps = [_FakeEntryPoint("my_typer_plugin", plug)]
    runner = CliRunner()
    with _patch_entry_points({CLI_ENTRY_POINT_GROUP: eps}):
        # Invalidate any prior registration record so the helper picks
        # up the freshly-mocked entry-point table.
        cli._registered_cli_plugins.discard("my_typer_plugin")
        cli._register_cli_plugins()
        result = runner.invoke(cli.app, ["ext", "my_typer_plugin", "ping"])
    assert result.exit_code == 0, result.stderr
    assert "typer-plugin-pong" in result.stdout


def test_ext_subapp_invokes_click_plugin() -> None:
    """End-to-end: register a Click plugin and invoke it under
    ``gauntlet ext my_click_plugin``. The Typer ↔ Click bridge wraps
    the click command in a passthrough Typer command."""
    from gauntlet import cli

    eps = [_FakeEntryPoint("my_click_plugin", _click_plugin)]
    runner = CliRunner()
    with _patch_entry_points({CLI_ENTRY_POINT_GROUP: eps}):
        cli._registered_cli_plugins.discard("my_click_plugin")
        cli._register_cli_plugins()
        result = runner.invoke(cli.app, ["ext", "my_click_plugin"])
    assert result.exit_code == 0, result.stderr
    assert "click-plugin-pong" in result.stdout


def test_ext_subapp_help_lists_no_plugins_when_none_installed() -> None:
    """``gauntlet ext --help`` succeeds even with zero plugins
    installed — the sub-app is mounted unconditionally so the namespace
    is discoverable."""
    from gauntlet.cli import app

    runner = CliRunner()
    with _patch_entry_points({CLI_ENTRY_POINT_GROUP: []}):
        result = runner.invoke(app, ["ext", "--help"])
    assert result.exit_code == 0
    assert "Third-party CLI extensions" in result.stdout


# ──────────────────────────────────────────────────────────────────────
# Cross-group hygiene — every discover_* helper handles failures the
# same way (regression guard for the shared _load_entry_points helper).
# ──────────────────────────────────────────────────────────────────────


def test_every_discover_helper_drops_failed_plugin() -> None:
    """Each new discover_* helper inherits :func:`_load_entry_points`'s
    failure-swallow behaviour. One regression here would imply the
    shared helper drifted."""
    bad = _FakeEntryPoint("bad", ImportError("nope"), dist_name="boom-pkg")
    helpers_and_groups = [
        (discover_axis_plugins, AXIS_ENTRY_POINT_GROUP),
        (discover_sampler_plugins, SAMPLER_ENTRY_POINT_GROUP),
        (discover_sink_plugins, SINK_ENTRY_POINT_GROUP),
        (discover_cli_plugins, CLI_ENTRY_POINT_GROUP),
    ]
    for helper, group in helpers_and_groups:
        helper.cache_clear()
        with (
            _patch_entry_points({group: [bad]}),
            pytest.warns(RuntimeWarning, match=r"failed to load"),
        ):
            assert helper() == {}


def test_every_discover_helper_warns_on_duplicate_name() -> None:
    """Duplicate entry-point names emit one RuntimeWarning naming both
    distributions; first-seen wins."""

    def _factory() -> PerturbationAxis:  # pragma: no cover - not invoked
        return _make_plugin_axis()

    helpers_and_groups = [
        (discover_axis_plugins, AXIS_ENTRY_POINT_GROUP),
        (discover_sampler_plugins, SAMPLER_ENTRY_POINT_GROUP),
        (discover_sink_plugins, SINK_ENTRY_POINT_GROUP),
        (discover_cli_plugins, CLI_ENTRY_POINT_GROUP),
    ]
    for helper, group in helpers_and_groups:
        helper.cache_clear()
        first = _FakeEntryPoint("dup", _factory, dist_name="first-pkg")
        second = _FakeEntryPoint("dup", _factory, dist_name="second-pkg")
        with (
            _patch_entry_points({group: [first, second]}),
            pytest.warns(
                RuntimeWarning,
                match=r"registered by both 'first-pkg' and 'second-pkg'",
            ),
        ):
            result = helper()
        assert set(result) == {"dup"}


def test_dogfood_axes_discovered_after_install() -> None:
    """``[project.entry-points."gauntlet.axes"]`` registers gauntlet's
    own torch-free axes. After ``uv pip install -e .`` they must
    surface through :func:`discover_axis_plugins` — symmetric to the
    policies / envs dogfood smoke tests in :mod:`tests.test_plugins`.
    """
    discover_axis_plugins.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plugins = discover_axis_plugins()
    assert {"lighting_intensity", "distractor_count"} <= set(plugins), (
        f"Dogfood axis entry points missing — got {sorted(plugins)}. "
        "Did you re-install after pyproject.toml changes? "
        "Try: uv pip install -e ."
    )


def test_dogfood_samplers_discovered_after_install() -> None:
    """The first-party ``cartesian`` sampler entry point must
    materialise. LHS / Sobol / adversarial are NOT dogfooded (LHS /
    Sobol could be added later; adversarial requires a non-zero-arg
    constructor and is out of scope for the plugin contract)."""
    discover_sampler_plugins.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plugins = discover_sampler_plugins()
    assert "cartesian" in plugins, (
        f"Dogfood sampler entry point missing — got {sorted(plugins)}. "
        "Did you re-install after pyproject.toml changes? "
        "Try: uv pip install -e ."
    )


def test_every_discover_helper_caches_result() -> None:
    """A second call without :meth:`cache_clear` does not re-invoke
    :func:`importlib.metadata.entry_points`."""
    helpers_and_groups = [
        (discover_axis_plugins, AXIS_ENTRY_POINT_GROUP),
        (discover_sampler_plugins, SAMPLER_ENTRY_POINT_GROUP),
        (discover_sink_plugins, SINK_ENTRY_POINT_GROUP),
        (discover_cli_plugins, CLI_ENTRY_POINT_GROUP),
    ]
    for helper, group in helpers_and_groups:
        helper.cache_clear()
        with (
            _patch_entry_points({group: []}) as mocked,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            first = helper()
            second = helper()
            assert first is second
            assert mocked.call_count == 1
