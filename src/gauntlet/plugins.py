"""Entry-point discovery for third-party gauntlet extensions.

See ``docs/polish-exploration-plugin-system.md`` and
``docs/extension-points.md`` for the full design; quick recap:

* Six ``[project.entry-points]`` groups are read:

    - ``gauntlet.policies`` — :class:`gauntlet.policy.base.Policy` classes.
    - ``gauntlet.envs``     — :class:`gauntlet.env.base.GauntletEnv` classes.
    - ``gauntlet.axes``     — zero-arg callables returning a
      :class:`gauntlet.env.perturbation.PerturbationAxis`.
    - ``gauntlet.samplers`` — :class:`gauntlet.suite.sampling.Sampler`
      classes (zero-arg constructible).
    - ``gauntlet.sinks``    — :class:`gauntlet.runner.sinks.EpisodeSink`
      classes accepting the standard ``run_name`` / ``suite_name`` /
      ``config`` constructor kwargs.
    - ``gauntlet.cli``      — Typer or Click command objects exposed
      under ``gauntlet ext <name>``.

* Discovery is cached via :func:`functools.lru_cache` — the entry-point
  table is fixed for the lifetime of the process under normal use.
  Tests that mock :func:`importlib.metadata.entry_points` must call
  ``cache_clear()`` on every ``discover_*_plugins`` helper they touch
  in fixture teardown.
* ``ep.load()`` failures are wrapped in a clear :class:`RuntimeWarning`
  and the offending entry point is dropped from the returned dict — the
  rest of gauntlet keeps working.

This module deliberately does NOT consult the built-in registry tables.
Built-in-vs-plugin precedence (built-ins always win on collision) is
enforced one layer up — see :mod:`gauntlet.policy.registry`,
:mod:`gauntlet.env.registry`, :mod:`gauntlet.suite.sampling`,
:mod:`gauntlet.env.perturbation`, and the CLI ``ext`` group resolver.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any, cast

from gauntlet.env.base import GauntletEnv
from gauntlet.env.perturbation.base import PerturbationAxis
from gauntlet.policy.base import Policy
from gauntlet.runner.sinks import EpisodeSink
from gauntlet.suite.sampling import Sampler

__all__ = [
    "AXIS_ENTRY_POINT_GROUP",
    "CLI_ENTRY_POINT_GROUP",
    "ENV_ENTRY_POINT_GROUP",
    "POLICY_ENTRY_POINT_GROUP",
    "SAMPLER_ENTRY_POINT_GROUP",
    "SINK_ENTRY_POINT_GROUP",
    "discover_axis_plugins",
    "discover_cli_plugins",
    "discover_env_plugins",
    "discover_policy_plugins",
    "discover_sampler_plugins",
    "discover_sink_plugins",
    "warn_on_collision",
]

POLICY_ENTRY_POINT_GROUP = "gauntlet.policies"
ENV_ENTRY_POINT_GROUP = "gauntlet.envs"
AXIS_ENTRY_POINT_GROUP = "gauntlet.axes"
SAMPLER_ENTRY_POINT_GROUP = "gauntlet.samplers"
SINK_ENTRY_POINT_GROUP = "gauntlet.sinks"
CLI_ENTRY_POINT_GROUP = "gauntlet.cli"


def _entry_point_dist_name(ep: EntryPoint) -> str:
    """Return the distribution name owning ``ep``, or ``'<unknown>'``.

    ``EntryPoint.dist`` is populated when the entry point comes from
    an installed distribution — the normal case. Hand-built EntryPoint
    instances (e.g. in tests) may leave it ``None``; we fall back to
    a placeholder so error messages never crash on an attribute lookup.
    """
    dist = getattr(ep, "dist", None)
    if dist is None:
        return "<unknown>"
    name = getattr(dist, "name", None)
    if isinstance(name, str) and name:
        return name
    return "<unknown>"


def _load_entry_points(group: str) -> dict[str, Any]:
    """Discover and load every entry point in ``group``.

    Returns a ``name -> loaded-object`` dict. Entry points that raise
    on :meth:`EntryPoint.load` are wrapped in a clear
    :class:`RuntimeWarning` (so the failure is visible) and dropped from
    the returned dict. Gauntlet itself stays operational either way.

    Duplicate names within ``group`` (rare — would require two different
    distributions both registering the same name) keep the first-seen
    entry; subsequent duplicates emit a :class:`RuntimeWarning` naming
    both distributions so the user can resolve the conflict by
    uninstalling one.
    """
    discovered: dict[str, Any] = {}
    seen_dists: dict[str, str] = {}
    for ep in entry_points(group=group):
        if ep.name in discovered:
            other = seen_dists.get(ep.name, "<unknown>")
            warnings.warn(
                f"gauntlet plugin name {ep.name!r} is registered by both "
                f"{other!r} and {_entry_point_dist_name(ep)!r}; using the "
                f"first ({other!r}). Uninstall one to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # wrap *any* plugin failure
            warnings.warn(
                f"Plugin {ep.name!r} from {_entry_point_dist_name(ep)!r} "
                f"failed to load: {exc!r}. Dropping from registry.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        discovered[ep.name] = obj
        seen_dists[ep.name] = _entry_point_dist_name(ep)
    return discovered


@lru_cache(maxsize=1)
def discover_policy_plugins() -> dict[str, type[Policy]]:
    """Return ``name -> Policy class`` for every installed policy plugin.

    Reads the ``gauntlet.policies`` entry-point group. Failed entry
    points are skipped with a :class:`RuntimeWarning`; the returned
    dict only contains successfully-loaded plugins.

    The result is cached for the lifetime of the process. Tests that
    inject fake entry points via :func:`unittest.mock.patch` on
    :func:`importlib.metadata.entry_points` must call
    :meth:`discover_policy_plugins.cache_clear` in fixture teardown
    to avoid leaking the mocked result into other tests.
    """
    raw = _load_entry_points(POLICY_ENTRY_POINT_GROUP)
    # We do not runtime-check that each loaded object satisfies the
    # ``Policy`` Protocol — that would require constructing an instance,
    # which is exactly the kind of side effect entry-point discovery is
    # meant to defer. The structural check happens at first
    # ``policy.act(obs)`` inside the Runner. The cast widens the
    # ``Any`` table to the documented public type.
    return cast(dict[str, type[Policy]], raw)


@lru_cache(maxsize=1)
def discover_env_plugins() -> dict[str, type[GauntletEnv]]:
    """Return ``name -> GauntletEnv class`` for every installed env plugin.

    Reads the ``gauntlet.envs`` entry-point group. Failed entry points
    are skipped with a :class:`RuntimeWarning`; the returned dict only
    contains successfully-loaded plugins.

    See :func:`discover_policy_plugins` for cache-invalidation guidance.
    """
    raw = _load_entry_points(ENV_ENTRY_POINT_GROUP)
    # See discover_policy_plugins() for the cast rationale.
    return cast(dict[str, type[GauntletEnv]], raw)


@lru_cache(maxsize=1)
def discover_axis_plugins() -> dict[str, Callable[[], PerturbationAxis]]:
    """Return ``name -> zero-arg axis factory`` for every installed axis plugin.

    Reads the ``gauntlet.axes`` entry-point group. Each entry must be a
    zero-arg callable that returns a fresh
    :class:`gauntlet.env.perturbation.PerturbationAxis` — same shape as
    the built-in factory functions in
    :mod:`gauntlet.env.perturbation.axes` (e.g. :func:`lighting_intensity`,
    :func:`distractor_count`). The :attr:`PerturbationAxis.name` returned
    by the callable should match the entry-point key — :func:`axis_for`
    validates the match at first lookup and surfaces a clear error on
    drift. We do NOT call the factory at discovery time; the structural
    check happens on first :func:`axis_for` lookup.

    Plugin axis authors must also ship a backend env via the
    ``gauntlet.envs`` group whose
    :meth:`gauntlet.env.base.GauntletEnv.set_perturbation` recognises
    the axis name — the built-in tabletop backends gate on a
    backend-private ``AXIS_NAMES`` ClassVar set and raise on unknown
    names. See ``docs/extension-points.md`` for the full recipe.

    Failed entry points are skipped with a :class:`RuntimeWarning`; the
    returned dict only contains successfully-loaded plugins. See
    :func:`discover_policy_plugins` for cache-invalidation guidance.
    """
    raw = _load_entry_points(AXIS_ENTRY_POINT_GROUP)
    # See discover_policy_plugins() for the cast rationale; runtime
    # signature is checked at first ``axis_for`` call.
    return cast(dict[str, Callable[[], PerturbationAxis]], raw)


@lru_cache(maxsize=1)
def discover_sampler_plugins() -> dict[str, type[Sampler]]:
    """Return ``name -> Sampler class`` for every installed sampler plugin.

    Reads the ``gauntlet.samplers`` entry-point group. Each entry must
    be a zero-arg constructible class implementing the
    :class:`gauntlet.suite.sampling.Sampler` Protocol — the same
    contract :class:`gauntlet.suite.sampling.CartesianSampler` and the
    built-in :class:`gauntlet.suite.lhs.LatinHypercubeSampler` /
    :class:`gauntlet.suite.sobol.SobolSampler` /
    :class:`gauntlet.suite.adversarial.AdversarialSampler` satisfy
    today (the ``adversarial`` built-in is the one exception — its
    constructor takes a pilot Report — and is not exposed through this
    plugin path; plugin samplers must be zero-arg constructible).

    The structural ``sample(suite, rng) -> list[SuiteCell]`` check
    happens on first :func:`gauntlet.suite.sampling.build_sampler`
    dispatch, not at discovery time.

    Failed entry points are skipped with a :class:`RuntimeWarning`; the
    returned dict only contains successfully-loaded plugins. See
    :func:`discover_policy_plugins` for cache-invalidation guidance.
    """
    raw = _load_entry_points(SAMPLER_ENTRY_POINT_GROUP)
    # See discover_policy_plugins() for the cast rationale.
    return cast(dict[str, type[Sampler]], raw)


@lru_cache(maxsize=1)
def discover_sink_plugins() -> dict[str, type[EpisodeSink]]:
    """Return ``name -> EpisodeSink class`` for every installed sink plugin.

    Reads the ``gauntlet.sinks`` entry-point group. Each entry must be a
    class implementing the :class:`gauntlet.runner.sinks.EpisodeSink`
    Protocol whose constructor accepts the standard kwargs
    ``run_name: str``, ``suite_name: str``, and
    ``config: dict[str, Any] | None`` — same contract the built-in
    :class:`gauntlet.runner.sinks.WandbSink` and
    :class:`gauntlet.runner.sinks.MlflowSink` satisfy. Plugin sinks are
    instantiated with these three kwargs by the CLI ``--sink`` plumbing
    in :mod:`gauntlet.cli`; sinks that need richer configuration should
    pull from environment variables or the ``config`` mapping rather
    than introduce a custom constructor signature.

    The structural ``log_episode`` / ``close`` check happens on first
    invocation, not at discovery time. Failed entry points are skipped
    with a :class:`RuntimeWarning`; the returned dict only contains
    successfully-loaded plugins. See :func:`discover_policy_plugins` for
    cache-invalidation guidance.
    """
    raw = _load_entry_points(SINK_ENTRY_POINT_GROUP)
    # See discover_policy_plugins() for the cast rationale.
    return cast(dict[str, type[EpisodeSink]], raw)


@lru_cache(maxsize=1)
def discover_cli_plugins() -> dict[str, Any]:
    """Return ``name -> command object`` for every installed CLI plugin.

    Reads the ``gauntlet.cli`` entry-point group. Each entry must be one
    of:

    * a :class:`typer.Typer` sub-app (preferred — gauntlet's own CLI is
      Typer-based, so a plugin Typer composes naturally);
    * a :class:`click.Command` or :class:`click.Group` (gauntlet adapts
      these via :func:`typer.main.get_command` at registration time —
      Typer is Click underneath, so the bridge is one-line and stable).

    Plugin commands are added to the top-level CLI under the
    ``gauntlet ext`` namespace (e.g. an entry point named ``mycommand``
    becomes ``gauntlet ext mycommand``); namespacing keeps the
    first-party command surface clean and avoids the silent shadowing
    footgun a plugin would otherwise enable. The return type is
    :class:`typing.Any` because Typer / Click commands have no shared
    runtime type — the registration shim in :mod:`gauntlet.cli` does
    the duck-type dispatch.

    Failed entry points are skipped with a :class:`RuntimeWarning`; the
    returned dict only contains successfully-loaded plugins. See
    :func:`discover_policy_plugins` for cache-invalidation guidance.
    """
    return _load_entry_points(CLI_ENTRY_POINT_GROUP)


def warn_on_collision(
    *,
    name: str,
    group: str,
    builtin_obj: object,
    plugin_obj: object,
    plugin_dist: str,
) -> None:
    """Emit a :class:`RuntimeWarning` if a plugin shadows a built-in.

    Called by the policy / env registry resolvers when a plugin entry
    point and a built-in name coincide. The check is **identity-based**
    (``is``) — gauntlet's own dogfooded entry points load the same class
    object the built-in table already holds, so they never warn. A real
    third-party shadow loads a different class and triggers the warning.
    """
    if plugin_obj is builtin_obj:
        return
    warnings.warn(
        f"plugin {name!r} from {plugin_dist!r} (group {group!r}) shadows the "
        f"built-in {name!r}; using the built-in. Rename the plugin's entry "
        f"point or uninstall it to silence this warning.",
        RuntimeWarning,
        stacklevel=3,
    )
