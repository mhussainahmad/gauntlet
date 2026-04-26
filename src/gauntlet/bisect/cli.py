"""Typer subcommand for ``gauntlet bisect`` — B-39.

Thin wrapper over :func:`gauntlet.bisect.bisect.bisect`. The CLI's
job is purely shape-shifting:

* parse ``--good`` / ``--bad`` / ``--intermediate`` (repeatable) into
  the engine's ``ckpt_list``;
* turn each checkpoint id into a zero-arg policy factory via
  :func:`gauntlet.policy.registry.resolve_policy_factory` (so the
  bisect treats every checkpoint as a full ``--policy`` spec --
  see :mod:`gauntlet.bisect.bisect` docstring's anti-feature note);
* load the suite from the YAML path;
* construct a :class:`functools.partial` over :class:`Runner` baking
  in the requested ``n_workers`` / ``cache_dir`` / ``policy_id`` /
  ``max_steps`` so the engine's ``runner_factory`` is a clean
  zero-arg callable;
* drive the engine and write the result as JSON to ``--output``.

The subcommand is registered against the top-level :data:`gauntlet.cli.app`
from inside :mod:`gauntlet.cli` (mirroring the ``compare`` / ``diff``
registration pattern -- both live as `@app.command(...)` siblings).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from gauntlet.bisect.bisect import (
    BisectError,
    BisectResult,
    bisect,
)
from gauntlet.diff.paired import PairingError
from gauntlet.policy.base import Policy
from gauntlet.policy.registry import PolicySpecError, resolve_policy_factory
from gauntlet.runner import Runner
from gauntlet.suite import load_suite

__all__ = ["register"]


def _default_resolver(spec: str) -> Callable[[], Policy]:
    """Module-level dispatch to :func:`resolve_policy_factory`.

    Re-binding this function via :func:`unittest.mock.patch` (or
    :func:`register`'s ``spec_to_factory`` override) is how tests
    swap in a fake resolver. We dispatch through the module-level
    name (rather than capturing :func:`resolve_policy_factory` in a
    closure) so the patch site is the one the inner CLI body
    actually consults at invocation time.
    """
    return resolve_policy_factory(spec)


def _build_runner_factory(
    *,
    n_workers: int,
    cache_dir: Path | None,
    max_steps: int | None,
    policy_id_for_cache: str | None,
) -> Callable[[], Runner]:
    """Bake the long-lived Runner kwargs into a zero-arg factory.

    The bisect engine wants a zero-arg callable so it can construct
    a fresh :class:`Runner` per candidate. We pre-resolve every kwarg
    here so the factory is pure capture-by-closure -- the engine
    never sees ``n_workers`` / ``cache_dir`` etc. directly. When
    ``cache_dir`` is set the caller MUST pass ``max_steps`` because
    the cache key contract (B-40) depends on it; the Runner enforces
    this at construction time and we let that error propagate.

    Note that ``policy_id`` here is overridden per-candidate by the
    bisect engine via the resolver result -- this default is only
    used when the engine is invoked outside the bisect (defence in
    depth; the per-candidate override always wins).
    """

    def factory() -> Runner:
        return Runner(
            n_workers=n_workers,
            cache_dir=cache_dir,
            max_steps=max_steps,
            policy_id=policy_id_for_cache,
        )

    return factory


def _resolve_ckpt_list(good: str, intermediates: list[str], bad: str) -> list[str]:
    """Assemble ``[good, *intermediates, bad]`` -- the engine's input shape.

    No deduplication, no reordering: the user's order is the
    bisect's order. Duplicates are accepted (the engine handles
    them). An empty intermediates list is fine -- the engine then
    runs only the two anchors and reports ``first_bad = bad``
    if the bad anchor regressed against the good anchor at the
    paired-CRN confidence level.
    """
    return [good, *intermediates, bad]


def _write_result(out: Path, result: BisectResult) -> None:
    """Serialise the :class:`BisectResult` and write to ``out`` as UTF-8 JSON.

    Mirrors :func:`gauntlet.cli._write_json` -- pydantic
    ``model_dump(mode='json')`` then ``json.dumps(..., allow_nan=False)``
    so the resulting file always parses with the strict-mode
    :func:`json.loads`.
    """
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def register(
    app: typer.Typer,
    *,
    spec_to_factory: Callable[[str], Callable[[], Policy]] | None = None,
) -> None:
    """Attach the ``gauntlet bisect`` subcommand to ``app``.

    Args:
        app: The top-level :class:`typer.Typer` from :mod:`gauntlet.cli`.
        spec_to_factory: Optional override for the policy-spec
            resolver. When ``None`` (the default) the CLI dispatches
            via :func:`gauntlet.policy.registry.resolve_policy_factory`,
            which accepts ``"random"`` / ``"scripted"`` /
            ``"module.path:attr"`` / plugin names. Tests inject a
            stub resolver that maps fake checkpoint ids to synthetic
            policies so the runner stays MuJoCo-free.
    """
    # Capture the explicit override (if any) at register time. When
    # ``spec_to_factory is None`` we keep ``override`` as ``None`` and
    # fall back to the module-level :func:`_default_resolver` at
    # invocation time -- that indirection is what lets
    # :func:`unittest.mock.patch` swap the resolver in tests without
    # touching :func:`register`.
    override = spec_to_factory

    @app.command("bisect")
    def bisect_cmd(
        good: Annotated[
            str,
            typer.Option(
                "--good",
                help=(
                    "Known-good checkpoint id. Resolved as a --policy spec "
                    "('random', 'scripted', 'module.path:attr', or a "
                    "registered plugin name) -- gauntlet does not load "
                    "raw weight files itself; the user supplies a factory "
                    "that knows how. See the B-39 backlog entry for the "
                    "anti-feature framing."
                ),
            ),
        ],
        bad: Annotated[
            str,
            typer.Option(
                "--bad",
                help="Known-bad checkpoint id. Same resolution as --good.",
            ),
        ],
        suite_path: Annotated[
            Path,
            typer.Option(
                "--suite",
                help="Path to a suite YAML file.",
                exists=False,  # checked manually for a friendlier message
                dir_okay=False,
                file_okay=True,
            ),
        ],
        target_cell: Annotated[
            int,
            typer.Option(
                "--target-cell",
                help=(
                    "The SuiteCell.index to bisect against. Per the B-39 "
                    "spec the bisect tracks ONE failing cell; multi-cell "
                    "bisection is out of scope."
                ),
                min=0,
            ),
        ],
        out: Annotated[
            Path,
            typer.Option(
                "--output",
                "-o",
                help="Output JSON path. Parent directory is created if missing.",
            ),
        ],
        intermediate: Annotated[
            list[str] | None,
            typer.Option(
                "--intermediate",
                help=(
                    "Intermediate checkpoint id between --good and --bad. "
                    "Repeatable. Resolution and ordering match --good. "
                    "Empty (default) means the bisect runs only the two "
                    "anchors -- correct but uninformative; pass at least "
                    "one intermediate to actually binary-search."
                ),
            ),
        ] = None,
        episodes_per_step: Annotated[
            int | None,
            typer.Option(
                "--episodes-per-step",
                help=(
                    "Override the suite's episodes_per_cell for every "
                    "bisect step. Default (unset) re-uses the suite's own "
                    "value -- the user already calibrated it for a single-"
                    "policy run, so the default is sensible."
                ),
                min=1,
            ),
        ] = None,
        n_workers: Annotated[
            int,
            typer.Option(
                "--n-workers",
                "-w",
                help="Number of worker processes per Runner (>=1).",
                min=1,
            ),
        ] = 1,
        cache_dir: Annotated[
            Path | None,
            typer.Option(
                "--cache-dir",
                help=(
                    "Directory for the per-Episode rollout cache (B-40). "
                    "When set, every (suite, axis_config, env_seed, "
                    "policy_id, max_steps) cell is looked up before "
                    "dispatch. Each candidate uses its checkpoint id as "
                    "policy_id so paired runs against the same checkpoint "
                    "cache-hit. Requires --max-steps."
                ),
            ),
        ] = None,
        max_steps: Annotated[
            int | None,
            typer.Option(
                "--max-steps",
                help=(
                    "Per-episode env step cap baked into the cache key. "
                    "Required when --cache-dir is set; ignored otherwise."
                ),
                min=1,
            ),
        ] = None,
    ) -> None:
        """Binary-search a checkpoint list for the first regression at TARGET-CELL."""
        if not suite_path.is_file():
            raise typer.BadParameter(f"suite file not found: {suite_path}")

        try:
            suite = load_suite(suite_path)
        except (ValidationError, ValueError) as exc:
            raise typer.BadParameter(f"{suite_path}: invalid suite YAML: {exc}") from exc
        except OSError as exc:
            raise typer.BadParameter(f"{suite_path}: could not read file: {exc}") from exc

        intermediates = list(intermediate or [])
        ckpt_list = _resolve_ckpt_list(good, intermediates, bad)

        # Per-checkpoint policy factory. Resolve via the override
        # captured at register time, falling back to
        # :func:`_default_resolver` (which dispatches through the
        # module-level :func:`resolve_policy_factory` import). Wrap
        # so an unknown spec produces a clean BadParameter rather
        # than a bare PolicySpecError leaking through.
        active_resolver = override if override is not None else _default_resolver

        def policy_factory_resolver(spec: str) -> Callable[[], Policy]:
            try:
                return active_resolver(spec)
            except PolicySpecError as exc:
                raise typer.BadParameter(f"checkpoint {spec!r}: {exc}") from exc

        runner_factory = _build_runner_factory(
            n_workers=n_workers,
            cache_dir=cache_dir,
            max_steps=max_steps,
            # Defence in depth: the bisect engine never re-uses this
            # default (it always passes a per-candidate factory built
            # by the resolver), but if a future caller invokes the
            # runner directly they should still get a meaningful
            # cache key. Use the good anchor's id as a stand-in.
            policy_id_for_cache=good,
        )

        try:
            result = bisect(
                ckpt_list=ckpt_list,
                policy_factory_resolver=policy_factory_resolver,
                suite=suite,
                target_cell_id=target_cell,
                runner_factory=runner_factory,
                episodes_per_step=episodes_per_step,
            )
        except BisectError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except PairingError as exc:
            # CRN violation -- runner-level bug, not a user error;
            # surface as exit-1 with the engine's diagnostic verbatim
            # (the helper already names the (cell, episode) coordinates
            # at which seed-derivation diverged).
            typer.echo(f"error: bisect: paired-CRN violation: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        _write_result(out, result)
        typer.echo(
            f"bisect: first_bad={result.first_bad} "
            f"target_cell_delta={result.target_cell_delta:+.3f} "
            f"[{result.target_cell_delta_ci_low:+.3f}, "
            f"{result.target_cell_delta_ci_high:+.3f}] "
            f"steps={len(result.steps)} -> {out}",
            err=True,
        )
