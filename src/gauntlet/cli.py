"""User-facing CLI — see ``GAUNTLET_SPEC.md`` §5 task 9.

Three subcommands wire the rest of the harness end-to-end:

* :func:`run` — load a suite YAML, resolve a ``--policy``, drive the
  :class:`gauntlet.runner.Runner`, and emit ``episodes.json`` /
  ``report.json`` / ``report.html`` into ``--out``.
* :func:`report` — re-render an HTML report from either an
  ``episodes.json`` (rebuilds the Report) or a pre-built
  ``report.json`` (loads as-is). Auto-detects by peeking at the JSON
  top-level type (``list`` vs ``dict``).
* :func:`compare` — diff two runs and emit ``compare.json`` plus a
  short stderr summary. The HTML companion is deferred to Phase 2;
  Phase 1 callers script off the JSON.

Stdout is reserved for machine-readable output (none in Phase 1).
Status messages and errors all go to stderr via
:func:`typer.echo(..., err=True)`. Every error path raises
:class:`typer.Exit` with a non-zero code so wrappers (Make, CI) can key
off exit status.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from pydantic import ValidationError

from gauntlet.env.base import GauntletEnv
from gauntlet.env.tabletop import TabletopEnv
from gauntlet.policy.registry import PolicySpecError, resolve_policy_factory
from gauntlet.report import Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.suite import load_suite

__all__ = ["app"]


app = typer.Typer(
    name="gauntlet",
    help="Evaluation harness for learned robot policies.",
    no_args_is_help=True,
    add_completion=False,
)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers (kept module-level so they pickle under spawn).
# ──────────────────────────────────────────────────────────────────────


def _echo_err(msg: str) -> None:
    """Write *msg* to stderr; thin wrapper to keep call sites short."""
    typer.echo(msg, err=True)


def _fail(msg: str, *, code: int = 1) -> typer.Exit:
    """Emit an error message and return a ``typer.Exit`` to raise."""
    _echo_err(f"error: {msg}")
    return typer.Exit(code=code)


def _read_json(path: Path) -> Any:
    """Load JSON from *path* with a uniform error envelope.

    Returns the parsed JSON tree (``Any``); the auto-detect logic in
    :func:`_load_report_or_episodes` decides whether it is a Report or
    an Episode list.
    """
    if not path.is_file():
        raise _fail(f"file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise _fail(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc


def _load_report_or_episodes(path: Path) -> Report:
    """Auto-detect ``episodes.json`` vs ``report.json`` and return a Report.

    Top-level ``list`` → episodes (rebuild via :func:`build_report`);
    top-level ``dict`` → assume it is a serialized :class:`Report`.
    Anything else is an error.
    """
    raw = _read_json(path)
    if isinstance(raw, list):
        episodes = _episodes_from_dicts(raw, source=path)
        try:
            return build_report(episodes)
        except ValueError as exc:
            raise _fail(f"{path}: cannot build report: {exc}") from exc
    if isinstance(raw, dict):
        try:
            return Report.model_validate(raw)
        except ValidationError as exc:
            raise _fail(f"{path}: not a valid report.json: {exc}") from exc
    raise _fail(
        f"{path}: top-level JSON must be a list (episodes) or dict (report); "
        f"got {type(raw).__name__}"
    )


def _episodes_from_dicts(raw: list[Any], *, source: Path) -> list[Episode]:
    """Validate a list of episode dicts, re-raising as a clean CLI error."""
    episodes: list[Episode] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _fail(f"{source}: episode {i} is not an object (got {type(item).__name__})")
        try:
            episodes.append(Episode.model_validate(item))
        except ValidationError as exc:
            raise _fail(f"{source}: episode {i} failed validation: {exc}") from exc
    return episodes


def _write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* as UTF-8 JSON, NaN-safe.

    All floats are passed through :func:`_nan_to_none` (lifted from
    :mod:`gauntlet.report.html`) so the resulting file always parses
    with the strict-mode :func:`json.loads` (no ``allow_nan``).
    """
    cleaned = _nan_to_none(payload)
    path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _episodes_to_dicts(episodes: list[Episode]) -> list[dict[str, Any]]:
    """Serialise episodes via Pydantic's JSON mode for round-tripping."""
    return [ep.model_dump(mode="json") for ep in episodes]


def _make_env_factory(env_max_steps: int | None) -> Callable[[], GauntletEnv] | None:
    """Build an env factory honouring the hidden ``--env-max-steps`` knob.

    Returns ``None`` when the user didn't override max_steps so the
    Runner uses its own default. ``functools.partial`` over the class
    pickles cleanly under ``spawn`` — important even though the test
    suite only exercises ``-w 1``.

    Return type is widened to ``GauntletEnv`` (the Runner's Protocol-typed
    factory signature) via ``cast``. Subpackage-specific backends flow
    through the same seam in later steps.
    """
    if env_max_steps is None:
        return None
    return cast(
        "Callable[[], GauntletEnv]",
        partial(TabletopEnv, max_steps=env_max_steps),
    )


# ──────────────────────────────────────────────────────────────────────
# `run` subcommand.
# ──────────────────────────────────────────────────────────────────────


@app.command("run")
def run(
    suite_path: Annotated[
        Path,
        typer.Argument(
            help="Path to a suite YAML file.",
            exists=False,  # checked manually for a friendlier message
            dir_okay=False,
            file_okay=True,
        ),
    ],
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Policy spec: 'random', 'scripted', or 'module.path:attr'.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory; created if missing.",
        ),
    ],
    n_workers: Annotated[
        int,
        typer.Option(
            "--n-workers",
            "-w",
            min=1,
            help="Number of worker processes (>=1).",
        ),
    ] = 1,
    seed_override: Annotated[
        int | None,
        typer.Option(
            "--seed-override",
            help="Override the suite's master seed with this value.",
        ),
    ] = None,
    no_html: Annotated[
        bool,
        typer.Option(
            "--no-html",
            help="Skip rendering report.html; emit JSON artefacts only.",
        ),
    ] = False,
    env_max_steps: Annotated[
        int | None,
        typer.Option(
            "--env-max-steps",
            help="(Hidden / test hook) Override TabletopEnv max_steps for fast tests.",
            hidden=True,
            min=1,
        ),
    ] = None,
) -> None:
    """Execute a suite and write episodes / report artefacts to ``--out``."""
    if not suite_path.is_file():
        raise _fail(f"suite file not found: {suite_path}")

    try:
        suite = load_suite(suite_path)
    except (ValidationError, ValueError) as exc:
        raise _fail(f"{suite_path}: invalid suite YAML: {exc}") from exc
    except OSError as exc:
        raise _fail(f"{suite_path}: could not read file: {exc}") from exc

    if seed_override is not None:
        # Pydantic models are immutable-ish; build a copy with the override.
        suite = suite.model_copy(update={"seed": seed_override})

    try:
        policy_factory = resolve_policy_factory(policy)
    except PolicySpecError as exc:
        raise _fail(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)

    env_factory = _make_env_factory(env_max_steps)
    runner = Runner(n_workers=n_workers, env_factory=env_factory)

    try:
        episodes = runner.run(policy_factory=policy_factory, suite=suite)
    except ValueError as exc:
        raise _fail(f"runner failed: {exc}") from exc

    try:
        report = build_report(episodes)
    except ValueError as exc:
        raise _fail(f"could not build report: {exc}") from exc

    episodes_path = out / "episodes.json"
    report_json_path = out / "report.json"
    report_html_path = out / "report.html"

    _write_json(episodes_path, _episodes_to_dicts(episodes))
    _write_json(report_json_path, report.model_dump(mode="json"))

    if not no_html:
        write_html(report, report_html_path)

    summary = (
        f"Wrote {len(episodes)} episodes / {len(report.per_cell)} cells "
        f"-> {out} (success: {report.overall_success_rate * 100:.1f}%)"
    )
    _echo_err(summary)


# ──────────────────────────────────────────────────────────────────────
# `report` subcommand.
# ──────────────────────────────────────────────────────────────────────


@app.command("report")
def report(
    results_path: Annotated[
        Path,
        typer.Argument(
            help="Path to either an episodes.json or a report.json file.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output HTML path; defaults to 'report.html' next to the input.",
        ),
    ] = None,
) -> None:
    """Render an HTML report from an episodes.json or a report.json file."""
    rep = _load_report_or_episodes(results_path)

    if out is None:
        out_path = results_path.parent / "report.html"
    else:
        out_path = out
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)

    write_html(rep, out_path)
    _echo_err(f"Wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────
# `compare` subcommand.
# ──────────────────────────────────────────────────────────────────────


def _cell_key(perturbation_config: dict[str, float]) -> frozenset[tuple[str, float]]:
    """Stable hashable identity for a per-cell perturbation config."""
    return frozenset(perturbation_config.items())


def _build_compare(
    report_a: Report,
    report_b: Report,
    *,
    threshold: float,
    min_cell_size: int,
) -> dict[str, Any]:
    """Build the ``compare.json`` payload for two reports.

    A regression (improvement) is a per-cell perturbation_config that
    appears in *both* reports with at least ``min_cell_size`` episodes
    on both sides AND a success-rate delta exceeding ``threshold`` in
    the negative (positive) direction.
    """
    cells_a = {_cell_key(c.perturbation_config): c for c in report_a.per_cell}
    cells_b = {_cell_key(c.perturbation_config): c for c in report_b.per_cell}
    shared_keys = cells_a.keys() & cells_b.keys()

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for key in shared_keys:
        ca = cells_a[key]
        cb = cells_b[key]
        if ca.n_episodes < min_cell_size or cb.n_episodes < min_cell_size:
            continue
        delta = cb.success_rate - ca.success_rate
        if delta < -threshold:
            regressions.append(
                {
                    "axis_combination": dict(ca.perturbation_config),
                    "rate_a": ca.success_rate,
                    "rate_b": cb.success_rate,
                    "delta": delta,
                    "n_episodes_a": ca.n_episodes,
                    "n_episodes_b": cb.n_episodes,
                }
            )
        elif delta > threshold:
            improvements.append(
                {
                    "axis_combination": dict(ca.perturbation_config),
                    "rate_a": ca.success_rate,
                    "rate_b": cb.success_rate,
                    "delta": delta,
                    "n_episodes_a": ca.n_episodes,
                    "n_episodes_b": cb.n_episodes,
                }
            )

    # Stable order: worst regression / best improvement first, then by
    # axis_combination repr for tie-breaking.
    regressions.sort(key=lambda r: (r["delta"], repr(r["axis_combination"])))
    improvements.sort(key=lambda r: (-r["delta"], repr(r["axis_combination"])))

    return {
        "a": {
            "name": report_a.suite_name,
            "overall_success_rate": report_a.overall_success_rate,
            "n_episodes": report_a.n_episodes,
        },
        "b": {
            "name": report_b.suite_name,
            "overall_success_rate": report_b.overall_success_rate,
            "n_episodes": report_b.n_episodes,
        },
        "delta_success_rate": report_b.overall_success_rate - report_a.overall_success_rate,
        "threshold": threshold,
        "min_cell_size": min_cell_size,
        "regressions": regressions,
        "improvements": improvements,
    }


@app.command("compare")
def compare(
    results_a: Annotated[
        Path,
        typer.Argument(help="First run: episodes.json or report.json."),
    ],
    results_b: Annotated[
        Path,
        typer.Argument(help="Second run: episodes.json or report.json."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output JSON path; defaults to 'compare.json' next to results_b.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Minimum |delta success_rate| to flag as a regression/improvement.",
            min=0.0,
        ),
    ] = 0.1,
    min_cell_size: Annotated[
        int,
        typer.Option(
            "--min-cell-size",
            help="Minimum n_episodes per cell (in BOTH runs) to be considered.",
            min=1,
        ),
    ] = 5,
) -> None:
    """Diff two runs and emit compare.json (HTML companion deferred to Phase 2)."""
    report_a = _load_report_or_episodes(results_a)
    report_b = _load_report_or_episodes(results_b)

    if report_a.suite_name != report_b.suite_name:
        _echo_err(
            f"warning: comparing across suites — a={report_a.suite_name!r} vs "
            f"b={report_b.suite_name!r}"
        )

    payload = _build_compare(
        report_a,
        report_b,
        threshold=threshold,
        min_cell_size=min_cell_size,
    )

    out_path = out if out is not None else results_b.parent / "compare.json"
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, payload)

    delta_pct = payload["delta_success_rate"] * 100
    _echo_err(f"Wrote {out_path}")
    _echo_err(f"  a: {report_a.suite_name} ({report_a.overall_success_rate * 100:.1f}%)")
    _echo_err(f"  b: {report_b.suite_name} ({report_b.overall_success_rate * 100:.1f}%)")
    _echo_err(f"  delta success_rate: {delta_pct:+.1f}%")
    _echo_err(
        f"  regressions: {len(payload['regressions'])}  "
        f"improvements: {len(payload['improvements'])}"
    )


# ``python -m gauntlet.cli`` parity with the installed entry point.
if __name__ == "__main__":  # pragma: no cover
    app()
