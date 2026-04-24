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
from rich.console import Console
from rich.theme import Theme

from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory
from gauntlet.policy.registry import PolicySpecError, resolve_policy_factory
from gauntlet.replay import OverrideError, parse_override, replay_one
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
#
# Status output goes through a rich ``Console`` on stderr. When stderr
# is not a TTY (CI, CliRunner, ``2>file``) rich auto-strips ANSI so the
# plain-text substrings tests assert against survive unchanged; on a
# real terminal you get colored prefixes, success-rate bands, and signed
# deltas. ``NO_COLOR=1`` (standard) also disables styling.

_THEME = Theme(
    {
        "err": "bold red",
        "warn": "bold yellow",
        "ok": "bold green",
        "path": "cyan",
        "pct.good": "green",
        "pct.mid": "yellow",
        "pct.bad": "red",
        "delta.up": "green",
        "delta.down": "red",
        "delta.zero": "dim",
    }
)

# ``soft_wrap=True`` so rich does not hard-wrap long lines to width and
# split tokens like ``tabletop-pybullet`` that stderr substring tests
# assert on. ``highlight=False`` suppresses rich's default auto-
# highlighting of numbers / paths (we apply explicit styles below).
_console = Console(stderr=True, theme=_THEME, highlight=False, soft_wrap=True)


def _echo_err(msg: str) -> None:
    """Write *msg* to stderr through the shared rich Console.

    ``msg`` may contain rich markup (e.g. ``"[ok]done[/]"``); markup is
    parsed and styled on a TTY and stripped to plain text otherwise.
    """
    _console.print(msg)


def _fail(msg: str, *, code: int = 1) -> typer.Exit:
    """Emit an error message and return a ``typer.Exit`` to raise."""
    _echo_err(f"[err]error:[/] {msg}")
    return typer.Exit(code=code)


def _fmt_success_rate(rate: float) -> str:
    """Colour a fractional success rate (0-1) by band for stderr summaries."""
    pct = rate * 100.0
    if pct >= 80.0:
        style = "pct.good"
    elif pct >= 50.0:
        style = "pct.mid"
    else:
        style = "pct.bad"
    return f"[{style}]{pct:.1f}%[/]"


def _fmt_signed_pct(delta: float) -> str:
    """Colour a signed fractional delta by direction (green up / red down)."""
    pct = delta * 100.0
    if delta > 0:
        style = "delta.up"
    elif delta < 0:
        style = "delta.down"
    else:
        style = "delta.zero"
    return f"[{style}]{pct:+.1f}%[/]"


def _fmt_path(path: Path) -> str:
    """Style a filesystem path so it stands out in stderr summaries."""
    return f"[path]{path}[/]"


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


def _make_env_factory(
    suite_env: str,
    env_max_steps: int | None,
) -> Callable[[], GauntletEnv] | None:
    """Build an env factory honouring the hidden ``--env-max-steps`` knob.

    Dispatches on ``suite_env`` via the env registry so a
    ``tabletop-pybullet`` suite actually produces
    :class:`PyBulletTabletopEnv`, not MuJoCo's :class:`TabletopEnv`.
    Previously this helper hard-coded ``TabletopEnv`` and the CLI
    silently misdispatched pybullet YAMLs through MuJoCo (RFC-005 §11.2
    / RFC-006 §9 bullet 1 — "CLI ``suite.env`` dispatch").

    Returns ``None`` when the user didn't override ``max_steps`` so the
    Runner's own registry dispatch kicks in unchanged. When set, the
    factory is ``functools.partial(backend_cls, max_steps=N)`` — the
    ``max_steps`` kwarg is part of both backends' constructor surface
    (RFC-005 §3.1 / RFC-006 §3.1), so wrapping is backend-agnostic.
    ``partial`` over a class pickles cleanly under ``spawn``.
    """
    if env_max_steps is None:
        return None
    backend = get_env_factory(suite_env)
    return cast(
        "Callable[[], GauntletEnv]",
        partial(backend, max_steps=env_max_steps),
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
    record_trajectories: Annotated[
        Path | None,
        typer.Option(
            "--record-trajectories",
            help=(
                "Directory to dump per-episode NPZ trajectories into. "
                "Defaults to OFF — when unset the Runner is byte-identical "
                "to Phase 1. Feed the directory into ``gauntlet monitor "
                "train`` / ``gauntlet monitor score``."
            ),
        ),
    ] = None,
    env_max_steps: Annotated[
        int | None,
        typer.Option(
            "--env-max-steps",
            help="(Hidden / test hook) Override the backend env's max_steps for fast tests.",
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

    env_factory = _make_env_factory(suite.env, env_max_steps)
    runner = Runner(
        n_workers=n_workers,
        env_factory=env_factory,
        trajectory_dir=record_trajectories,
    )

    try:
        episodes = runner.run(policy_factory=policy_factory, suite=suite)
    except ValueError as exc:
        raise _fail(f"runner failed: {exc}") from exc

    try:
        report = build_report(episodes, suite_env=suite.env)
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
        f"[ok]Wrote[/] {len(episodes)} episodes / {len(report.per_cell)} cells "
        f"-> {_fmt_path(out)} (success: {_fmt_success_rate(report.overall_success_rate)})"
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
    _echo_err(f"[ok]Wrote[/] {_fmt_path(out_path)}")


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
    allow_cross_backend: Annotated[
        bool,
        typer.Option(
            "--allow-cross-backend",
            help=(
                "Opt-in to comparing reports from different backends "
                "(e.g. 'tabletop' vs 'tabletop-pybullet'). Cross-backend "
                "comparison measures simulator drift, NOT policy regression "
                "— see RFC-005 §7.4."
            ),
        ),
    ] = False,
) -> None:
    """Diff two runs and emit compare.json (HTML companion deferred to Phase 2)."""
    report_a = _load_report_or_episodes(results_a)
    report_b = _load_report_or_episodes(results_b)

    # Cross-backend guard — RFC-005 §11.3 / §12 Q2. Only fires when
    # both reports carry a suite_env (Phase-2-written reports) and the
    # values differ. Pre-RFC-005 reports have None and are silently
    # accepted; mixing an old + a new report where only one has env
    # info is also silently accepted (we have no evidence of drift).
    if (
        report_a.suite_env is not None
        and report_b.suite_env is not None
        and report_a.suite_env != report_b.suite_env
    ):
        if not allow_cross_backend:
            raise _fail(
                f"cross-backend compare: a.suite_env={report_a.suite_env!r} "
                f"but b.suite_env={report_b.suite_env!r}. Cross-backend "
                f"comparison measures simulator drift, NOT policy "
                f"regression (RFC-005 §7.4). Pass --allow-cross-backend "
                f"to override."
            )
        _echo_err(
            f"[warn]warning:[/] cross-backend compare — a.suite_env="
            f"{report_a.suite_env!r} vs b.suite_env={report_b.suite_env!r}. "
            f"This measures simulator drift, NOT policy regression."
        )

    if report_a.suite_name != report_b.suite_name:
        _echo_err(
            f"[warn]warning:[/] comparing across suites — a={report_a.suite_name!r} vs "
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

    _echo_err(f"[ok]Wrote[/] {_fmt_path(out_path)}")
    _echo_err(f"  a: {report_a.suite_name} ({_fmt_success_rate(report_a.overall_success_rate)})")
    _echo_err(f"  b: {report_b.suite_name} ({_fmt_success_rate(report_b.overall_success_rate)})")
    _echo_err(f"  delta success_rate: {_fmt_signed_pct(payload['delta_success_rate'])}")
    n_regressions = len(payload["regressions"])
    n_improvements = len(payload["improvements"])
    regressions_style = "delta.down" if n_regressions else "delta.zero"
    improvements_style = "delta.up" if n_improvements else "delta.zero"
    _echo_err(
        f"  regressions: [{regressions_style}]{n_regressions}[/]  "
        f"improvements: [{improvements_style}]{n_improvements}[/]"
    )


# ──────────────────────────────────────────────────────────────────────
# `monitor` subcommand group — drift detector (torch-backed).
# ──────────────────────────────────────────────────────────────────────
#
# The subcommands call into ``gauntlet.monitor.train`` /
# ``gauntlet.monitor.score``. Both lazily import torch at module scope
# and raise an ``ImportError`` with an install hint when the extra is
# missing — we surface that as a clean CLI error so users get the hint
# on the same line, not after a traceback.


monitor_app = typer.Typer(
    name="monitor",
    help="Runtime drift detector (requires the 'monitor' extra).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(monitor_app)


def _fail_monitor_extra(exc: ImportError) -> typer.Exit:
    """Turn the install-hint ``ImportError`` into a clean CLI error."""
    return _fail(str(exc))


@monitor_app.command("train")
def monitor_train(
    trajectory_dir: Annotated[
        Path,
        typer.Argument(
            help="Reference trajectory directory (from ``gauntlet run --record-trajectories``).",
            exists=False,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="AE checkpoint directory; created if missing.",
        ),
    ],
    epochs: Annotated[
        int,
        typer.Option("--epochs", min=1, help="Training epochs over the reference set."),
    ] = 50,
    latent_dim: Annotated[
        int,
        typer.Option("--latent-dim", min=1, help="AE bottleneck dimension."),
    ] = 8,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Minibatch size."),
    ] = 256,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Adam learning rate."),
    ] = 1e-3,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Torch RNG seed for bit-identical reruns."),
    ] = 0,
    reference_suite: Annotated[
        str | None,
        typer.Option(
            "--reference-suite",
            help="Optional suite name echoed into the checkpoint's config.json.",
        ),
    ] = None,
) -> None:
    """Fit a :class:`StateAutoencoder` on the reference trajectories."""
    if not trajectory_dir.is_dir():
        raise _fail(f"trajectory dir not found: {trajectory_dir}")
    try:
        from gauntlet.monitor.train import train_ae
    except ImportError as exc:
        raise _fail_monitor_extra(exc) from exc

    try:
        train_ae(
            trajectory_dir,
            out_dir=out,
            reference_suite=reference_suite,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise _fail(f"monitor train failed: {exc}") from exc

    _echo_err(f"[ok]Wrote[/] AE checkpoint -> {_fmt_path(out)}")


@monitor_app.command("score")
def monitor_score(
    episodes_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the candidate sweep's episodes.json.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    trajectory_dir: Annotated[
        Path,
        typer.Argument(
            help="Candidate trajectory directory (matching the episodes.json above).",
            dir_okay=True,
            file_okay=False,
        ),
    ],
    ae: Annotated[
        Path,
        typer.Option(
            "--ae",
            help="AE checkpoint directory from ``gauntlet monitor train``.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output drift.json path."),
    ],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            min=0,
            help="How many most-OOD episode indices to list in top_ood_episodes.",
        ),
    ] = 10,
) -> None:
    """Score a candidate sweep against a trained AE and emit drift.json."""
    if not episodes_path.is_file():
        raise _fail(f"episodes file not found: {episodes_path}")
    if not trajectory_dir.is_dir():
        raise _fail(f"trajectory dir not found: {trajectory_dir}")
    if not ae.is_dir():
        raise _fail(f"AE checkpoint dir not found: {ae}")

    try:
        from gauntlet.monitor.score import score_drift
    except ImportError as exc:
        raise _fail_monitor_extra(exc) from exc

    try:
        drift = score_drift(episodes_path, trajectory_dir, ae, top_k=top_k)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise _fail(f"monitor score failed: {exc}") from exc

    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, drift.model_dump(mode="json"))
    cand = drift.candidate_reconstruction_error_mean
    ref = drift.reference_reconstruction_error_mean
    # Higher reconstruction error on the candidate = more drift.
    cand_style = "delta.down" if cand > ref else ("delta.up" if cand < ref else "delta.zero")
    _echo_err(
        f"[ok]Wrote[/] drift.json -> {_fmt_path(out)} "
        f"(n_episodes={drift.n_episodes}, "
        f"candidate mean=[{cand_style}]{cand:.4f}[/] "
        f"vs reference mean={ref:.4f})"
    )


# ──────────────────────────────────────────────────────────────────────
# `replay` subcommand — see ``docs/phase2-rfc-004-trajectory-replay.md``.
# ──────────────────────────────────────────────────────────────────────


def _parse_episode_id(spec: str) -> tuple[int, int]:
    """Parse a ``"cell:episode"`` id into a ``(cell_index, episode_index)``.

    The format mirrors RFC §3 ("topology-free, trivially copy-pasteable
    and stable under jq filtering"). We reject anything with other than
    exactly one colon, or with non-integer halves.
    """
    if spec.count(":") != 1:
        raise _fail(f"--episode-id {spec!r}: expected 'CELL:EPISODE' with exactly one ':'")
    left, right = spec.split(":", 1)
    try:
        cell_index = int(left)
        episode_index = int(right)
    except ValueError as exc:
        raise _fail(f"--episode-id {spec!r}: both halves must be integers") from exc
    if cell_index < 0 or episode_index < 0:
        raise _fail(f"--episode-id {spec!r}: both halves must be non-negative")
    return cell_index, episode_index


def _available_ids_preview(episodes: list[Episode], limit: int = 10) -> str:
    """Render a terse list of available ``(cell:episode)`` pairs."""
    pairs = sorted({(ep.cell_index, ep.episode_index) for ep in episodes})
    head = pairs[:limit]
    rendered = ", ".join(f"{c}:{e}" for c, e in head)
    if len(pairs) > limit:
        rendered += f", ... ({len(pairs) - limit} more)"
    return rendered


@app.command("replay")
def replay(
    episodes_path: Annotated[
        Path,
        typer.Argument(
            help="Path to an episodes.json emitted by a prior 'gauntlet run'.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    suite_path: Annotated[
        Path,
        typer.Option(
            "--suite",
            help="Path to the suite YAML the run used.",
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
    episode_id: Annotated[
        str,
        typer.Option(
            "--episode-id",
            help="Target episode as 'CELL:EPISODE' (e.g. '12:3').",
        ),
    ],
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            help=(
                "Repeatable axis override 'AXIS=VALUE'. "
                "AXIS must be declared in the suite; VALUE must lie within "
                "the declared axis envelope. Off-grid values are allowed."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output JSON path; defaults to 'replay.json' next to EPISODES_JSON.",
        ),
    ] = None,
    env_max_steps: Annotated[
        int | None,
        typer.Option(
            "--env-max-steps",
            help="(Hidden / test hook) Override the backend env's max_steps for fast tests.",
            hidden=True,
            min=1,
        ),
    ] = None,
) -> None:
    """Re-simulate one Episode, optionally with axis overrides."""
    cell_index, episode_index = _parse_episode_id(episode_id)

    raw = _read_json(episodes_path)
    if not isinstance(raw, list):
        raise _fail(
            f"{episodes_path}: expected a list of Episode objects (got {type(raw).__name__})"
        )
    episodes = _episodes_from_dicts(raw, source=episodes_path)

    target: Episode | None = None
    for ep in episodes:
        if ep.cell_index == cell_index and ep.episode_index == episode_index:
            target = ep
            break
    if target is None:
        raise _fail(
            f"--episode-id {episode_id!r}: no matching episode in {episodes_path}; "
            f"available: {_available_ids_preview(episodes)}"
        )

    if not suite_path.is_file():
        raise _fail(f"suite file not found: {suite_path}")
    try:
        suite = load_suite(suite_path)
    except (ValidationError, ValueError) as exc:
        raise _fail(f"{suite_path}: invalid suite YAML: {exc}") from exc
    except OSError as exc:
        raise _fail(f"{suite_path}: could not read file: {exc}") from exc

    if suite.name != target.suite_name:
        raise _fail(
            f"suite name mismatch: episode.suite_name={target.suite_name!r} vs "
            f"suite.name={suite.name!r} ({suite_path})"
        )

    overrides: dict[str, float] = {}
    for raw_override in override or []:
        try:
            axis_name, value = parse_override(raw_override)
        except OverrideError as exc:
            raise _fail(str(exc)) from exc
        if axis_name in overrides:
            raise _fail(
                f"--override {axis_name!r} passed more than once; overrides must be unique per axis"
            )
        overrides[axis_name] = value

    try:
        policy_factory = resolve_policy_factory(policy)
    except PolicySpecError as exc:
        raise _fail(str(exc)) from exc

    env_factory = _make_env_factory(suite.env, env_max_steps)

    try:
        replayed = replay_one(
            target=target,
            suite=suite,
            policy_factory=policy_factory,
            overrides=overrides,
            env_factory=env_factory,
        )
    except OverrideError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(f"replay failed: {exc}") from exc

    out_path = out if out is not None else episodes_path.parent / "replay.json"
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "episode_id": f"{cell_index}:{episode_index}",
        "suite_name": target.suite_name,
        "policy": policy,
        "overrides": overrides,
        "original": target.model_dump(mode="json"),
        "replayed": replayed.model_dump(mode="json"),
    }
    _write_json(out_path, payload)

    delta_reward = replayed.total_reward - target.total_reward
    original_style = "delta.up" if target.success else "delta.down"
    replayed_style = "delta.up" if replayed.success else "delta.down"
    if delta_reward > 0:
        reward_delta_style = "delta.up"
    elif delta_reward < 0:
        reward_delta_style = "delta.down"
    else:
        reward_delta_style = "delta.zero"
    _echo_err(f"[ok]Wrote[/] {_fmt_path(out_path)}")
    _echo_err(
        f"  episode {cell_index}:{episode_index}  overrides: {overrides if overrides else '{}'}"
    )
    _echo_err(
        f"  original: success=[{original_style}]{target.success}[/] "
        f"steps={target.step_count} reward={target.total_reward:.4f}"
    )
    _echo_err(
        f"  replayed: success=[{replayed_style}]{replayed.success}[/] "
        f"steps={replayed.step_count} reward={replayed.total_reward:.4f} "
        f"(delta [{reward_delta_style}]{delta_reward:+.4f}[/])"
    )


# ──────────────────────────────────────────────────────────────────────
# `ros2` subcommand group — RFC-010.
# ──────────────────────────────────────────────────────────────────────
#
# The subcommands call into ``gauntlet.ros2.publisher`` /
# ``gauntlet.ros2.recorder``. Both lazily import rclpy at module scope
# and raise an ``ImportError`` with the apt / Docker install hint when
# the user hasn't installed ROS 2 — we surface that as a clean CLI
# error so users get the hint on the same line, not after a traceback.
#
# ``--dry-run`` on ``publish`` short-circuits the rclpy import entirely
# so users can preview the JSON payloads without installing ROS 2.


ros2_app = typer.Typer(
    name="ros2",
    help="ROS 2 integration — publish episode summaries / record robot topics.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(ros2_app)


def _fail_ros2_extra(exc: ImportError) -> typer.Exit:
    """Turn the rclpy install-hint ``ImportError`` into a clean CLI error."""
    return _fail(str(exc))


@ros2_app.command("publish")
def ros2_publish(
    episodes_path: Annotated[
        Path,
        typer.Argument(
            help="Path to an episodes.json emitted by a prior 'gauntlet run'.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    topic: Annotated[
        str,
        typer.Option(
            "--topic",
            help="ROS 2 topic to publish on. Defaults to /gauntlet/episodes.",
        ),
    ] = "/gauntlet/episodes",
    node_name: Annotated[
        str,
        typer.Option(
            "--node-name",
            help="ROS 2 node name. Defaults to gauntlet_episode_publisher.",
        ),
    ] = "gauntlet_episode_publisher",
    qos_depth: Annotated[
        int,
        typer.Option(
            "--qos-depth",
            min=1,
            help="QoS history depth. Defaults to 10.",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Render the JSON payloads to stderr without contacting rclpy. "
                "Useful for previewing the wire format on a machine without "
                "ROS 2 installed."
            ),
        ),
    ] = False,
) -> None:
    """Publish each Episode in EPISODES_JSON to a ROS 2 topic.

    Wire format: JSON serialised inside ``std_msgs/msg/String`` (RFC-010
    §5). One message per Episode. Requires the ``rclpy`` Python bindings
    on PATH unless ``--dry-run`` is passed; install ROS 2 via your
    system package manager (``apt install ros-<distro>-rclpy``) or via
    the official Docker image (``osrf/ros:<distro>-desktop``).
    """
    if not episodes_path.is_file():
        raise _fail(f"episodes file not found: {episodes_path}")
    if not topic:
        raise _fail("--topic must be a non-empty string")

    raw = _read_json(episodes_path)
    if not isinstance(raw, list):
        raise _fail(
            f"{episodes_path}: expected a list of Episode objects (got {type(raw).__name__})"
        )
    episodes = _episodes_from_dicts(raw, source=episodes_path)

    if dry_run:
        # Lazy schema import — torch-/rclpy-free; works on every install.
        from gauntlet.ros2.schema import Ros2EpisodePayload

        for ep in episodes:
            payload = Ros2EpisodePayload.from_episode(ep)
            _echo_err(payload.model_dump_json())
        _echo_err(
            f"[ok]Dry-run[/] published {len(episodes)} payloads to stderr "
            f"(topic [path]{topic}[/], rclpy NOT contacted)"
        )
        return

    try:
        from gauntlet.ros2.publisher import Ros2EpisodePublisher
    except ImportError as exc:
        raise _fail_ros2_extra(exc) from exc

    try:
        publisher = Ros2EpisodePublisher(
            topic=topic,
            node_name=node_name,
            qos_depth=qos_depth,
        )
    except ValueError as exc:
        raise _fail(f"ros2 publish failed: {exc}") from exc

    try:
        for ep in episodes:
            publisher.publish_episode(ep)
    finally:
        publisher.close()

    _echo_err(
        f"[ok]Published[/] {len(episodes)} episodes to [path]{topic}[/] (node [path]{node_name}[/])"
    )


@ros2_app.command("record")
def ros2_record(
    topic: Annotated[
        str,
        typer.Option(
            "--topic",
            help="ROS 2 topic to subscribe to (required).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="JSONL output path (one received message per line).",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    duration: Annotated[
        float,
        typer.Option(
            "--duration",
            help=(
                "Soft cap in seconds. 0.0 means run forever (until Ctrl-C). Defaults to 30 seconds."
            ),
        ),
    ] = 30.0,
    node_name: Annotated[
        str,
        typer.Option(
            "--node-name",
            help="ROS 2 node name. Defaults to gauntlet_rollout_recorder.",
        ),
    ] = "gauntlet_rollout_recorder",
    qos_depth: Annotated[
        int,
        typer.Option(
            "--qos-depth",
            min=1,
            help="QoS history depth. Defaults to 10.",
        ),
    ] = 10,
) -> None:
    """Subscribe to TOPIC and dump received messages to OUT as JSONL.

    One JSON object per line: ``{"timestamp": <float>, "topic": <str>,
    "data": <str>}``. ``data`` is :func:`str` of the message payload —
    lossy but generic across message types (RFC-010 §6 / §11).

    Requires the ``rclpy`` Python bindings on PATH; install ROS 2 via
    ``apt install ros-<distro>-rclpy`` or the official Docker image
    (``osrf/ros:<distro>-desktop``).
    """
    if not topic:
        raise _fail("--topic must be a non-empty string")
    if duration < 0.0:
        raise _fail(f"--duration must be >= 0; got {duration}")

    try:
        from gauntlet.ros2.recorder import Ros2RolloutRecorder
    except ImportError as exc:
        raise _fail_ros2_extra(exc) from exc

    try:
        with Ros2RolloutRecorder(
            topic=topic,
            out_path=out,
            node_name=node_name,
            duration_s=duration,
            qos_depth=qos_depth,
        ) as recorder:
            n_received = recorder.spin_until_done()
    except ValueError as exc:
        raise _fail(f"ros2 record failed: {exc}") from exc

    _echo_err(f"[ok]Recorded[/] {n_received} messages from [path]{topic}[/] -> {_fmt_path(out)}")


# ``python -m gauntlet.cli`` parity with the installed entry point.
if __name__ == "__main__":  # pragma: no cover
    app()
