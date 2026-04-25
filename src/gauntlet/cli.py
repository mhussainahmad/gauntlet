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
import math
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, TypeAlias, cast

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.theme import Theme

from gauntlet.diff import (
    PairingError,
    compute_paired_cells,
    diff_reports,
    render_text,
)
from gauntlet.env.base import GauntletEnv
from gauntlet.env.registry import get_env_factory
from gauntlet.policy.registry import PolicySpecError, resolve_policy_factory
from gauntlet.replay import OverrideError, parse_override, replay_one
from gauntlet.report import CellBreakdown, Report, build_report, write_html
from gauntlet.report.html import _nan_to_none
from gauntlet.runner import Episode, Runner
from gauntlet.runner.provenance import compute_suite_hash
from gauntlet.runner.worker import TrajectoryFormat as _TrajectoryFormatLiteral
from gauntlet.suite import LintFinding, lint_suite, load_suite
from gauntlet.suite.schema import Suite

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


# Recursive type for JSON-style nested data — same shape as
# :data:`gauntlet.report.html._JsonValue`. Narrowed from ``Any`` so the
# CLI's JSON read / validate / write helpers carry the actual structure
# they accept (``json.load`` produces; pydantic ``model_dump(mode="json")``
# returns; the report-package ``_nan_to_none`` walker accepts).
_JsonValue: TypeAlias = (
    float
    | int
    | str
    | bool
    | None
    | dict[str, "_JsonValue"]
    | list["_JsonValue"]
    | tuple["_JsonValue", ...]
)


def _read_json(path: Path) -> _JsonValue:
    """Load JSON from *path* with a uniform error envelope.

    Returns the parsed JSON tree as a recursive :data:`_JsonValue`; the
    auto-detect logic in :func:`_load_report_or_episodes` narrows it to
    a Report (top-level ``dict``) or an Episode list (top-level
    ``list``) before constructing pydantic models.
    """
    if not path.is_file():
        raise _fail(f"file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return cast("_JsonValue", json.load(fh))
    except json.JSONDecodeError as exc:
        raise _fail(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc


def _load_report_or_episodes(path: Path) -> Report:
    """Auto-detect ``episodes.json`` vs ``report.json`` and return a Report.

    Top-level ``list`` → episodes (rebuild via :func:`build_report`);
    top-level ``dict`` → assume it is a serialized :class:`Report`.
    Anything else is an error.
    """
    report, _ = _load_report_with_episodes(path)
    return report


def _load_report_with_episodes(path: Path) -> tuple[Report, list[Episode] | None]:
    """Load a report + (optionally) the original episode list.

    The B-08 paired-comparison path needs the *per-episode* pass/fail
    triples (``cell_index``, ``episode_index``, ``success``) plus the
    ``master_seed`` echoed onto each Episode's metadata; the per-cell
    aggregates carried by :class:`Report` are not enough on their own.
    This helper exposes both: the rebuilt :class:`Report` (for the
    legacy unpaired path that already drives compare/diff) AND the
    original :class:`Episode` list (returned only when the input file
    was an ``episodes.json``; ``report.json`` carries no per-episode
    detail and the second tuple element is ``None``).
    """
    raw = _read_json(path)
    if isinstance(raw, list):
        episodes = _episodes_from_dicts(raw, source=path)
        try:
            return (build_report(episodes), episodes)
        except ValueError as exc:
            raise _fail(f"{path}: cannot build report: {exc}") from exc
    if isinstance(raw, dict):
        try:
            return (Report.model_validate(raw), None)
        except ValidationError as exc:
            raise _fail(f"{path}: not a valid report.json: {exc}") from exc
    raise _fail(
        f"{path}: top-level JSON must be a list (episodes) or dict (report); "
        f"got {type(raw).__name__}"
    )


def _episodes_have_master_seed(episodes: list[Episode]) -> bool:
    """Return True iff every episode carries an int ``master_seed``.

    The auto-paired path requires the runner-stamped
    ``metadata["master_seed"]`` echo. Older / hand-built episode lists
    may omit it; in that case the auto path silently falls back to the
    legacy unpaired comparison (the explicit ``--paired`` flag still
    surfaces a clear PairingError so the user knows why).
    """
    for ep in episodes:
        seed = ep.metadata.get("master_seed")
        if not isinstance(seed, int):
            return False
    return True


def _resolve_pairing_mode(
    requested: bool | None,
    episodes_a: list[Episode] | None,
    episodes_b: list[Episode] | None,
    path_a: Path,
    path_b: Path,
) -> bool:
    """Map ``--paired/--no-paired`` tri-state into a concrete on/off decision.

    Rules (B-08):

    * ``--no-paired`` (``requested=False``) → never pair, even if both
      sides have episodes. Matches the explicit user opt-out.
    * ``--paired`` (``requested=True``) → require episodes on both
      sides; if either side is a ``report.json`` (no per-episode
      detail), raise a clean :class:`typer.Exit` so the user knows
      exactly which file lacks the per-episode pass/fail triples that
      drive the McNemar / paired-CI computation. (``master_seed``
      mismatches surface later, inside :func:`compute_paired_cells`.)
    * ``requested is None`` (the default) → auto: pair iff both sides
      loaded as episode lists AND every episode carries the runner-
      stamped ``master_seed`` (older / hand-built fixtures omit it).
      Avoids surprising the legacy ``compare report-a.json report-b.json``
      invocation while making the CRN reduction free for the common
      ``compare episodes-a.json episodes-b.json`` flow.
    """
    if requested is False:
        return False
    if requested is True:
        if episodes_a is None or episodes_b is None:
            missing = []
            if episodes_a is None:
                missing.append(str(path_a))
            if episodes_b is None:
                missing.append(str(path_b))
            raise _fail(
                f"--paired requires per-episode data (episodes.json) on both sides; "
                f"got report.json for: {', '.join(missing)}. Re-run with the "
                f"episodes.json output, or drop --paired for the independent path."
            )
        return True
    if episodes_a is None or episodes_b is None:
        return False
    return _episodes_have_master_seed(episodes_a) and _episodes_have_master_seed(episodes_b)


def _episodes_from_dicts(raw: list[_JsonValue], *, source: Path) -> list[Episode]:
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


def _write_json(path: Path, payload: _JsonValue) -> None:
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


def _episodes_to_dicts(episodes: list[Episode]) -> list[dict[str, _JsonValue]]:
    """Serialise episodes via Pydantic's JSON mode for round-tripping."""
    return [cast("dict[str, _JsonValue]", ep.model_dump(mode="json")) for ep in episodes]


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
# `repro.json` writer (B-22 — episode-level seed manifest).
# ──────────────────────────────────────────────────────────────────────


def _episode_repro_id(episode: Episode) -> str:
    """Canonical human-friendly id used in ``gauntlet repro <id>``.

    Mirrors the wording the spec gives ("e.g. ``cell_3_episode_7``").
    Both halves are integers so the shape ``cell_{int}_episode_{int}``
    is unambiguous and stable under round-trip through ``repro.json``.
    """
    return f"cell_{episode.cell_index}_episode_{episode.episode_index}"


def _build_repro_payload(
    *,
    episodes: list[Episode],
    suite: Suite,
    suite_path: Path,
    policy: str,
    seed_override: int | None,
    env_max_steps: int | None,
) -> dict[str, _JsonValue]:
    """Assemble the ``repro.json`` payload for one ``gauntlet run``.

    Keys at the top level:

    * ``schema_version`` — bumped on incompatible field changes.
    * ``suite_path`` — the YAML path the user passed (anchored to the
      cwd of the run for portability across machines is left to the
      user; we record what was given).
    * ``suite_name`` / ``suite_env`` — quick human cross-check.
    * ``policy`` — the resolver string the user passed (``random``,
      ``module.path:attr``, etc.). ``gauntlet repro`` reuses this to
      reconstruct the policy without a separate flag.
    * ``seed_override`` / ``env_max_steps`` — echo of the run-time CLI
      knobs so a subsequent repro matches end-to-end.
    * ``episodes`` — one entry per Episode keyed off
      :func:`_episode_repro_id`.

    Per-episode fields (``axis_config`` is a copy of
    :attr:`Episode.perturbation_config` keyed by axis name):

    * ``episode_id``, ``cell_index``, ``episode_index``
    * ``env_seed`` (alias of :attr:`Episode.seed`)
    * ``policy_seed`` (mirror of ``env_seed``: both streams derive from
      the same SeedSequence node — see runner.runner module docstring)
    * ``axis_config``
    * ``gauntlet_version`` / ``suite_hash`` / ``git_commit``

    The output is written with ``sort_keys=True`` so byte-comparing two
    ``repro.json`` files is a meaningful regression signal.
    """
    per_episode: list[dict[str, _JsonValue]] = []
    for ep in episodes:
        per_episode.append(
            {
                "episode_id": _episode_repro_id(ep),
                "cell_index": ep.cell_index,
                "episode_index": ep.episode_index,
                "env_seed": ep.seed,
                "policy_seed": ep.seed,
                "axis_config": cast("_JsonValue", dict(ep.perturbation_config)),
                "gauntlet_version": ep.gauntlet_version,
                "suite_hash": ep.suite_hash,
                "git_commit": ep.git_commit,
            }
        )
    payload: dict[str, _JsonValue] = {
        "schema_version": 1,
        "suite_path": str(suite_path),
        "suite_name": suite.name,
        "suite_env": suite.env,
        "suite_hash": compute_suite_hash(suite),
        "policy": policy,
        "seed_override": seed_override,
        "env_max_steps": env_max_steps,
        "episodes": cast("_JsonValue", per_episode),
    }
    return payload


def _write_repro_json(path: Path, payload: dict[str, _JsonValue]) -> None:
    """Write ``repro.json`` with deterministic ordering.

    Uses ``sort_keys=True`` (unlike :func:`_write_json`) so two repro
    manifests produced from the same run on different machines hash
    identically. NaN / inf cleansing is unnecessary here — every value
    in the payload is an int / str / None / nested dict, never a float
    that could be non-finite.
    """
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
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
                "Directory to dump per-episode trajectories into. "
                "Defaults to OFF — when unset the Runner is byte-identical "
                "to Phase 1. Format is controlled by --trajectory-format. "
                "Feed the directory into ``gauntlet monitor train`` / "
                "``gauntlet monitor score`` (NPZ) or DuckDB / pandas / "
                "polars (Parquet)."
            ),
        ),
    ] = None,
    trajectory_format: Annotated[
        str,
        typer.Option(
            "--trajectory-format",
            help=(
                "Trajectory output format when --record-trajectories is "
                "set. 'npz' (default) keeps the byte-identical pre-B-23 "
                "behaviour: one np.savez_compressed per episode, no "
                "pyarrow dependency. 'parquet' writes one Parquet file "
                "per episode (requires the [parquet] extra: "
                "`pip install \"gauntlet[parquet]\"`). 'both' writes "
                "both side-by-side. Ignored when --record-trajectories "
                "is unset."
            ),
            case_sensitive=False,
        ),
    ] = "npz",
    env_max_steps: Annotated[
        int | None,
        typer.Option(
            "--env-max-steps",
            help="(Hidden / test hook) Override the backend env's max_steps for fast tests.",
            hidden=True,
            min=1,
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help=(
                "Directory for the per-Episode rollout cache. When set, every "
                "(suite, axis_config, env_seed, policy, max_steps) cell is "
                "looked up before dispatch — a hit returns the cached Episode "
                "without re-rolling. Defaults to OFF (no cache lookup, "
                "byte-identical to no-cache runs). See "
                "docs/polish-exploration-incremental-cache.md."
            ),
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help=(
                "Force-disable the rollout cache even if --cache-dir is set "
                "(useful for wrapper scripts that bake in a default cache "
                "path)."
            ),
        ),
    ] = False,
    cache_stats: Annotated[
        bool,
        typer.Option(
            "--cache-stats",
            help="Print cache hit / miss / put counts to stderr after the run.",
        ),
    ] = False,
    junit: Annotated[
        Path | None,
        typer.Option(
            "--junit",
            help=(
                "Also emit a JUnit-style XML at PATH (one <testcase> per "
                "episode). Drop into any CI (GitHub Actions, Jenkins, "
                "GitLab) without custom glue. Parent directory is created "
                "if missing. See backlog B-24."
            ),
        ),
    ] = None,
    prune_after_cells: Annotated[
        int | None,
        typer.Option(
            "--prune-after-cells",
            help=(
                "B-26 Optuna-style early-stop: check the running success-rate "
                "Wilson 95%% CI every N cells; abort the run when the CI clearly "
                "straddles --prune-min-success from above or below. PRUNING "
                "DESTROYS FAIR SAMPLING — only use when you want a quick "
                "pass/fail screen, not a full report. Must be paired with "
                "--prune-min-success. Force-disabled (warn-and-continue) when "
                "the suite uses sampling=sobol."
            ),
            min=1,
        ),
    ] = None,
    prune_min_success: Annotated[
        float | None,
        typer.Option(
            "--prune-min-success",
            help=(
                "B-26 pass/fail threshold for early-stop pruning, in [0, 1]. "
                "PRUNING DESTROYS FAIR SAMPLING — only use when you want a "
                "quick pass/fail screen, not a full report. Must be paired "
                "with --prune-after-cells."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = None,
    measure_action_consistency: Annotated[
        bool,
        typer.Option(
            "--measure-action-consistency",
            help=(
                "B-18 mode-collapse metric: every Nth rollout step, sample 8 "
                "candidate actions from the policy and report mean per-axis "
                "variance on Episode.action_variance (aggregated as "
                "mean_action_variance per failure cluster). Only sampleable "
                "policies (Random, π0, SmolVLA) expose the column; greedy "
                "policies (Scripted, OpenVLA-greedy) honestly report None. "
                "Off by default — adds ~8x policy-inference cost on measured "
                "steps."
            ),
        ),
    ] = False,
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

    # Resolve the cache configuration. ``--no-cache`` always wins (so a
    # wrapper script that bakes in --cache-dir can be opted out per
    # invocation); when --cache-dir is honoured, --env-max-steps must
    # be set because the cache key depends on max_steps and we have no
    # other way to derive it.
    effective_cache_dir = None if no_cache else cache_dir
    if effective_cache_dir is not None and env_max_steps is None:
        raise _fail(
            "--cache-dir requires --env-max-steps to be set; the cache key "
            "depends on max_steps and the env factory does not expose it."
        )

    try:
        runner = Runner(
            n_workers=n_workers,
            env_factory=env_factory,
            trajectory_dir=record_trajectories,
            cache_dir=effective_cache_dir,
            policy_id=policy if effective_cache_dir is not None else None,
            max_steps=env_max_steps if effective_cache_dir is not None else None,
            measure_action_consistency=measure_action_consistency,
        )
    except ValueError as exc:
        raise _fail(f"runner config invalid: {exc}") from exc

    # B-26: surface "only one of the prune flags was passed" as a
    # config error before the runner does, so the user sees a clean
    # CLI message instead of "runner failed: ...".
    if (prune_after_cells is None) ^ (prune_min_success is None):
        raise _fail(
            "--prune-after-cells and --prune-min-success must be supplied "
            "together; got "
            f"--prune-after-cells={prune_after_cells!r}, "
            f"--prune-min-success={prune_min_success!r}.",
        )

    # Validate --trajectory-format up-front so a typo surfaces as a
    # clean CLI error before the Runner spawns workers (and so the
    # Literal narrowing below is honest under mypy --strict).
    fmt_normalised = trajectory_format.lower()
    if fmt_normalised not in ("npz", "parquet", "both"):
        raise _fail(
            f"--trajectory-format must be one of {{npz, parquet, both}}; got {trajectory_format!r}."
        )
    fmt_literal = cast(_TrajectoryFormatLiteral, fmt_normalised)

    try:
        episodes = runner.run(
            policy_factory=policy_factory,
            suite=suite,
            prune_after_cells=prune_after_cells,
            prune_min_success=prune_min_success,
            trajectory_format=fmt_literal,
        )
    except ValueError as exc:
        raise _fail(f"runner failed: {exc}") from exc

    try:
        report = build_report(episodes, suite_env=suite.env, sampling=suite.sampling)
    except ValueError as exc:
        raise _fail(f"could not build report: {exc}") from exc

    # B-26: stamp the prune cell count onto the report after build_report
    # runs (build_report has no notion of pruning by design — the schema
    # field is the boundary).
    if runner.last_pruned_at_cell is not None:
        report = report.model_copy(update={"pruned_at_cell": runner.last_pruned_at_cell})

    episodes_path = out / "episodes.json"
    report_json_path = out / "report.json"
    report_html_path = out / "report.html"
    repro_path = out / "repro.json"

    _write_json(episodes_path, cast("_JsonValue", _episodes_to_dicts(episodes)))
    _write_json(report_json_path, cast("_JsonValue", report.model_dump(mode="json")))

    # B-22: episode-level seed manifest with full provenance. Always
    # written next to ``episodes.json`` so ``gauntlet repro`` can pick
    # it up by ``--out`` directory alone (no extra flag required).
    _write_repro_json(
        repro_path,
        _build_repro_payload(
            episodes=episodes,
            suite=suite,
            suite_path=suite_path,
            policy=policy,
            seed_override=seed_override,
            env_max_steps=env_max_steps,
        ),
    )

    if not no_html:
        write_html(report, report_html_path)

    if junit is not None:
        # Lazy import — keeps `gauntlet --help` snappy for the common
        # no-JUnit run path (mirrors aggregate / dashboard / realsim).
        from gauntlet.report.junit import to_junit_xml

        if junit.parent and not junit.parent.exists():
            junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_bytes(to_junit_xml(episodes, suite.name))
        _echo_err(f"[ok]Wrote[/] JUnit XML -> {_fmt_path(junit)}")

    summary = (
        f"[ok]Wrote[/] {len(episodes)} episodes / {len(report.per_cell)} cells "
        f"-> {_fmt_path(out)} (success: {_fmt_success_rate(report.overall_success_rate)})"
    )
    _echo_err(summary)

    if cache_stats:
        # Always emit a stats line when --cache-stats is set, even when
        # the cache is disabled — zeros across the board signal "you
        # asked for stats but you also disabled the cache".
        s = runner.cache_stats()
        _echo_err(f"  cache: hits={s['hits']} misses={s['misses']} puts={s['puts']}")


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


def _resolve_drift_suite_hash(
    episodes_a: list[Episode] | None,
    episodes_b: list[Episode] | None,
) -> str:
    """Derive the shared ``suite_hash`` for the B-29 drift map.

    Both sides MUST be loaded as ``episodes.json`` (so the per-episode
    ``suite_hash`` provenance field is available) AND every episode
    on both sides MUST agree on the same hash. Cross-backend reports
    that disagree on the suite payload would silently emit a
    half-attributed drift map otherwise — fail loudly here instead.
    """
    if not episodes_a or not episodes_b:
        raise _fail(
            "--drift-map requires episodes.json (not report.json) on "
            "both sides — drift_map.json's suite_hash is derived from "
            "the per-episode provenance trio (B-22)"
        )
    hashes_a = {ep.suite_hash for ep in episodes_a if ep.suite_hash is not None}
    hashes_b = {ep.suite_hash for ep in episodes_b if ep.suite_hash is not None}
    if not hashes_a or not hashes_b:
        raise _fail(
            "--drift-map requires every episode to carry suite_hash "
            "(B-22 provenance); pre-B-22 episodes.json files cannot "
            "establish a shared suite identity"
        )
    if len(hashes_a) > 1 or len(hashes_b) > 1:
        raise _fail(
            "--drift-map requires a single suite_hash per side; "
            f"a={sorted(hashes_a)!r} b={sorted(hashes_b)!r}"
        )
    (hash_a,) = hashes_a
    (hash_b,) = hashes_b
    if hash_a != hash_b:
        raise _fail(
            "--drift-map requires identical suite_hash across "
            f"backends; a={hash_a!r} vs b={hash_b!r}"
        )
    return hash_a


def _episodes_suite_env(episodes: list[Episode]) -> str | None:
    """Pull the ``suite_env`` slug off an episode list's metadata.

    ``Runner`` stamps ``suite_env`` onto every Episode's metadata
    dict (RFC-005 §12 Q2); when episodes.json is loaded back
    :func:`build_report` is called without the env hint so the rebuilt
    Report carries ``suite_env=None``. The drift map needs the slug,
    so look it up here without touching the rest of the CLI's
    report-rebuilding path.
    """
    envs: set[str] = set()
    for ep in episodes:
        env = ep.metadata.get("suite_env")
        if isinstance(env, str):
            envs.add(env)
    if len(envs) == 1:
        (only,) = envs
        return only
    return None


def _build_compare(
    report_a: Report,
    report_b: Report,
    *,
    threshold: float,
    min_cell_size: int,
    episodes_a: list[Episode] | None = None,
    episodes_b: list[Episode] | None = None,
) -> dict[str, _JsonValue]:
    """Build the ``compare.json`` payload for two reports.

    A regression (improvement) is a per-cell perturbation_config that
    appears in *both* reports with at least ``min_cell_size`` episodes
    on both sides AND a success-rate delta exceeding ``threshold`` in
    the negative (positive) direction.

    When ``episodes_a`` / ``episodes_b`` are provided AND both runs
    share the same ``master_seed`` (via the deterministic seed-derivation
    in :func:`gauntlet.runner.runner.Runner._build_work_items`), the
    payload is enriched with the B-08 paired CRN bracket per cell and
    a top-level ``paired: true`` flag. ``master_seed`` mismatches are a
    hard :class:`PairingError` raised by
    :func:`gauntlet.diff.compute_paired_cells` and surfaced verbatim
    by the caller.
    """
    cells_a = {_cell_key(c.perturbation_config): c for c in report_a.per_cell}
    cells_b = {_cell_key(c.perturbation_config): c for c in report_b.per_cell}
    shared_keys = cells_a.keys() & cells_b.keys()

    paired_payload: dict[str, _JsonValue] | None = None
    paired_by_config: dict[frozenset[tuple[str, float]], dict[str, _JsonValue]] = {}
    if episodes_a is not None and episodes_b is not None:
        paired = compute_paired_cells(episodes_a, episodes_b, suite_name=report_a.suite_name)
        paired_payload = cast("dict[str, _JsonValue]", json.loads(paired.model_dump_json()))
        for paired_cell in paired.cells:
            paired_by_config[_cell_key(paired_cell.perturbation_config)] = {
                "delta_ci_low": paired_cell.delta_ci_low,
                "delta_ci_high": paired_cell.delta_ci_high,
                "mcnemar_p_value": paired_cell.mcnemar.p_value,
                "n_paired": paired_cell.n_paired,
            }

    def _row(
        ca: CellBreakdown,
        cb: CellBreakdown,
        delta: float,
        key: frozenset[tuple[str, float]],
    ) -> dict[str, _JsonValue]:
        row: dict[str, _JsonValue] = {
            "axis_combination": dict(ca.perturbation_config),
            "rate_a": ca.success_rate,
            "rate_b": cb.success_rate,
            "delta": delta,
            "n_episodes_a": ca.n_episodes,
            "n_episodes_b": cb.n_episodes,
        }
        paired_row = paired_by_config.get(key)
        if paired_row is not None:
            row["paired"] = True
            row.update(paired_row)
        return row

    regressions: list[dict[str, _JsonValue]] = []
    improvements: list[dict[str, _JsonValue]] = []
    for key in shared_keys:
        ca = cells_a[key]
        cb = cells_b[key]
        if ca.n_episodes < min_cell_size or cb.n_episodes < min_cell_size:
            continue
        delta = cb.success_rate - ca.success_rate
        if delta < -threshold:
            regressions.append(_row(ca, cb, delta, key))
        elif delta > threshold:
            improvements.append(_row(ca, cb, delta, key))

    # Stable order: worst regression / best improvement first, then by
    # axis_combination repr for tie-breaking. ``r["delta"]`` is built as
    # a Python float two stanzas above; the cast narrows the
    # ``_JsonValue`` lookup so the arithmetic / tuple comparison type-
    # check (every payload in the list is constructed in this function).
    regressions.sort(key=lambda r: (cast("float", r["delta"]), repr(r["axis_combination"])))
    improvements.sort(key=lambda r: (-cast("float", r["delta"]), repr(r["axis_combination"])))

    payload: dict[str, _JsonValue] = {
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
        "paired": paired_payload is not None,
        # ``regressions`` / ``improvements`` are constructed locally as
        # ``list[dict[str, _JsonValue]]``; ``list`` is invariant in its
        # parameter so the cast bridges to ``_JsonValue`` (which the
        # downstream JSON writer + Chart.js consumer treat structurally).
        "regressions": cast("_JsonValue", regressions),
        "improvements": cast("_JsonValue", improvements),
    }
    if paired_payload is not None:
        payload["paired_comparison"] = paired_payload
    return payload


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
    paired: Annotated[
        bool | None,
        typer.Option(
            "--paired/--no-paired",
            help=(
                "Common-random-numbers (CRN) paired comparison (B-08). "
                "Default: auto-on when both inputs are episodes.json files "
                "sharing the same master_seed. --paired forces it (errors "
                "out if seeds mismatch or input is report.json); --no-paired "
                "forces the legacy independent-Wilson path."
            ),
        ),
    ] = None,
    github_summary: Annotated[
        Path | None,
        typer.Option(
            "--github-summary",
            help=(
                "Also emit a GitHub Actions step-summary markdown blob at "
                "PATH (failure-first table of regressions / improvements). "
                "Pipe into $GITHUB_STEP_SUMMARY in your workflow. Parent "
                "directory is created if missing. See backlog B-24."
            ),
        ),
    ] = None,
    drift_map: Annotated[
        Path | None,
        typer.Option(
            "--drift-map",
            help=(
                "Also emit a cross-backend embodiment-transfer drift map "
                "at PATH (per-axis-value rate-delta table; SIMPLER-style "
                "control / visual disparity). Requires --allow-cross-backend. "
                "When set, also prints the top-5 most-divergent axis values "
                "to stderr. See backlog B-29."
            ),
        ),
    ] = None,
    drift_map_policy_label: Annotated[
        str | None,
        typer.Option(
            "--drift-map-policy-label",
            help=(
                "Required with --drift-map. Free-form policy identifier "
                "stamped onto the emitted drift_map.json (no canonical "
                "policy_label exists on Report — the user supplies it)."
            ),
        ),
    ] = None,
) -> None:
    """Diff two runs and emit compare.json (HTML companion deferred to Phase 2)."""
    report_a, episodes_a = _load_report_with_episodes(results_a)
    report_b, episodes_b = _load_report_with_episodes(results_b)

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

    use_paired = _resolve_pairing_mode(paired, episodes_a, episodes_b, results_a, results_b)
    try:
        payload = _build_compare(
            report_a,
            report_b,
            threshold=threshold,
            min_cell_size=min_cell_size,
            episodes_a=episodes_a if use_paired else None,
            episodes_b=episodes_b if use_paired else None,
        )
    except PairingError as exc:
        raise _fail(str(exc)) from exc

    out_path = out if out is not None else results_b.parent / "compare.json"
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, payload)

    if github_summary is not None:
        # Lazy import — keeps `gauntlet --help` snappy for the common
        # no-summary path (mirrors aggregate / dashboard / realsim).
        from gauntlet.compare import to_github_summary

        if github_summary.parent and not github_summary.parent.exists():
            github_summary.parent.mkdir(parents=True, exist_ok=True)
        github_summary.write_text(to_github_summary(payload), encoding="utf-8")
        _echo_err(f"[ok]Wrote[/] GitHub summary -> {_fmt_path(github_summary)}")

    if drift_map is not None:
        # B-29: cross-backend embodiment-transfer drift map. Gated
        # behind --allow-cross-backend (a same-backend "drift map" is
        # a category error: the metric measures simulator drift, not
        # policy regression). policy_label is required because no
        # canonical field carries it on Report; suite_hash is derived
        # from the underlying Episode.suite_hash so a mismatched suite
        # is rejected at the same gate as the cross-backend check.
        if not allow_cross_backend:
            raise _fail(
                "--drift-map requires --allow-cross-backend "
                "(drift map measures sim-vs-sim disparity, not "
                "policy regression — see backlog B-29)"
            )
        if drift_map_policy_label is None:
            raise _fail(
                "--drift-map requires --drift-map-policy-label "
                "(no canonical policy_label exists on Report; the "
                "user supplies the identifier stamped onto drift_map.json)"
            )
        suite_hash = _resolve_drift_suite_hash(episodes_a, episodes_b)
        # ``_load_report_with_episodes`` rebuilds the Report from
        # episodes.json without threading suite_env (build_report's env
        # parameter is set on the live ``run`` path only). For the
        # drift map we need the slug — pull it off the per-episode
        # metadata stamped by the Runner so the cross-backend gate
        # inside :func:`compute_drift_map` has the data it needs.
        # ``cast`` here narrows ``episodes_a`` / ``_b`` from
        # ``list[Episode] | None`` to ``list[Episode]``: the
        # ``_resolve_drift_suite_hash`` call above already errored out
        # if either side was None (or empty), so by this point both
        # sides are guaranteed to be a populated list.
        episodes_a_list = cast("list[Episode]", episodes_a)
        episodes_b_list = cast("list[Episode]", episodes_b)
        if report_a.suite_env is None:
            env_a = _episodes_suite_env(episodes_a_list)
            if env_a is not None:
                report_a = report_a.model_copy(update={"suite_env": env_a})
        if report_b.suite_env is None:
            env_b = _episodes_suite_env(episodes_b_list)
            if env_b is not None:
                report_b = report_b.model_copy(update={"suite_env": env_b})

        # Lazy import — same rationale as github_summary above.
        from gauntlet.compare import compute_drift_map, render_drift_map_table

        try:
            dmap = compute_drift_map(
                report_a,
                report_b,
                policy_label=drift_map_policy_label,
                suite_hash=suite_hash,
            )
        except ValueError as exc:
            raise _fail(str(exc)) from exc

        if drift_map.parent and not drift_map.parent.exists():
            drift_map.parent.mkdir(parents=True, exist_ok=True)
        _write_json(drift_map, cast("_JsonValue", dmap.model_dump(mode="json")))
        _echo_err(f"[ok]Wrote[/] drift map -> {_fmt_path(drift_map)}")
        render_drift_map_table(dmap, _console, limit=5)

    _echo_err(f"[ok]Wrote[/] {_fmt_path(out_path)}")
    _echo_err(f"  a: {report_a.suite_name} ({_fmt_success_rate(report_a.overall_success_rate)})")
    _echo_err(f"  b: {report_b.suite_name} ({_fmt_success_rate(report_b.overall_success_rate)})")
    # ``payload`` is built by ``_build_compare`` directly above; the
    # casts narrow the recursive ``_JsonValue`` lookups so the float / len
    # call-sites type-check (the structure is local-and-known).
    _echo_err(
        f"  delta success_rate: {_fmt_signed_pct(cast('float', payload['delta_success_rate']))}"
    )
    n_regressions = len(cast("list[object]", payload["regressions"]))
    n_improvements = len(cast("list[object]", payload["improvements"]))
    regressions_style = "delta.down" if n_regressions else "delta.zero"
    improvements_style = "delta.up" if n_improvements else "delta.zero"
    _echo_err(
        f"  regressions: [{regressions_style}]{n_regressions}[/]  "
        f"improvements: [{improvements_style}]{n_improvements}[/]"
    )
    # B-08 paired-mode echo. When the CRN bracket is attached the
    # report.json carries `paired: true` plus a per-cell `delta_ci_low/
    # high` and `mcnemar_p_value` — surface a one-line status so the
    # user knows the variance-reduced statistics fired.
    if cast("bool", payload["paired"]):
        _echo_err("  paired: [ok]on[/] (CRN bracket + McNemar attached)")
    else:
        _echo_err("  paired: [warn]off[/] (independent-Wilson bracket)")


# ──────────────────────────────────────────────────────────────────────
# `diff` subcommand — git-diff-style structured per-axis report deltas.
# ──────────────────────────────────────────────────────────────────────
#
# ``compare`` answers a yes/no question (did anything regress beyond the
# threshold?). ``diff`` answers "what moved?" — per-axis rate deltas,
# per-cell flips, and the failure-cluster set difference. The default
# render is human-readable text via :func:`render_text`; ``--json`` emits
# the :class:`ReportDiff.model_dump_json(indent=2)` for machine
# consumption.
#
# Both inputs accept either an ``episodes.json`` (rebuilt into a Report)
# or a ``report.json`` — same auto-detect as ``compare`` / ``report``.


@app.command("diff")
def diff(
    results_a: Annotated[
        Path,
        typer.Argument(help="First run: episodes.json or report.json."),
    ],
    results_b: Annotated[
        Path,
        typer.Argument(help="Second run: episodes.json or report.json."),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit ReportDiff.model_dump_json(indent=2) on stdout for "
                "machine consumption instead of the human-readable text."
            ),
        ),
    ] = False,
    cell_flip_threshold: Annotated[
        float,
        typer.Option(
            "--cell-flip-threshold",
            help=(
                "Inclusive minimum |b.success_rate - a.success_rate| to "
                "surface a per-cell flip. Two-sided."
            ),
            min=0.0,
        ),
    ] = 0.10,
    cluster_intensify_threshold: Annotated[
        float,
        typer.Option(
            "--cluster-intensify-threshold",
            help=(
                "Inclusive minimum (b.lift - a.lift) for a shared failure "
                "cluster to be reported as intensified. One-sided (lift growth)."
            ),
            min=0.0,
        ),
    ] = 0.5,
    paired: Annotated[
        bool | None,
        typer.Option(
            "--paired/--no-paired",
            help=(
                "Common-random-numbers (CRN) paired diff (B-08). Default: "
                "auto-on when both inputs are episodes.json files sharing "
                "the same master_seed. --paired forces it (errors out if "
                "seeds mismatch or input is report.json); --no-paired "
                "forces the legacy unpaired path."
            ),
        ),
    ] = None,
    show_noise: Annotated[
        bool,
        typer.Option(
            "--show-noise/--no-show-noise",
            help=(
                "B-20: surface cell-flips tagged ``within_noise`` in the "
                "rendered diff. Off by default — noise flips clutter the "
                "output without representing a real regression. The "
                "``--json`` output always includes them regardless."
            ),
        ),
    ] = False,
) -> None:
    """Emit a git-diff-style structured delta between two runs."""
    report_a, episodes_a = _load_report_with_episodes(results_a)
    report_b, episodes_b = _load_report_with_episodes(results_b)

    if report_a.suite_name != report_b.suite_name:
        _echo_err(
            f"[warn]warning:[/] diffing across suites — a={report_a.suite_name!r} vs "
            f"b={report_b.suite_name!r}"
        )

    use_paired = _resolve_pairing_mode(paired, episodes_a, episodes_b, results_a, results_b)
    paired_comparison = None
    if use_paired and episodes_a is not None and episodes_b is not None:
        try:
            paired_comparison = compute_paired_cells(
                episodes_a, episodes_b, suite_name=report_a.suite_name
            )
        except PairingError as exc:
            raise _fail(str(exc)) from exc

    try:
        result = diff_reports(
            report_a,
            report_b,
            a_label=str(results_a),
            b_label=str(results_b),
            cell_flip_threshold=cell_flip_threshold,
            cluster_intensify_threshold=cluster_intensify_threshold,
            paired_comparison=paired_comparison,
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    if json_output:
        # Stdout is reserved for machine-readable payloads (CLI module
        # docstring); rich.Console default is stderr. Use typer.echo to
        # stdout so ``gauntlet diff a b --json | jq`` works unchanged.
        typer.echo(result.model_dump_json(indent=2))
    else:
        # Plain-text render goes through the rich Console so a TTY gets
        # markup-free output (the renderer emits no ANSI itself) and the
        # console handles soft-wrap / NO_COLOR. Print to stdout via
        # typer.echo so users can redirect ``> diff.txt`` cleanly.
        # ``--show-noise`` (B-20) toggles whether within-noise flips are
        # surfaced in the rendered output (JSON always carries them).
        typer.echo(render_text(result, show_noise=show_noise), nl=False)

    # Always emit a one-line stderr summary so the user can spot the
    # headline even when the body is redirected to a file / pipe.
    paired_tag = " (paired)" if result.paired else ""
    _echo_err(
        f"[ok]Diffed[/]{paired_tag} {_fmt_path(results_a)} -> {_fmt_path(results_b)}: "
        f"overall {_fmt_signed_pct(result.overall_success_rate_delta)}, "
        f"cell flips: {len(result.cell_flips)}, "
        f"clusters +{len(result.cluster_added)}/-{len(result.cluster_removed)}/"
        f"!{len(result.cluster_intensified)}"
    )


# ──────────────────────────────────────────────────────────────────────
# `aggregate` subcommand — fleet-wide failure-mode clustering.
# ──────────────────────────────────────────────────────────────────────
#
# Reads every ``report.json`` recursively under DIR and writes a
# fleet meta-report to ``<out>/fleet_report.json`` (and, by default,
# ``<out>/fleet_report.html``). The persistence-threshold knob keys
# the cross-run cluster algorithm — see
# ``docs/phase3-rfc-019-fleet-aggregate.md`` for the design.


@app.command("aggregate")
def aggregate(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Directory to recursively scan for report.json files.",
            exists=False,  # checked manually for a friendlier message
            file_okay=False,
            dir_okay=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory; created if missing. Receives fleet_report.json (+ .html).",
        ),
    ],
    persistence_threshold: Annotated[
        float,
        typer.Option(
            "--persistence-threshold",
            help=(
                "Minimum fraction of runs a failure cluster must appear in to be "
                "flagged as persistent. Range [0.0, 1.0]; comparison is inclusive."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    html: Annotated[
        bool,
        typer.Option(
            "--html/--no-html",
            help="Render fleet_report.html alongside the JSON. Defaults to ON.",
        ),
    ] = True,
) -> None:
    """Aggregate every report.json under DIR into a fleet meta-report."""
    if not directory.is_dir():
        raise _fail(f"directory not found: {directory}")

    # Lazy import — keeps `gauntlet --help` snappy and avoids pulling
    # the jinja2 template loader into unrelated subcommand startup.
    from gauntlet.aggregate import (
        aggregate_directory,
        write_fleet_html,
    )

    try:
        fleet = aggregate_directory(
            directory,
            persistence_threshold=persistence_threshold,
        )
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)
    fleet_json_path = out / "fleet_report.json"
    fleet_html_path = out / "fleet_report.html"

    _write_json(fleet_json_path, fleet.model_dump(mode="json"))
    if html:
        write_fleet_html(fleet, fleet_html_path)

    _echo_err(f"[ok]Wrote[/] {_fmt_path(fleet_json_path)}")
    if html:
        _echo_err(f"[ok]Wrote[/] {_fmt_path(fleet_html_path)}")
    _echo_err(
        f"  fleet: {fleet.n_runs} runs / {fleet.n_total_episodes} episodes "
        f"(mean success: {_fmt_success_rate(fleet.mean_success_rate)})"
    )
    n_clusters = len(fleet.persistent_failure_clusters)
    cluster_style = "delta.down" if n_clusters else "delta.zero"
    _echo_err(
        f"  persistent failure clusters: [{cluster_style}]{n_clusters}[/] "
        f"(threshold >={persistence_threshold:.2f})"
    )


# ──────────────────────────────────────────────────────────────────────
# `aggregate-sim-real` subcommand — paired sim/real correlation (B-28).
# ──────────────────────────────────────────────────────────────────────
#
# SureSim (arxiv 2510.04354, Oct 2025) and SIMPLER (arxiv 2405.05941,
# CoRL 2024) both demonstrate that paired sim-real correlation is the
# gold-standard sim-to-real validity signal. This subcommand ingests
# two directories of ``episodes.json`` files (sim side, real side),
# matches by ``(suite_hash, cell_index, episode_index)``, and writes a
# per-axis correlation table to ``<out>/sim_real_report.json``.


@app.command("aggregate-sim-real")
def aggregate_sim_real(
    sim_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of sim-side episodes.json files (recursively scanned).",
            exists=False,  # checked manually for a friendlier message
            file_okay=False,
            dir_okay=True,
        ),
    ],
    real_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of real-side episodes.json files (recursively scanned).",
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
            help="Output directory; created if missing. Receives sim_real_report.json.",
        ),
    ],
    top_n: Annotated[
        int,
        typer.Option(
            "--top-n",
            min=1,
            help="How many axes to print on stderr, sorted by |correlation|.",
        ),
    ] = 10,
) -> None:
    """Pair sim/real episodes and emit a per-axis correlation report (B-28)."""
    if not sim_dir.is_dir():
        raise _fail(f"sim directory not found: {sim_dir}")
    if not real_dir.is_dir():
        raise _fail(f"real directory not found: {real_dir}")

    # Lazy import — keeps `gauntlet --help` snappy.
    from gauntlet.aggregate import compute_sim_real_correlation

    try:
        report = compute_sim_real_correlation(sim_dir, real_dir)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    out.mkdir(parents=True, exist_ok=True)
    report_json_path = out / "sim_real_report.json"
    _write_json(report_json_path, report.model_dump(mode="json"))

    _echo_err(f"[ok]Wrote[/] {_fmt_path(report_json_path)}")
    _echo_err(
        f"  paired episodes: {report.n_paired_total} "
        f"(unmatched: sim={report.n_unmatched_sim}, real={report.n_unmatched_real})"
    )
    _echo_err(f"  overall correlation: {_fmt_correlation(report.overall_correlation)}")

    if not report.per_axis:
        _echo_err("  no axes with paired data")
        return

    # Sort by absolute correlation descending — anti-correlated axes
    # (negative r) and tightly correlated axes (positive r) both rise
    # to the top; the middle (~0) is the "matters in sim, not in real"
    # band that SureSim flags.
    def _abs_corr(axis: str) -> float:
        c = report.per_axis[axis].correlation
        return abs(c) if math.isfinite(c) else -1.0

    ordered = sorted(report.per_axis.keys(), key=_abs_corr, reverse=True)
    _echo_err(f"  top {min(top_n, len(ordered))} axes by |correlation|:")
    for axis in ordered[:top_n]:
        row = report.per_axis[axis]
        _echo_err(
            f"    [path]{axis}[/]: r={_fmt_correlation(row.correlation)} "
            f"(sim {_fmt_success_rate(row.sim_mean)} vs "
            f"real {_fmt_success_rate(row.real_mean)}; "
            f"n={row.n_paired_episodes})"
        )


def _fmt_correlation(corr: float) -> str:
    """Render a Pearson correlation with band colouring; NaN renders as a dash."""
    if not math.isfinite(corr):
        return "[delta.zero]nan[/]"
    abs_c = abs(corr)
    if abs_c >= 0.7:
        style = "pct.good"
    elif abs_c >= 0.3:
        style = "pct.mid"
    else:
        style = "pct.bad"
    return f"[{style}]{corr:+.3f}[/]"


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
# `monitor conformal` subcommand group — B-01 conformal failure detector.
# ──────────────────────────────────────────────────────────────────────
#
# Numpy-only sibling of the AE drift detector above. ``fit`` consumes a
# successful-rollout calibration set and writes a tiny ``detector.json``
# with the per-policy threshold; ``score`` consumes a candidate
# ``episodes.json`` and writes a copy with ``failure_score`` /
# ``failure_alarm`` populated. Mirrors the ``monitor train`` / ``score``
# flow exactly (see above) so users learn one CLI shape.

monitor_conformal_app = typer.Typer(
    name="conformal",
    help="Conformal-calibrated failure-prediction detector (B-01).",
    no_args_is_help=True,
    add_completion=False,
)
monitor_app.add_typer(monitor_conformal_app)


@monitor_conformal_app.command("fit")
def monitor_conformal_fit(
    calibration_path: Annotated[
        Path,
        typer.Argument(
            help="Calibration episodes.json — successful rollouts from a samplable policy.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output detector.json path."),
    ],
    alpha: Annotated[
        float,
        typer.Option(
            "--alpha",
            help="Target false-positive rate; threshold is the (1-alpha) conformal quantile.",
        ),
    ] = 0.05,
    successful_only: Annotated[
        bool,
        typer.Option(
            "--successful-only/--all-episodes",
            help=(
                "Restrict calibration to episodes with success=True (default). "
                "Pass --all-episodes to include every episode regardless of outcome."
            ),
        ),
    ] = True,
) -> None:
    """Fit a :class:`ConformalFailureDetector` on a calibration set."""
    from gauntlet.monitor.conformal import ConformalFailureDetector

    raw = _read_json(calibration_path)
    if not isinstance(raw, list):
        raise _fail(
            f"{calibration_path}: expected an episodes.json (top-level list); "
            f"got {type(raw).__name__}",
        )
    episodes = _episodes_from_dicts(raw, source=calibration_path)
    if successful_only:
        episodes = [ep for ep in episodes if ep.success]

    try:
        detector = ConformalFailureDetector.fit(episodes, alpha=alpha)
    except ValueError as exc:
        raise _fail(f"monitor conformal fit failed: {exc}") from exc

    detector.save(out)
    _echo_err(
        f"[ok]Wrote[/] detector.json -> {_fmt_path(out)} "
        f"(n_calibration={detector.n_calibration}, "
        f"alpha={detector.alpha:g}, threshold={detector.threshold:.4g})"
    )


@monitor_conformal_app.command("score")
def monitor_conformal_score(
    episodes_path: Annotated[
        Path,
        typer.Argument(
            help="Candidate episodes.json to score.",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    detector_path: Annotated[
        Path,
        typer.Option(
            "--detector",
            help="Detector JSON written by ``gauntlet monitor conformal fit``.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output episodes.json path with scores populated."),
    ],
) -> None:
    """Score an episodes.json against a fitted detector and emit a copy."""
    from gauntlet.monitor.conformal import ConformalFailureDetector

    raw = _read_json(episodes_path)
    if not isinstance(raw, list):
        raise _fail(
            f"{episodes_path}: expected an episodes.json (top-level list); "
            f"got {type(raw).__name__}",
        )
    episodes = _episodes_from_dicts(raw, source=episodes_path)

    try:
        detector = ConformalFailureDetector.load(detector_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise _fail(f"could not load detector: {exc}") from exc

    n_alarm = 0
    n_scored = 0
    for ep in episodes:
        score, alarm = detector.score(ep)
        ep.failure_score = score
        ep.failure_alarm = alarm
        if score is not None:
            n_scored += 1
        if alarm:
            n_alarm += 1

    _write_json(out, cast("_JsonValue", _episodes_to_dicts(episodes)))
    _echo_err(
        f"[ok]Wrote[/] {_fmt_path(out)} "
        f"(n_episodes={len(episodes)}, n_scored={n_scored}, "
        f"n_alarms={n_alarm}, threshold={detector.threshold:.4g})"
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

    payload: dict[str, _JsonValue] = {
        "episode_id": f"{cell_index}:{episode_index}",
        "suite_name": target.suite_name,
        "policy": policy,
        # ``overrides`` is ``dict[str, float]`` (parsed CLI flags); the
        # cast bridges to ``_JsonValue`` (``dict`` is invariant in its
        # value parameter so a structural narrow won't infer through).
        "overrides": cast("_JsonValue", overrides),
        "original": cast("_JsonValue", target.model_dump(mode="json")),
        "replayed": cast("_JsonValue", replayed.model_dump(mode="json")),
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
# `repro` subcommand (B-22) — episode-level reproducibility check.
# ──────────────────────────────────────────────────────────────────────


# Fields compared between the original Episode and the freshly-rolled
# repro Episode for the bit-identity assertion. We intentionally exclude
# the provenance trio (``gauntlet_version`` / ``suite_hash`` /
# ``git_commit``) and ``video_path`` — the whole point of ``gauntlet
# repro`` is to detect drift between the originating run's checkout and
# the current checkout, which would otherwise spuriously fail every
# repro that happens to be run from a different commit. The list mirrors
# every field that is part of the determinism contract documented in
# ``GAUNTLET_SPEC.md`` §6 plus :class:`Episode`'s identity tuple.
_REPRO_BITWISE_FIELDS: tuple[str, ...] = (
    "suite_name",
    "cell_index",
    "episode_index",
    "seed",
    "perturbation_config",
    "success",
    "terminated",
    "truncated",
    "step_count",
    "total_reward",
)


def _episode_diff_summary(original: Episode, replayed: Episode) -> list[str]:
    """Return one human-readable line per mismatching field.

    Empty list means bit-identical on the determinism-contract subset.
    """
    diffs: list[str] = []
    for field in _REPRO_BITWISE_FIELDS:
        a = getattr(original, field)
        b = getattr(replayed, field)
        if a != b:
            diffs.append(f"  {field}: original={a!r} repro={b!r}")
    return diffs


@app.command("repro")
def repro(
    episode_id: Annotated[
        str,
        typer.Argument(
            help=(
                "Episode id from a prior run, e.g. 'cell_3_episode_7'. "
                "Looked up in <out>/repro.json."
            ),
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help=(
                "Run output directory containing repro.json + "
                "episodes.json (the same path the original 'gauntlet run' "
                "wrote to)."
            ),
        ),
    ] = Path("out"),
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
    """Re-run one Episode in the current checkout and assert bit-identity.

    Reads ``<out>/repro.json`` for the seed manifest and
    ``<out>/episodes.json`` for the original Episode record, reconstructs
    the env + policy with the recorded seeds and axis_config, runs ONE
    episode, and exits 0 on bit-identical match (state-only:
    ``seed`` / ``success`` / ``step_count`` / ``total_reward`` / etc.)
    or 1 on mismatch with a diff summary on stderr. Provenance fields
    (``gauntlet_version`` / ``suite_hash`` / ``git_commit``) are
    intentionally excluded from the comparison — drift there is the
    *signal* the user is checking against, not a determinism failure.
    """
    repro_path = out / "repro.json"
    episodes_path = out / "episodes.json"
    if not repro_path.is_file():
        raise _fail(f"repro manifest not found: {repro_path}")
    if not episodes_path.is_file():
        raise _fail(f"episodes file not found: {episodes_path}")

    raw_repro = _read_json(repro_path)
    if not isinstance(raw_repro, dict):
        raise _fail(f"{repro_path}: expected a JSON object (got {type(raw_repro).__name__})")

    raw_episodes_payload = raw_repro.get("episodes")
    if not isinstance(raw_episodes_payload, list):
        raise _fail(
            f"{repro_path}: 'episodes' key missing or not a list "
            f"(got {type(raw_episodes_payload).__name__})"
        )

    entry: dict[str, _JsonValue] | None = None
    for raw_entry in raw_episodes_payload:
        if not isinstance(raw_entry, dict):
            continue
        if raw_entry.get("episode_id") == episode_id:
            entry = raw_entry
            break
    if entry is None:
        available = [
            r.get("episode_id")
            for r in raw_episodes_payload
            if isinstance(r, dict) and isinstance(r.get("episode_id"), str)
        ]
        preview = ", ".join(cast("list[str]", available[:10]))
        if len(available) > 10:
            preview += f", ... ({len(available) - 10} more)"
        raise _fail(f"episode_id {episode_id!r} not found in {repro_path}; available: {preview}")

    suite_path_raw = raw_repro.get("suite_path")
    policy_raw = raw_repro.get("policy")
    if not isinstance(suite_path_raw, str) or not isinstance(policy_raw, str):
        raise _fail(
            f"{repro_path}: 'suite_path' and 'policy' must both be strings "
            f"(got {type(suite_path_raw).__name__} / {type(policy_raw).__name__})"
        )
    suite_path = Path(suite_path_raw)
    policy = policy_raw

    seed_override_raw = raw_repro.get("seed_override")
    if seed_override_raw is not None and not isinstance(seed_override_raw, int):
        raise _fail(
            f"{repro_path}: 'seed_override' must be int or null "
            f"(got {type(seed_override_raw).__name__})"
        )
    seed_override: int | None = seed_override_raw

    # ``env_max_steps`` from the manifest is the run-time value; an
    # explicit CLI flag wins so users testing on slow hardware can
    # bound the rollout further. None on both sides means "use the
    # backend default" (matches Runner's contract).
    manifest_max_steps = raw_repro.get("env_max_steps")
    if manifest_max_steps is not None and not isinstance(manifest_max_steps, int):
        raise _fail(
            f"{repro_path}: 'env_max_steps' must be int or null "
            f"(got {type(manifest_max_steps).__name__})"
        )
    effective_max_steps: int | None = (
        env_max_steps if env_max_steps is not None else manifest_max_steps
    )

    if not suite_path.is_file():
        raise _fail(
            f"suite file referenced by repro manifest not found: {suite_path}. "
            f"Run 'gauntlet repro' from a checkout that contains the original "
            f"suite YAML, or move the YAML back to its original path."
        )
    try:
        suite = load_suite(suite_path)
    except (ValidationError, ValueError) as exc:
        raise _fail(f"{suite_path}: invalid suite YAML: {exc}") from exc
    except OSError as exc:
        raise _fail(f"{suite_path}: could not read file: {exc}") from exc

    if seed_override is not None:
        suite = suite.model_copy(update={"seed": seed_override})

    # Locate the original Episode in episodes.json so we can compare
    # against the bit-identity contract. The cell_index / episode_index
    # in the manifest are the lookup key.
    cell_index_raw = entry.get("cell_index")
    episode_index_raw = entry.get("episode_index")
    if not isinstance(cell_index_raw, int) or not isinstance(episode_index_raw, int):
        raise _fail(f"{repro_path}: entry {episode_id!r} missing integer cell_index/episode_index")
    cell_index: int = cell_index_raw
    episode_index: int = episode_index_raw

    raw_episodes = _read_json(episodes_path)
    if not isinstance(raw_episodes, list):
        raise _fail(
            f"{episodes_path}: expected a list of Episode objects "
            f"(got {type(raw_episodes).__name__})"
        )
    episodes = _episodes_from_dicts(raw_episodes, source=episodes_path)
    original: Episode | None = None
    for ep in episodes:
        if ep.cell_index == cell_index and ep.episode_index == episode_index:
            original = ep
            break
    if original is None:
        raise _fail(f"episode {cell_index}:{episode_index} not found in {episodes_path}")

    try:
        policy_factory = resolve_policy_factory(policy)
    except PolicySpecError as exc:
        raise _fail(str(exc)) from exc

    env_factory = _make_env_factory(suite.env, effective_max_steps)

    try:
        replayed = replay_one(
            target=original,
            suite=suite,
            policy_factory=policy_factory,
            overrides=None,
            env_factory=env_factory,
        )
    except OverrideError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(f"repro failed: {exc}") from exc

    diffs = _episode_diff_summary(original, replayed)
    if not diffs:
        _echo_err(
            f"[ok]repro match[/] {episode_id} "
            f"(seed={original.seed} success={original.success} "
            f"reward={original.total_reward:.4f})"
        )
        return

    _echo_err(f"[err]repro mismatch[/] {episode_id}: {len(diffs)} field(s) differ")
    for line in diffs:
        _echo_err(line)
    raise typer.Exit(code=1)


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


# ──────────────────────────────────────────────────────────────────────
# `dashboard` subcommand group — Phase 3 Task 20 (RFC 020 §3.2).
# ──────────────────────────────────────────────────────────────────────
#
# ``build`` — recursively scan a directory of ``report.json`` files
#             and materialise a self-contained static SPA
#             (``index.html`` + ``dashboard.js`` + ``dashboard.css``)
#             into the output directory. The Python API
#             (:func:`gauntlet.dashboard.build_dashboard`) is the
#             pipeline; this subcommand is the thin wrapper RFC 020
#             §3.2 specified.


dashboard_app = typer.Typer(
    name="dashboard",
    help="Self-contained static SPA indexing every report.json under a directory.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(dashboard_app)


@dashboard_app.command("build")
def dashboard_build(
    runs_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to recursively scan for report.json files.",
            exists=False,  # checked manually for a friendlier message
            file_okay=False,
            dir_okay=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help=(
                "Output directory; created if missing. Receives "
                "index.html + dashboard.js + dashboard.css."
            ),
        ),
    ],
    title: Annotated[
        str,
        typer.Option(
            "--title",
            help=(
                "Dashboard title rendered into <title> and the page header. "
                "Autoescaped so user-supplied values are safe."
            ),
        ),
    ] = "Gauntlet Dashboard",
) -> None:
    """Build a self-contained dashboard SPA from a directory of run artefacts."""
    if not runs_dir.is_dir():
        raise _fail(f"runs-dir not found: {runs_dir}")

    # Lazy import — keeps `gauntlet --help` snappy and avoids pulling
    # the jinja2 template loader into unrelated subcommand startup.
    from gauntlet.dashboard import build_dashboard

    try:
        build_dashboard(runs_dir, out, title=title)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    index_html = out / "index.html"
    _echo_err(f"[ok]Wrote[/] {_fmt_path(index_html)}")
    # Count what we just embedded so the user sees a one-line summary
    # without having to open the HTML. ``discover_reports`` is cheap
    # (single rglob) and the path-list is the source of truth for
    # ``n_runs`` in the embedded index.
    from gauntlet.dashboard import discover_reports

    paths = discover_reports(runs_dir)
    _echo_err(f"  dashboard: {len(paths)} runs -> {_fmt_path(out)}")


# ──────────────────────────────────────────────────────────────────────
# `realsim` subcommand group — Phase 3 Task 18 (RFC 021).
# ──────────────────────────────────────────────────────────────────────
#
# ``ingest``  — validate a directory of robot camera frames + a
#               calibration JSON and emit a self-contained scene
#               directory (``manifest.json`` + frame copies / symlinks).
# ``info``    — dump a one-screen summary of an existing scene
#               directory's manifest.
#
# The renderer Protocol + module-local registry live next to the
# pipeline (:mod:`gauntlet.realsim.renderer`); a future RFC decides
# whether a CLI ``realsim render`` subcommand makes sense alongside
# whatever first concrete renderer implementation lands. The schema
# changes nothing for that follow-up.


realsim_app = typer.Typer(
    name="realsim",
    help="Real-to-sim scene reconstruction inputs (gaussian-splatting renderer deferred).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(realsim_app)


@realsim_app.command("ingest")
def realsim_ingest(
    frames_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory holding the raw camera frames referenced by --calib.",
            exists=False,  # checked manually for a friendlier message
            dir_okay=True,
            file_okay=False,
        ),
    ],
    calib: Annotated[
        Path,
        typer.Option(
            "--calib",
            help="Path to the calibration JSON (intrinsics + per-frame poses).",
            dir_okay=False,
            file_okay=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output scene directory; created if missing.",
            dir_okay=True,
            file_okay=False,
        ),
    ],
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Freeform metadata tag (robot id, log id, etc.).",
        ),
    ] = None,
    symlink: Annotated[
        bool,
        typer.Option(
            "--symlink",
            help="Symlink frames into <out> instead of copying. Default OFF.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow overwriting an existing manifest at <out>/manifest.json.",
        ),
    ] = False,
) -> None:
    """Ingest a directory of frames + calibration into a scene directory."""
    if not frames_dir.is_dir():
        raise _fail(f"frames_dir not found: {frames_dir}")
    if not calib.is_file():
        raise _fail(f"calibration file not found: {calib}")

    # Lazy import — keeps `gauntlet --help` snappy and pulls the
    # pipeline / IO modules only when the subcommand actually runs.
    from gauntlet.realsim import (
        IngestionError,
        SceneIOError,
        ingest_frames,
        save_scene,
    )

    try:
        scene = ingest_frames(frames_dir, calib, source=source)
    except IngestionError as exc:
        raise _fail(str(exc)) from exc

    try:
        manifest_path = save_scene(
            scene,
            out,
            frames_dir=frames_dir,
            symlink=symlink,
            overwrite=overwrite,
        )
    except SceneIOError as exc:
        raise _fail(str(exc)) from exc

    _echo_err(f"[ok]Wrote[/] {_fmt_path(manifest_path)}")
    _echo_err(
        f"  scene: {len(scene.frames)} frames / {len(scene.intrinsics)} intrinsics "
        f"(source: {scene.source!r})"
    )


@realsim_app.command("info")
def realsim_info(
    scene_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a scene directory (containing manifest.json).",
            exists=False,  # checked manually for a friendlier message
            dir_okay=True,
            file_okay=False,
        ),
    ],
) -> None:
    """Print a one-screen summary of SCENE_DIR's manifest."""
    if not scene_dir.is_dir():
        raise _fail(f"scene_dir not found: {scene_dir}")

    from gauntlet.realsim import SceneIOError, load_scene

    try:
        scene = load_scene(scene_dir)
    except SceneIOError as exc:
        raise _fail(str(exc)) from exc

    intrinsics_ids = sorted(scene.intrinsics.keys())
    timestamps = [f.timestamp for f in scene.frames]
    if timestamps:
        time_summary = f"{min(timestamps):.3f}s -> {max(timestamps):.3f}s"
    else:
        time_summary = "(no frames)"

    _echo_err(f"[ok]scene:[/] {_fmt_path(scene_dir)}")
    _echo_err(f"  version: {scene.version}")
    _echo_err(f"  source: {scene.source!r}")
    _echo_err(f"  intrinsics: {len(intrinsics_ids)} ({', '.join(intrinsics_ids) or '-'})")
    _echo_err(f"  frames: {len(scene.frames)}")
    _echo_err(f"  time range: {time_summary}")


# ──────────────────────────────────────────────────────────────────────
# `suite` subcommand group — Phase 3 polish (B-25).
# ──────────────────────────────────────────────────────────────────────
#
# ``check`` — lint a suite YAML for the common authoring footguns the
#             schema validators do not catch (unused axes, cartesian
#             explosion, visual-only axes on a state-only backend,
#             insufficient ``episodes_per_cell`` for tight CIs, empty
#             suites). The Python API
#             (:func:`gauntlet.suite.lint_suite`) is the lint engine;
#             this subcommand is the thin wrapper.
#
# Exit codes:
#   * 0 — clean OR warnings only.
#   * 1 — load failure OR at least one ``error`` finding.


suite_app = typer.Typer(
    name="suite",
    help="Suite YAML utilities.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(suite_app)


def _print_lint_finding(finding: LintFinding) -> None:
    """Render one :class:`LintFinding` to stderr through the rich Console.

    Rich treats square-bracket runs as markup, so the rule name is
    wrapped in escaped brackets (``\\[name]``) — keeps the visual hint
    without colliding with rich's ``[style]`` syntax. Tests assert on
    the raw rule string surviving that escape unmangled.
    """
    prefix = "[err]error:[/]" if finding.severity == "error" else "[warn]warning:[/]"
    _echo_err(rf"{prefix} \[{finding.rule}] {finding.message}")


@suite_app.command("check")
def suite_check(
    suite_path: Annotated[
        Path,
        typer.Argument(
            help="Path to a suite YAML file.",
            exists=False,  # checked manually for a friendlier message
            dir_okay=False,
            file_okay=True,
        ),
    ],
) -> None:
    """Lint a suite YAML for common authoring footguns.

    Exits 0 on a clean suite or one with warnings only. Exits 1 if the
    suite fails to load OR if any rule produces an ``error`` finding
    (currently only the ``visual-only-axis-on-state-only-backend``
    rule). Warnings are printed regardless of exit code.
    """
    if not suite_path.is_file():
        raise _fail(f"suite file not found: {suite_path}")

    try:
        suite = load_suite(suite_path)
    except (ValidationError, ValueError) as exc:
        raise _fail(f"{suite_path}: invalid suite YAML: {exc}") from exc
    except OSError as exc:
        raise _fail(f"{suite_path}: could not read file: {exc}") from exc

    findings = lint_suite(suite)

    for finding in findings:
        _print_lint_finding(finding)

    n_errors = sum(1 for f in findings if f.severity == "error")
    n_warnings = sum(1 for f in findings if f.severity == "warning")

    if n_errors == 0 and n_warnings == 0:
        _echo_err(f"[ok]ok[/] {_fmt_path(suite_path)}: no lint issues")
        return

    summary = f"{_fmt_path(suite_path)}: {n_errors} error(s), {n_warnings} warning(s)"
    if n_errors > 0:
        _echo_err(f"[err]failed[/] {summary}")
        raise typer.Exit(code=1)
    _echo_err(f"[warn]ok with warnings[/] {summary}")


# ``python -m gauntlet.cli`` parity with the installed entry point.
if __name__ == "__main__":  # pragma: no cover
    app()
