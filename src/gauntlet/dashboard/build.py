"""Builder for the self-contained dashboard SPA — see RFC 020.

Reads every ``report.json`` recursively under a directory, extracts
the index needed by the SPA (per-run summary + per-axis aggregates +
file-mtime for the time series + sibling-``report.html`` href when
present), embeds the index into the Jinja-rendered ``index.html`` as
an inline JSON literal, and copies the static assets
(``dashboard.js`` + ``dashboard.css``) into the output directory.

The output directory contains exactly three files:

* ``index.html`` — the SPA shell with the embedded JSON literal.
* ``dashboard.js`` — chart wiring + filter UI (vanilla JS, no
  build step, Chart.js loaded from CDN inside ``index.html``).
* ``dashboard.css`` — the (small) stylesheet.

The SPA is openable from a ``file://`` path with no web server (per
RFC §3); embedding the JSON literal sidesteps Chromium's CORS
rejection of same-origin file fetches.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, TypeAlias

from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import ValidationError

from gauntlet.report.schema import Report

__all__ = [
    "build_dashboard",
    "build_dashboard_index",
    "discover_reports",
]


# Recursive JSON value alias — narrows ``Any`` for the NaN walker so
# the same shape goes in and out. Mirrors
# :data:`gauntlet.aggregate.html._JsonValue`.
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


_ENV = Environment(
    loader=PackageLoader("gauntlet.dashboard", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "xml", "jinja")),
    trim_blocks=False,
    lstrip_blocks=False,
)


# Static asset filenames inside the package's ``static/`` resource
# directory. Listed as a tuple so the build step copies them
# explicitly (and so the test asserting "the output dir contains
# exactly these files" stays in sync with the builder).
_STATIC_ASSETS = ("dashboard.js", "dashboard.css")


# ---------------------------------------------------------------------------
# Discovery + index extraction.
# ---------------------------------------------------------------------------


def discover_reports(directory: Path) -> list[Path]:
    """Recursively find files literally named ``report.json``.

    Returns an absolute, sorted list of matches (sort order is
    Python's default lexical sort on ``Path``, matching
    :func:`gauntlet.aggregate.discover_run_files` for consistency).

    Raises:
        FileNotFoundError: if *directory* does not exist or is not a
            directory.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    return sorted(directory.rglob("report.json"))


def _nan_to_none(value: _JsonValue) -> _JsonValue:
    """Recursively replace non-finite floats with ``None``.

    Same contract as :func:`gauntlet.aggregate.html._nan_to_none` —
    Chart.js / ``JSON.parse`` reject literal ``nan`` / ``inf`` but
    accept ``null`` (rendered as a gap in line charts and a gray bar
    in bar charts). The dashboard inherits the per-run reports'
    ``ser_json_inf_nan="strings"`` config when those reports go
    through ``model_dump(mode="json")``, but the per-axis aggregate
    we build below carries Python ``float("nan")`` for empty buckets;
    this walker keeps the embedded JSON literal valid in either path.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_nan_to_none(v) for v in value)
    return value


def _per_axis_aggregate(reports: list[Report]) -> dict[str, dict[str, dict[str, float]]]:
    """Pool per-axis ``counts`` / ``successes`` across reports.

    Returns a dict keyed by axis name; each value is
    ``{"rates": {axis_value: rate}, "counts": {axis_value: count}}``.
    The axis-value keys are stringified floats (``str(float(v))``) so
    the resulting nested dict round-trips through
    :func:`json.dumps` without a custom encoder; the JS side uses
    them directly as Chart.js labels.

    Empty fleet (``reports == []``) returns an empty dict.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    successes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_axes: list[str] = []
    for rep in reports:
        for ab in rep.per_axis:
            if ab.name not in counts:
                seen_axes.append(ab.name)
            for value, c in ab.counts.items():
                counts[ab.name][str(value)] += c
            for value, s in ab.successes.items():
                successes[ab.name][str(value)] += s

    out: dict[str, dict[str, dict[str, float]]] = {}
    for axis in seen_axes:
        axis_counts = counts[axis]
        axis_successes = successes[axis]
        sorted_keys = sorted(axis_counts.keys(), key=float)
        rates: dict[str, float] = {}
        for k in sorted_keys:
            n = axis_counts[k]
            rates[k] = axis_successes[k] / n if n > 0 else float("nan")
        out[axis] = {
            "rates": rates,
            "counts": {k: float(axis_counts[k]) for k in sorted_keys},
        }
    return out


def _summary(reports: list[Report]) -> dict[str, float | int]:
    """Compute the index card scalars (n_runs / n_episodes / mean / std).

    Empty input returns zeros across the board so the SPA can render
    "no runs" without a special-case branch.
    """
    n_runs = len(reports)
    n_total_episodes = sum(r.n_episodes for r in reports)
    if n_runs == 0:
        return {
            "n_runs": 0,
            "n_total_episodes": 0,
            "mean_success_rate": 0.0,
            "std_success_rate": 0.0,
        }
    rates = [r.overall_success_rate for r in reports]
    mean_rate = sum(rates) / n_runs
    variance = sum((r - mean_rate) ** 2 for r in rates) / n_runs
    std_rate = math.sqrt(variance)
    return {
        "n_runs": n_runs,
        "n_total_episodes": n_total_episodes,
        "mean_success_rate": mean_rate,
        "std_success_rate": std_rate,
    }


def build_dashboard_index(
    paths: list[Path],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """Convert a list of ``report.json`` paths into the SPA's index dict.

    The returned dict has this top-level shape (consumed by
    ``dashboard.js``):

    * ``runs`` — list of per-run dicts with ``run_id``,
      ``policy_label``, ``suite_name``, ``env``, ``n_episodes``,
      ``n_success``, ``success_rate``, ``mtime`` (epoch seconds, from
      ``path.stat().st_mtime`` — RFC §6), ``source_file`` (relative
      to *base_dir*), and ``report_html`` (relative href to a sibling
      ``report.html`` if present, else ``None``).
    * ``per_axis_aggregate`` — see :func:`_per_axis_aggregate`.
    * ``summary`` — see :func:`_summary`.
    * ``envs`` / ``suite_names`` / ``policy_labels`` — sorted unique
      values for the filter ``<select>`` boxes.

    Args:
        paths: ``report.json`` paths (absolute, e.g. as returned by
            :func:`discover_reports`).
        base_dir: scan root; per-run ``source_file`` is recorded
            relative to this directory so the embedded JSON is
            portable.

    Raises:
        ValueError: if any file fails to parse as JSON or fails
            Pydantic validation as a :class:`Report` — the failure
            mode mirrors :func:`gauntlet.aggregate.aggregate_directory`,
            with the offending path baked into the error message.
    """
    runs: list[dict[str, Any]] = []
    reports: list[Report] = []
    for path in paths:
        rel = path.relative_to(base_dir).as_posix()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
        try:
            report = Report.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"{rel}: not a valid report.json: {exc}") from exc
        reports.append(report)
        # ``policy_label`` and ``run_id`` derive from the run dir name
        # (the parent of ``report.json``) — same convention as
        # :func:`gauntlet.aggregate.aggregate_directory`. Files placed
        # directly in ``base_dir`` get the file stem as a fallback so
        # the label is never empty.
        parent = path.parent.name or path.stem
        sibling_html = path.parent / "report.html"
        report_html_rel: str | None = None
        if sibling_html.is_file():
            report_html_rel = sibling_html.relative_to(base_dir).as_posix()
        runs.append(
            {
                "run_id": parent,
                "policy_label": parent,
                "suite_name": report.suite_name,
                "env": report.suite_env,
                "n_episodes": report.n_episodes,
                "n_success": report.n_success,
                "success_rate": report.overall_success_rate,
                "mtime": path.stat().st_mtime,
                "source_file": rel,
                "report_html": report_html_rel,
            }
        )

    envs = sorted({r["env"] for r in runs if r["env"] is not None})
    suite_names = sorted({r["suite_name"] for r in runs})
    policy_labels = sorted({r["policy_label"] for r in runs})

    return {
        "summary": _summary(reports),
        "runs": runs,
        "per_axis_aggregate": _per_axis_aggregate(reports),
        "envs": envs,
        "suite_names": suite_names,
        "policy_labels": policy_labels,
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def build_dashboard(
    reports_dir: Path,
    out: Path,
    *,
    title: str = "Gauntlet Dashboard",
) -> None:
    """Materialise the self-contained dashboard SPA.

    Writes ``<out>/index.html``, ``<out>/dashboard.js``, and
    ``<out>/dashboard.css``. Re-runs over the same ``out`` overwrite
    the three files in place; other files in ``out`` are left
    untouched (we never ``rmtree`` user dirs).

    Args:
        reports_dir: directory to recursively scan for ``report.json``
            files.
        out: output directory; created if missing.
        title: dashboard title (rendered into ``<title>`` and the
            page header). Autoescaped by Jinja so user-supplied
            titles are safe to embed.

    Raises:
        FileNotFoundError: if *reports_dir* does not exist.
        ValueError: if no ``report.json`` files are found, or any
            file is malformed JSON, or any file fails Pydantic
            validation as a :class:`Report`.
    """
    paths = discover_reports(reports_dir)
    if not paths:
        raise ValueError(f"no report.json files found under {reports_dir}")

    index = build_dashboard_index(paths, base_dir=reports_dir)
    cleaned = _nan_to_none(index)

    template = _ENV.get_template("dashboard.html.jinja")
    rendered: str = template.render(title=title, data=cleaned)

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(rendered, encoding="utf-8")

    static_root = resources.files("gauntlet.dashboard").joinpath("static")
    for asset in _STATIC_ASSETS:
        src = static_root.joinpath(asset)
        with resources.as_file(src) as src_path:
            shutil.copyfile(src_path, out / asset)
