"""Static HTML report generator — see ``GAUNTLET_SPEC.md`` §5 task 8.

Produces a single self-contained HTML file from a :class:`Report`. The
template (``templates/report.html.jinja``) ships with the package via
:class:`jinja2.PackageLoader`. Chart.js loads from CDN at view time, so
the only Python-side dependency is :mod:`jinja2`.

Design rules from spec §6 enforced here:

* The failure-clusters table leads the rendered HTML; the overall
  success rate is never headlined above it.
* The wall-of-numbers per-cell table is collapsed inside a
  ``<details>`` element by default.

NaN handling: :meth:`Report.model_dump` (mode="json") emits Python's
literal ``NaN`` for empty heatmap cells, which is invalid JSON and would
break ``JSON.parse`` on the JS side. We walk the dump and replace any
non-finite float with ``None`` (→ ``null`` in JSON). The JS treats
``null`` as "no data" and renders the cell gray.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

from jinja2 import Environment, PackageLoader, select_autoescape

from gauntlet.report.schema import Report

__all__ = ["render_html", "write_html"]


# Recursive type for the JSON-style structure pydantic's
# ``model_dump(mode="json")`` returns. Narrowed from ``Any`` so the
# ``_nan_to_none`` walker carries the actual shape it accepts. Tuples
# appear because pydantic encodes ``tuple[...]`` fields as Python tuples
# in dump mode (the v2 contract); the walker normalises them back to a
# tuple of normalised members so a tuple in == tuple out is preserved.
# Same shape as :data:`gauntlet.aggregate.html._JsonValue`; intentionally
# duplicated to keep the two report packages decoupled.
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


# Single module-level environment — building one per call would re-load
# the template from disk every time. ``autoescape`` is on for HTML/XML
# extensions so suite names containing markup are escaped (XSS hardening
# even though suite names are author-controlled).
_ENV = Environment(
    loader=PackageLoader("gauntlet.report", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "xml", "jinja")),
    trim_blocks=False,
    lstrip_blocks=False,
)


def _nan_to_none(value: _JsonValue) -> _JsonValue:
    """Recursively replace non-finite floats with ``None``.

    ``Report.model_dump(mode="json")`` returns a JSON-style nested dict
    but emits ``float('nan')`` as the literal Python ``nan`` (which the
    stdlib :mod:`json` and the browser's ``JSON.parse`` both reject). We
    convert NaN/±inf to ``None`` so the embedded ``<script>`` block is
    always valid JSON. ``None`` round-trips to ``null``, which the JS
    renders as a "no data" gray cell.
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


def render_html(report: Report) -> str:
    """Render *report* to a self-contained HTML string.

    The output is a complete HTML document (``<!DOCTYPE html>`` …
    ``</html>``) with all CSS inline and Chart.js loaded from CDN. No
    server, build step, or local file dependency required to view it.

    Args:
        report: A fully populated :class:`Report` (typically from
            :func:`gauntlet.report.build_report`).

    Returns:
        A string containing the rendered HTML.
    """
    template = _ENV.get_template("report.html.jinja")
    data = _nan_to_none(report.model_dump(mode="json"))
    rendered: str = template.render(report=report, data=data)
    return rendered


def write_html(report: Report, path: Path) -> None:
    """Render *report* and write it to *path* as UTF-8.

    Convenience wrapper around :func:`render_html`. The parent directory
    is NOT created — callers are expected to manage their own output
    layout (the CLI in task 9 owns ``--out``-dir creation).
    """
    path.write_text(render_html(report), encoding="utf-8")
