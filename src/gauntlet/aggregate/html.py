"""Static HTML renderer for :class:`FleetReport`.

Mirrors :mod:`gauntlet.report.html` — same Jinja autoescape, same
NaN→null normalisation for the embedded JSON block, same self-
contained-page contract (Chart.js loaded from CDN at view time).

Design rules from spec §6 enforced here:

* The persistent-failure-cluster table leads the rendered HTML; the
  fleet mean success rate appears later (in the summary card alongside
  the failure rate, never as the lone headline).
* The wall-of-numbers per-run table is collapsed inside a ``<details>``
  element by default.

XSS surface: ``policy_label``, ``source_file``, ``suite_name``, axis
names, and cluster axis names all flow from user-controlled inputs.
:class:`jinja2.Environment(autoescape=...)` covers every ``{{ ... }}``
interpolation; the embedded ``<script id="fleet-data">`` block uses
the same ``|tojson`` escape pattern the per-run report uses to defeat
``</script>`` breakouts. Tests in :mod:`tests.test_fleet_aggregate`
pin both behaviours.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

from jinja2 import Environment, PackageLoader, select_autoescape

from gauntlet.aggregate.schema import FleetReport

__all__ = ["render_fleet_html", "write_fleet_html"]


# Recursive type for the JSON-style structure pydantic's
# ``model_dump(mode="json")`` returns. Narrowed from ``Any`` so the
# ``_nan_to_none`` walker carries the actual shape it accepts. Tuples
# appear because pydantic encodes ``tuple[...]`` fields as Python tuples
# in dump mode (the v2 contract); the walker normalises them back to a
# tuple of normalised members so a tuple in == tuple out is preserved.
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
    loader=PackageLoader("gauntlet.aggregate", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html", "xml", "jinja")),
    trim_blocks=False,
    lstrip_blocks=False,
)


def _nan_to_none(value: _JsonValue) -> _JsonValue:
    """Recursively replace non-finite floats with ``None``.

    The fleet template embeds the full :class:`FleetReport` as JSON for
    Chart.js; ``model_dump(mode="json")`` would emit Python's literal
    ``nan`` (rejected by both :mod:`json` and the browser's
    ``JSON.parse``). ``None`` round-trips to ``null``, which the JS
    treats as "no data" (gray bar / placeholder text).

    Same shape as :func:`gauntlet.report.html._nan_to_none`. We avoid
    importing from there to keep the two report packages decoupled.
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


def render_fleet_html(fleet: FleetReport) -> str:
    """Render *fleet* to a self-contained HTML string.

    The output is a complete HTML document (``<!DOCTYPE html>`` …
    ``</html>``) with all CSS inline and Chart.js loaded from CDN.
    """
    template = _ENV.get_template("fleet_report.html.jinja")
    data = _nan_to_none(fleet.model_dump(mode="json"))
    rendered: str = template.render(fleet=fleet, data=data)
    return rendered


def write_fleet_html(fleet: FleetReport, path: Path) -> None:
    """Render *fleet* and write it to *path* as UTF-8."""
    path.write_text(render_fleet_html(fleet), encoding="utf-8")
