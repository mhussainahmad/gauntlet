"""Self-contained static SPA dashboard — see RFC 020 / spec §7.

Public surface:

* :func:`build_dashboard` — given a directory of ``report.json``
  files, materialise a self-contained dashboard SPA (``index.html``
  + ``dashboard.js`` + ``dashboard.css``) into an output directory.
* :func:`discover_reports` — recursive ``report.json`` glob (mirrors
  :func:`gauntlet.aggregate.discover_run_files`).
* :func:`build_dashboard_index` — pure helper that converts a list
  of discovered ``report.json`` paths into the JSON-serialisable
  index the SPA consumes. Exposed for testing and for downstream
  notebook / custom-renderer use.

The HTML is openable from a ``file://`` path with no web server
(per RFC §3); all run data is embedded inline as a JSON literal so
there is no second HTTP fetch to fail under file:// CORS.
"""

from __future__ import annotations

from gauntlet.dashboard.build import build_dashboard as build_dashboard
from gauntlet.dashboard.build import build_dashboard_index as build_dashboard_index
from gauntlet.dashboard.build import discover_reports as discover_reports

__all__ = [
    "build_dashboard",
    "build_dashboard_index",
    "discover_reports",
]
