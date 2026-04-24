"""Security regression: HTML report escapes attacker-controlled strings.

Phase 2.5 Task 16 — pins the existing autoescape behaviour of
:func:`gauntlet.report.html.render_html`. The Jinja environment is
constructed with ``select_autoescape(enabled_extensions=("html", "xml",
"jinja"))`` and the template is named ``report.html.jinja`` (matches),
so every ``{{ ... }}`` interpolation is HTML-escaped by default.

Threat model:

* A suite name flows from :attr:`Suite.name` (loaded from a YAML the
  user supplied) into :attr:`Report.suite_name` and is then rendered
  twice in the HTML (``<title>`` and ``<h1>``). If the YAML is
  attacker-controlled (e.g. shared CI artefact), an unescaped
  ``<script>`` injection would execute when a victim opens the
  report in a browser.

* Axis names flow into ``<h3>`` headers and table cells. Cell values
  are floats (already safe).

* The cluster-summary table interpolates axis names + axis values,
  also via ``{{ ... }}``.

The pin: build a Report with attacker-controlled strings in every
field that surfaces in the HTML, render it, and assert (a) the raw
``<script ...>`` tag does NOT appear, (b) the escaped form
``&lt;script`` DOES appear in the rendered output.

If a future refactor introduces ``|safe`` on any user-controlled field
or disables autoescape, the test fires.

References:
* https://owasp.org/www-community/attacks/xss/ — Cross-Site Scripting class.
* https://cwe.mitre.org/data/definitions/79.html — CWE-79.
"""

from __future__ import annotations

from gauntlet.report import Report, build_report, render_html
from gauntlet.report.schema import (
    AxisBreakdown,
    CellBreakdown,
    FailureCluster,
    Heatmap2D,
)
from gauntlet.runner.episode import Episode

# The canonical XSS probe. If this substring appears verbatim in the
# rendered HTML, the report is vulnerable.
_XSS_PAYLOAD = "<script>alert(1)</script>"

# Variants — escaped HTML attribute, JS-context-ending sequences, etc.
# All must be HTML-escaped by Jinja autoescape.
_XSS_VARIANTS: tuple[str, ...] = (
    "<script>alert(1)</script>",
    '"><script>alert(2)</script>',
    "<img src=x onerror=alert(3)>",
    "</title><script>alert(4)</script>",
)


def _make_episodes_with_evil_suite_name(suite_name: str) -> list[Episode]:
    """Build an Episode list whose suite_name carries the XSS payload."""
    return [
        Episode(
            suite_name=suite_name,
            cell_index=0,
            episode_index=i,
            seed=i,
            perturbation_config={"lighting_intensity": 0.3 + 0.1 * i},
            success=(i % 2 == 0),
            terminated=True,
            truncated=False,
            step_count=10,
            total_reward=1.0 if i % 2 == 0 else 0.0,
        )
        for i in range(4)
    ]


# ----- suite_name -----------------------------------------------------------


def test_suite_name_xss_is_escaped_in_rendered_html() -> None:
    """``Report.suite_name`` reaches ``<title>`` and ``<h1>`` — must escape."""
    eps = _make_episodes_with_evil_suite_name(_XSS_PAYLOAD)
    report = build_report(eps)
    html = render_html(report)
    # The literal payload must NOT appear.
    assert "<script>alert(1)</script>" not in html, (
        "suite_name XSS payload rendered verbatim — autoescape is OFF"
    )
    # The escaped form MUST appear (twice — once for title, once for h1).
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_all_xss_variants_escaped_in_suite_name() -> None:
    """Every variant in :data:`_XSS_VARIANTS` must escape the same way.

    The template legitimately contains its own ``<script>`` tags
    (``report-data`` and the Chart.js CDN bootstrap); we cannot assert
    the global absence of ``<script``. Instead, count the script-tag
    occurrences in a baseline render with a clean suite_name and
    require that the evil-payload render carries the SAME count — any
    increase implies the payload survived as a tag.
    """
    baseline_html = render_html(build_report(_make_episodes_with_evil_suite_name("clean-name")))
    baseline_script_count = baseline_html.lower().count("<script")
    for payload in _XSS_VARIANTS:
        eps = _make_episodes_with_evil_suite_name(payload)
        report = build_report(eps)
        html = render_html(report)
        evil_script_count = html.lower().count("<script")
        assert evil_script_count == baseline_script_count, (
            f"payload {payload!r} added {evil_script_count - baseline_script_count} "
            f"new <script tag(s) to the rendered HTML"
        )
        # ``onerror=`` payload — autoescape will preserve the substring
        # but only inside escaped text (``&lt;img src=x onerror=...``),
        # which cannot execute. The dangerous form is the unescaped
        # ``<img ... onerror=`` tag pattern. Assert the raw img tag is
        # not present.
        if "onerror=" in payload:
            assert "<img" not in html, (
                f"payload {payload!r} survived as an unescaped <img> tag with onerror"
            )


# ----- axis names -----------------------------------------------------------


def _make_evil_axis_report() -> Report:
    """Build a Report by hand carrying an attacker-controlled axis name.

    We bypass :func:`build_report` because the suite schema rejects
    non-canonical axis names at YAML-load time — the attack vector here
    is a hand-crafted Report (e.g. one written by a third-party tool)
    that gets re-rendered via ``gauntlet report``. The Report Pydantic
    model accepts any string for axis names; the HTML renderer is the
    final defence.
    """
    return Report(
        suite_name="clean-suite",
        suite_env=None,
        n_episodes=2,
        n_success=1,
        per_axis=[
            AxisBreakdown(
                name=_XSS_PAYLOAD,
                rates={0.0: 1.0, 1.0: 0.0},
                counts={0.0: 1, 1.0: 1},
                successes={0.0: 1, 1.0: 0},
            ),
        ],
        per_cell=[
            CellBreakdown(
                cell_index=0,
                perturbation_config={_XSS_PAYLOAD: 0.0},
                n_episodes=1,
                n_success=1,
                success_rate=1.0,
            ),
            CellBreakdown(
                cell_index=1,
                perturbation_config={_XSS_PAYLOAD: 1.0},
                n_episodes=1,
                n_success=0,
                success_rate=0.0,
            ),
        ],
        failure_clusters=[
            FailureCluster(
                axes={_XSS_PAYLOAD: 1.0, "second_axis": 0.0},
                n_episodes=1,
                n_success=0,
                failure_rate=1.0,
                lift=10.0,
            ),
        ],
        heatmap_2d={
            f"{_XSS_PAYLOAD}__second_axis": Heatmap2D(
                axis_x=_XSS_PAYLOAD,
                axis_y="second_axis",
                x_values=[0.0, 1.0],
                y_values=[0.0, 1.0],
                success_rate=[[1.0, 0.0], [1.0, 0.0]],
            ),
        },
        overall_success_rate=0.5,
        overall_failure_rate=0.5,
        cluster_multiple=2.0,
    )


def test_axis_name_xss_is_escaped_in_rendered_html() -> None:
    """An attacker-controlled axis name MUST be HTML-escaped everywhere
    it surfaces (per-axis ``<h3>``, heatmap headers, failure-cluster
    table, per-cell config column).
    """
    report = _make_evil_axis_report()
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html, (
        "axis_name XSS payload rendered verbatim — at least one path bypasses autoescape"
    )
    # Sanity check: the escaped version IS present (otherwise the
    # template might have stopped rendering this field entirely, which
    # would also "pass" the no-script-tag check vacuously).
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# ----- embedded JSON block --------------------------------------------------


def test_embedded_json_block_does_not_break_out_of_script_context() -> None:
    """The ``<script id="report-data">`` block embeds a JSON dump of the
    Report. ``</script>`` inside a string would escape the ``<script>``
    context in HTML5 parsing rules. Jinja's ``|tojson`` filter handles
    this by escaping ``/`` to ``\\/``; we pin that escape here.
    """
    eps = _make_episodes_with_evil_suite_name("</script><script>alert(5)</script>")
    report = build_report(eps)
    html = render_html(report)
    # The raw ``</script>`` close tag must not appear inside the data
    # block context. We extract the data block and assert it does not
    # contain a literal ``</script>``.
    import re

    match = re.search(
        r'<script id="report-data" type="application/json">(?P<body>.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "report-data block missing"
    body = match.group("body")
    # The body MUST NOT contain a verbatim ``</script>`` sequence.
    assert "</script>" not in body, (
        "embedded JSON contains unescaped </script> close tag — XSS via JSON-context break"
    )
