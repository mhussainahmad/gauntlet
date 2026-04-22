"""HTML report generator tests — see ``GAUNTLET_SPEC.md`` §5 task 8 and §6.

These tests build synthetic :class:`Episode` lists + :func:`build_report`
to construct test reports — they do NOT spin up a Runner. The contract
under test is the rendered HTML's structure and content; the analyze
layer is exercised in :mod:`tests.test_report`.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from gauntlet.report import Report, build_report, render_html, write_html
from gauntlet.runner.episode import Episode

# ────────────────────────────────────────────────────────────────────────
# Episode + Report fixtures
# ────────────────────────────────────────────────────────────────────────

_SUITE = "test-suite-html"


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    suite_name: str = _SUITE,
    seed: int = 0,
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def _report_with_clusters() -> Report:
    """Two-axis grid where (lighting=1.5, camera=1.0) is a clear failure cluster.

    The grid mirrors ``test_failure_cluster_fires_when_rate_exceeds_threshold``
    from :mod:`tests.test_report` — guarantees ``failure_clusters`` is
    non-empty so the cluster table renders.
    """
    eps: list[Episode] = []
    cell = 0
    for lighting in (0.3, 1.5):
        for camera in (0.0, 1.0):
            for i in range(3):
                success = not (lighting == 1.5 and camera == 1.0)
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=i,
                        success=success,
                        config={"lighting_intensity": lighting, "camera_offset_x": camera},
                    )
                )
            cell += 1
    return build_report(eps, cluster_multiple=2.0, min_cluster_size=3)


def _report_all_success_single_axis() -> Report:
    """All-success, single-axis suite: empty failure_clusters AND empty heatmap_2d."""
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"lighting_intensity": 0.3}),
        _ep(cell_index=0, episode_index=1, success=True, config={"lighting_intensity": 0.3}),
        _ep(cell_index=1, episode_index=0, success=True, config={"lighting_intensity": 1.5}),
    ]
    return build_report(eps)


def _report_with_nan_heatmap() -> Report:
    """Two-axis report with an empty heatmap cell — exercises NaN→null path."""
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"a": 0.0, "b": 0.0}),
        _ep(cell_index=0, episode_index=1, success=False, config={"a": 0.0, "b": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"a": 1.0, "b": 1.0}),
    ]
    return build_report(eps)


# ────────────────────────────────────────────────────────────────────────
# Helpers for asserting on the embedded JSON.
# ────────────────────────────────────────────────────────────────────────

_REPORT_DATA_RE = re.compile(
    r'<script id="report-data" type="application/json">(?P<body>.*?)</script>',
    re.DOTALL,
)


def _extract_embedded_json(html: str) -> dict[str, object]:
    """Pull the ``<script id="report-data">`` block out of *html* and parse it."""
    match = _REPORT_DATA_RE.search(html)
    assert match is not None, "embedded report-data <script> block missing"
    body = match.group("body")
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    return parsed


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_render_html_returns_doctype_str() -> None:
    """Test 1 — render_html returns a str containing <!DOCTYPE html>."""
    html = render_html(_report_with_clusters())
    assert isinstance(html, str)
    assert html.startswith("<!DOCTYPE html>")


def test_render_html_contains_suite_name_success_rate_and_n_episodes() -> None:
    """Test 2 — suite name, overall success rate, and n_episodes appear."""
    report = _report_with_clusters()
    html = render_html(report)
    assert report.suite_name in html
    # n_episodes is rendered literally in the summary card.
    assert str(report.n_episodes) in html
    # Overall pass rate is shown as "X.X%" in the per-axis subtitle.
    expected_pct = f"{report.overall_success_rate * 100:.1f}%"
    assert expected_pct in html


def test_render_html_embeds_parseable_json_with_top_level_keys() -> None:
    """Test 3 — the <script id="report-data"> block parses to the expected dict."""
    report = _report_with_clusters()
    html = render_html(report)
    parsed = _extract_embedded_json(html)
    expected = {
        "suite_name",
        "n_episodes",
        "n_success",
        "per_axis",
        "per_cell",
        "failure_clusters",
        "heatmap_2d",
        "overall_success_rate",
        "overall_failure_rate",
        "cluster_multiple",
    }
    assert expected.issubset(parsed.keys())


def test_failure_cluster_table_renders_when_clusters_exist() -> None:
    """Test 4 — at least one cluster's axis combination appears in the HTML."""
    report = _report_with_clusters()
    html = render_html(report)
    # The failure cluster (lighting=1.5, camera=1.0) is the certainty.
    # The exact rendered token will be `lighting_intensity=1.5`.
    assert "lighting_intensity=1.5" in html
    assert "camera_offset_x=1.0" in html
    # Both pieces of the combination should appear connected by the
    # logical-and glyph as in the spec example.
    assert "∧" in html


def test_empty_state_message_renders_when_no_clusters() -> None:
    """Test 5 — empty-state copy renders when failure_clusters is empty."""
    report = _report_all_success_single_axis()
    assert report.failure_clusters == []
    html = render_html(report)
    assert "No statistically significant failure clusters" in html


def test_per_axis_canvas_elements_exist_one_per_axis() -> None:
    """Test 6 — one ``<canvas id="axis-...">`` per axis in report.per_axis."""
    report = _report_with_clusters()
    html = render_html(report)
    for axis in report.per_axis:
        assert f'<canvas id="axis-{axis.name}"' in html


def test_heatmap_section_renders_one_block_per_pair() -> None:
    """Test 7 — heatmap section renders one block per pair; absent when empty."""
    report_with_pairs = _report_with_clusters()
    html_pairs = render_html(report_with_pairs)
    for key, heat in report_with_pairs.heatmap_2d.items():
        assert f'data-heatmap-key="{key}"' in html_pairs
        assert f"{heat.axis_x} × {heat.axis_y}" in html_pairs  # noqa: RUF001

    report_no_pairs = _report_all_success_single_axis()
    assert report_no_pairs.heatmap_2d == {}
    html_no_pairs = render_html(report_no_pairs)
    # Strip <script> blocks before asserting "Axis-pair heatmaps" /
    # `data-heatmap-key` are absent — the JS itself references that
    # attribute name (querySelectorAll), and we only care that the body
    # section is empty when there are no pairs.
    body_only = re.sub(
        r"<script\b[^>]*>.*?</script>", "", html_no_pairs, flags=re.DOTALL | re.IGNORECASE
    )
    assert "data-heatmap-key" not in body_only
    assert "Axis-pair heatmaps" not in body_only


def test_write_html_round_trips_to_disk(tmp_path: Path) -> None:
    """Test 8 — write_html() produces a file equal to render_html() output."""
    report = _report_with_clusters()
    target = tmp_path / "out.html"
    write_html(report, target)
    on_disk = target.read_text(encoding="utf-8")
    assert on_disk == render_html(report)


def test_html_escapes_script_in_suite_name() -> None:
    """Test 9 — autoescape protects against XSS via the suite name field."""
    eps = [
        _ep(
            cell_index=0,
            episode_index=0,
            success=True,
            config={"lighting_intensity": 0.5},
            suite_name="<script>alert(1)</script>",
        )
    ]
    report = build_report(eps)
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    # The escaped form must be present (in the <h1> title at minimum).
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_self_contained_no_local_stylesheet_links() -> None:
    """Test 10 — no <link rel="stylesheet"> references; only CDN script allowed."""
    html = render_html(_report_with_clusters())
    # Spec: no <link rel="stylesheet"> at all (CSS is inline).
    assert "<link" not in html.lower() or 'rel="stylesheet"' not in html.lower()
    # External script tags must point to the documented Chart.js CDN only.
    cdn = "cdn.jsdelivr.net/npm/chart.js"
    # Walk script tags; any tag with `src=` must reference the CDN.
    script_src_re = re.compile(r"<script\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
    srcs = script_src_re.findall(html)
    assert srcs, "expected at least one external <script src=...> for Chart.js"
    for src in srcs:
        assert cdn in src, f"unexpected external script src: {src!r}"


def test_render_html_is_deterministic() -> None:
    """Test 11 — two consecutive calls yield byte-identical output."""
    report = _report_with_clusters()
    assert render_html(report) == render_html(report)


def test_round_trip_via_embedded_json_yields_top_level_fields() -> None:
    """Test 12 — Report.model_dump round-trip via embedded <script> matches."""
    report = _report_with_clusters()
    html = render_html(report)
    parsed = _extract_embedded_json(html)
    # Top-level scalar fields survive the JSON embed faithfully.
    assert parsed["suite_name"] == report.suite_name
    assert parsed["n_episodes"] == report.n_episodes
    assert parsed["n_success"] == report.n_success
    assert parsed["overall_failure_rate"] == pytest.approx(report.overall_failure_rate)
    assert parsed["overall_success_rate"] == pytest.approx(report.overall_success_rate)
    # Failure cluster count survives.
    assert isinstance(parsed["failure_clusters"], list)
    assert len(parsed["failure_clusters"]) == len(report.failure_clusters)


# ────────────────────────────────────────────────────────────────────────
# NaN handling — separately tested per spec.
# ────────────────────────────────────────────────────────────────────────


def test_embedded_json_replaces_nan_with_null() -> None:
    """Empty heatmap cells (NaN) become JSON null so JSON.parse cannot fail."""
    report = _report_with_nan_heatmap()
    html = render_html(report)
    parsed = _extract_embedded_json(html)
    matrix = parsed["heatmap_2d"]["a__b"]["success_rate"]  # type: ignore[index]
    # The (a=0.0, b=1.0) and (a=1.0, b=0.0) cells were absent → null after dump.
    flat = [v for row in matrix for v in row]
    assert None in flat
    # And no literal "NaN" leaked into the embedded JSON body.
    body = _REPORT_DATA_RE.search(html)
    assert body is not None
    assert "NaN" not in body.group("body")


# ────────────────────────────────────────────────────────────────────────
# Bonus: structural sanity checks.
# ────────────────────────────────────────────────────────────────────────


class _TagBalanceParser(HTMLParser):
    """Tiny structural sanity check — open and close tags balance."""

    # Void elements per HTML5 — no closing tag expected.
    _VOID = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self._VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            self.errors.append(f"close {tag} with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"close {tag} but top is {self.stack[-1]}")
            return
        self.stack.pop()


def test_html_tags_are_well_balanced() -> None:
    """Lightweight structural sanity check — html parser sees balanced tags."""
    html = render_html(_report_with_clusters())
    parser = _TagBalanceParser()
    parser.feed(html)
    parser.close()
    assert parser.errors == [], parser.errors
    assert parser.stack == [], f"unclosed: {parser.stack}"


def test_render_html_handles_asymmetric_heatmap_axes() -> None:
    """Regression - non-square heatmaps (e.g. 3 x-values by 2 y-values) render OK.

    A previous version of the template indexed ``success_rate`` with
    ``loop.index0`` from the inner loop, which raises ``IndexError`` on
    any heatmap whose axes have different cardinalities. The realistic
    suite YAML in the spec is 4 lighting steps by 3 camera steps.
    """
    eps: list[Episode] = []
    cell = 0
    # 3 lighting values by 2 camera values - fully populated grid so no
    # NaN lands in the heatmap (independent assertion path).
    for lighting in (0.0, 0.5, 1.0):
        for camera in (0.0, 1.0):
            for i in range(2):
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=i,
                        success=(i == 0),
                        config={"lighting_intensity": lighting, "camera_offset_x": camera},
                    )
                )
            cell += 1
    report = build_report(eps)
    h = report.heatmap_2d["lighting_intensity__camera_offset_x"]
    assert len(h.x_values) == 3
    assert len(h.y_values) == 2
    # The render itself is the contract under test — must not raise.
    html = render_html(report)
    assert "<!DOCTYPE html>" in html


def test_per_cell_section_is_collapsed_by_default() -> None:
    """§6 — wall of numbers is collapsed; <details> has no `open` attribute."""
    html = render_html(_report_with_clusters())
    # Find every <details ...> tag and assert none of them are auto-opened.
    details_re = re.compile(r"<details\b([^>]*)>", re.IGNORECASE)
    matches = details_re.findall(html)
    assert matches, "expected at least one <details> block (per-cell table)"
    for attrs in matches:
        assert " open" not in attrs.lower(), f"<details{attrs}> must not be open"
