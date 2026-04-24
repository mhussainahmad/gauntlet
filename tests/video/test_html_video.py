"""HTML report video-embed tests for the rollout-video Polish feature.

Marked ``@pytest.mark.video`` — runs only in the dedicated
``video-tests`` CI job. The rendered HTML's ``<video>`` element + the
relative ``src`` attribute are the contract under test; the actual
encoding is covered by ``test_video_writer.py``.
"""

from __future__ import annotations

import re

import pytest

from gauntlet.report import build_report, render_html
from gauntlet.runner.episode import Episode

pytestmark = pytest.mark.video


_SUITE = "video-html-test"


def _ep(
    *,
    cell_index: int,
    episode_index: int,
    success: bool,
    config: dict[str, float],
    video_path: str | None = None,
) -> Episode:
    return Episode(
        suite_name=_SUITE,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=cell_index * 10 + episode_index,
        perturbation_config=dict(config),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
        video_path=video_path,
    )


def _report_with_failure_videos() -> tuple[str, str]:
    """Build a 2x2 grid where ``(lighting=1.5, camera=1.0)`` cluster fails.

    Each failed episode carries a unique ``video_path`` so the
    rendered HTML embeds three distinct ``<video>`` elements.
    Returns ``(rendered_html, expected_failure_path)`` for the first
    failure-cluster video.
    """
    eps: list[Episode] = []
    cell = 0
    expected = "videos/episode_cell0001_ep0000_seed10.mp4"
    for lighting in (0.3, 1.5):
        for camera in (0.0, 1.0):
            for i in range(3):
                fail = lighting == 1.5 and camera == 1.0
                # Successful episodes get videos too (so the per-cell
                # column also exercises a path), but the failure-cluster
                # row should embed only the failed-episode videos.
                video_path: str | None = (
                    f"videos/episode_cell{cell:04d}_ep{i:04d}_seed{cell * 10 + i}.mp4"
                )
                eps.append(
                    _ep(
                        cell_index=cell,
                        episode_index=i,
                        success=not fail,
                        config={"lighting_intensity": lighting, "camera_offset_x": camera},
                        video_path=video_path,
                    )
                )
            cell += 1
    report = build_report(eps, cluster_multiple=2.0, min_cluster_size=3)
    return render_html(report), expected


def test_failure_cluster_renders_video_elements() -> None:
    """The failure-clusters row carries ``<video>`` thumbnails for each failed episode."""
    html, _ = _report_with_failure_videos()
    # Every failed-episode video appears as a <source src="..."> in the
    # failure-clusters table.
    assert '<source src="videos/episode_cell0003_ep0000_seed30.mp4"' in html
    assert '<source src="videos/episode_cell0003_ep0001_seed31.mp4"' in html
    assert '<source src="videos/episode_cell0003_ep0002_seed32.mp4"' in html
    # The header column for failure-cluster videos was rendered.
    assert ">Failed-rollout videos<" in html
    # And the per-cell wall-of-numbers also got a videos column when
    # any cell carries video.
    assert ">Rollout videos<" in html


def test_video_thumbnails_have_correct_attributes() -> None:
    """Each ``<video>`` element is sized + lazy-loaded for fast page load."""
    html, _ = _report_with_failure_videos()
    # Find every <video ...> opening tag in the rendered HTML.
    tags = re.findall(r"<video\b[^>]*>", html)
    assert tags, "expected at least one <video> tag in the rendered HTML"
    for tag in tags:
        # ``preload="metadata"`` keeps the page lightweight — the
        # browser fetches just the MP4 header until the user hits play.
        assert 'preload="metadata"' in tag, tag
        # ``controls`` makes the video playable from the page itself.
        assert "controls" in tag, tag


def test_no_video_report_has_no_video_columns() -> None:
    """Pre-PR-shape report (no video_paths) renders without video columns."""
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, config={"a": 0.0}),
        _ep(cell_index=0, episode_index=1, success=False, config={"a": 0.0}),
        _ep(cell_index=0, episode_index=2, success=False, config={"a": 0.0}),
        _ep(cell_index=1, episode_index=0, success=True, config={"a": 1.0}),
    ]
    report = build_report(eps, cluster_multiple=1.0, min_cluster_size=2)
    html = render_html(report)
    # The video header columns are absent because no episode carries
    # a ``video_path``. This is the load-bearing backwards-compat
    # invariant for the HTML report.
    assert ">Failed-rollout videos<" not in html
    assert ">Rollout videos<" not in html
    assert "<video " not in html
