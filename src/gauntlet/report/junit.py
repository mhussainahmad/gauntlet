"""JUnit-style XML export for ``gauntlet run`` — see backlog B-24.

One ``<testcase>`` per :class:`gauntlet.runner.Episode`. ``classname``
is ``"{suite_name}.cell_{cell_index}"`` and ``name`` is
``"episode_{episode_index}_seed_{seed}"`` so the (cell, episode, seed)
identity round-trips into any JUnit consumer (GitHub Actions, Jenkins,
Buildkite, GitLab) without bespoke parsing. Failed episodes carry a
``<failure>`` child whose ``message`` echoes the cell's
``perturbation_config`` for greppability across CI logs.

Time per testcase is currently always ``0`` because :class:`Episode`
does not record wall-clock duration — when the runner adds a
``duration_s`` field this module will pick it up automatically (see
``_episode_time``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from gauntlet.runner import Episode

__all__ = ["to_junit_xml"]


def _episode_time(episode: Episode) -> float:
    """Wall-clock duration of *episode* in seconds, or ``0.0`` if unknown.

    ``Episode`` does not currently record runtime; we look in
    ``metadata`` for an opt-in ``duration_s`` numeric and fall back to
    ``0.0`` so the JUnit ``time`` attribute is always populated.
    """
    raw = episode.metadata.get("duration_s") if episode.metadata else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return 0.0


def _failure_message(episode: Episode) -> str:
    """Human + grep friendly one-liner describing why an episode failed."""
    cfg = ", ".join(f"{k}={v}" for k, v in sorted(episode.perturbation_config.items()))
    return (
        f"Episode failed (terminated={episode.terminated}, "
        f"truncated={episode.truncated}, reward={episode.total_reward:.4f}, "
        f"steps={episode.step_count}); axis_config: {{{cfg}}}"
    )


def to_junit_xml(episodes: list[Episode], suite_name: str) -> bytes:
    """Render a list of :class:`Episode` as a JUnit XML byte-string.

    Returns a UTF-8 encoded XML document with a single ``<testsuite>``
    wrapping one ``<testcase>`` per episode. ``failures`` and ``tests``
    on the root element are populated from the inputs; ``time`` is the
    sum of per-episode times (currently zero — see ``_episode_time``).

    The output is deterministic for a given ordered ``episodes`` list:
    no timestamps, no hostnames, no environment-derived attributes.
    """
    total_time = sum(_episode_time(ep) for ep in episodes)
    n_failures = sum(1 for ep in episodes if not ep.success)

    suite_el = ET.Element(
        "testsuite",
        {
            "name": suite_name,
            "tests": str(len(episodes)),
            "failures": str(n_failures),
            "errors": "0",
            "skipped": "0",
            "time": f"{total_time:.4f}",
        },
    )

    for ep in episodes:
        case = ET.SubElement(
            suite_el,
            "testcase",
            {
                "classname": f"{suite_name}.cell_{ep.cell_index}",
                "name": f"episode_{ep.episode_index}_seed_{ep.seed}",
                "time": f"{_episode_time(ep):.4f}",
            },
        )
        if not ep.success:
            ET.SubElement(
                case,
                "failure",
                {"message": _failure_message(ep), "type": "EpisodeFailure"},
            )

    body: bytes = ET.tostring(suite_el, encoding="utf-8", xml_declaration=False)
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + body
