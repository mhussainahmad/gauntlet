"""Tests for :mod:`gauntlet.report.junit` — backlog B-24."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gauntlet.cli import app
from gauntlet.report.junit import to_junit_xml
from gauntlet.runner import Episode


def _ep(
    *,
    cell_index: int = 0,
    episode_index: int = 0,
    success: bool = True,
    seed: int = 42,
    config: dict[str, float] | None = None,
    suite_name: str = "tiny",
) -> Episode:
    return Episode(
        suite_name=suite_name,
        cell_index=cell_index,
        episode_index=episode_index,
        seed=seed,
        perturbation_config=dict(config or {}),
        success=success,
        terminated=success,
        truncated=False,
        step_count=10,
        total_reward=1.0 if success else 0.0,
    )


def _parse(xml_bytes: bytes) -> ET.Element:
    """Parse the JUnit XML round-trip and return the root element."""
    return ET.fromstring(xml_bytes)


def test_empty_episodes_yields_zero_count_testsuite() -> None:
    xml = to_junit_xml([], suite_name="empty-suite")
    root = _parse(xml)
    assert root.tag == "testsuite"
    assert root.attrib["name"] == "empty-suite"
    assert root.attrib["tests"] == "0"
    assert root.attrib["failures"] == "0"
    assert list(root) == []


def test_all_pass_emits_no_failure_elements() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True),
        _ep(cell_index=0, episode_index=1, success=True, seed=43),
        _ep(cell_index=1, episode_index=0, success=True, seed=44),
    ]
    xml = to_junit_xml(eps, suite_name="pass-suite")
    root = _parse(xml)
    assert root.attrib["tests"] == "3"
    assert root.attrib["failures"] == "0"
    cases = root.findall("testcase")
    assert len(cases) == 3
    for case in cases:
        assert case.find("failure") is None


def test_all_fail_marks_every_testcase_as_failure() -> None:
    eps = [_ep(cell_index=0, episode_index=i, success=False, seed=100 + i) for i in range(3)]
    xml = to_junit_xml(eps, suite_name="fail-suite")
    root = _parse(xml)
    assert root.attrib["tests"] == "3"
    assert root.attrib["failures"] == "3"
    for case in root.findall("testcase"):
        failure = case.find("failure")
        assert failure is not None
        assert failure.attrib["type"] == "EpisodeFailure"


def test_mixed_pass_fail_partitions_correctly() -> None:
    eps = [
        _ep(cell_index=0, episode_index=0, success=True, seed=1),
        _ep(cell_index=0, episode_index=1, success=False, seed=2),
        _ep(cell_index=1, episode_index=0, success=True, seed=3),
        _ep(cell_index=1, episode_index=1, success=False, seed=4),
    ]
    xml = to_junit_xml(eps, suite_name="mixed")
    root = _parse(xml)
    assert root.attrib["tests"] == "4"
    assert root.attrib["failures"] == "2"
    failed = [c for c in root.findall("testcase") if c.find("failure") is not None]
    assert {c.attrib["name"] for c in failed} == {
        "episode_1_seed_2",
        "episode_1_seed_4",
    }


def test_failure_message_contains_axis_config_for_grep() -> None:
    eps = [
        _ep(
            cell_index=2,
            episode_index=5,
            success=False,
            seed=99,
            config={"lighting_intensity": 0.3, "object_mass": 1.5},
        )
    ]
    xml = to_junit_xml(eps, suite_name="grep-me")
    root = _parse(xml)
    case = root.find("testcase")
    assert case is not None
    assert case.attrib["classname"] == "grep-me.cell_2"
    assert case.attrib["name"] == "episode_5_seed_99"
    failure = case.find("failure")
    assert failure is not None
    msg = failure.attrib["message"]
    assert "lighting_intensity=0.3" in msg
    assert "object_mass=1.5" in msg
    assert "axis_config" in msg


def test_round_trip_parses_as_valid_xml_with_declaration() -> None:
    eps = [_ep(cell_index=0, episode_index=0, success=True)]
    xml = to_junit_xml(eps, suite_name="rt")
    # Declaration present.
    assert xml.startswith(b'<?xml version="1.0" encoding="utf-8"?>\n')
    # Round-trip parse.
    root = _parse(xml)
    assert root.tag == "testsuite"
    # Re-serialise and re-parse — must remain stable.
    re_serialised = ET.tostring(root, encoding="utf-8")
    root2 = ET.fromstring(re_serialised)
    assert root2.attrib == root.attrib


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_run_cli_emits_junit_when_flag_set(runner: CliRunner, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "name: junit-cli-suite\n"
        "env: tabletop\n"
        "episodes_per_cell: 1\n"
        "seed: 7\n"
        "axes:\n"
        "  lighting_intensity:\n"
        "    values: [1.0]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    junit_path = tmp_path / "nested" / "junit.xml"

    result = runner.invoke(
        app,
        [
            "run",
            str(suite_path),
            "--policy",
            "random",
            "--out",
            str(out_dir),
            "-w",
            "1",
            "--env-max-steps",
            "5",
            "--no-html",
            "--junit",
            str(junit_path),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert junit_path.is_file()
    root = _parse(junit_path.read_bytes())
    assert root.tag == "testsuite"
    assert root.attrib["name"] == "junit-cli-suite"
    assert int(root.attrib["tests"]) >= 1
