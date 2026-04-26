"""Tests for the raw capture-dir parser — Phase 3 Task 18.

Drives :func:`gauntlet.realsim.load_real_scene` over synthetic
capture-dir fixtures (tmp_path with hand-built ``intrinsics.json`` /
``extrinsics.json`` / ``manifest.json``). No on-disk fixture files —
every JSON payload is constructed in-test so the schema contract is
visible at the call-site.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gauntlet.realsim.scene_input import (
    INTRINSICS_REQUIRED_KEYS,
    RealSceneInput,
    RealSceneInputError,
    load_real_scene,
    validate_scene_input,
)

# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _good_intrinsics() -> dict[str, float]:
    return {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640.0,
        "height": 480.0,
    }


def _write_capture_dir(
    tmp_path: Path,
    *,
    intrinsics: dict[str, float] | None = None,
    extrinsics: list[list[list[float]]] | dict[str, Any] | None = None,
    sidecars: dict[str, Any] | None = None,
    skip_intrinsics: bool = False,
    skip_extrinsics: bool = False,
) -> Path:
    """Write a synthetic capture-dir and return its root."""
    capture = tmp_path / "capture"
    capture.mkdir()
    if not skip_intrinsics:
        (capture / "intrinsics.json").write_text(
            json.dumps(intrinsics if intrinsics is not None else _good_intrinsics()),
            encoding="utf-8",
        )
    if not skip_extrinsics:
        payload: list[list[list[float]]] | dict[str, Any]
        if extrinsics is None:
            payload = [_identity_matrix(), _identity_matrix(), _identity_matrix()]
        else:
            payload = extrinsics
        (capture / "extrinsics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    if sidecars is not None:
        (capture / "manifest.json").write_text(
            json.dumps(sidecars),
            encoding="utf-8",
        )
    return capture


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


def test_load_real_scene_happy_path(tmp_path: Path) -> None:
    """A 4-frame synthetic capture-dir parses cleanly."""
    capture = _write_capture_dir(
        tmp_path,
        extrinsics=[_identity_matrix() for _ in range(4)],
    )
    scene = load_real_scene(capture)
    assert isinstance(scene, RealSceneInput)
    assert scene.capture_dir == capture.resolve()
    assert set(scene.intrinsics.keys()) >= INTRINSICS_REQUIRED_KEYS
    assert len(scene.extrinsics_per_frame) == 4
    for m in scene.extrinsics_per_frame:
        assert m.shape == (4, 4)
        assert m.dtype == np.float64
    assert scene.depth_maps is None
    assert scene.point_cloud is None
    assert scene.metadata == {}
    assert validate_scene_input(scene) == []


def test_load_real_scene_with_sidecars_and_metadata(tmp_path: Path) -> None:
    """A capture-dir with depth maps, point cloud, and metadata round-trips."""
    capture = _write_capture_dir(
        tmp_path,
        extrinsics=[_identity_matrix(), _identity_matrix()],
        sidecars={
            "depth_maps": ["depth/0.exr", "depth/1.exr"],
            "point_cloud": "points.ply",
            "metadata": {"capture_device": "robot_a", "software": "polycam-1.2"},
        },
    )
    # Materialise the sidecar files so the validator doesn't warn about
    # missing files.
    (capture / "depth").mkdir()
    (capture / "depth" / "0.exr").write_text("stub", encoding="utf-8")
    (capture / "depth" / "1.exr").write_text("stub", encoding="utf-8")
    (capture / "points.ply").write_text("stub", encoding="utf-8")

    scene = load_real_scene(capture)
    assert scene.depth_maps is not None
    assert len(scene.depth_maps) == 2
    assert scene.point_cloud is not None
    assert scene.point_cloud.name == "points.ply"
    assert scene.metadata["capture_device"] == "robot_a"
    assert validate_scene_input(scene) == []


def test_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    """``RealSceneInput.to_dict`` produces JSON-encodable output."""
    capture = _write_capture_dir(tmp_path)
    scene = load_real_scene(capture)
    payload = scene.to_dict()
    # Every value must round-trip through ``json.dumps`` without an
    # encoder; this is the load-bearing contract for the dataclass.
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["intrinsics"] == _good_intrinsics()
    assert len(decoded["extrinsics_per_frame"]) == 3
    assert decoded["depth_maps"] is None
    assert decoded["point_cloud"] is None


# ---------------------------------------------------------------------------
# Hard parse failures.
# ---------------------------------------------------------------------------


def test_load_real_scene_missing_intrinsics_raises(tmp_path: Path) -> None:
    capture = _write_capture_dir(tmp_path, skip_intrinsics=True)
    with pytest.raises(RealSceneInputError, match=r"intrinsics\.json"):
        load_real_scene(capture)


def test_load_real_scene_missing_extrinsics_raises(tmp_path: Path) -> None:
    capture = _write_capture_dir(tmp_path, skip_extrinsics=True)
    with pytest.raises(RealSceneInputError, match=r"extrinsics\.json"):
        load_real_scene(capture)


def test_load_real_scene_unknown_capture_dir_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such-dir"
    with pytest.raises(RealSceneInputError, match="capture_dir not found"):
        load_real_scene(bogus)


def test_load_real_scene_intrinsics_missing_required_key(tmp_path: Path) -> None:
    """``fx`` missing from intrinsics is a hard error."""
    bad = _good_intrinsics()
    bad.pop("fx")
    capture = _write_capture_dir(tmp_path, intrinsics=bad)
    with pytest.raises(RealSceneInputError, match="missing required keys"):
        load_real_scene(capture)


def test_load_real_scene_intrinsics_non_finite_raises(tmp_path: Path) -> None:
    """NaN / inf intrinsics are a hard error."""
    bad = _good_intrinsics()
    bad["fx"] = float("inf")
    capture = _write_capture_dir(tmp_path, intrinsics=bad)
    with pytest.raises(RealSceneInputError, match="must be finite"):
        load_real_scene(capture)


def test_load_real_scene_extrinsics_wrong_shape_raises(tmp_path: Path) -> None:
    """A 3x4 matrix is rejected."""
    capture = _write_capture_dir(
        tmp_path,
        extrinsics=[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
    )
    with pytest.raises(RealSceneInputError, match="must be 4x4"):
        load_real_scene(capture)


def test_load_real_scene_extrinsics_bad_bottom_row_raises(tmp_path: Path) -> None:
    """A matrix with a non-canonical bottom row is rejected."""
    bad = _identity_matrix()
    bad[3] = [0.0, 0.0, 0.0, 0.5]
    capture = _write_capture_dir(tmp_path, extrinsics=[bad])
    with pytest.raises(RealSceneInputError, match="bottom row"):
        load_real_scene(capture)


def test_load_real_scene_extrinsics_empty_list_raises(tmp_path: Path) -> None:
    capture = _write_capture_dir(tmp_path, extrinsics=[])
    with pytest.raises(RealSceneInputError, match="at least one frame"):
        load_real_scene(capture)


def test_load_real_scene_extrinsics_dict_form_with_frames_key(tmp_path: Path) -> None:
    """A ``{"frames": [...]}`` wrapper is accepted (NeRFStudio convention)."""
    capture = _write_capture_dir(
        tmp_path,
        extrinsics={"frames": [_identity_matrix(), _identity_matrix()]},
    )
    scene = load_real_scene(capture)
    assert len(scene.extrinsics_per_frame) == 2


def test_load_real_scene_invalid_intrinsics_json_raises(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "intrinsics.json").write_text("{not json", encoding="utf-8")
    (capture / "extrinsics.json").write_text(
        json.dumps([_identity_matrix()]),
        encoding="utf-8",
    )
    with pytest.raises(RealSceneInputError, match="invalid JSON"):
        load_real_scene(capture)


# ---------------------------------------------------------------------------
# Path-traversal / safe_join boundary tests.
# ---------------------------------------------------------------------------


def test_load_real_scene_rejects_traversal_in_point_cloud(tmp_path: Path) -> None:
    """A ``../etc/passwd`` ``point_cloud`` value is caught by safe_join."""
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"point_cloud": "../etc/passwd"},
    )
    with pytest.raises(RealSceneInputError, match="escapes the capture directory"):
        load_real_scene(capture)


def test_load_real_scene_rejects_traversal_in_depth_maps(tmp_path: Path) -> None:
    """A traversal value buried in ``depth_maps`` is caught by safe_join."""
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"depth_maps": ["depth/0.exr", "../../etc/passwd"]},
    )
    with pytest.raises(RealSceneInputError, match="escapes the capture directory"):
        load_real_scene(capture)


def test_load_real_scene_rejects_absolute_point_cloud_path(tmp_path: Path) -> None:
    """An absolute ``point_cloud`` path is rejected by safe_join."""
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"point_cloud": "/etc/passwd"},
    )
    with pytest.raises(RealSceneInputError, match="escapes the capture directory"):
        load_real_scene(capture)


def test_load_real_scene_rejects_non_string_point_cloud(tmp_path: Path) -> None:
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"point_cloud": 123},
    )
    with pytest.raises(RealSceneInputError, match="must be a string path"):
        load_real_scene(capture)


def test_load_real_scene_rejects_non_dict_metadata(tmp_path: Path) -> None:
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"metadata": "not-a-dict"},
    )
    with pytest.raises(RealSceneInputError, match=r"metadata.*must be an object"):
        load_real_scene(capture)


# ---------------------------------------------------------------------------
# Soft validation warnings.
# ---------------------------------------------------------------------------


def test_validate_scene_input_clean_returns_empty_list(tmp_path: Path) -> None:
    capture = _write_capture_dir(tmp_path)
    scene = load_real_scene(capture)
    assert validate_scene_input(scene) == []


def test_validate_scene_input_single_frame_warns(tmp_path: Path) -> None:
    capture = _write_capture_dir(tmp_path, extrinsics=[_identity_matrix()])
    scene = load_real_scene(capture)
    warnings = validate_scene_input(scene)
    assert any("frame(s)" in w and "needs >= 2" in w for w in warnings)


def test_validate_scene_input_partial_distortion_warns(tmp_path: Path) -> None:
    intrinsics = _good_intrinsics()
    intrinsics["k1"] = 0.01
    intrinsics["k2"] = -0.002
    # k3 missing -> partial distortion -> warning.
    capture = _write_capture_dir(tmp_path, intrinsics=intrinsics)
    scene = load_real_scene(capture)
    warnings = validate_scene_input(scene)
    assert any("partial distortion" in w for w in warnings)


def test_validate_scene_input_aggregates_multiple_warnings(tmp_path: Path) -> None:
    """Validator collects ALL findings, not just the first.

    Triggers three independent warnings simultaneously (single-frame
    extrinsics + length-mismatched depth maps + missing depth-map file
    on disk) and asserts the validator surfaces every one of them.
    """
    capture = _write_capture_dir(
        tmp_path,
        extrinsics=[_identity_matrix()],  # single-frame -> warning #1
        sidecars={
            # 2 entries vs 1 frame -> warning #2 (length mismatch)
            # second file missing on disk -> warning #3
            "depth_maps": ["d0.exr", "d1.exr"],
        },
    )
    (capture / "d0.exr").write_text("stub", encoding="utf-8")
    # Deliberately do NOT create d1.exr.
    scene = load_real_scene(capture)
    warnings = validate_scene_input(scene)
    # Three independent findings should all surface.
    assert any("needs >= 2" in w for w in warnings)
    assert any("1:1 alignment" in w for w in warnings)
    assert any("missing on disk" in w for w in warnings)
    # Sanity: more than one finding (no early-return).
    assert len(warnings) >= 3


def test_validate_scene_input_missing_point_cloud_when_depth_present(
    tmp_path: Path,
) -> None:
    capture = _write_capture_dir(
        tmp_path,
        extrinsics=[_identity_matrix(), _identity_matrix()],
        sidecars={"depth_maps": ["d0.exr", "d1.exr"]},
    )
    (capture / "d0.exr").write_text("stub", encoding="utf-8")
    (capture / "d1.exr").write_text("stub", encoding="utf-8")
    scene = load_real_scene(capture)
    warnings = validate_scene_input(scene)
    assert any("no point_cloud aggregate" in w for w in warnings)


def test_validate_scene_input_missing_point_cloud_file_warns(tmp_path: Path) -> None:
    capture = _write_capture_dir(
        tmp_path,
        sidecars={"point_cloud": "missing.ply"},
    )
    scene = load_real_scene(capture)
    warnings = validate_scene_input(scene)
    assert any("point_cloud declared but file is missing on disk" in w for w in warnings)


# ---------------------------------------------------------------------------
# Frozen-ness of the dataclass.
# ---------------------------------------------------------------------------


def test_real_scene_input_is_frozen(tmp_path: Path) -> None:
    """``RealSceneInput`` is frozen — assignment raises."""
    capture = _write_capture_dir(tmp_path)
    scene = load_real_scene(capture)
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        scene.capture_dir = Path("/tmp")  # type: ignore[misc]
