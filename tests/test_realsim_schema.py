"""Schema-level tests for the real-to-sim stub — Phase 3 Task 18.

Pin the field shape, the validation surface (positive focal lengths,
finite pose entries, bottom-row tolerance, intrinsics-id resolution),
and the JSON round-trip. The pipeline / IO / CLI tests live in their
own modules.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from gauntlet.realsim.schema import (
    POSE_BOTTOM_ROW_TOLERANCE,
    SCENE_SCHEMA_VERSION,
    CameraFrame,
    CameraIntrinsics,
    Pose,
    Scene,
)

# ---------------------------------------------------------------------------
# CameraIntrinsics.
# ---------------------------------------------------------------------------


def _good_intrinsics(**overrides: object) -> CameraIntrinsics:
    base: dict[str, object] = {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
    }
    base.update(overrides)
    return CameraIntrinsics(**base)  # type: ignore[arg-type]


def test_intrinsics_happy_path() -> None:
    intr = _good_intrinsics()
    assert intr.width == 640
    assert intr.distortion == []


@pytest.mark.parametrize("field", ["fx", "fy"])
def test_intrinsics_focal_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: 0.0})
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: -1.0})
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: float("inf")})


@pytest.mark.parametrize("field", ["cx", "cy"])
def test_intrinsics_principal_point_must_be_finite(field: str) -> None:
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: float("nan")})
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: float("inf")})


@pytest.mark.parametrize("field", ["width", "height"])
def test_intrinsics_resolution_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: 0})
    with pytest.raises(ValidationError):
        _good_intrinsics(**{field: -1})


def test_intrinsics_distortion_finite() -> None:
    with pytest.raises(ValidationError):
        _good_intrinsics(distortion=[0.1, float("nan"), 0.0])


def test_intrinsics_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        CameraIntrinsics.model_validate(
            {
                "fx": 600.0,
                "fy": 600.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480,
                "rogue": "no",
            }
        )


def test_intrinsics_round_trip() -> None:
    intr = _good_intrinsics(distortion=[-0.1, 0.05, 0.0, 0.0, 0.0])
    rebuilt = CameraIntrinsics.model_validate(intr.model_dump(mode="json"))
    assert rebuilt == intr


# ---------------------------------------------------------------------------
# Pose.
# ---------------------------------------------------------------------------


def _identity_pose() -> Pose:
    return Pose(
        matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_pose_identity_validates() -> None:
    pose = _identity_pose()
    assert pose.matrix[0][0] == 1.0


def test_pose_must_be_4x4() -> None:
    with pytest.raises(ValidationError):
        Pose(matrix=[[1.0, 0.0, 0.0]])
    with pytest.raises(ValidationError):
        Pose(matrix=[[1.0, 0.0, 0.0, 0.0]] * 3)
    with pytest.raises(ValidationError):
        # 4 rows but a short row in the middle.
        Pose(
            matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )


def test_pose_entries_must_be_finite() -> None:
    bad = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, float("nan")],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValidationError):
        Pose(matrix=bad)
    bad[2][3] = float("inf")
    with pytest.raises(ValidationError):
        Pose(matrix=bad)


def test_pose_bottom_row_tolerance_accepts_tiny_jitter() -> None:
    # Within tolerance — accepted.
    jitter = POSE_BOTTOM_ROW_TOLERANCE / 2.0
    pose = Pose(
        matrix=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [jitter, -jitter, 0.0, 1.0 + jitter],
        ]
    )
    assert math.isclose(pose.matrix[3][3], 1.0, abs_tol=POSE_BOTTOM_ROW_TOLERANCE)


def test_pose_bottom_row_rejects_off_tolerance() -> None:
    far = POSE_BOTTOM_ROW_TOLERANCE * 100.0
    with pytest.raises(ValidationError) as exc:
        Pose(
            matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [far, 0.0, 0.0, 1.0],
            ]
        )
    assert "bottom row" in str(exc.value)


def test_pose_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        Pose.model_validate(
            {
                "matrix": _identity_pose().matrix,
                "extra": True,
            }
        )


def test_pose_round_trip() -> None:
    pose = _identity_pose()
    rebuilt = Pose.model_validate(pose.model_dump(mode="json"))
    assert rebuilt == pose


# ---------------------------------------------------------------------------
# CameraFrame.
# ---------------------------------------------------------------------------


def _good_frame(**overrides: object) -> CameraFrame:
    base: dict[str, object] = {
        "path": "frames/0001.png",
        "timestamp": 0.0,
        "intrinsics_id": "wrist",
        "pose": _identity_pose(),
    }
    base.update(overrides)
    return CameraFrame(**base)  # type: ignore[arg-type]


def test_frame_happy_path() -> None:
    frame = _good_frame()
    assert frame.path == "frames/0001.png"
    assert frame.intrinsics_id == "wrist"


def test_frame_path_must_be_relative() -> None:
    with pytest.raises(ValidationError):
        _good_frame(path="/abs/0001.png")
    with pytest.raises(ValidationError):
        _good_frame(path="")
    with pytest.raises(ValidationError):
        _good_frame(path="frames\\0001.png")
    with pytest.raises(ValidationError):
        _good_frame(path="../escape.png")


def test_frame_timestamp_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        _good_frame(timestamp=float("nan"))
    with pytest.raises(ValidationError):
        _good_frame(timestamp=float("inf"))


def test_frame_intrinsics_id_nonempty() -> None:
    with pytest.raises(ValidationError):
        _good_frame(intrinsics_id="")


# ---------------------------------------------------------------------------
# Scene.
# ---------------------------------------------------------------------------


def _good_scene(*, n_frames: int = 2) -> Scene:
    intr = _good_intrinsics()
    frames = [_good_frame(path=f"frames/{i:04d}.png", timestamp=float(i)) for i in range(n_frames)]
    return Scene(
        version=SCENE_SCHEMA_VERSION,
        source="unit-test",
        intrinsics={"wrist": intr},
        frames=frames,
    )


def test_scene_happy_path() -> None:
    scene = _good_scene()
    assert scene.version == SCENE_SCHEMA_VERSION
    assert len(scene.frames) == 2
    assert "wrist" in scene.intrinsics


def test_scene_unknown_intrinsics_id_rejected() -> None:
    intr = _good_intrinsics()
    with pytest.raises(ValidationError) as exc:
        Scene(
            intrinsics={"wrist": intr},
            frames=[_good_frame(intrinsics_id="scene")],
        )
    msg = str(exc.value)
    assert "scene" in msg
    assert "wrist" in msg


def test_scene_version_floor() -> None:
    intr = _good_intrinsics()
    with pytest.raises(ValidationError):
        Scene(version=0, intrinsics={"wrist": intr}, frames=[])


def test_scene_version_future_rejected() -> None:
    intr = _good_intrinsics()
    with pytest.raises(ValidationError) as exc:
        Scene(
            version=SCENE_SCHEMA_VERSION + 1,
            intrinsics={"wrist": intr},
            frames=[],
        )
    assert "newer than this gauntlet" in str(exc.value)


def test_scene_default_source_is_none() -> None:
    intr = _good_intrinsics()
    scene = Scene(intrinsics={"wrist": intr}, frames=[])
    assert scene.source is None


def test_scene_extra_field_forbidden() -> None:
    intr = _good_intrinsics()
    payload = Scene(intrinsics={"wrist": intr}, frames=[]).model_dump(mode="json")
    payload["sneaky"] = True
    with pytest.raises(ValidationError):
        Scene.model_validate(payload)


def test_scene_round_trip() -> None:
    scene = _good_scene(n_frames=3)
    rebuilt = Scene.model_validate(scene.model_dump(mode="json"))
    assert rebuilt == scene
    # The version field round-trips literally — no silent re-stamping.
    assert rebuilt.version == SCENE_SCHEMA_VERSION


def test_scene_empty_frames_allowed() -> None:
    """A scene with zero frames is structurally valid (calibration-only)."""
    intr = _good_intrinsics()
    scene = Scene(intrinsics={"wrist": intr}, frames=[])
    assert scene.frames == []
