"""Phase 2 — InputQualityChecker unit tests (synthetic frames, no camera required)."""

from __future__ import annotations

import numpy as np
import pytest

import backend.api.input_quality as iq_mod
from backend.api.input_quality import InputQualityChecker, QualityCode
from backend.api.landmark_engine import (
    KEY_POINTS,
    LandmarkResult,
    NormalizedLandmarks,
    PoseAngles,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(h: int = 240, w: int = 320, brightness: int = 120) -> np.ndarray:
    """Return a solid-colour BGR frame (sharp by definition — uniform colour → 0 Laplacian).
    Use a textured variant for blur tests."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_textured_frame(h: int = 240, w: int = 320, brightness: int = 120) -> np.ndarray:
    """Checkerboard so Laplacian variance is high (sharp)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # 8×8 checkerboard
    for r in range(h):
        for c in range(w):
            if (r // 8 + c // 8) % 2 == 0:
                frame[r, c] = brightness
            else:
                frame[r, c] = brightness // 2
    return frame


def _make_result(
    roll: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    z_spread: float = 0.5,
) -> LandmarkResult:
    """Minimal LandmarkResult with controllable pose and z-spread."""
    pts = np.zeros((478, 3), dtype=np.float32)

    # Place KEY_POINTS at known positions with requested z-spread
    anchor_keys = list(KEY_POINTS.keys())
    for i, key in enumerate(anchor_keys):
        idx = KEY_POINTS[key]
        pts[idx, 2] = float(i) * (z_spread / max(len(anchor_keys) - 1, 1))

    # Nose at origin, eyes horizontal (as normalisation guarantees)
    pts[KEY_POINTS["nose_tip"]] = [0.0, 0.0, pts[KEY_POINTS["nose_tip"], 2]]
    pts[KEY_POINTS["eye_left"]] = [-0.5, 0.0, pts[KEY_POINTS["eye_left"], 2]]
    pts[KEY_POINTS["eye_right"]] = [0.5, 0.0, pts[KEY_POINTS["eye_right"], 2]]

    return LandmarkResult(
        raw=[],
        normalized=NormalizedLandmarks(points=pts, ipd=80.0),
        pose=PoseAngles(roll=roll, yaw=yaw, pitch=pitch),
        frame_hw=(240, 320),
    )


checker = InputQualityChecker()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ok_path() -> None:
    frame = _make_textured_frame()
    result = _make_result()
    status = checker.check(frame, result)
    assert status.code == QualityCode.OK
    assert status.is_usable
    assert status.reason is None


def test_face_not_detected() -> None:
    frame = _make_textured_frame()
    status = checker.check(frame, None)
    assert status.code == QualityCode.UNRELIABLE
    assert status.reason == "face_not_detected"
    assert not status.is_usable


def test_low_light() -> None:
    dark_frame = _make_textured_frame(brightness=10)
    result = _make_result()
    status = checker.check(dark_frame, result)
    assert status.code == QualityCode.UNRELIABLE
    assert status.reason == "low_light"


def test_occlusion_small_z_range() -> None:
    frame = _make_textured_frame()
    result = _make_result(z_spread=0.0)   # all z=0 → z_range < MIN_Z_RANGE
    status = checker.check(frame, result)
    assert status.code == QualityCode.UNRELIABLE
    assert status.reason == "occlusion"


def test_camera_blur() -> None:
    # Solid-colour frame has 0 Laplacian variance → blur
    frame = _make_frame()    # uniform, no texture
    result = _make_result()
    status = checker.check(frame, result)
    assert status.code == QualityCode.UNRELIABLE
    assert status.reason == "camera_blur"


def test_extreme_roll() -> None:
    frame = _make_textured_frame()
    result = _make_result(roll=20.0)
    status = checker.check(frame, result)
    assert status.code == QualityCode.DEGRADED
    assert "extreme_pose" in (status.reason or "")
    assert status.is_usable   # DEGRADED is still usable


def test_extreme_yaw() -> None:
    frame = _make_textured_frame()
    result = _make_result(yaw=30.0)
    status = checker.check(frame, result)
    assert status.code == QualityCode.DEGRADED


def test_extreme_pitch() -> None:
    frame = _make_textured_frame()
    result = _make_result(pitch=-25.0)
    status = checker.check(frame, result)
    assert status.code == QualityCode.DEGRADED


def test_threshold_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm tests can override module-level thresholds."""
    monkeypatch.setattr(iq_mod, "MAX_ROLL_DEG", 90.0)
    frame = _make_textured_frame()
    result = _make_result(roll=20.0)
    status = checker.check(frame, result)
    assert status.code == QualityCode.OK
