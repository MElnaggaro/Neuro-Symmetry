"""Phase 3 — feature extractor unit tests (no camera, no MediaPipe model required)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.api.feature_extractor import (
    FEATURE_NAMES,
    N_FEATURES,
    _ALPHA,
    _BETA,
    _bilateral_distances,
    _ear,
    _EAR_LEFT,
    _EAR_RIGHT,
    compute_symmetry_score,
    extract_features,
)
from backend.api.landmark_engine import (
    KEY_POINTS,
    LEFT_INDICES,
    MIRROR_PAIRS,
    RIGHT_INDICES,
    LandmarkResult,
    NormalizedLandmarks,
    PoseAngles,
)
from backend.api.texture_engine import TextureEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_symmetric_pts() -> np.ndarray:
    """
    478-point array where every bilateral pair is perfectly mirrored.
    Nose at origin; eyes at ±0.5 on x.
    """
    pts = np.zeros((478, 3), dtype=np.float32)

    # Place LEFT_INDICES at (−x, y, z) and RIGHT_INDICES at (+x, y, z)
    for i, (l, r) in enumerate(MIRROR_PAIRS):
        x = 0.1 * (i % 5 + 1)
        y = 0.05 * i
        z = 0.02 * i
        pts[l] = [-x, y, z]
        pts[r] = [ x, y, z]

    # Key points
    pts[KEY_POINTS["nose_tip"]]   = [0.0,  0.0, 0.0]
    pts[KEY_POINTS["eye_left"]]   = [-0.5, 0.0, 0.0]
    pts[KEY_POINTS["eye_right"]]  = [ 0.5, 0.0, 0.0]
    pts[KEY_POINTS["brow_left"]]  = [-0.5, -0.8, 0.0]
    pts[KEY_POINTS["brow_right"]] = [ 0.5, -0.8, 0.0]
    pts[KEY_POINTS["mouth_left"]] = [-0.3,  0.4, 0.0]
    pts[KEY_POINTS["mouth_right"]]= [ 0.3,  0.4, 0.0]

    # EAR points for left eye (p1=33, p2=160, p3=158, p4=133, p5=153, p6=144)
    pts[33]  = [-0.50, 0.0,   0.0]   # outer corner
    pts[133] = [-0.20, 0.0,   0.0]   # inner corner  → eye width = 0.30
    pts[160] = [-0.35, -0.05, 0.0]   # upper 1
    pts[158] = [-0.35, -0.04, 0.0]   # upper 2
    pts[153] = [-0.35,  0.05, 0.0]   # lower 1
    pts[144] = [-0.35,  0.04, 0.0]   # lower 2

    # EAR points for right eye (p1=263, p2=387, p3=385, p4=362, p5=380, p6=373)
    pts[263] = [ 0.50, 0.0,   0.0]
    pts[362] = [ 0.20, 0.0,   0.0]
    pts[387] = [ 0.35, -0.05, 0.0]
    pts[385] = [ 0.35, -0.04, 0.0]
    pts[380] = [ 0.35,  0.05, 0.0]
    pts[373] = [ 0.35,  0.04, 0.0]

    return pts


def _make_mock_result(pts: np.ndarray, ipd: float = 80.0) -> LandmarkResult:
    mock_raw = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    return LandmarkResult(
        raw=mock_raw,
        normalized=NormalizedLandmarks(points=pts, ipd=ipd),
        pose=PoseAngles(roll=0.0, yaw=0.0, pitch=0.0),
        frame_hw=(480, 640),
    )


def _make_frame(h: int = 480, w: int = 640, brightness: int = 128) -> np.ndarray:
    import numpy as _np
    frame = _np.zeros((h, w, 3), dtype=_np.uint8)
    for r in range(h):
        for c in range(w):
            frame[r, c] = brightness if (r // 8 + c // 8) % 2 == 0 else brightness // 2
    return frame


class _FakeTextureEngine:
    """Always returns a fixed S_texture, regardless of frame content."""
    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    def compute(self, frame_bgr, result) -> float:
        return self._score


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_feature_names_length(self) -> None:
        assert len(FEATURE_NAMES) == N_FEATURES == 50

    def test_feature_names_unique(self) -> None:
        assert len(set(FEATURE_NAMES)) == N_FEATURES

    def test_weights_sum_to_one(self) -> None:
        assert abs(_ALPHA + _BETA - 1.0) < 1e-9


class TestBilateralDistances:
    def test_symmetric_face_gives_near_zero_distances(self) -> None:
        pts = _make_symmetric_pts()
        dists = _bilateral_distances(pts)
        assert dists.shape == (len(MIRROR_PAIRS),)
        assert float(dists.max()) < 1e-5, f"expected ~0, got max={dists.max():.6f}"

    def test_asymmetric_face_gives_nonzero_distances(self) -> None:
        pts = _make_symmetric_pts()
        # Shift one left landmark off-mirror
        l_idx = LEFT_INDICES[0]
        pts[l_idx, 1] += 0.2   # y offset → breaks symmetry
        dists = _bilateral_distances(pts)
        assert dists[0] > 0.1

    def test_output_dtype_and_shape(self) -> None:
        pts = _make_symmetric_pts()
        dists = _bilateral_distances(pts)
        assert dists.dtype == np.float32
        assert dists.shape == (40,)

    def test_mirror_negates_x_only(self) -> None:
        """If right.x = 0.3, mirrored = (−0.3, y, z); dist to left at (−0.3, y, z) = 0."""
        pts = np.zeros((478, 3), dtype=np.float32)
        l, r = MIRROR_PAIRS[0]   # (33, 263)
        pts[l] = [-0.3, 0.1, 0.05]
        pts[r] = [ 0.3, 0.1, 0.05]
        dists = _bilateral_distances(pts)
        assert abs(dists[0]) < 1e-6


class TestEAR:
    def test_symmetric_ears_equal(self) -> None:
        pts = _make_symmetric_pts()
        ear_l = _ear(pts, _EAR_LEFT)
        ear_r = _ear(pts, _EAR_RIGHT)
        assert abs(ear_l - ear_r) < 1e-5, f"L={ear_l:.5f} R={ear_r:.5f}"

    def test_ear_positive(self) -> None:
        pts = _make_symmetric_pts()
        assert _ear(pts, _EAR_LEFT) > 0
        assert _ear(pts, _EAR_RIGHT) > 0

    def test_closed_eye_gives_near_zero(self) -> None:
        pts = _make_symmetric_pts()
        # Collapse upper/lower to same y
        pts[160, 1] = pts[144, 1] = 0.0
        pts[158, 1] = pts[153, 1] = 0.0
        ear = _ear(pts, _EAR_LEFT)
        assert ear < 0.01


class TestExtractFeatures:
    def test_output_shape(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine())
        assert features.shape == (50,)

    def test_output_dtype(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine())
        assert features.dtype == np.float32

    def test_bilateral_slice_near_zero_for_symmetric_face(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine(1.0))
        bilateral = features[:40]
        assert float(bilateral.max()) < 1e-5, (
            f"symmetric face should have ~0 bilateral distances, max={bilateral.max():.6f}"
        )

    def test_ear_delta_zero_for_symmetric_face(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine())
        ear_delta = features[42]
        assert abs(ear_delta) < 1e-5, f"EAR delta should be 0 for symmetric face, got {ear_delta}"

    def test_brow_height_delta_zero_for_symmetric_face(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine())
        assert abs(features[45]) < 1e-5

    def test_mouth_x_offset_near_zero_for_centered_mouth(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine())
        assert abs(features[47]) < 1e-5

    def test_texture_score_propagated(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine(0.7))
        assert abs(features[48] - 0.7) < 1e-6

    def test_symmetry_error_near_zero_for_symmetric_face(self) -> None:
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine(1.0))
        # error = alpha * 0 + beta * (1-1) = 0
        assert abs(features[49]) < 1e-5


class TestComputeSymmetryScore:
    def test_zero_error_gives_score_one(self) -> None:
        features = np.zeros(50, dtype=np.float32)
        score = compute_symmetry_score(features)
        assert abs(score - 1.0) < 1e-6

    def test_score_in_range(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            features = rng.random(50).astype(np.float32)
            score = compute_symmetry_score(features)
            assert 0.0 < score <= 1.0

    def test_neutral_symmetric_face_score_above_0_90(self) -> None:
        """Symmetric face + perfect texture → score > 0.90."""
        pts = _make_symmetric_pts()
        result = _make_mock_result(pts)
        frame = _make_frame()
        features = extract_features(result, frame, texture_engine=_FakeTextureEngine(1.0))
        score = compute_symmetry_score(features)
        assert score > 0.90, f"expected score > 0.90 for symmetric face, got {score:.4f}"

    def test_higher_error_gives_lower_score(self) -> None:
        f_low  = np.zeros(50, dtype=np.float32)
        f_low[49] = 0.1
        f_high = np.zeros(50, dtype=np.float32)
        f_high[49] = 0.5
        assert compute_symmetry_score(f_low) > compute_symmetry_score(f_high)

    def test_exponential_formula(self) -> None:
        features = np.zeros(50, dtype=np.float32)
        features[49] = 0.3
        expected = math.exp(-0.3)
        assert abs(compute_symmetry_score(features) - expected) < 1e-6
