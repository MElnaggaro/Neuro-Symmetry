"""
Feature Extractor — 50-dimensional geometric + texture feature vector.

Feature layout (N_FEATURES = 50):
  [0:40]  bilateral mirror distances   — one per MIRROR_PAIR (40 pairs)
  [40]    EAR_left                     — Eye Aspect Ratio, left eye
  [41]    EAR_right                    — Eye Aspect Ratio, right eye
  [42]    EAR_delta                    — |EAR_left - EAR_right|
  [43]    brow_height_left             — brow Y offset from nose (normalised)
  [44]    brow_height_right
  [45]    brow_height_delta            — |left - right|
  [46]    mouth_y_delta                — |mouth_left.y - mouth_right.y|
  [47]    mouth_x_offset               — |mouth_left.x + mouth_right.x| (0 = centred)
  [48]    texture_score                — S_texture ∈ [0, 1]
  [49]    symmetry_error               — α·mean(dists) + β·(1−S_texture)

Fused symmetry score: score = exp(−features[49]) ∈ (0, 1]

Normalised-space conventions (see landmark_engine.py):
  • nose_tip at (0, 0, z)
  • eye_left at (−0.5, ~0, z), eye_right at (+0.5, ~0, z)
  • mirror of right landmark = (−x, y, z)
  • bilateral_dist ≈ 0 for a perfectly symmetric face
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from backend.api.landmark_engine import (
    KEY_POINTS,
    LEFT_INDICES,
    MIRROR_PAIRS,
    RIGHT_INDICES,
    LandmarkResult,
)
from backend.api.texture_engine import TextureEngine

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 50

# Fused score weights (Phase 3 default; optimised in Phase 4)
_ALPHA = 0.65   # geometric weight
_BETA  = 0.35   # texture weight

# EAR landmark indices — (outer_corner, upper1, upper2, inner_corner, lower1, lower2)
_EAR_LEFT  = (33,  160, 158, 133, 153, 144)
_EAR_RIGHT = (263, 387, 385, 362, 380, 373)

# Region labels aligned with MIRROR_PAIRS order (used to build FEATURE_NAMES)
_REGION_LABELS: list[str] = (
    ["eye_outline"]   * 9 +
    ["eye_aperture"]  * 4 +
    ["eyebrow"]       * 10 +
    ["nose"]          * 5 +
    ["mouth"]         * 4 +
    ["jaw"]           * 5 +
    ["forehead"]      * 3
)

FEATURE_NAMES: list[str] = (
    [f"{region}_{l:03d}" for (l, _), region in zip(MIRROR_PAIRS, _REGION_LABELS)] +
    [
        "ear_left", "ear_right", "ear_delta",
        "brow_height_left", "brow_height_right", "brow_height_delta",
        "mouth_y_delta", "mouth_x_offset",
        "texture_score", "symmetry_error",
    ]
)

assert len(FEATURE_NAMES) == N_FEATURES, f"{len(FEATURE_NAMES)} != {N_FEATURES}"

# ── Internal helpers ──────────────────────────────────────────────────────────

def _bilateral_distances(pts: np.ndarray) -> np.ndarray:
    """
    Compute 40 bilateral mirror distances in normalised space.

    For each (left_idx, right_idx) in MIRROR_PAIRS:
        mirror(right) = (−right.x, right.y, right.z)
        dist = ||pts[left_idx] − mirror(pts[right_idx])||

    Returns shape (40,) float32. Approaches 0 for symmetric faces.
    """
    left_pts  = pts[LEFT_INDICES]           # (40, 3)
    right_pts = pts[RIGHT_INDICES]          # (40, 3)
    right_mirrored        = right_pts.copy()
    right_mirrored[:, 0] *= -1.0            # negate x to mirror
    return np.linalg.norm(left_pts - right_mirrored, axis=1).astype(np.float32)


def _ear(pts: np.ndarray, indices: tuple[int, int, int, int, int, int]) -> float:
    """
    Eye Aspect Ratio: (||p2−p6|| + ||p3−p5||) / (2·||p1−p4||).
    Uses 2-D (x, y) distances; returns 0.0 if the eye width is degenerate.
    """
    p1, p2, p3, p4, p5, p6 = [pts[i, :2] for i in indices]
    width = float(np.linalg.norm(p1 - p4))
    if width < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width))


# ── Public API ────────────────────────────────────────────────────────────────

_texture_engine = TextureEngine()


def extract_features(
    result: LandmarkResult,
    frame_bgr: np.ndarray,
    texture_engine: Optional[TextureEngine] = None,
) -> np.ndarray:
    """
    Extract a 50-dim float32 feature vector from a LandmarkResult + BGR frame.

    Parameters
    ----------
    result      : LandmarkResult from LandmarkEngine.process()
    frame_bgr   : The source frame (same one that produced `result`)
    texture_engine: Override the module-level TextureEngine (for testing)

    Returns
    -------
    np.ndarray of shape (50,), dtype float32
    """
    pts = result.normalized.points   # (478, 3) float32, nose-centred IPD-scaled

    # ── Features 0–39: bilateral mirror distances ──────────────────────────
    feat_bilateral = _bilateral_distances(pts)   # (40,)

    # ── Features 40–42: Eye Aspect Ratio ──────────────────────────────────
    ear_l = _ear(pts, _EAR_LEFT)
    ear_r = _ear(pts, _EAR_RIGHT)
    ear_d = abs(ear_l - ear_r)

    # ── Features 43–45: Eyebrow height (distance above nose in normalised space)
    # In normalised y-axis: upward = more negative (nose at 0, brows above)
    brow_h_l = float(-pts[KEY_POINTS["brow_left"],  1])   # negate: larger = higher
    brow_h_r = float(-pts[KEY_POINTS["brow_right"], 1])
    brow_d   = abs(brow_h_l - brow_h_r)

    # ── Features 46–47: Mouth deviation ───────────────────────────────────
    m_l = pts[KEY_POINTS["mouth_left"]]
    m_r = pts[KEY_POINTS["mouth_right"]]
    mouth_y_delta  = float(abs(m_l[1] - m_r[1]))
    # x_offset: how far the mouth midpoint is from x=0 (the nose axis)
    mouth_x_offset = float(abs((m_l[0] + m_r[0]) / 2.0))

    # ── Feature 48: texture score ──────────────────────────────────────────
    engine = texture_engine or _texture_engine
    s_texture = engine.compute(frame_bgr, result)

    # ── Feature 49: fused symmetry error ──────────────────────────────────
    symmetry_error = (
        _ALPHA * float(feat_bilateral.mean()) +
        _BETA  * (1.0 - s_texture)
    )

    derived = np.array(
        [ear_l, ear_r, ear_d,
         brow_h_l, brow_h_r, brow_d,
         mouth_y_delta, mouth_x_offset,
         s_texture, symmetry_error],
        dtype=np.float32,
    )

    return np.concatenate([feat_bilateral, derived])


def compute_symmetry_score(features: np.ndarray) -> float:
    """
    Derive the fused symmetry score from a feature vector.

    score = exp(−features[49]) ∈ (0, 1]
      1.0 = perfectly symmetric
      0.0 = maximally asymmetric
    """
    symmetry_error = float(features[49])
    return math.exp(-symmetry_error)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    import cv2 as _cv2

    from backend.api.landmark_engine import LandmarkEngine
    from datasets.config import CELEBA_IMGS_ALIGNED

    img_path = sys.argv[1] if len(sys.argv) > 1 else str(CELEBA_IMGS_ALIGNED / "000001.jpg")
    frame = _cv2.imread(img_path)
    if frame is None:
        print(f"ERROR: cannot read {img_path}")
        sys.exit(1)

    with LandmarkEngine() as engine:
        result = engine.process(frame)

    if result is None:
        print("No face detected — cannot extract features.")
        sys.exit(1)

    t0 = time.perf_counter()
    features = extract_features(result, frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    score = compute_symmetry_score(features)

    print(f"Image       : {img_path}")
    print(f"Shape       : {features.shape}  dtype={features.dtype}")
    print(f"Score       : {score:.4f}  (1.0 = perfectly symmetric)")
    print(f"Elapsed     : {elapsed_ms:.2f} ms")
    print(f"\nTop asymmetric features (bilateral distances):")
    bilateral = features[:40]
    top_idx = np.argsort(bilateral)[::-1][:5]
    for i in top_idx:
        print(f"  [{i:02d}] {FEATURE_NAMES[i]:30s}  {bilateral[i]:.4f}")
    print(f"\nDerived features:")
    for i in range(40, N_FEATURES):
        print(f"  [{i}] {FEATURE_NAMES[i]:30s}  {features[i]:.4f}")
