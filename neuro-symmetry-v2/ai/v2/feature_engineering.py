"""
V2 geometric feature extractor — 25 asymmetry features from LandmarkResult.

Works on result.normalized.points (478, 3) in the nose-centred,
IPD-scaled, roll-corrected coordinate system:
  • x = 0  →  face midline (nose axis)
  • x < 0  →  left side of image   (patient's right anatomically)
  • x > 0  →  right side of image  (patient's left anatomically)
  • y > 0  →  downward  (image convention)
  • scale  →  1 unit = 1 IPD

Landmark conventions are consistent with MIRROR_PAIRS in landmark_engine.py:
  LEFT_INDICES  (negative x)  →  first element of each mirror pair
  RIGHT_INDICES (positive x)  →  second element of each mirror pair
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from backend.api.landmark_engine import LandmarkResult

# ── Eye contour approximation (EAR hexagon, ordered around the eye) ──────────
# Points: outer_corner → upper1 → upper2 → inner_corner → lower1 → lower2
_EAR_LEFT_IDX  = (33,  160, 158, 133, 153, 144)   # left eye  (negative-x group)
_EAR_RIGHT_IDX = (263, 387, 385, 362, 380, 373)   # right eye (positive-x group)

N_V2_FEATURES = 25
EPS = 1e-8


def _poly_area(pts: np.ndarray) -> float:
    """Shoelace formula area for an ordered polygon. pts: (N, 2)."""
    n  = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    )


def _angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Signed angle (°) of vector p1→p2 from the positive-x axis."""
    d = p2 - p1
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def extract_v2_features(result: LandmarkResult) -> np.ndarray:
    """
    Compute 25 geometric asymmetry features from a LandmarkResult.

    Parameters
    ----------
    result : LandmarkResult from LandmarkEngine.process()

    Returns
    -------
    np.ndarray of shape (25,), dtype float32
    """
    pts = result.normalized.points   # (478, 3)  nose-centred, IPD-scaled
    f: List[float] = []

    # ── 1–2  Eye area ratio (shoelace on 6-point EAR polygon) ────────────────
    la = _poly_area(pts[list(_EAR_LEFT_IDX),  :2])
    ra = _poly_area(pts[list(_EAR_RIGHT_IDX), :2])
    f += [la / (ra + EPS),
          abs(la - ra) / (la + ra + EPS)]                    # normalised diff

    # ── 3–4  Eye vertical openness (outer top → outer bottom, per eye) ───────
    lh = abs(pts[160, 1] - pts[153, 1]) + EPS   # left:  upper1 – lower1
    rh = abs(pts[387, 1] - pts[380, 1]) + EPS   # right: upper1 – lower1
    f += [lh / rh, abs(lh - rh) / (lh + rh)]

    # ── 5–6  Eye horizontal width (outer – inner corner) ─────────────────────
    lw = abs(pts[33,  0] - pts[133, 0]) + EPS   # left
    rw = abs(pts[263, 0] - pts[362, 0]) + EPS   # right
    f += [lw / rw, abs(lw - rw) / (lw + rw)]

    # ── 7–8  Eyebrow peak height (y-coord; more negative = higher up) ────────
    # MIRROR_PAIR (105, 334): 105 = left brow peak, 334 = right brow peak
    lb_y = pts[105, 1]   # left brow peak y  (negative = higher)
    rb_y = pts[334, 1]   # right brow peak y
    # Signed: positive means left brow is lower than right
    brow_asym = lb_y - rb_y
    f += [brow_asym, abs(brow_asym)]

    # ── 9    Eyebrow inner-corner y-asymmetry ─────────────────────────────────
    # MIRROR_PAIR (55, 285): 55 = left brow inner, 285 = right brow inner
    f.append(pts[55, 1] - pts[285, 1])

    # ── 10–11  Mouth corner tilt angle ────────────────────────────────────────
    ml = pts[61,  :2]    # mouth left  (negative x)
    mr = pts[291, :2]    # mouth right (positive x)
    # Positive angle = right corner is lower (image-space drooping)
    m_angle = _angle_deg(ml, mr)
    f += [m_angle, abs(m_angle)]

    # ── 12–14  Mouth corner y-deviation from inter-corner midpoint ────────────
    m_cy = (ml[1] + mr[1]) * 0.5
    ld, rd = ml[1] - m_cy, mr[1] - m_cy
    f += [ld, rd, abs(ld - rd)]

    # ── 15    Mouth centroid x-offset from face midline (x = 0) ──────────────
    f.append(float((ml[0] + mr[0]) * 0.5))

    # ── 16–18  Nasolabial fold angles (from origin/nose to mouth corners) ────
    # In normalised space, nose tip is at (0, 0, z) — i.e. the origin
    nl_l = _angle_deg(np.zeros(2), ml)    # angle to left  mouth corner
    nl_r = _angle_deg(np.zeros(2), mr)    # angle to right mouth corner
    f += [nl_l, nl_r, abs(abs(nl_l) - abs(nl_r))]

    # ── 19    Nose ala width asymmetry ────────────────────────────────────────
    # MIRROR_PAIR (45, 275): 45 = left ala, 275 = right ala
    l_ala_x = abs(pts[45,  0]) + EPS   # distance from midline
    r_ala_x = abs(pts[275, 0]) + EPS
    f.append(abs(l_ala_x - r_ala_x) / (l_ala_x + r_ala_x))

    # ── 20    Cheek x-distance symmetry ──────────────────────────────────────
    # MIRROR_PAIR (234, 454): 234 = left cheek, 454 = right cheek
    l_ck = abs(pts[234, 0]) + EPS
    r_ck = abs(pts[454, 0]) + EPS
    f.append(abs(l_ck - r_ck) / (l_ck + r_ck))

    # ── 21    Chin x-offset from face midline ─────────────────────────────────
    f.append(float(pts[152, 0]))

    # ── 22    Mouth width / inter-outer-eye-corner width ─────────────────────
    mouth_w = abs(mr[0] - ml[0]) + EPS
    eye_w   = abs(pts[33, 0] - pts[263, 0]) + EPS
    f.append(mouth_w / eye_w)

    # ── 23    Cheek y-asymmetry (height difference between cheekbones) ────────
    f.append(float(pts[234, 1] - pts[454, 1]))

    # ── 24    Inter-eye centre x-offset from midline ──────────────────────────
    f.append(float((pts[33, 0] + pts[263, 0]) * 0.5))

    # ── 25    Eye width × openness area proxy ratio ───────────────────────────
    l_area_proxy = lw * lh
    r_area_proxy = rw * rh
    f.append(l_area_proxy / (r_area_proxy + EPS))

    assert len(f) == N_V2_FEATURES, f"Expected {N_V2_FEATURES}, got {len(f)}"
    return np.array(f, dtype=np.float32)


V2_FEATURE_NAMES: list[str] = [
    "v2_eye_area_ratio",    "v2_eye_area_norm_diff",
    "v2_eye_h_ratio",       "v2_eye_h_norm_diff",
    "v2_eye_w_ratio",       "v2_eye_w_norm_diff",
    "v2_brow_height_asym",  "v2_brow_height_abs",
    "v2_brow_inner_y_asym",
    "v2_mouth_angle",       "v2_mouth_angle_abs",
    "v2_mouth_ld",          "v2_mouth_rd",          "v2_mouth_corner_abs",
    "v2_mouth_cx_offset",
    "v2_nl_left_angle",     "v2_nl_right_angle",    "v2_nl_asym",
    "v2_nose_ala_asym",
    "v2_cheek_x_asym",
    "v2_chin_x_offset",
    "v2_mouth_w_ratio",
    "v2_cheek_y_asym",
    "v2_eye_cx_offset",
    "v2_eye_area_proxy_ratio",
]

assert len(V2_FEATURE_NAMES) == N_V2_FEATURES
