"""
Input Quality Gate — frames are scored before landmark extraction.

QualityCode  : OK | DEGRADED | UNRELIABLE
QualityStatus: (code, reason) — is_usable = code != UNRELIABLE
InputQualityChecker.check(frame_bgr, result) -> QualityStatus

Checks (in order, first failure wins):
  1. face_not_detected   → UNRELIABLE  (result is None)
  2. low_light           → UNRELIABLE  (frame mean luminance < 40)
  3. occlusion           → UNRELIABLE  (z-range of nose/eye/mouth landmarks < 0.01)
  4. camera_blur         → UNRELIABLE  (Laplacian variance < 80)
  5. extreme_pose        → DEGRADED    (|roll|>15° or |yaw|>25° or |pitch|>20°)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from backend.api.landmark_engine import KEY_POINTS, LandmarkResult


# ── Public types ──────────────────────────────────────────────────────────────

class QualityCode(str, Enum):
    OK         = "OK"
    DEGRADED   = "DEGRADED"    # usable but with reduced accuracy
    UNRELIABLE = "UNRELIABLE"  # discard this frame


@dataclass(frozen=True)
class QualityStatus:
    code:   QualityCode
    reason: Optional[str] = None   # human-readable; None when code is OK

    @property
    def is_usable(self) -> bool:
        return self.code != QualityCode.UNRELIABLE


# ── Thresholds (module-level so tests can override) ───────────────────────────

MIN_LUMINANCE   = 40.0    # mean pixel value (0–255)
MIN_LAPLACIAN   = 80.0    # variance of Laplacian (sharpness proxy)
MIN_Z_RANGE     = 0.01    # normalised z-range across face anchor points
MAX_ROLL_DEG    = 15.0
MAX_YAW_DEG     = 25.0
MAX_PITCH_DEG   = 20.0

# Subset of KEY_POINTS used for the occlusion / z-range check
_OCCLUSION_KEYS = ("eye_left", "eye_right", "nose_tip", "mouth_left", "mouth_right")


# ── Checker ───────────────────────────────────────────────────────────────────

class InputQualityChecker:
    """
    Stateless per-frame quality gate.

    Usage:
        checker = InputQualityChecker()
        status = checker.check(frame_bgr, landmark_result)
        if not status.is_usable:
            continue
    """

    def check(
        self,
        frame_bgr: np.ndarray,
        result: Optional[LandmarkResult],
    ) -> QualityStatus:
        # 1. Face not detected
        if result is None:
            return QualityStatus(QualityCode.UNRELIABLE, "face_not_detected")

        # 2. Low light — use mean of V channel (value in HSV)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if float(hsv[:, :, 2].mean()) < MIN_LUMINANCE:
            return QualityStatus(QualityCode.UNRELIABLE, "low_light")

        # 3. Occlusion proxy — z-spread of face anchor points in normalised space
        pts = result.normalized.points
        z_vals = np.array([pts[KEY_POINTS[k], 2] for k in _OCCLUSION_KEYS])
        if float(z_vals.max() - z_vals.min()) < MIN_Z_RANGE:
            return QualityStatus(QualityCode.UNRELIABLE, "occlusion")

        # 4. Camera / motion blur — Laplacian variance on greyscale frame
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if lap_var < MIN_LAPLACIAN:
            return QualityStatus(QualityCode.UNRELIABLE, "camera_blur")

        # 5. Extreme head pose → DEGRADED (usable, but flag it)
        p = result.pose
        if abs(p.roll) > MAX_ROLL_DEG or abs(p.yaw) > MAX_YAW_DEG or abs(p.pitch) > MAX_PITCH_DEG:
            return QualityStatus(
                QualityCode.DEGRADED,
                f"extreme_pose roll={p.roll:+.1f} yaw={p.yaw:+.1f} pitch={p.pitch:+.1f}",
            )

        return QualityStatus(QualityCode.OK)
