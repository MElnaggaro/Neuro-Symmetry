"""
Input Quality Gate — frames are scored before landmark extraction.

QualityCode  : OK | DEGRADED | UNRELIABLE
QualityStatus: (code, reason) — is_usable = code != UNRELIABLE
InputQualityChecker.check(frame_bgr, result) -> QualityStatus

All threshold values are sourced from ``backend.core.config.QualitySettings``
so a deployment can tune the gate via env vars (NS_QUALITY__MIN_LUMINANCE etc.)
without touching code.

Checks (in order — first failure wins):
  1. face_not_detected   → UNRELIABLE  (result is None)
  2. low_light           → UNRELIABLE  (mean V < quality.min_luminance)
  3. occlusion           → UNRELIABLE  (z-range of anchors < quality.min_z_range)
  4. camera_blur         → UNRELIABLE  (Laplacian variance < quality.min_laplacian)
  5. extreme_pose        → DEGRADED    (|roll/yaw/pitch| above max_*_deg)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from backend.api.landmark_engine import KEY_POINTS, LandmarkResult
from backend.core import QualitySettings, get_settings


# ── Public types ──────────────────────────────────────────────────────────────

class QualityCode(str, Enum):
    OK         = "OK"
    DEGRADED   = "DEGRADED"
    UNRELIABLE = "UNRELIABLE"


@dataclass(frozen=True)
class QualityStatus:
    code:   QualityCode
    reason: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        return self.code != QualityCode.UNRELIABLE


# Subset of KEY_POINTS used for the occlusion / z-range check
_OCCLUSION_KEYS: tuple[str, ...] = (
    "eye_left", "eye_right", "nose_tip", "mouth_left", "mouth_right",
)


# ── Checker ───────────────────────────────────────────────────────────────────

class InputQualityChecker:
    """
    Quality gate with blur hysteresis.

    Blur below min_laplacian is DEGRADED (still usable — EMA-smoothed landmarks
    carry the frame) for up to _BLUR_GRACE_FRAMES consecutive blurry frames.
    Only after the grace window is exhausted does it become UNRELIABLE.

    Hard failures (face_not_detected, low_light, occlusion) are always UNRELIABLE.
    Extreme pose is always DEGRADED (usable).

    Parameters
    ----------
    settings : optional QualitySettings
        Override the active configuration (used by tests).

    Usage::

        checker = InputQualityChecker()
        status = checker.check(frame_bgr, landmark_result)
        if not status.is_usable:
            continue
    """

    # Allow this many consecutive blurry frames before returning UNRELIABLE.
    # At 30 fps this means ~100 ms of blur (3 frames) before the frame is dropped.
    _BLUR_GRACE_FRAMES: int = 3

    def __init__(self, settings: Optional[QualitySettings] = None) -> None:
        self._cfg        = settings or get_settings().quality
        self._blur_streak: int = 0   # consecutive blurry frames seen

    def reset(self) -> None:
        """Clear blur-streak state — call between sessions."""
        self._blur_streak = 0

    def check(
        self,
        frame_bgr: np.ndarray,
        result:    Optional[LandmarkResult],
    ) -> QualityStatus:
        cfg = self._cfg

        # 1. Face not detected — hard failure, always UNRELIABLE
        if result is None:
            self._blur_streak = 0
            return QualityStatus(QualityCode.UNRELIABLE, "face_not_detected")

        # 2. Low light — hard failure
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if float(hsv[:, :, 2].mean()) < cfg.min_luminance:
            self._blur_streak = 0
            return QualityStatus(QualityCode.UNRELIABLE, "low_light")

        # 3. Occlusion proxy — z-spread of face anchor points (normalised space)
        pts = result.normalized.points
        z_vals = np.array([pts[KEY_POINTS[k], 2] for k in _OCCLUSION_KEYS])
        if float(z_vals.max() - z_vals.min()) < cfg.min_z_range:
            self._blur_streak = 0
            return QualityStatus(QualityCode.UNRELIABLE, "occlusion")

        # 4. Camera / motion blur — Laplacian variance on greyscale frame.
        #    DEGRADED for up to _BLUR_GRACE_FRAMES consecutive blurry frames so
        #    that transient camera shake does not break temporal continuity.
        #    EMA-smoothed landmarks from LandmarkEngine remain usable during this
        #    grace window.  Only a sustained blur burst becomes UNRELIABLE.
        grey    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if lap_var < cfg.min_laplacian:
            self._blur_streak += 1
            if self._blur_streak <= self._BLUR_GRACE_FRAMES:
                return QualityStatus(
                    QualityCode.DEGRADED,
                    f"camera_blur lap={lap_var:.0f} (grace {self._blur_streak}/{self._BLUR_GRACE_FRAMES})",
                )
            return QualityStatus(QualityCode.UNRELIABLE, f"camera_blur lap={lap_var:.0f}")

        self._blur_streak = 0   # clear streak on a clean frame

        # 5. Extreme head pose → DEGRADED (usable but flagged)
        p = result.pose
        if (abs(p.roll)  > cfg.max_roll_deg
                or abs(p.yaw)   > cfg.max_yaw_deg
                or abs(p.pitch) > cfg.max_pitch_deg):
            return QualityStatus(
                QualityCode.DEGRADED,
                f"extreme_pose roll={p.roll:+.1f} yaw={p.yaw:+.1f} pitch={p.pitch:+.1f}",
            )

        return QualityStatus(QualityCode.OK)