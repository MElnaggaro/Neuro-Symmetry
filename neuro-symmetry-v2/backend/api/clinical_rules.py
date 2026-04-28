"""
Clinical Rules — FAST protocol risk-level classification.

FAST Signal Matrix
------------------
  Face    | Speech   | Arm   | Trajectory            | Risk
  --------|----------|-------|------------------------|----------
  Normal  | —        | —     | —                     | NORMAL
  Mild    | —        | —     | —                     | MILD
  Mild    | Slurred  | —     | —                     | HIGH_RISK
  Severe  | —        | —     | —                     | HIGH_RISK
  Any     | Slurred  | Drift | —                     | CRITICAL
  Severe  | Slurred  | —     | —                     | CRITICAL
  Any > 0 | —        | —     | COLLAPSE / SUDDEN_DROP | CRITICAL

Speech and arm inputs are reserved for future multi-modal sensor
integration.  Pass None (default) to evaluate face + trajectory only.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    NORMAL    = "NORMAL"
    MILD      = "MILD"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL  = "CRITICAL"


_CRITICAL_TRAJECTORIES = frozenset({"COLLAPSE", "SUDDEN_DROP"})


def apply_fast_logic(
    face_class_id: int,
    trajectory:    str,
    speech_slur:   Optional[bool] = None,
    arm_drift:     Optional[bool] = None,
) -> RiskLevel:
    """
    Apply the FAST protocol and trajectory-escalation rules.

    Parameters
    ----------
    face_class_id : classifier output — 0 = Normal, 1 = Mild, 2 = Severe
    trajectory    : Trajectory enum value as a string (e.g. "STABLE")
    speech_slur   : True if speech slurring detected (future sensor input)
    arm_drift     : True if arm drift detected       (future sensor input)

    Returns
    -------
    RiskLevel enum value
    """
    # Trajectory escalation: any pathological face + collapse trajectory → CRITICAL
    if face_class_id > 0 and trajectory in _CRITICAL_TRAJECTORIES:
        return RiskLevel.CRITICAL

    # Multi-modal FAST table (evaluated in severity order)
    if speech_slur and arm_drift:
        return RiskLevel.CRITICAL
    if face_class_id == 2 and speech_slur:
        return RiskLevel.CRITICAL
    if face_class_id == 2:
        return RiskLevel.HIGH_RISK
    if face_class_id == 1 and speech_slur:
        return RiskLevel.HIGH_RISK
    if face_class_id == 1:
        return RiskLevel.MILD

    return RiskLevel.NORMAL
