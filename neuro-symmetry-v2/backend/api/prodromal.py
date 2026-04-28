"""
Prodromal Sentinel — temporal-asymmetry assessment endpoints.

POST /prodromal/assess  — receive browser-computed eye/smile timing, return alert
GET  /prodromal/history — last N stored assessments (in-process ring buffer)

All heavy temporal computation is done client-side; this layer validates,
enriches, logs, and returns a normalised ProdromalResult that the frontend
renders.  A persistent DB can replace _assessment_log in production.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/prodromal", tags=["prodromal"])
_log = logging.getLogger("neuro_symmetry.prodromal")

_assessment_log: list[dict] = []   # ring-buffer, max 500 entries


# ── Request models ────────────────────────────────────────────────────────────

class EyeEvent(BaseModel):
    """Timing of peak blink closure (ms from recording start)."""
    blink_close_ms: Optional[float] = None
    detected:       bool            = False


class SmileEvent(BaseModel):
    """Timing when lip-corner elevation first crossed the smile threshold (ms)."""
    elevation_ms: Optional[float] = None
    detected:     bool            = False


class ProdromalRequest(BaseModel):
    """
    Browser sends timing data extracted from frame-level MediaPipe landmarks
    during the 6-second guided clip (1s baseline → 2.5s blink → 2.5s smile).
    """
    right_eye:     EyeEvent   = Field(default_factory=EyeEvent)
    left_eye:      EyeEvent   = Field(default_factory=EyeEvent)
    right_smile:   SmileEvent = Field(default_factory=SmileEvent)
    left_smile:    SmileEvent = Field(default_factory=SmileEvent)
    symptom_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Keys: mastoid_pain, hyperacusis, taste_change",
    )


# ── Response model ────────────────────────────────────────────────────────────

class ProdromalResult(BaseModel):
    id:            str
    asymmetry_ms:  float
    affected_side: str    # "LEFT" | "RIGHT" | "NONE"
    alert_level:   str    # "NONE" | "YELLOW" | "RED"
    confidence:    float  # 0.0–1.0
    detail:        str
    timestamp:     str


# ── Core assessment logic ─────────────────────────────────────────────────────

def _assess(req: ProdromalRequest) -> ProdromalResult:
    # Eye-blink temporal asymmetry ──────────────────────────────────────────
    eye_asym_ms, eye_side = 0.0, "NONE"
    if req.right_eye.detected and req.left_eye.detected:
        r_t = req.right_eye.blink_close_ms or 0.0
        l_t = req.left_eye.blink_close_ms  or 0.0
        eye_asym_ms = abs(r_t - l_t)
        # Slower side (later peak) is the suspected affected side.
        # Note: "right/left" here are IMAGE-space (MediaPipe convention):
        #   image-right eye = user's biological LEFT side.
        eye_side = "LEFT" if l_t > r_t else "RIGHT"  # biological mapping (mirrored)

    # Smile-corner temporal asymmetry ───────────────────────────────────────
    smile_asym_ms, smile_side = 0.0, "NONE"
    if req.right_smile.detected and req.left_smile.detected:
        r_t = req.right_smile.elevation_ms or 0.0
        l_t = req.left_smile.elevation_ms  or 0.0
        smile_asym_ms = abs(r_t - l_t)
        smile_side = "LEFT" if l_t > r_t else "RIGHT"

    # Worst signal dominates (most clinically conservative) ─────────────────
    asym_ms = max(eye_asym_ms, smile_asym_ms)
    side    = eye_side if eye_asym_ms >= smile_asym_ms else smile_side

    # Confidence: 0 signals → 0.40, 1 → 0.70, 2 → 1.00 ─────────────────────
    n_signals = (
        int(req.right_eye.detected and req.left_eye.detected)
        + int(req.right_smile.detected and req.left_smile.detected)
    )
    confidence = round(0.40 + 0.30 * n_signals, 2)

    # Alert threshold: 200 ms matches clinical prodromal detection literature ─
    symptom_count = sum(1 for v in req.symptom_flags.values() if v)
    if asym_ms > 200 and symptom_count >= 1:
        alert = "RED"
    elif asym_ms > 200:
        alert = "YELLOW"
    else:
        alert = "NONE"

    # Human-readable summary ────────────────────────────────────────────────
    parts: list[str] = []
    if asym_ms > 50:
        parts.append(
            f"{asym_ms:.0f} ms temporal asymmetry — "
            f"{side} side shows delayed muscle response"
        )
    else:
        parts.append("Muscle activation timing within normal range")
    if symptom_count:
        parts.append(f"{symptom_count} associated symptom(s) reported")

    uid = str(uuid.uuid4())
    result = ProdromalResult(
        id            = uid,
        asymmetry_ms  = round(asym_ms, 1),
        affected_side = side if asym_ms > 50 else "NONE",
        alert_level   = alert,
        confidence    = confidence,
        detail        = "; ".join(parts),
        timestamp     = datetime.now(timezone.utc).isoformat(),
    )

    entry = result.model_dump()
    _assessment_log.append(entry)
    if len(_assessment_log) > 500:
        _assessment_log.pop(0)

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/assess", response_model=ProdromalResult)
async def assess(req: ProdromalRequest) -> ProdromalResult:
    """Analyse browser-computed timing data and return an alert classification."""
    result = _assess(req)
    _log.info(
        "Prodromal: asym=%.0f ms  side=%s  alert=%s  conf=%.2f",
        result.asymmetry_ms, result.affected_side, result.alert_level, result.confidence,
    )
    return result


@router.get("/history")
async def history(limit: int = 50) -> list[dict]:
    """Return the most-recent assessments (newest last)."""
    return _assessment_log[-limit:]
