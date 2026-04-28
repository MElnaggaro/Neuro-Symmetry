"""
FastAPI backend — Phase 5 + Phase 6.

Full intelligence pipeline per frame:
  Frame → InputQualityChecker → LandmarkEngine → FeatureExtractor
        → DecisionEngine (inference) → ZScoreBaseline (anomaly)
        → TemporalEngine → ChangeDetector → TrajectoryEngine
        → ClinicalRules (FAST) → ConfirmationLayer → XAIEngine
        → JSON response

Endpoints
---------
GET  /health          → liveness probe + model availability flags
POST /analyze         → single-frame analysis (multipart image upload)
POST /calibrate       → feed one calibration frame into per-session baseline
POST /session/reset   → clear rolling session state (REST endpoint)
WS   /stream          → real-time per-frame analysis over WebSocket

POST /analyze  and  POST /calibrate
  Request : multipart/form-data, field "file" — any OpenCV-readable image

WS /stream  message types (JSON):
  {"image": "<base64>"}                         → analyse frame
  {"type": "calibrate", "image": "<base64>"}   → calibration frame
  {"type": "reset"}                             → reset session state

AnalysisResponse schema
-----------------------
  quality_code        "OK" | "DEGRADED" | "UNRELIABLE"
  quality_reason      null or string
  class_id            0 | 1 | 2  (null when UNRELIABLE)
  class_label         "Normal" | "Mild" | "Severe"
  confidence          max softmax probability
  probabilities       [p_normal, p_mild, p_severe]
  symmetry_score      exp(−symmetry_error) ∈ (0, 1]
  anomaly_score       z-score vs. personal baseline (null if uncalibrated)
  risk_level          "NORMAL" | "MILD" | "HIGH_RISK" | "CRITICAL"
  trajectory          "STABLE" | "LINEAR_DECLINE" | "SUDDEN_DROP" |
                      "OSCILLATING" | "COLLAPSE"
  affected_side       "LEFT" | "RIGHT" | "BILATERAL"
  onset_seconds       seconds since first alert (null if normal)
  xai                 list of {feature, contribution, level} ranked by impact
  alert               true when confirmation state is CONFIRMED
  change_detected     true when z-score anomaly vs. rolling baseline fires
  confirmation_state  "NORMAL" | "PENDING" | "CONFIRMED"
  ema_score           exponential moving average of symmetry score
  latency_ms          end-to-end pipeline time
"""

from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.calibrate_baseline import BaselineCalibrator, UserBaseline
from backend.api.change_detection import ChangeDetector
from backend.api.prodromal import router as _prodromal_router
from backend.api.clinical_rules import RiskLevel, apply_fast_logic
from backend.api.confirmation_layer import ConfirmationLayer, ConfirmationState
from backend.api.decision_engine import DecisionEngine
from backend.api.feature_extractor import FEATURE_NAMES, compute_symmetry_score, extract_features
from backend.api.input_quality import InputQualityChecker
from backend.api.landmark_engine import LandmarkEngine
from backend.api.temporal_engine import TemporalEngine
from backend.api.trajectory_engine import Trajectory, TrajectoryEngine
from backend.api.xai_engine import XAIFeature, explain_prediction

_log = logging.getLogger("neuro_symmetry.backend.api")

_ASSUMED_FPS      = 30.0   # used to convert onset_frame → onset_seconds
_MIN_CALIB_FRAMES = 30     # minimum calibration frames before baseline activates


# ── Per-session state ─────────────────────────────────────────────────────────

class _SessionState:
    """
    All stateful pipeline components scoped to a single analysis session.

    One instance is kept module-level for the REST endpoint; each WebSocket
    connection receives its own instance for full isolation.
    """

    def __init__(self) -> None:
        self.change_detector:   ChangeDetector         = ChangeDetector()
        self.confirmation:      ConfirmationLayer       = ConfirmationLayer()
        self.temporal_engine:   TemporalEngine          = TemporalEngine()
        self.trajectory_engine: TrajectoryEngine        = TrajectoryEngine()
        self.baseline:          Optional[UserBaseline]       = None
        self.calibrator:        Optional[BaselineCalibrator] = None
        self.frame_count:       int = 0

    def reset(self) -> None:
        """Reset rolling state. Baseline and calibrator are preserved."""
        self.change_detector.reset()
        self.confirmation.reset()
        self.temporal_engine.reset()
        self.trajectory_engine.reset()
        self.frame_count = 0


# ── Application-level singletons ──────────────────────────────────────────────

class _AppState:
    landmark_engine: Optional[LandmarkEngine]  = None
    quality_checker: InputQualityChecker        = InputQualityChecker()
    decision_engine: Optional[DecisionEngine]  = None
    rest_session:    _SessionState              = _SessionState()


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state.landmark_engine = LandmarkEngine()
        _log.info("LandmarkEngine loaded.")
    except FileNotFoundError as exc:
        _log.warning("LandmarkEngine not loaded: %s", exc)
        _state.landmark_engine = None

    _state.decision_engine = DecisionEngine()
    yield

    if _state.landmark_engine:
        _state.landmark_engine.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Neuro-Symmetry API",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(_prodromal_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────────────────

class XAIFeatureOut(BaseModel):
    feature:      str
    contribution: float
    level:        str    # "HIGH" | "MEDIUM" | "LOW"


class AnalysisResponse(BaseModel):
    quality_code:       str
    quality_reason:     Optional[str]             = None
    class_id:           Optional[int]             = None
    class_label:        Optional[str]             = None
    confidence:         Optional[float]           = None
    probabilities:      Optional[list[float]]     = None
    symmetry_score:     Optional[float]           = None
    anomaly_score:      Optional[float]           = None
    risk_level:         Optional[str]             = None
    trajectory:         Optional[str]             = None
    affected_side:      Optional[str]             = None
    onset_seconds:      Optional[float]           = None
    xai:                list[XAIFeatureOut]       = []
    alert:              bool                      = False
    change_detected:    bool                      = False
    confirmation_state: str                       = "NORMAL"
    ema_score:          Optional[float]           = None
    latency_ms:         float                     = 0.0


class StreamFrameResponse(AnalysisResponse):
    frame: int = 0


class CalibrateResponse(BaseModel):
    frames_recorded: int
    ready:           bool
    min_frames:      int = _MIN_CALIB_FRAMES
    message:         str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_pipeline(
    frame_bgr: np.ndarray,
    session:   _SessionState,
) -> AnalysisResponse:
    """
    Execute the full intelligence pipeline for one frame.

    All stateful components (temporal, trajectory, confirmation, change
    detection) are read from and written back to *session* so that each
    caller maintains independent rolling state.
    """
    t0 = time.perf_counter()
    session.frame_count += 1

    if _state.landmark_engine is None:
        raise HTTPException(
            503,
            detail=(
                "Landmark model not loaded. "
                "Run: python -m backend.api.landmark_engine --download"
            ),
        )

    result  = _state.landmark_engine.process(frame_bgr)
    quality = _state.quality_checker.check(frame_bgr, result)

    if not quality.is_usable:
        return AnalysisResponse(
            quality_code=quality.code.value,
            quality_reason=quality.reason,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    features = extract_features(result, frame_bgr)
    score    = compute_symmetry_score(features)
    decision = _state.decision_engine.infer(features)

    # ── Personalised baseline (anomaly z-score) ───────────────────────────────
    anomaly_score: Optional[float] = None
    if session.baseline is not None:
        anomaly_score = round(session.baseline.anomaly_score(features), 4)

    # ── Temporal tracking (EMA, slope, onset) ─────────────────────────────────
    temporal_state = session.temporal_engine.update(score, decision.class_id)

    # ── Change detection ──────────────────────────────────────────────────────
    change_event = session.change_detector.update(score)

    # ── Trajectory classification ─────────────────────────────────────────────
    trajectory: Trajectory = session.trajectory_engine.update(score)

    # ── FAST clinical risk level ──────────────────────────────────────────────
    risk_level: RiskLevel = apply_fast_logic(decision.class_id, trajectory.value)

    # ── Confirmation gate ─────────────────────────────────────────────────────
    session.confirmation.update(decision.class_id)
    alert = session.confirmation.state == ConfirmationState.CONFIRMED

    # ── XAI + affected side ───────────────────────────────────────────────────
    xai_features, affected_side = explain_prediction(
        features, FEATURE_NAMES, decision.probabilities, decision.class_id,
    )

    # ── Onset seconds ─────────────────────────────────────────────────────────
    onset_seconds: Optional[float] = None
    if temporal_state.onset_frame is not None:
        onset_seconds = round(temporal_state.duration_frames / _ASSUMED_FPS, 1)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if alert:
        _log.warning(
            "ALERT CONFIRMED | frame=%d  risk=%s  trajectory=%s  onset=%.1fs  latency=%.1fms",
            session.frame_count, risk_level.value, trajectory.value,
            onset_seconds or 0.0, latency_ms,
        )

    return AnalysisResponse(
        quality_code=quality.code.value,
        quality_reason=quality.reason,
        class_id=decision.class_id,
        class_label=decision.class_label,
        confidence=round(decision.confidence, 4),
        probabilities=[round(p, 4) for p in decision.probabilities],
        symmetry_score=round(score, 4),
        anomaly_score=anomaly_score,
        risk_level=risk_level.value,
        trajectory=trajectory.value,
        affected_side=affected_side,
        onset_seconds=onset_seconds,
        xai=[
            XAIFeatureOut(feature=f.feature, contribution=f.contribution, level=f.level)
            for f in xai_features
        ],
        alert=alert,
        change_detected=change_event is not None,
        confirmation_state=session.confirmation.state.value,
        ema_score=round(temporal_state.ema_score, 4),
        latency_ms=latency_ms,
    )


def _record_calibration_frame(
    frame_bgr: np.ndarray,
    session:   _SessionState,
    user_id:   str = "default",
) -> CalibrateResponse:
    """
    Shared logic for REST and WebSocket calibration frame ingestion.
    Returns a CalibrateResponse reflecting current calibrator state.
    """
    if _state.landmark_engine is None:
        raise HTTPException(503, "Landmark engine not loaded.")

    result  = _state.landmark_engine.process(frame_bgr)
    quality = _state.quality_checker.check(frame_bgr, result)

    if not quality.is_usable:
        n = session.calibrator.n_samples if session.calibrator else 0
        return CalibrateResponse(
            frames_recorded=n,
            ready=session.baseline is not None,
            message=f"Frame rejected: {quality.reason or quality.code.value}",
        )

    if session.calibrator is None:
        session.calibrator = BaselineCalibrator(user_id)

    features = extract_features(result, frame_bgr)
    session.calibrator.record(features)
    n = session.calibrator.n_samples

    if n >= _MIN_CALIB_FRAMES and session.baseline is None:
        session.baseline = session.calibrator.compute()
        _log.info("Baseline calibrated from %d frames (user=%s).", n, user_id)
        return CalibrateResponse(
            frames_recorded=n,
            ready=True,
            message=f"Baseline calibrated from {n} frames. Anomaly scoring active.",
        )

    return CalibrateResponse(
        frames_recorded=n,
        ready=session.baseline is not None,
        message=f"Recorded {n}/{_MIN_CALIB_FRAMES} calibration frames.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe + model availability."""
    return {
        "status": "ok",
        "landmark_engine": _state.landmark_engine is not None,
        "decision_engine": (
            _state.decision_engine.has_model if _state.decision_engine else False
        ),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Analyse a single face image.

    Accepts any OpenCV-readable format (JPEG, PNG, BMP, WEBP).
    Uses the shared REST session — call /session/reset between subjects.
    """
    data = await file.read()
    frame = _decode_image(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Send JPEG, PNG, BMP, or WEBP.")
    return _run_pipeline(frame, _state.rest_session)


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(file: UploadFile = File(...)) -> CalibrateResponse:
    """
    Feed one frame into the REST-session baseline calibrator.

    Send ~30–60 frames of a neutral face.  Once enough frames have been
    collected the baseline activates automatically, and subsequent /analyze
    responses will include a personalised ``anomaly_score``.
    """
    data = await file.read()
    frame = _decode_image(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")
    return _record_calibration_frame(frame, _state.rest_session, user_id="default")


@app.post("/session/reset")
async def session_reset() -> dict:
    """
    Reset REST-session rolling state (temporal, trajectory, change detection,
    confirmation).  Baseline and calibrator are preserved.
    """
    _state.rest_session.reset()
    _log.info("REST session state reset.")
    return {"status": "reset"}


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """
    Real-time WebSocket stream.

    Each connection gets fully isolated session state (temporal baseline,
    trajectory history, confirmation window, change detector).

    Message types
    -------------
    {"image": "<base64>"}
        Analyse one frame.  Reply: AnalysisResponse + {"frame": N}.

    {"type": "calibrate", "image": "<base64>"}
        Feed a calibration frame.  Reply: CalibrateResponse JSON.

    {"type": "reset"}
        Reset rolling session state.  Reply: {"status": "reset"}.
    """
    await ws.accept()

    session:     _SessionState = _SessionState()
    frame_count: int           = 0

    try:
        while True:
            msg   = await ws.receive_json()
            mtype = msg.get("type", "analyze")

            # ── Session reset ─────────────────────────────────────────────────
            if mtype == "reset":
                session.reset()
                await ws.send_json({"status": "reset"})
                continue

            # ── Calibration frame ─────────────────────────────────────────────
            if mtype == "calibrate":
                b64 = msg.get("image", "")
                if not b64:
                    await ws.send_json({"error": "missing 'image' field"})
                    continue
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    await ws.send_json({"error": "invalid base64"})
                    continue
                frame = _decode_image(raw)
                if frame is None:
                    await ws.send_json({"error": "could not decode image"})
                    continue
                resp = _record_calibration_frame(frame, session, user_id="ws_session")
                payload = resp.model_dump()
                payload["type"] = "calibrate"
                await ws.send_json(payload)
                continue

            # ── Inference frame ───────────────────────────────────────────────
            b64 = msg.get("image", "")
            if not b64:
                await ws.send_json({"error": "missing 'image' field"})
                continue
            try:
                raw = base64.b64decode(b64)
            except Exception:
                await ws.send_json({"error": "invalid base64 encoding"})
                continue
            frame = _decode_image(raw)
            if frame is None:
                await ws.send_json({"error": "could not decode image"})
                continue

            response = _run_pipeline(frame, session)
            payload  = response.model_dump()
            payload["frame"] = frame_count
            frame_count += 1
            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
