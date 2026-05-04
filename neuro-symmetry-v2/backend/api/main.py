"""
FastAPI backend for Neuro-Symmetry V2.

The application shell owns transport concerns, image analysis, session state,
and router wiring. V2 model inference is isolated in ``backend.api.predict`` so
the backend no longer imports deleted V1 modules from ``ai/``.

Endpoints
---------
GET  /health          liveness probe and model availability flags
POST /api/v1/predict  V2 ONNX inference over a raw 50-D feature vector
POST /analyze         single-frame image analysis
POST /session/reset   clear rolling REST-session state
WS   /ws/stream       real-time per-frame analysis over WebSocket
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
from pydantic import BaseModel, Field

from ai.calibrate_baseline import BaselineCalibrator, UserBaseline
from backend.api.change_detection import ChangeDetector
from backend.api.clinical_rules import RiskLevel, apply_fast_logic
from backend.api.confirmation_layer import ConfirmationLayer, ConfirmationState
from backend.api.decision_engine import DecisionEngine
from backend.api.feature_extractor import FEATURE_NAMES, compute_symmetry_score, extract_features
from backend.api.input_quality import InputQualityChecker
from backend.api.landmark_engine import LandmarkEngine
from backend.api.predict import model_status as _predict_model_status
from backend.api.predict import router as _predict_router
from backend.api.prodromal import router as _prodromal_router
from backend.api.temporal_engine import TemporalEngine
from backend.api.trajectory_engine import Trajectory, TrajectoryEngine
from backend.api.xai_engine import explain_prediction
from backend.core import get_settings

_settings = get_settings()
logging.basicConfig(level=_settings.log_level)
_log = logging.getLogger("neuro_symmetry.backend.api")


class _SessionState:
    """
    Stateful analysis components scoped to a single subject/session.

    The old V1 personalised baseline calibration path has been removed. The
    remaining state is lightweight rolling state used by the V2 backend UX:
    temporal smoothing, change detection, trajectory tracking, and alert
    confirmation.
    """

    def __init__(self) -> None:
        self.change_detector: ChangeDetector = ChangeDetector()
        self.confirmation: ConfirmationLayer = ConfirmationLayer()
        self.temporal_engine: TemporalEngine = TemporalEngine()
        self.trajectory_engine: TrajectoryEngine = TrajectoryEngine()
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset rolling state for the current subject/session."""
        self.change_detector.reset()
        self.confirmation.reset()
        self.temporal_engine.reset()
        self.trajectory_engine.reset()
        self.frame_count = 0


class _AppState:
    landmark_engine: Optional[LandmarkEngine] = None
    quality_checker: InputQualityChecker = InputQualityChecker()
    decision_engine: Optional[DecisionEngine] = None
    rest_session: _SessionState = _SessionState()


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


app = FastAPI(
    title=_settings.api_title,
    version=_settings.api_version,
    debug=_settings.debug,
    lifespan=lifespan,
)

app.include_router(_predict_router)
app.include_router(_prodromal_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_methods=_settings.cors_allow_methods,
    allow_headers=_settings.cors_allow_headers,
    allow_credentials=_settings.cors_allow_credentials,
)


class XAIFeatureOut(BaseModel):
    feature: str
    contribution: float
    level: str


class AnalysisResponse(BaseModel):
    quality_code: str
    quality_reason: Optional[str] = None
    class_id: Optional[int] = None
    class_label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[list[float]] = None
    symmetry_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    risk_level: Optional[str] = None
    trajectory: Optional[str] = None
    affected_side: Optional[str] = None
    onset_seconds: Optional[float] = None
    xai: list[XAIFeatureOut] = Field(default_factory=list)
    alert: bool = False
    change_detected: bool = False
    confirmation_state: str = "NORMAL"
    ema_score: Optional[float] = None
    latency_ms: float = 0.0


class StreamFrameResponse(AnalysisResponse):
    frame: int = 0


def _decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_pipeline(
    frame_bgr:       np.ndarray,
    session:         _SessionState,
    quality_checker: Optional["InputQualityChecker"] = None,
    baseline:        Optional[UserBaseline] = None,
) -> AnalysisResponse:
    """
    Execute the image-analysis pipeline for one frame.

    The dedicated V2 ONNX model endpoint lives at ``/api/v1/predict``. This
    image path keeps the existing feature extraction, quality gates, temporal
    state, clinical rules, confirmation, and XAI flow intact.
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

    if _state.decision_engine is None:
        raise HTTPException(503, detail="Decision engine not initialised.")

    result  = _state.landmark_engine.process(frame_bgr)
    checker = quality_checker if quality_checker is not None else _state.quality_checker
    quality = checker.check(frame_bgr, result)

    if not quality.is_usable:
        return AnalysisResponse(
            quality_code=quality.code.value,
            quality_reason=quality.reason,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    features = extract_features(result, frame_bgr)
    score = compute_symmetry_score(features)
    decision = _state.decision_engine.infer(features)

    temporal_state = session.temporal_engine.update(score, decision.class_id)
    change_event = session.change_detector.update(score)
    trajectory: Trajectory = session.trajectory_engine.update(score)
    risk_level: RiskLevel = apply_fast_logic(decision.class_id, trajectory.value)

    # Personal anomaly: deviation from this user's calibrated baseline (z-score
    # mean over the 50-D feature vector). Replaces the change-detector z-score
    # when a baseline is available so the displayed anomaly is personalised.
    personal_anomaly: Optional[float] = (
        baseline.anomaly_score(features) if baseline is not None else None
    )

    session.confirmation.update(decision.class_id)
    alert = session.confirmation.state == ConfirmationState.CONFIRMED

    xai_features, affected_side = explain_prediction(
        features,
        FEATURE_NAMES,
        decision.probabilities,
        decision.class_id,
    )

    onset_seconds: Optional[float] = None
    if temporal_state.onset_frame is not None:
        onset_seconds = round(temporal_state.duration_frames / _settings.session.assumed_fps, 1)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if alert:
        _log.warning(
            "ALERT CONFIRMED | frame=%d risk=%s trajectory=%s onset=%.1fs latency=%.1fms",
            session.frame_count,
            risk_level.value,
            trajectory.value,
            onset_seconds or 0.0,
            latency_ms,
        )

    return AnalysisResponse(
        quality_code=quality.code.value,
        quality_reason=quality.reason,
        class_id=decision.class_id,
        class_label=decision.class_label,
        confidence=round(decision.confidence, 4),
        probabilities=[round(p, 4) for p in decision.probabilities],
        symmetry_score=round(score, 4),
        anomaly_score=(
            round(personal_anomaly, 4) if personal_anomaly is not None
            else (round(change_event.z_score, 4) if change_event else None)
        ),
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


@app.get("/health")
async def health() -> dict:
    """Liveness probe plus backend model availability."""
    return {
        "status": "ok",
        "landmark_engine": _state.landmark_engine is not None,
        "decision_engine": (
            _state.decision_engine.has_model if _state.decision_engine else False
        ),
        "v2_predict": _predict_model_status(),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Analyse a single face image.

    Accepts any OpenCV-readable format: JPEG, PNG, BMP, or WEBP. Uses the
    shared REST session; call ``/session/reset`` between subjects.
    """
    data = await file.read()
    frame = _decode_image(data)
    if frame is None:
        raise HTTPException(400, detail="Could not decode image. Send JPEG, PNG, BMP, or WEBP.")
    return _run_pipeline(frame, _state.rest_session)


@app.post("/session/reset")
async def session_reset() -> dict:
    """Reset rolling REST-session state."""
    _state.rest_session.reset()
    _state.quality_checker.reset()
    if _state.landmark_engine is not None:
        _state.landmark_engine.reset_session()
    _log.info("REST session state reset.")
    return {"status": "reset"}


@app.websocket("/stream")
@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    """
    Real-time WebSocket stream.

    Message types:
    - {"image": "<base64>"} analyzes one frame
    - {"type": "reset"} clears rolling state for this connection
    """
    await ws.accept()
    session = _SessionState()
    # Calibration state for this connection. We accumulate FEATURE VECTORS
    # (not raw frames) because the baseline z-score lives in feature space.
    calibrator = BaselineCalibrator(user_id=f"ws-{id(ws):x}")
    user_baseline: Optional[UserBaseline] = None
    calibration_min = _settings.session.min_calib_frames
    # Per-connection quality checker with independent blur-streak state
    ws_quality_checker = _state.quality_checker.__class__(get_settings().quality)

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type", "analyze")

            if mtype == "reset":
                session.reset()
                ws_quality_checker.reset()
                calibrator.reset()
                user_baseline = None
                await ws.send_json({"status": "reset"})
                continue

            if mtype == "calibrate":
                b64 = msg.get("image", "")
                if not b64:
                    await ws.send_json({"error": "missing 'image' field"})
                    continue
                try:
                    raw = base64.b64decode(b64, validate=True)
                except Exception:
                    await ws.send_json({"error": "invalid base64 encoding"})
                    continue
                frame = _decode_image(raw)
                if frame is None:
                    await ws.send_json({"error": "could not decode image"})
                    continue

                # Real calibration: run landmark detection + quality gate, then
                # accumulate the 50-D feature vector. Frames where the face is
                # missing or the quality is unusable do NOT count toward the
                # required minimum — otherwise a baseline could be built from
                # blurry / faceless frames and corrupt the personal z-score.
                if _state.landmark_engine is None:
                    await ws.send_json({"error": "landmark engine not available"})
                    continue

                lm_result = _state.landmark_engine.process(frame)
                quality = ws_quality_checker.check(frame, lm_result)
                if not quality.is_usable or lm_result is None:
                    await ws.send_json({
                        "type": "calibrate",
                        "frames_recorded": calibrator.n_samples,
                        "ready": False,
                        "min_frames": calibration_min,
                        "message": (
                            f"Skipped frame ({quality.reason or 'no face'}); "
                            f"hold steady — {calibrator.n_samples}/{calibration_min}"
                        ),
                    })
                    continue

                features = extract_features(lm_result, frame)
                calibrator.record(features)

                ready = calibrator.n_samples >= calibration_min
                if ready and user_baseline is None:
                    try:
                        user_baseline = calibrator.compute()
                        _log.info(
                            "Personal baseline ready (n=%d, mean‖μ‖=%.3f).",
                            calibrator.n_samples,
                            float(np.linalg.norm(user_baseline.mean)),
                        )
                    except RuntimeError as exc:
                        # Should never happen now (n_samples >= calibration_min
                        # and _MIN_SAMPLES are both checked), but log just in case.
                        _log.warning("Baseline compute failed: %s", exc)
                        ready = False

                await ws.send_json({
                    "type": "calibrate",
                    "frames_recorded": calibrator.n_samples,
                    "ready": ready,
                    "min_frames": calibration_min,
                    "message": (
                        "Baseline ready"
                        if ready
                        else f"Recording {calibrator.n_samples}/{calibration_min}"
                    ),
                })
                continue

            if mtype != "analyze":
                await ws.send_json({"error": f"unsupported message type: {mtype}"})
                continue

            b64 = msg.get("image", "")
            if not b64:
                await ws.send_json({"error": "missing 'image' field"})
                continue

            try:
                raw = base64.b64decode(b64, validate=True)
            except Exception:
                await ws.send_json({"error": "invalid base64 encoding"})
                continue

            frame = _decode_image(raw)
            if frame is None:
                await ws.send_json({"error": "could not decode image"})
                continue

            response = _run_pipeline(frame, session, ws_quality_checker, user_baseline)
            payload = response.model_dump()
            payload["frame"] = session.frame_count
            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
