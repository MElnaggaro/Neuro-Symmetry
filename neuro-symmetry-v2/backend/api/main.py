"""
FastAPI backend — Phase 5.

Endpoints
---------
GET  /health    → service liveness + model availability
POST /analyze   → single-frame analysis (multipart image upload)
WS   /stream    → real-time per-frame analysis over WebSocket

POST /analyze
  Request : multipart/form-data  field "file" — any OpenCV-readable image
  Response: AnalysisResponse JSON (see schema below)

WS /stream
  Client → Server: {"image": "<base64-encoded image bytes>"}
  Server → Client: AnalysisResponse JSON + {"frame": N}

AnalysisResponse fields
-----------------------
  quality_code       : "OK" | "DEGRADED" | "UNRELIABLE"
  quality_reason     : null or string (e.g. "camera_blur")
  class_id           : 0 | 1 | 2  (null when UNRELIABLE)
  class_label        : "Normal" | "Mild" | "Severe"  (null when UNRELIABLE)
  confidence         : max softmax probability  (null when UNRELIABLE)
  probabilities      : [p_normal, p_mild, p_severe]  (null when UNRELIABLE)
  symmetry_score     : float ∈ [0, 1]  (null when UNRELIABLE)
  change_detected    : bool — z-score anomaly vs. rolling baseline
  confirmation_state : "NORMAL" | "PENDING" | "CONFIRMED"
  latency_ms         : end-to-end pipeline time in milliseconds
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.api.change_detection import ChangeDetector
from backend.api.confirmation_layer import ConfirmationLayer
from backend.api.decision_engine import DecisionEngine
from backend.api.feature_extractor import compute_symmetry_score, extract_features
from backend.api.input_quality import InputQualityChecker
from backend.api.landmark_engine import LandmarkEngine


# ── Application state ─────────────────────────────────────────────────────────

class _AppState:
    landmark_engine: Optional[LandmarkEngine] = None
    quality_checker: InputQualityChecker = InputQualityChecker()
    decision_engine: Optional[DecisionEngine] = None


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state.landmark_engine = LandmarkEngine()
        print("[startup] LandmarkEngine loaded.")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schema ───────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    quality_code:       str
    quality_reason:     Optional[str]         = None
    class_id:           Optional[int]         = None
    class_label:        Optional[str]         = None
    confidence:         Optional[float]       = None
    probabilities:      Optional[list[float]] = None
    symmetry_score:     Optional[float]       = None
    change_detected:    bool                  = False
    confirmation_state: str                   = "NORMAL"
    latency_ms:         float                 = 0.0


class StreamFrameResponse(AnalysisResponse):
    frame: int = 0


# ── Shared state for REST endpoint (single-user stateful baseline) ─────────────

_rest_change_detector = ChangeDetector()
_rest_confirmation    = ConfirmationLayer()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_pipeline(
    frame_bgr:       np.ndarray,
    change_detector: ChangeDetector,
    confirmation:    ConfirmationLayer,
) -> AnalysisResponse:
    """Quality → landmarks → features → inference → change → confirm."""
    t0 = time.perf_counter()

    if _state.landmark_engine is None:
        raise HTTPException(
            503,
            detail=(
                "Landmark model not loaded. "
                "Download it: python -m backend.api.landmark_engine --download"
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

    features = extract_features(result, frame_bgr)
    score    = compute_symmetry_score(features)
    decision = _state.decision_engine.infer(features)

    change_event = change_detector.update(score)
    confirmation.update(decision.class_id)

    return AnalysisResponse(
        quality_code=quality.code.value,
        quality_reason=quality.reason,
        class_id=decision.class_id,
        class_label=decision.class_label,
        confidence=round(decision.confidence, 4),
        probabilities=[round(p, 4) for p in decision.probabilities],
        symmetry_score=round(score, 4),
        change_detected=change_event is not None,
        confirmation_state=confirmation.state.value,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
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
    The REST endpoint shares a rolling baseline across calls for change detection.
    For per-session baselines, use the WebSocket /stream endpoint.
    """
    data = await file.read()
    frame = _decode_image(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Send JPEG, PNG, BMP, or WEBP.")

    return _run_pipeline(frame, _rest_change_detector, _rest_confirmation)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """
    Real-time WebSocket stream with per-connection state.

    Client sends JSON:  {"image": "<base64-encoded image bytes>"}
    Server replies with AnalysisResponse + {"frame": N}

    Each connection gets its own ChangeDetector and ConfirmationLayer so
    baselines are isolated per user session.
    """
    await ws.accept()

    change_detector = ChangeDetector()
    confirmation    = ConfirmationLayer()
    frame_count     = 0

    try:
        while True:
            msg = await ws.receive_json()
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

            response = _run_pipeline(frame, change_detector, confirmation)
            payload  = response.model_dump()
            payload["frame"] = frame_count
            frame_count += 1
            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
