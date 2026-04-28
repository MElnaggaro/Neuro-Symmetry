"""
V2 Production Inference Pipeline.

Strict architectural rules
--------------------------
1. Pydantic schemas for all I/O contracts
2. All tunable parameters loaded from environment variables
3. Python logging module only (print() forbidden in production paths)
4. Structured as a drop-in FastAPI worker (Docker-ready)

Environment variables
---------------------
V2_MODEL_PATH           path to model_v2_calibrated.pt          [required]
V2_DEVICE               "cpu" | "cuda"                          [cpu]
V2_TEMPERATURE          override temperature scalar             [from checkpoint]
V2_THRESHOLD_NORMAL     override Normal  threshold              [from checkpoint]
V2_THRESHOLD_MILD       override Mild    threshold              [from checkpoint]
V2_THRESHOLD_SEVERE     override Severe  threshold              [from checkpoint]
V2_TEMPORAL_WINDOW      deque size for temporal smoother        [10]
V2_REQUIRED_FRAMES      consecutive frames to confirm label     [5]
V2_USE_V2_FEATURES      "1" to also extract 25-D geometric feats [0]

Drop-in replacement for backend/api/decision_engine.py DecisionEngine.
Call V2DecisionEngine.infer(features_50d) for immediate migration with
zero changes to the existing backend — it accepts the same 50-D vector.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ai.v2.model_v2 import CalibratedModelV2
from ai.v2.post_hoc import TemporalSmoother, threshold_predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger("neuro_symmetry.v2.inference")

CLASS_NAMES = {0: "Normal", 1: "Mild", 2: "Severe"}


# ── Configuration ─────────────────────────────────────────────────────────────

def _env(key: str, default, cast=str):
    return cast(os.environ.get(key, default))


try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class InferenceConfig(BaseSettings):
        model_path:        str   = Field(env="V2_MODEL_PATH")
        device:            str   = Field(default="cpu",  env="V2_DEVICE")
        temperature:       Optional[float] = Field(default=None, env="V2_TEMPERATURE")
        threshold_normal:  Optional[float] = Field(default=None, env="V2_THRESHOLD_NORMAL")
        threshold_mild:    Optional[float] = Field(default=None, env="V2_THRESHOLD_MILD")
        threshold_severe:  Optional[float] = Field(default=None, env="V2_THRESHOLD_SEVERE")
        temporal_window:   int   = Field(default=10, env="V2_TEMPORAL_WINDOW")
        required_frames:   int   = Field(default=5,  env="V2_REQUIRED_FRAMES")
        use_v2_features:   bool  = Field(default=False, env="V2_USE_V2_FEATURES")

        class Config:
            env_file = ".env"

except ImportError:
    @dataclass
    class InferenceConfig:
        model_path:       str   = field(default_factory=lambda: os.environ["V2_MODEL_PATH"])
        device:           str   = field(default_factory=lambda: _env("V2_DEVICE", "cpu"))
        temperature:      Optional[float] = None
        threshold_normal: Optional[float] = None
        threshold_mild:   Optional[float] = None
        threshold_severe: Optional[float] = None
        temporal_window:  int   = field(default_factory=lambda: _env("V2_TEMPORAL_WINDOW", 10, int))
        required_frames:  int   = field(default_factory=lambda: _env("V2_REQUIRED_FRAMES",  5, int))
        use_v2_features:  bool  = field(default_factory=lambda: _env("V2_USE_V2_FEATURES",  "0") == "1")


# ── I/O Schemas ───────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, Field as PField

    class V2PredictRequest(BaseModel):
        features_50d: List[float] = PField(
            ..., description="50-D feature vector from existing V1 pipeline"
        )
        features_25d: Optional[List[float]] = PField(
            None, description="25-D V2 geometric features (optional)"
        )
        frame_id: Optional[int] = None

    class V2PredictResponse(BaseModel):
        frame_id:               Optional[int]
        probabilities:          List[float]
        raw_prediction:         int
        thresholded_prediction: int
        temporal_prediction:    int
        confidence:             float
        is_pathological:        bool
        class_label:            str
        temporal_warmed_up:     bool

except ImportError:
    @dataclass
    class V2PredictRequest:
        features_50d: List[float]
        features_25d: Optional[List[float]] = None
        frame_id:     Optional[int] = None

    @dataclass
    class V2PredictResponse:
        frame_id:               Optional[int]
        probabilities:          List[float]
        raw_prediction:         int
        thresholded_prediction: int
        temporal_prediction:    int
        confidence:             float
        is_pathological:        bool
        class_label:            str
        temporal_warmed_up:     bool


# ── Core Pipeline ─────────────────────────────────────────────────────────────

class V2InferencePipeline:
    """
    Stateful inference pipeline — one instance per video stream.
    Thread-safety: use one instance per worker process in multi-worker deployments.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config   = config
        self._device   = torch.device(config.device)
        self._model    = self._load_model()
        self._smoother = TemporalSmoother(config.temporal_window, config.required_frames)

        # Thresholds: env override → checkpoint → safe default
        ckpt_thresh = self._model.thresholds
        self._thresholds = [
            config.threshold_normal  if config.threshold_normal  is not None else ckpt_thresh[0],
            config.threshold_mild    if config.threshold_mild    is not None else ckpt_thresh[1],
            config.threshold_severe  if config.threshold_severe  is not None else ckpt_thresh[2],
        ]

        # Scaler from checkpoint (if available)
        self._scaler_mean = (
            np.array(self._model.scaler_mean, dtype=np.float32)
            if self._model.scaler_mean else None
        )
        self._scaler_std = (
            np.array(self._model.scaler_std, dtype=np.float32)
            if self._model.scaler_std else None
        )

        _log.info(
            "V2 pipeline ready | device=%s thresholds=%s window=%d req=%d",
            config.device, [f"{t:.3f}" for t in self._thresholds],
            config.temporal_window, config.required_frames,
        )

    def _load_model(self) -> CalibratedModelV2:
        path = Path(self._config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"V2 model not found: {path}")

        # Override temperature from env if set
        model = CalibratedModelV2.load(path, device=self._config.device)
        if self._config.temperature is not None:
            model.temperature.data.fill_(self._config.temperature)
            _log.info("Temperature overridden by env: %.4f", self._config.temperature)

        model.eval()
        _log.info("Model loaded from %s", path)
        return model

    def _normalise(self, feats: np.ndarray) -> np.ndarray:
        if self._scaler_mean is not None and self._scaler_std is not None:
            return ((feats - self._scaler_mean) / (self._scaler_std + 1e-8)).astype(np.float32)
        return feats.astype(np.float32)

    @torch.no_grad()
    def predict(self, request: "V2PredictRequest") -> "V2PredictResponse":
        # Assemble feature vector
        feats = np.array(request.features_50d, dtype=np.float32)
        if request.features_25d is not None:
            feats = np.concatenate([feats, np.array(request.features_25d, dtype=np.float32)])

        feats_norm = self._normalise(feats)
        x          = torch.from_numpy(feats_norm).unsqueeze(0).to(self._device)

        logits, _  = self._model(x)
        probs      = F.softmax(logits, dim=-1).cpu().numpy()[0]

        raw_pred    = int(np.argmax(probs))
        thresh_pred = threshold_predict(probs, self._thresholds)
        temp_pred   = self._smoother.update(thresh_pred)
        confidence  = float(probs.max())
        pathological = temp_pred > 0

        if pathological:
            _log.warning(
                "PATHOLOGICAL frame=%s raw=%s thresh=%s temporal=%s conf=%.3f",
                request.frame_id,
                CLASS_NAMES[raw_pred], CLASS_NAMES[thresh_pred], CLASS_NAMES[temp_pred],
                confidence,
            )
        else:
            _log.info("frame=%s pred=%s conf=%.3f",
                      request.frame_id, CLASS_NAMES[temp_pred], confidence)

        return V2PredictResponse(
            frame_id=request.frame_id,
            probabilities=probs.tolist(),
            raw_prediction=raw_pred,
            thresholded_prediction=thresh_pred,
            temporal_prediction=temp_pred,
            confidence=confidence,
            is_pathological=pathological,
            class_label=CLASS_NAMES[temp_pred],
            temporal_warmed_up=self._smoother.is_warmed_up,
        )

    def reset_session(self) -> None:
        self._smoother.reset()
        _log.info("Temporal state reset.")


# ── V1-compatible drop-in wrapper ─────────────────────────────────────────────

class V2DecisionEngine:
    """
    Drop-in replacement for backend/api/decision_engine.py DecisionEngine.
    Accepts the same 50-D feature vector, returns a DecisionResult-compatible object.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        search_dir = model_dir or Path("ai/checkpoints")
        pt_path    = search_dir / "model_v2_calibrated.pt"

        if not pt_path.exists():
            _log.warning("V2 model not found at %s — V2DecisionEngine disabled.", pt_path)
            self._pipeline = None
            return

        os.environ.setdefault("V2_MODEL_PATH", str(pt_path))
        self._pipeline = V2InferencePipeline(InferenceConfig())
        _log.info("V2DecisionEngine ready.")

    @property
    def has_model(self) -> bool:
        return self._pipeline is not None

    def infer(self, features: np.ndarray):
        """
        Accepts 50-D float32 feature vector.
        Returns an object with .class_id, .class_label, .confidence, .probabilities.
        """
        from backend.api.decision_engine import DecisionResult

        if self._pipeline is None:
            # Graceful fallback to threshold-based decision
            from backend.api.decision_engine import _threshold_decision
            return _threshold_decision(features)

        features = np.asarray(features, dtype=np.float32).ravel()
        req  = V2PredictRequest(features_50d=features.tolist())
        resp = self._pipeline.predict(req)
        return DecisionResult(
            class_id=resp.temporal_prediction,
            class_label=resp.class_label,
            confidence=resp.confidence,
            probabilities=resp.probabilities,
            source="v2_pytorch",
        )

    def reset_session(self) -> None:
        if self._pipeline:
            self._pipeline.reset_session()


# ── FastAPI microservice (Docker-ready) ───────────────────────────────────────

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException

    _v2_pipeline: Optional[V2InferencePipeline] = None

    @asynccontextmanager
    async def lifespan_v2(app: FastAPI):
        global _v2_pipeline
        try:
            _v2_pipeline = V2InferencePipeline(InferenceConfig())
            _log.info("FastAPI V2 worker ready.")
        except Exception as exc:
            _log.error("Failed to init V2 pipeline: %s", exc)
        yield
        _log.info("FastAPI V2 worker shutting down.")

    v2_app = FastAPI(
        title="Neuro-Symmetry V2 Inference",
        version="2.0.0",
        lifespan=lifespan_v2,
    )

    @v2_app.get("/health")
    async def v2_health() -> dict:
        return {"status": "ok", "model_loaded": _v2_pipeline is not None}

    @v2_app.post("/predict", response_model=V2PredictResponse)
    async def v2_predict(req: V2PredictRequest) -> V2PredictResponse:
        if _v2_pipeline is None:
            raise HTTPException(503, "V2 pipeline not initialised")
        return _v2_pipeline.predict(req)

    @v2_app.post("/session/reset")
    async def v2_reset_session() -> dict:
        if _v2_pipeline is None:
            raise HTTPException(503, "V2 pipeline not initialised")
        _v2_pipeline.reset_session()
        return {"status": "reset"}

except ImportError:
    _log.warning("FastAPI not installed — HTTP entrypoint disabled.")
