"""
V2 ONNX inference endpoint.

This module is intentionally independent from ``ai/`` Python modules. It loads
the exported ONNX graph plus checkpoint metadata and exposes a small typed API
for raw 50-D MediaPipe feature vectors.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.core import get_settings

_log = logging.getLogger("neuro_symmetry.backend.predict")

router = APIRouter(prefix="/api/v1", tags=["v2-inference"])

ClassLabel = Literal["Normal", "Mild", "Severe"]
FeatureVector50 = Annotated[list[float], Field(min_length=50, max_length=50)]
CLASS_LABELS: tuple[ClassLabel, ClassLabel, ClassLabel] = ("Normal", "Mild", "Severe")
THRESHOLD_PRIORITY: tuple[int, int, int] = (2, 1, 0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: FeatureVector50 = Field(
        ...,
        description="Raw 50-D MediaPipe feature vector.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_: ClassLabel = Field(alias="class")
    probability: float = Field(ge=0.0, le=1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = logits64 - np.max(logits64)
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def _coerce_vector(
    payload: object,
    *,
    name: str,
    expected_len: int,
) -> np.ndarray:
    if payload is None:
        raise ValueError(f"{name} is missing")

    vector = np.asarray(payload, dtype=np.float32).reshape(-1)
    if vector.shape[0] != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {vector.shape[0]}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"V2 metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint_metadata(checkpoints_dir: Path) -> dict:
    pt_path = checkpoints_dir / "model_v2_calibrated.pt"
    if not pt_path.exists():
        return {}

    try:
        import torch

        return torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        _log.warning("Could not load V2 checkpoint metadata from %s: %s", pt_path, exc)
        return {}


def _load_scaler_from_pickle(checkpoints_dir: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    scaler_path = checkpoints_dir / "feature_scaler.pkl"
    if not scaler_path.exists():
        return None, None

    try:
        import joblib

        scaler = joblib.load(scaler_path)
        return (
            np.asarray(scaler.mean_, dtype=np.float32),
            np.asarray(scaler.scale_, dtype=np.float32),
        )
    except Exception as exc:
        _log.warning("Could not load fallback scaler from %s: %s", scaler_path, exc)
        return None, None


def _load_scaler(
    metrics: dict,
    checkpoint_metadata: dict,
    checkpoints_dir: Path,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = metrics.get("scaler_mean")
    std = metrics.get("scaler_std")

    if mean is None or std is None:
        mean = checkpoint_metadata.get("scaler_mean")
        std = checkpoint_metadata.get("scaler_std")

    if mean is None or std is None:
        mean_arr, std_arr = _load_scaler_from_pickle(checkpoints_dir)
    else:
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)

    if mean_arr is None or std_arr is None:
        raise RuntimeError(
            "V2 scaler_mean/scaler_std are missing. Add them to metrics_v2.json "
            "or keep model_v2_calibrated.pt alongside model_v2.onnx."
        )

    if mean_arr.shape != (input_dim,) or std_arr.shape != (input_dim,):
        raise RuntimeError(
            "V2 scaler shape mismatch: "
            f"mean={mean_arr.shape}, std={std_arr.shape}, expected=({input_dim},)"
        )

    std_arr = np.where(std_arr == 0.0, 1.0, std_arr).astype(np.float32)
    return mean_arr.astype(np.float32), std_arr


def _load_thresholds(metrics: dict, checkpoint_metadata: dict) -> np.ndarray:
    thresholds = metrics.get("thresholds") or checkpoint_metadata.get("thresholds")
    try:
        vector = _coerce_vector(thresholds, name="thresholds", expected_len=3)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    if np.any((vector < 0.0) | (vector > 1.0)):
        raise RuntimeError("V2 thresholds must be in the [0, 1] range")
    return vector


def _threshold_predict(probabilities: np.ndarray, thresholds: np.ndarray) -> int:
    for class_id in THRESHOLD_PRIORITY:
        if probabilities[class_id] >= thresholds[class_id]:
            return class_id
    return int(np.argmax(probabilities))


class V2OnnxPredictor:
    def __init__(self, *, model_path: Path, metrics_path: Path, checkpoints_dir: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"V2 ONNX model not found: {model_path}")

        metrics = _load_json(metrics_path)
        checkpoint_metadata = _load_checkpoint_metadata(checkpoints_dir)

        self.input_dim = int(metrics.get("input_dim") or checkpoint_metadata.get("input_dim") or 50)
        if self.input_dim != 50:
            raise RuntimeError(f"Expected a 50-D V2 model, got input_dim={self.input_dim}")

        self.thresholds = _load_thresholds(metrics, checkpoint_metadata)
        self.scaler_mean, self.scaler_std = _load_scaler(
            metrics,
            checkpoint_metadata,
            checkpoints_dir,
            self.input_dim,
        )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for /api/v1/predict") from exc

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        _log.info(
            "V2 ONNX predictor loaded | model=%s input=%s thresholds=%s",
            model_path,
            self.input_name,
            [round(float(t), 6) for t in self.thresholds],
        )

    def predict(self, features: list[float]) -> PredictResponse:
        raw = _coerce_vector(features, name="features", expected_len=self.input_dim)
        scaled = ((raw - self.scaler_mean) / self.scaler_std).astype(np.float32)

        logits = self.session.run(None, {self.input_name: scaled.reshape(1, -1)})[0][0]
        probabilities = _softmax(logits)
        class_id = _threshold_predict(probabilities, self.thresholds)

        return PredictResponse(
            class_=CLASS_LABELS[class_id],
            probability=float(probabilities[class_id]),
        )


@lru_cache(maxsize=1)
def get_predictor() -> V2OnnxPredictor:
    settings = get_settings()
    return V2OnnxPredictor(
        model_path=settings.checkpoints_dir / "model_v2.onnx",
        metrics_path=settings.checkpoints_dir / "metrics_v2.json",
        checkpoints_dir=settings.checkpoints_dir,
    )


def model_status() -> dict:
    settings = get_settings()
    model_path = settings.checkpoints_dir / "model_v2.onnx"
    metrics_path = settings.checkpoints_dir / "metrics_v2.json"
    return {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }


@router.post("/predict", response_model=PredictResponse, response_model_by_alias=True)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
        return predictor.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        _log.error("V2 predictor unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
