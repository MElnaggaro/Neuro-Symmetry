"""
Decision Engine — ONNX Runtime inference with threshold fallback.

Searches ai/checkpoints/ for a trained model in this order:
  1. model_real_calibrated.onnx  (trained on real data)
  2. model_calibrated.onnx       (trained on synthetic data)

Falls back to symmetry-score thresholds when no ONNX model is present,
so the API is functional before training completes.

Label convention
----------------
  0 = Normal
  1 = Mild asymmetry
  2 = Severe asymmetry
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


CLASS_LABELS = ["Normal", "Mild", "Severe"]

_MILD_THRESHOLD   = 0.75   # symmetry score below this → at least Mild
_SEVERE_THRESHOLD = 0.55   # symmetry score below this → Severe


@dataclass(frozen=True)
class DecisionResult:
    class_id:      int
    class_label:   str
    confidence:    float
    probabilities: list[float]   # [p_normal, p_mild, p_severe]
    source:        str           # "onnx" | "threshold"


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def _threshold_decision(features: np.ndarray) -> DecisionResult:
    score = float(np.exp(-features[49]))
    if score >= _MILD_THRESHOLD:
        probs, cid = [0.85, 0.12, 0.03], 0
    elif score >= _SEVERE_THRESHOLD:
        probs, cid = [0.10, 0.75, 0.15], 1
    else:
        probs, cid = [0.05, 0.20, 0.75], 2
    return DecisionResult(
        class_id=cid,
        class_label=CLASS_LABELS[cid],
        confidence=probs[cid],
        probabilities=probs,
        source="threshold",
    )


class DecisionEngine:
    """
    Inference engine — ONNX Runtime preferred, threshold fallback.

    Usage::
        engine = DecisionEngine()
        result = engine.infer(features)  # features: (50,) float32
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._session = None
        self._input_name = "features"

        search_dir = model_dir or (
            Path(__file__).resolve().parents[2] / "ai" / "checkpoints"
        )
        candidates = [
            search_dir / "model_real_calibrated.onnx",
            search_dir / "model_calibrated.onnx",
        ]

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(candidate),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                print(f"[DecisionEngine] Loaded: {candidate.name}")
                break
            except Exception as exc:
                print(f"[DecisionEngine] Could not load {candidate.name}: {exc}")

        if self._session is None:
            print("[DecisionEngine] No ONNX model found — using threshold fallback.")

        # Load scaler trained alongside the model (required for normalized features)
        self._scaler = None
        scaler_path = search_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            try:
                import joblib
                self._scaler = joblib.load(scaler_path)
                print(f"[DecisionEngine] Scaler loaded: {scaler_path.name}")
            except Exception as exc:
                print(f"[DecisionEngine] Could not load scaler: {exc}")

    @property
    def has_model(self) -> bool:
        return self._session is not None

    def infer(self, features: np.ndarray) -> DecisionResult:
        """
        Run inference on a 50-dim feature vector.
        Accepts shape (50,) or (1, 50).
        """
        features = np.asarray(features, dtype=np.float32).ravel()

        # Apply scaler if available (must match training pipeline)
        if self._scaler is not None:
            features = self._scaler.transform(features.reshape(1, -1))[0].astype(np.float32)

        if self._session is None:
            return _threshold_decision(features)

        x      = features.reshape(1, -1)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        probs  = _softmax(logits)
        cid    = int(probs.argmax())
        return DecisionResult(
            class_id=cid,
            class_label=CLASS_LABELS[cid],
            confidence=float(probs[cid]),
            probabilities=probs.tolist(),
            source="onnx",
        )
