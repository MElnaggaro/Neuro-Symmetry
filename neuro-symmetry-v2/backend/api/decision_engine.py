"""
Decision Engine — ONNX Runtime inference with threshold fallback.

All tuneable values (model search paths, severity thresholds, scaler filename)
are injected from ``backend.core.config.Settings``.

Label convention
----------------
  0 = Normal
  1 = Mild asymmetry
  2 = Severe asymmetry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core import Settings, get_settings

_log = logging.getLogger("neuro_symmetry.backend.decision_engine")

CLASS_LABELS: tuple[str, str, str] = ("Normal", "Mild", "Severe")


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


def _threshold_decision(features: np.ndarray, settings: Optional[Settings] = None) -> DecisionResult:
    """
    Symmetry-score-based fallback used when no ONNX model is available.

    score = exp(-symmetry_error) in (0, 1]
      score >= mild_threshold   -> Normal  (high symmetry)
      score >= severe_threshold -> Mild    (moderate asymmetry)
      score <  severe_threshold -> Severe  (strong asymmetry)

    Requires: mild_threshold > severe_threshold > 0. Validated at call time.
    """
    if settings is None:
        settings = get_settings()
    mild_t   = settings.decision.mild_threshold
    severe_t = settings.decision.severe_threshold

    # Guard: misconfigured thresholds would silently invert labels
    if not (mild_t > severe_t > 0):
        _log.error(
            "Invalid threshold config: mild=%.3f severe=%.3f — "
            "expected mild > severe > 0. Falling back to defaults (0.75 / 0.45).",
            mild_t, severe_t,
        )
        mild_t, severe_t = 0.75, 0.45

    score = float(np.exp(-features[49]))

    if score >= mild_t:
        probs, cid = [0.85, 0.12, 0.03], 0
    elif score >= severe_t:
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

    Parameters
    ----------
    model_dir : optional Path
        Override the directory where ONNX checkpoints are searched.
        Defaults to ``settings.checkpoints_dir``.
    settings  : optional Settings
        Override the active configuration (used by tests).

    Usage::

        engine = DecisionEngine()
        result = engine.infer(features)   # features: (50,) float32
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        settings:  Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._session = None
        self._input_name = "features"

        search_dir: Path = model_dir or self._settings.checkpoints_dir
        candidates: list[Path] = [
            search_dir / fname for fname in self._settings.decision.onnx_candidates
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
                _log.info("Model loaded: %s", candidate.name)
                break
            except Exception as exc:
                _log.warning("Could not load %s: %s", candidate.name, exc)

        if self._session is None:
            _log.warning("No ONNX model found — using threshold fallback.")

        # Load the scaler trained alongside the model (required for normalised features)
        self._scaler = None
        scaler_path = search_dir / self._settings.decision.scaler_filename
        if scaler_path.exists():
            try:
                import joblib
                self._scaler = joblib.load(scaler_path)
                _log.info("Scaler loaded: %s", scaler_path.name)
            except Exception as exc:
                _log.warning("Could not load scaler: %s", exc)

    @property
    def has_model(self) -> bool:
        return self._session is not None

    def infer(self, features: np.ndarray) -> DecisionResult:
        """Run inference on a 50-dim feature vector."""
        raw_features = np.asarray(features, dtype=np.float32).ravel()

        # ── Threshold fallback uses RAW features ────────────────────────
        # _threshold_decision computes exp(-features[49]) which expects the
        # original symmetry_error magnitude.  Feeding standard-scaled data
        # (mean ≈ 0, var ≈ 1) would produce arbitrary garbage scores.
        if self._session is None:
            return _threshold_decision(raw_features, self._settings)

        # ── ONNX path uses SCALED features ──────────────────────────────
        # The scaler must ONLY be applied to features entering the ONNX
        # session — never to the threshold fallback.
        scaled = raw_features.copy()
        if self._scaler is not None:
            scaled = self._scaler.transform(scaled.reshape(1, -1))[0].astype(np.float32)

        x      = scaled.reshape(1, -1)
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