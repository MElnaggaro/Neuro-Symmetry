"""
Decision Engine — ONNX Runtime inference with threshold fallback.

All tuneable values (model search paths, severity thresholds, scaler filename)
are injected from ``backend.core.config.Settings``.

Label convention
----------------
  0 = Normal
  1 = Mild asymmetry
  2 = Severe asymmetry

Inference path
--------------
The ONNX path mirrors ``backend.api.predict.V2OnnxPredictor``:
  1. Standardise the 50-D feature vector with the per-feature mean/std loaded
     from ``metrics_v2.json`` (no sklearn dep — pure numpy).
  2. Run the ONNX graph and softmax the logits.
  3. Pick the class via calibrated-threshold priority (Severe > Mild > Normal),
     falling back to argmax only if every threshold fails.

Why this matters for live detection
-----------------------------------
The V2 model was trained with heavy class imbalance and intentionally
calibrated for stroke-detection sensitivity. Its thresholds are extreme
(e.g. ~0.94 for Normal, ~0.07 for Mild, ~0.05 for Severe). Plain ``argmax``
on those probabilities silently collapses borderline-abnormal cases like
``[0.45, 0.35, 0.20]`` into Normal — which is exactly the failure mode the
calibration was designed to prevent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core import Settings, get_settings

_log = logging.getLogger("neuro_symmetry.backend.decision_engine")

CLASS_LABELS: tuple[str, str, str] = ("Normal", "Mild", "Severe")
# Severity-priority order: check Severe first, then Mild, then Normal.
# Matches V2OnnxPredictor in backend.api.predict.
_THRESHOLD_PRIORITY: tuple[int, int, int] = (2, 1, 0)


@dataclass(frozen=True)
class DecisionResult:
    class_id:      int
    class_label:   str
    confidence:    float
    probabilities: list[float]   # [p_normal, p_mild, p_severe]
    source:        str           # "onnx" | "threshold"


def _softmax(logits: np.ndarray) -> np.ndarray:
    # Use float64 for the shift/exp to avoid underflow on confident logits.
    logits64 = np.asarray(logits, dtype=np.float64).reshape(-1)
    e = np.exp(logits64 - logits64.max())
    return (e / e.sum()).astype(np.float32)


def _load_metrics(search_dir: Path) -> dict:
    metrics_path = search_dir / "metrics_v2.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("Could not parse %s: %s", metrics_path, exc)
        return {}


def _load_scaler_arrays(search_dir: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load (scaler_mean, scaler_std) as numpy arrays.

    Preference order:
      1. metrics_v2.json scaler_mean / scaler_std fields  (no extra deps)
      2. feature_scaler.pkl via joblib                     (legacy layout)
    """
    metrics = _load_metrics(search_dir)
    mean = metrics.get("scaler_mean")
    std  = metrics.get("scaler_std")

    if mean is not None and std is not None:
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr  = np.asarray(std,  dtype=np.float32)
        std_arr  = np.where(std_arr == 0.0, 1.0, std_arr).astype(np.float32)
        _log.info("Scaler arrays loaded from metrics_v2.json (n=%d).", mean_arr.size)
        return mean_arr, std_arr

    # Legacy: try the pickled sklearn scaler. Will quietly fail without sklearn,
    # which is fine — we'll log a warning and return (None, None).
    pkl_path = search_dir / "feature_scaler.pkl"
    if pkl_path.exists():
        try:
            import joblib
            scaler = joblib.load(pkl_path)
            mean_arr = np.asarray(scaler.mean_,  dtype=np.float32)
            std_arr  = np.asarray(scaler.scale_, dtype=np.float32)
            std_arr  = np.where(std_arr == 0.0, 1.0, std_arr).astype(np.float32)
            _log.info("Scaler arrays loaded from %s (legacy pickle).", pkl_path.name)
            return mean_arr, std_arr
        except Exception as exc:
            _log.warning("Could not load %s: %s", pkl_path.name, exc)

    return None, None


def _load_thresholds(search_dir: Path) -> Optional[np.ndarray]:
    """Load the per-class calibrated probability thresholds, if available."""
    metrics = _load_metrics(search_dir)
    thresholds = metrics.get("thresholds")
    if thresholds is None:
        return None
    arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3 or np.any((arr < 0.0) | (arr > 1.0)):
        _log.warning("Ignoring malformed thresholds=%s in metrics_v2.json", thresholds)
        return None
    _log.info("Calibrated thresholds loaded: %s", [round(float(t), 4) for t in arr])
    return arr


def _threshold_predict(probs: np.ndarray, thresholds: np.ndarray) -> int:
    """Pick the class id by severity-priority threshold check."""
    for class_id in _THRESHOLD_PRIORITY:
        if probs[class_id] >= thresholds[class_id]:
            return int(class_id)
    return int(np.argmax(probs))


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
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None
        self._thresholds:  Optional[np.ndarray] = None

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
            return

        # Load scaler arrays + calibrated thresholds from metrics_v2.json.
        # Pure-numpy: no sklearn / joblib dependency. Falls back to feature_scaler.pkl
        # only if metrics is missing the arrays (legacy layout).
        self._scaler_mean, self._scaler_std = _load_scaler_arrays(search_dir)
        self._thresholds = _load_thresholds(search_dir)

        if self._scaler_mean is None or self._scaler_std is None:
            _log.warning(
                "ONNX model loaded but no usable scaler found in %s — "
                "the model will see RAW features and produce meaningless probabilities. "
                "Add scaler_mean/scaler_std to metrics_v2.json.",
                search_dir,
            )
        if self._thresholds is None:
            _log.warning(
                "ONNX model loaded but calibrated thresholds are missing — "
                "falling back to argmax. Borderline abnormal cases will be "
                "mislabeled as Normal.",
            )

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

        # ── ONNX path: scale → infer → calibrated threshold decision ────
        if self._scaler_mean is not None and self._scaler_std is not None:
            scaled = ((raw_features - self._scaler_mean) / self._scaler_std).astype(np.float32)
        else:
            scaled = raw_features

        x      = scaled.reshape(1, -1)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        probs  = _softmax(logits)

        if self._thresholds is not None:
            cid = _threshold_predict(probs, self._thresholds)
        else:
            cid = int(probs.argmax())

        return DecisionResult(
            class_id=cid,
            class_label=CLASS_LABELS[cid],
            confidence=float(probs[cid]),
            probabilities=probs.tolist(),
            source="onnx",
        )