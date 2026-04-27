"""
Personalized baseline calibration — per-user z-score anomaly detection.

Workflow
--------
1. Run a ~2-minute calibration session (collect feature vectors from a neutral face).
2. Call BaselineCalibrator.compute() → UserBaseline.
3. At inference time: anomaly_score(features, baseline) → z-score.
   High z-score (> 3.0) means the current frame deviates from the user's personal normal.

Persistence
-----------
UserBaseline.save(dir, user_id) / UserBaseline.load(dir, user_id)
Saved as a compressed .npz next to model checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

N_FEATURES = 50
_MIN_SAMPLES = 30   # minimum frames needed for a valid baseline


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class UserBaseline:
    user_id: str
    mean:    np.ndarray   # (50,) float32
    std:     np.ndarray   # (50,) float32 — clipped to ≥ 1e-4 to avoid div/0

    def anomaly_score(self, features: np.ndarray) -> float:
        """
        Per-feature z-score, collapsed to a single scalar via L2 norm.
        Score ≈ 0 for a typical frame; > 3 signals a meaningful deviation.
        """
        z = (features - self.mean) / self.std                 # (50,)
        return float(np.linalg.norm(z) / np.sqrt(N_FEATURES)) # normalised by dim

    def save(self, directory: Path, user_id: Optional[str] = None) -> Path:
        uid = user_id or self.user_id
        path = directory / f"baseline_{uid}.npz"
        np.savez_compressed(path, mean=self.mean, std=self.std, user_id=np.array(uid))
        return path

    @classmethod
    def load(cls, directory: Path, user_id: str) -> "UserBaseline":
        path = directory / f"baseline_{user_id}.npz"
        data = np.load(path)
        return cls(
            user_id=str(data["user_id"]),
            mean=data["mean"].astype(np.float32),
            std=data["std"].astype(np.float32),
        )


# ── Calibrator ────────────────────────────────────────────────────────────────

class BaselineCalibrator:
    """
    Accumulates feature vectors during a calibration session and
    computes a personal normal distribution for the user.

    Usage
    -----
        cal = BaselineCalibrator("user_42")
        for frame in calibration_frames:
            features = extract_features(result, frame)
            cal.record(features)
        baseline = cal.compute()
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._frames: list[np.ndarray] = []

    def record(self, features: np.ndarray) -> None:
        """Add one 50-dim feature vector from a calibration frame."""
        if features.shape != (N_FEATURES,):
            raise ValueError(f"Expected shape ({N_FEATURES},), got {features.shape}")
        self._frames.append(features.astype(np.float32))

    @property
    def n_samples(self) -> int:
        return len(self._frames)

    def compute(self) -> UserBaseline:
        """
        Fit mean and std from accumulated frames.
        Raises RuntimeError if fewer than _MIN_SAMPLES frames were recorded.
        """
        if self.n_samples < _MIN_SAMPLES:
            raise RuntimeError(
                f"Need at least {_MIN_SAMPLES} calibration frames, "
                f"got {self.n_samples}."
            )
        mat = np.stack(self._frames, axis=0)          # (N, 50)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0).clip(min=1e-4)         # guard against constant features
        return UserBaseline(user_id=self.user_id, mean=mean, std=std)

    def reset(self) -> None:
        self._frames.clear()
