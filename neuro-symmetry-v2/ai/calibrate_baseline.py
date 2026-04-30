"""
Personalised baseline calibration.

Records N frames from a subject at rest, computes per-feature mean/std, and
provides an anomaly_score(frame) that measures how many standard deviations
the current frame deviates from that personal baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MIN_SAMPLES: int = 30   # minimum frames required to compute a valid baseline


@dataclass
class UserBaseline:
    user_id: str
    mean:    np.ndarray   # (N_FEATURES,) float32
    std:     np.ndarray   # (N_FEATURES,) float32

    def anomaly_score(self, frame: np.ndarray) -> float:
        """
        Mean z-score across features.  Higher → more deviant from personal baseline.
        std=0 features are skipped (constant feature → no anomaly signal).
        """
        diff = np.abs(frame.astype(np.float32) - self.mean)
        safe_std = np.where(self.std > 0, self.std, np.inf)
        z = diff / safe_std
        valid = np.isfinite(z)
        return float(z[valid].mean()) if valid.any() else 0.0

    def save(self, directory: Path, user_id: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / f"baseline_{user_id}.npz",
            mean=self.mean,
            std=self.std,
            user_id=np.array(user_id),
        )

    @classmethod
    def load(cls, directory: Path, user_id: str) -> "UserBaseline":
        path = Path(directory) / f"baseline_{user_id}.npz"
        data = np.load(path, allow_pickle=False)
        return cls(
            user_id=str(data["user_id"]),
            mean=data["mean"].astype(np.float32),
            std=data["std"].astype(np.float32),
        )


class BaselineCalibrator:
    """
    Accumulates frames for a subject and computes a UserBaseline.

    Usage::

        cal = BaselineCalibrator("patient-001")
        for frame_features in calibration_frames:
            cal.record(frame_features)
        baseline = cal.compute()
    """

    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._frames: list[np.ndarray] = []

    @property
    def n_samples(self) -> int:
        return len(self._frames)

    def record(self, frame: np.ndarray) -> None:
        self._frames.append(np.asarray(frame, dtype=np.float32))

    def compute(self) -> UserBaseline:
        if len(self._frames) < _MIN_SAMPLES:
            raise RuntimeError(
                f"Not enough calibration frames: need {_MIN_SAMPLES}, "
                f"got {len(self._frames)}"
            )
        stack = np.stack(self._frames, axis=0)   # (N, D)
        return UserBaseline(
            user_id=self._user_id,
            mean=stack.mean(axis=0).astype(np.float32),
            std=stack.std(axis=0).astype(np.float32),
        )

    def reset(self) -> None:
        self._frames.clear()


__all__ = ["BaselineCalibrator", "UserBaseline", "_MIN_SAMPLES"]
