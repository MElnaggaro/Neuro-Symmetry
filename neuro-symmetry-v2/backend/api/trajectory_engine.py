"""
Trajectory Engine — classifies symmetry score trends over time.

States
------
  STABLE          → score is flat or gently improving
  LINEAR_DECLINE  → steady negative slope (R² > r2_thresh, slope < slope_thresh)
  SUDDEN_DROP     → baseline − min(recent drop_window) > drop_threshold
  OSCILLATING    → high variance (std > osc_var_thresh), no clear slope
  COLLAPSE        → rolling mean < collapse_mean

All thresholds are sourced from ``backend.core.config.TrajectorySettings``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import linregress

from backend.core import TrajectorySettings, get_settings


class Trajectory(str, Enum):
    STABLE         = "STABLE"
    LINEAR_DECLINE = "LINEAR_DECLINE"
    SUDDEN_DROP    = "SUDDEN_DROP"
    OSCILLATING    = "OSCILLATING"
    COLLAPSE       = "COLLAPSE"


class TrajectoryEngine:
    """
    Per-session trajectory classifier.

    Call ``update(score)`` once per usable frame; returns the current
    ``Trajectory`` state.
    """

    def __init__(
        self,
        window:    Optional[int] = None,
        settings:  Optional[TrajectorySettings] = None,
    ) -> None:
        cfg = settings or get_settings().trajectory
        self._cfg     = cfg
        self._scores: deque[float] = deque(maxlen=window if window is not None else cfg.window_size)

    def update(self, score: float) -> Trajectory:
        self._scores.append(score)
        return self.classify(list(self._scores), settings=self._cfg)

    @staticmethod
    def classify(
        score_window: list[float],
        *,
        settings: Optional[TrajectorySettings] = None,
    ) -> Trajectory:
        """Classify a score window — pure function, exposed for unit tests."""
        cfg = settings or get_settings().trajectory
        n = len(score_window)
        if n < cfg.min_frames:
            return Trajectory.STABLE

        scores = np.array(score_window, dtype=np.float64)

        # COLLAPSE — sustained very low mean score
        if float(scores.mean()) < cfg.collapse_mean:
            return Trajectory.COLLAPSE

        # SUDDEN_DROP — large drop in recent window vs. earlier baseline
        if TrajectoryEngine.detect_sudden_drop(scores, cfg.drop_threshold, cfg.drop_window):
            return Trajectory.SUDDEN_DROP

        # Linear regression over full window
        x = np.arange(n, dtype=np.float64)
        slope, _, r_value, _, _ = linregress(x, scores)
        slope = float(slope)
        r_sq  = float(r_value ** 2)

        # OSCILLATING — high recent variance with no directional trend
        recent_window = min(n, 15)
        recent_std = float(scores[-recent_window:].std())
        if recent_std > cfg.osc_var_thresh and abs(slope) < cfg.osc_slope_abs:
            return Trajectory.OSCILLATING

        # LINEAR_DECLINE — sustained downward trend with good linear fit
        if slope < cfg.slope_thresh and r_sq > cfg.r2_thresh:
            return Trajectory.LINEAR_DECLINE

        return Trajectory.STABLE

    @staticmethod
    def detect_sudden_drop(
        scores:    np.ndarray,
        threshold: float,
        window:    int,
    ) -> bool:
        """True if min(last *window*) lies more than *threshold* below earlier mean."""
        n = len(scores)
        if n < window:
            return False
        earlier   = scores[:-window]
        baseline  = float(earlier.mean()) if len(earlier) > 0 else float(scores.mean())
        recent_min = float(scores[-window:].min())
        return (baseline - recent_min) > threshold

    def reset(self) -> None:
        self._scores.clear()