"""
Trajectory Engine — classifies symmetry score trends over time.

Uses scipy.stats.linregress for linear trend detection and delta-based
heuristics for collapse and sudden-drop patterns.

States
------
  STABLE          → score is flat or gently improving
  LINEAR_DECLINE  → steady negative slope (R² > 0.5, slope < −0.003 / frame)
  SUDDEN_DROP     → baseline − min(recent 30) > 0.25
  OSCILLATING     → high variance (std > 0.05), no clear slope
  COLLAPSE        → rolling mean < 0.30 (sustained severe asymmetry)
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
from scipy.stats import linregress


# ── Tuning constants ──────────────────────────────────────────────────────────

WINDOW_SIZE     = 60      # rolling history length (frames)
MIN_FRAMES      = 10      # minimum frames before non-STABLE states are reported
DROP_THRESHOLD  = 0.25    # minimum drop magnitude to flag SUDDEN_DROP
DROP_WINDOW     = 30      # frames over which the drop is measured
COLLAPSE_MEAN   = 0.30    # rolling mean below this threshold → COLLAPSE
SLOPE_THRESH    = -0.003  # per-frame slope below this → candidate LINEAR_DECLINE
R2_THRESH       = 0.50    # minimum R² for linear decline to be confirmed
OSC_VAR_THRESH  = 0.05    # recent-window std above this → OSCILLATING candidate
OSC_SLOPE_ABS   = 0.001   # |slope| must be below this for OSCILLATING


class Trajectory(str, Enum):
    STABLE         = "STABLE"
    LINEAR_DECLINE = "LINEAR_DECLINE"
    SUDDEN_DROP    = "SUDDEN_DROP"
    OSCILLATING    = "OSCILLATING"
    COLLAPSE       = "COLLAPSE"


class TrajectoryEngine:
    """
    Per-session trajectory classifier.

    Call ``update(score)`` once per usable frame.
    Returns the current ``Trajectory`` state.
    """

    def __init__(self, window: int = WINDOW_SIZE) -> None:
        self._scores: deque[float] = deque(maxlen=window)

    def update(self, score: float) -> Trajectory:
        """Ingest one symmetry score and return the current trajectory."""
        self._scores.append(score)
        return self.classify(list(self._scores))

    @staticmethod
    def classify(
        score_window:   list[float],
        min_frames:     int   = MIN_FRAMES,
        drop_threshold: float = DROP_THRESHOLD,
        drop_window:    int   = DROP_WINDOW,
        collapse_mean:  float = COLLAPSE_MEAN,
        slope_thresh:   float = SLOPE_THRESH,
        r2_thresh:      float = R2_THRESH,
        osc_var_thresh: float = OSC_VAR_THRESH,
        osc_slope_abs:  float = OSC_SLOPE_ABS,
    ) -> Trajectory:
        """
        Classify a score window.  Exposed as a static method for unit testing
        without instantiating a full engine.
        """
        n = len(score_window)
        if n < min_frames:
            return Trajectory.STABLE

        scores = np.array(score_window, dtype=np.float64)

        # COLLAPSE — sustained very low mean score
        if float(scores.mean()) < collapse_mean:
            return Trajectory.COLLAPSE

        # SUDDEN_DROP — large drop in recent window vs. earlier baseline
        if TrajectoryEngine.detect_sudden_drop(scores, drop_threshold, drop_window):
            return Trajectory.SUDDEN_DROP

        # Linear regression over full window
        x = np.arange(n, dtype=np.float64)
        slope, _, r_value, _, _ = linregress(x, scores)
        slope   = float(slope)
        r_sq    = float(r_value ** 2)

        # OSCILLATING — high recent variance with no directional trend
        recent_std = float(scores[-15:].std()) if n >= 15 else float(scores.std())
        if recent_std > osc_var_thresh and abs(slope) < osc_slope_abs:
            return Trajectory.OSCILLATING

        # LINEAR_DECLINE — sustained downward trend with good linear fit
        if slope < slope_thresh and r_sq > r2_thresh:
            return Trajectory.LINEAR_DECLINE

        return Trajectory.STABLE

    @staticmethod
    def detect_sudden_drop(
        scores:    np.ndarray,
        threshold: float = DROP_THRESHOLD,
        window:    int   = DROP_WINDOW,
    ) -> bool:
        """
        Returns True if the minimum score in the last *window* frames lies more
        than *threshold* below the mean of all earlier frames.

        Parameters
        ----------
        scores    : 1-D array of symmetry scores (chronological order)
        threshold : minimum drop magnitude (default 0.25)
        window    : number of recent frames to inspect (default 30)
        """
        n = len(scores)
        if n < window:
            return False
        earlier   = scores[:-window]
        baseline  = float(earlier.mean()) if len(earlier) > 0 else float(scores.mean())
        recent_min = float(scores[-window:].min())
        return (baseline - recent_min) > threshold

    def reset(self) -> None:
        self._scores.clear()
