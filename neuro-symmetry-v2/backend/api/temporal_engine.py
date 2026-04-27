"""
Temporal Engine — time-series tracking of facial asymmetry.

Maintains a rolling history of symmetry scores and class predictions,
computing:
  • EMA (exponential moving average) score — noise-smoothed baseline
  • Slope — linear trend over the last SLOPE_WINDOW frames (per-frame Δ)
  • Onset frame — when the first alert was detected
  • Duration — frames elapsed since onset

EMA update rule:  ema_t = α·score_t + (1−α)·ema_{t−1}
  α = 0.10 → slow-reacting, good for separating noise from trend

Slope sign convention:
  Positive  → score rising (asymmetry improving)
  Negative  → score falling (asymmetry worsening)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

EMA_ALPHA    = 0.10    # smoothing factor
SLOPE_WINDOW = 30      # frames over which slope is estimated
MIN_SLOPE_N  = 5       # minimum frames before reporting a slope


@dataclass(frozen=True)
class TemporalState:
    ema_score:      float          # exponential moving average of symmetry score
    slope:          float          # per-frame trend in score (+ = improving)
    onset_frame:    Optional[int]  # frame index when first alert appeared (None if normal)
    duration_frames: int           # frames elapsed since onset (0 if normal)
    frame_count:    int            # total frames processed by this engine


class TemporalEngine:
    """
    Per-session temporal tracker.

    Call ``update(score, class_id)`` once per usable frame.
    Returns a ``TemporalState`` every call.
    """

    def __init__(
        self,
        ema_alpha:    float = EMA_ALPHA,
        slope_window: int   = SLOPE_WINDOW,
    ) -> None:
        self._alpha       = ema_alpha
        self._slope_buf:  deque[float] = deque(maxlen=slope_window)
        self._ema:        Optional[float] = None
        self._onset_frame: Optional[int]  = None
        self._frame_count = 0

    def update(self, score: float, class_id: int) -> TemporalState:
        """
        Ingest one frame.  Returns updated TemporalState.

        Parameters
        ----------
        score    : symmetry score ∈ [0, 1]
        class_id : 0 = Normal, 1 = Mild, 2 = Severe
        """
        self._frame_count += 1
        self._slope_buf.append(score)

        # EMA — warm-start on first frame
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self._alpha * score + (1.0 - self._alpha) * self._ema

        # Onset tracking: latch on first alert; release if back to Normal
        if class_id != 0:
            if self._onset_frame is None:
                self._onset_frame = self._frame_count
        else:
            self._onset_frame = None

        duration = (
            self._frame_count - self._onset_frame
            if self._onset_frame is not None else 0
        )

        return TemporalState(
            ema_score=round(float(self._ema), 4),
            slope=self._compute_slope(),
            onset_frame=self._onset_frame,
            duration_frames=duration,
            frame_count=self._frame_count,
        )

    def _compute_slope(self) -> float:
        n = len(self._slope_buf)
        if n < MIN_SLOPE_N:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(self._slope_buf, dtype=np.float64)
        # First-degree polynomial fit → slope coefficient
        return float(np.polyfit(x, y, 1)[0])

    def reset(self) -> None:
        self._slope_buf.clear()
        self._ema = None
        self._onset_frame = None
        self._frame_count = 0
