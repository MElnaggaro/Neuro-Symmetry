"""
Temporal Engine — time-series tracking of facial asymmetry.

EMA (exponential moving average) score, slope, onset frame, duration.
Tuneable values come from ``backend.core.config.TemporalSettings``.

EMA update rule:  ema_t = α·score_t + (1−α)·ema_{t−1}
Slope sign convention:
  Positive  → score rising (asymmetry improving)
  Negative  → score falling (asymmetry worsening)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.core import TemporalSettings, get_settings


@dataclass(frozen=True)
class TemporalState:
    ema_score:       float
    slope:           float
    onset_frame:     Optional[int]
    duration_frames: int
    frame_count:     int


class TemporalEngine:
    """Per-session temporal tracker — call ``update(score, class_id)`` per frame."""

    def __init__(
        self,
        ema_alpha:    Optional[float] = None,
        slope_window: Optional[int]   = None,
        min_slope_n:  Optional[int]   = None,
        settings:     Optional[TemporalSettings] = None,
    ) -> None:
        cfg = settings or get_settings().temporal
        self._alpha       = ema_alpha    if ema_alpha    is not None else cfg.ema_alpha
        self._min_slope_n = min_slope_n  if min_slope_n  is not None else cfg.min_slope_n
        self._slope_buf:  deque[float] = deque(
            maxlen=slope_window if slope_window is not None else cfg.slope_window,
        )
        self._ema:         Optional[float] = None
        self._onset_frame: Optional[int]   = None
        self._frame_count                 = 0
        # Temporal hysteresis: require this many consecutive Normal frames
        # before resetting the onset tracker.  Prevents a single false-negative
        # frame (camera jitter, transient detection miss) from destroying the
        # continuous duration measurement.
        self._recovery_patience           = 15
        self._normal_streak               = 0

    def update(self, score: float, class_id: int) -> TemporalState:
        self._frame_count += 1
        self._slope_buf.append(score)

        if self._ema is None:
            self._ema = score
        else:
            self._ema = self._alpha * score + (1.0 - self._alpha) * self._ema

        # ── Onset tracking with temporal hysteresis ─────────────────────
        if class_id != 0:
            # Pathological frame → reset the Normal-streak counter and
            # start (or continue) tracking the onset.
            self._normal_streak = 0
            if self._onset_frame is None:
                self._onset_frame = self._frame_count
        else:
            # Normal frame → increment the recovery counter.  Only reset
            # the onset after a sustained run of Normal frames exceeds
            # the patience threshold.  This prevents a single jittery
            # false-negative from wiping the duration tracking.
            self._normal_streak += 1
            if self._normal_streak >= self._recovery_patience:
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
        if n < self._min_slope_n:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(self._slope_buf, dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])

    def reset(self) -> None:
        self._slope_buf.clear()
        self._ema = None
        self._onset_frame = None
        self._frame_count = 0
        self._normal_streak = 0
