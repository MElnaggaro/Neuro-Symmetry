"""
Change Detection — rolling-baseline anomaly detector.

Maintains a deque of symmetry scores. After a warmup period, fires a
ChangeEvent when the current score is more than Z_THRESHOLD standard
deviations below the rolling mean.

Algorithm
---------
  • Rolling window of size WINDOW (default 90 frames ≈ 3 s at 30 fps)
  • Detection starts after WARMUP frames
  • z = (score − µ) / σ  where µ, σ are computed over the window
  • ChangeEvent fired when z < −Z_THRESHOLD
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


Z_THRESHOLD = 2.5
WINDOW      = 90
WARMUP      = 30


@dataclass(frozen=True)
class ChangeEvent:
    score:    float   # score that triggered the event
    baseline: float   # rolling mean at time of event
    z_score:  float   # σ below mean (negative)


class ChangeDetector:
    """
    Per-session rolling z-score change detector.

    Call ``update(score)`` once per frame.
    Returns a ``ChangeEvent`` when an anomalous drop is detected, else None.
    """

    def __init__(
        self,
        window:      int   = WINDOW,
        warmup:      int   = WARMUP,
        z_threshold: float = Z_THRESHOLD,
    ) -> None:
        self._window      = window
        self._warmup      = warmup
        self._z_threshold = z_threshold
        self._scores: deque[float] = deque(maxlen=window)

    @property
    def ready(self) -> bool:
        """True once enough frames have been collected to detect changes."""
        return len(self._scores) >= self._warmup

    @property
    def n_frames(self) -> int:
        return len(self._scores)

    def update(self, score: float) -> Optional[ChangeEvent]:
        """Add a symmetry score. Returns ChangeEvent if anomaly detected."""
        self._scores.append(score)

        if not self.ready:
            return None

        arr  = np.array(self._scores, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(arr.std())

        if std < 1e-6:
            return None

        z = (score - mean) / std
        if z < -self._z_threshold:
            return ChangeEvent(score=score, baseline=mean, z_score=z)
        return None

    def reset(self) -> None:
        self._scores.clear()
