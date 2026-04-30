"""
Change Detection — rolling-baseline anomaly detector.

Maintains a deque of symmetry scores. After a warmup period, fires a
ChangeEvent when the current score is more than ``z_threshold`` standard
deviations below the rolling mean.

All tuning constants (window, warmup, z_threshold) are sourced from
``backend.core.config.ChangeDetectionSettings``.  Constructor kwargs override
the global settings (used by tests).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.core import ChangeDetectionSettings, get_settings


@dataclass(frozen=True)
class ChangeEvent:
    score:    float
    baseline: float
    z_score:  float


class ChangeDetector:
    """
    Per-session rolling z-score change detector.

    Call ``update(score)`` once per frame.
    Returns a ``ChangeEvent`` when an anomalous drop is detected, else None.
    """

    def __init__(
        self,
        window:      Optional[int]   = None,
        warmup:      Optional[int]   = None,
        z_threshold: Optional[float] = None,
        settings:    Optional[ChangeDetectionSettings] = None,
    ) -> None:
        cfg = settings or get_settings().change
        self._window      = window      if window      is not None else cfg.window
        self._warmup      = warmup      if warmup      is not None else cfg.warmup
        self._z_threshold = z_threshold if z_threshold is not None else cfg.z_threshold
        self._scores: deque[float] = deque(maxlen=self._window)

    @property
    def ready(self) -> bool:
        return len(self._scores) >= self._warmup

    @property
    def n_frames(self) -> int:
        return len(self._scores)

    def update(self, score: float) -> Optional[ChangeEvent]:
        """Add a symmetry score. Returns ChangeEvent if anomaly detected."""
        # ── Prevent Z-score data leakage ────────────────────────────────
        # Compute baseline statistics from the EXISTING history, EXCLUDING
        # the current score.  Appending before computing would drag the
        # mean toward the anomalous point and inflate the std, effectively
        # suppressing the Z-score and masking genuine anomalies.

        if not self.ready:
            # Still in warmup — just accumulate, no detection yet
            self._scores.append(score)
            return None

        # Baseline from history BEFORE the new observation
        arr  = np.array(self._scores, dtype=np.float64)
        mean = float(arr.mean())
        std  = float(arr.std())

        # NOW append the new score to the rolling window
        self._scores.append(score)

        if std < 1e-6:
            return None

        z = (score - mean) / std
        if z < -self._z_threshold:
            return ChangeEvent(score=score, baseline=mean, z_score=z)
        return None

    def reset(self) -> None:
        self._scores.clear()