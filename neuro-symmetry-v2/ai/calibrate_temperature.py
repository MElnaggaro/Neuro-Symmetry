"""
Temperature scaling and ECE computation (backward-compat shim).

The canonical implementation lives in ai.v2.post_hoc.TemperatureScaling.
This module re-exports compute_ece so legacy imports
  from ai.calibrate_temperature import compute_ece
continue to work without modification.
"""

from __future__ import annotations

import numpy as np

from ai.v2.post_hoc import TemperatureScaling


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error — lower is better (target < 0.05)."""
    return TemperatureScaling.compute_ece(probs, labels, n_bins)


__all__ = ["compute_ece"]
