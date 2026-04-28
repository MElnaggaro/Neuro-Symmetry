"""
XAI Engine — per-feature explainability for symmetry predictions.

Contribution model
------------------
Each feature is assigned a raw contribution proportional to how much it
pushes toward a pathological prediction:

  • Bilateral distances and delta features: risk ↑ when feature is large.
      contribution = abs(feature_value) × pathological_confidence

  • Polarity-inverted features (EAR, texture_score): risk ↑ when value is LOW.
      contribution = max(0, 0.5 − feature_value) × pathological_confidence

Contributions are normalised relative to the top contributor, then bucketed
into HIGH (> 60%) / MEDIUM (> 30%) / LOW (≤ 30%) tiers.

Affected-side detection
-----------------------
Votes are cast using the only features that carry side-directional information
after the absolute-value normalisation in the feature extractor:

  ear_left < ear_right    → left eye more closed      → LEFT vote
  ear_left > ear_right    → right eye more closed     → RIGHT vote
  brow_height_left < brow_height_right → left brow lower → LEFT vote
  brow_height_left > brow_height_right → right brow lower → RIGHT vote

Tie or no significant asymmetry → BILATERAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_N_TOP = 8          # number of features surfaced in the API response
_VOTE_MIN = 0.01    # minimum |left − right| to cast a side vote

# Features where a LOW value (not a HIGH one) indicates higher risk
_INVERSE_PREFIX = ("texture_score", "ear_left", "ear_right")


@dataclass(frozen=True)
class XAIFeature:
    feature:      str
    contribution: float
    level:        str    # "HIGH" | "MEDIUM" | "LOW"


def explain_prediction(
    features:      np.ndarray,
    feature_names: list[str],
    probs:         list[float],
    class_id:      int,
) -> tuple[list[XAIFeature], str]:
    """
    Compute feature contributions and infer the affected facial side.

    Parameters
    ----------
    features      : (50,) float32 feature vector
    feature_names : list of 50 feature name strings
    probs         : [p_normal, p_mild, p_severe]
    class_id      : argmax class (0 / 1 / 2)

    Returns
    -------
    (ranked_features, affected_side)
      ranked_features : list of XAIFeature, sorted by contribution descending
      affected_side   : "LEFT" | "RIGHT" | "BILATERAL"
    """
    pathological_prob = float(probs[1] + probs[2])   # combined non-normal confidence

    raw: list[tuple[str, float]] = []
    for name, val in zip(feature_names, features):
        fval = float(val)
        if any(name.startswith(p) for p in _INVERSE_PREFIX):
            contrib = max(0.0, 0.5 - fval) * pathological_prob
        else:
            contrib = abs(fval) * pathological_prob
        raw.append((name, contrib))

    raw.sort(key=lambda t: t[1], reverse=True)
    top   = raw[:_N_TOP]
    max_c = top[0][1] if top and top[0][1] > 1e-9 else 1.0

    result: list[XAIFeature] = []
    for name, c in top:
        rel = c / max_c
        level = "HIGH" if rel > 0.60 else "MEDIUM" if rel > 0.30 else "LOW"
        result.append(XAIFeature(feature=name, contribution=round(c, 4), level=level))

    affected_side = _detect_affected_side(features, feature_names)
    return result, affected_side


def _detect_affected_side(
    features:      np.ndarray,
    feature_names: list[str],
) -> str:
    """Vote LEFT / RIGHT / BILATERAL based on directional bilateral features."""
    name_idx = {n: i for i, n in enumerate(feature_names)}

    votes_left  = 0
    votes_right = 0

    def _vote(left_key: str, right_key: str, low_is_worse: bool = True) -> None:
        """
        Cast a side vote.

        low_is_worse=True  (default): lower value on a side means that side is worse
                                      (e.g. ear_left < ear_right → LEFT drooped).
        low_is_worse=False: higher value on a side means that side is worse.
        """
        nonlocal votes_left, votes_right
        if left_key not in name_idx or right_key not in name_idx:
            return
        left_val  = float(features[name_idx[left_key]])
        right_val = float(features[name_idx[right_key]])
        if abs(left_val - right_val) < _VOTE_MIN:
            return
        worse_is_left = (left_val < right_val) if low_is_worse else (left_val > right_val)
        if worse_is_left:
            votes_left  += 1
        else:
            votes_right += 1

    # EAR: lower = more closed = worse
    _vote("ear_left", "ear_right", low_is_worse=True)

    # Brow height: lower = more drooped = worse
    _vote("brow_height_left", "brow_height_right", low_is_worse=True)

    if votes_left == votes_right:
        return "BILATERAL"
    return "LEFT" if votes_left > votes_right else "RIGHT"
