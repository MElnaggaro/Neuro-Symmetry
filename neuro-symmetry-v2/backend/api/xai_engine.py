"""
Explainable AI helpers for the 50-D V2 facial-symmetry feature vector.

This module does not import the training stack or any deleted V1 files. It is
designed to sit beside the ONNX inference path and explain the already-extracted
features that were sent to the V2 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_N_TOP = 5
_SIDE_EPS = 0.01


@dataclass(frozen=True)
class XAIFeature:
    feature: str
    contribution: float
    level: str


def _pathological_probability(probabilities: Sequence[float], class_id: int) -> float:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probs.size >= 3:
        return float(max(probs[1], probs[2]))
    if 0 <= class_id < probs.size:
        return float(probs[class_id])
    return 0.0


def _level(contribution: float, top_contribution: float) -> str:
    if top_contribution <= 0.0:
        return "LOW"

    ratio = contribution / top_contribution
    if ratio >= 0.70:
        return "HIGH"
    if ratio >= 0.35:
        return "MEDIUM"
    return "LOW"


def _feature_name(feature_names: Sequence[str], index: int) -> str:
    if 0 <= index < len(feature_names):
        return feature_names[index]
    return f"feature_{index}"


def _rank_contributions(
    features: np.ndarray,
    feature_names: Sequence[str],
    pathological_prob: float,
) -> list[XAIFeature]:
    """
    Rank clinically interpretable asymmetry features.

    The ONNX graph does not expose gradients, so this uses deterministic
    domain-weighted attribution over the 50-D feature vector. Contributions are
    gated by pathological probability so Normal predictions produce near-zero
    explanation scores.
    """
    weights = np.ones(50, dtype=np.float32) * 0.35
    weights[0:40] = 0.45
    weights[40] = 0.50  # ear_left
    weights[41] = 0.50  # ear_right
    weights[42] = 1.20  # ear_delta
    weights[43] = 0.55  # brow_height_left
    weights[44] = 0.55  # brow_height_right
    weights[45] = 1.10  # brow_height_delta
    weights[46] = 1.00  # mouth_y_delta
    weights[47] = 0.85  # mouth_x_offset
    weights[48] = 0.80  # low texture score
    weights[49] = 1.35  # fused symmetry_error

    signal = np.abs(features).astype(np.float32)
    signal[48] = max(0.0, 1.0 - float(features[48]))
    signal[49] = max(0.0, float(features[49]))

    raw = signal * weights * max(0.0, min(pathological_prob, 1.0))
    order = np.argsort(raw)[::-1]
    top_score = float(raw[order[0]]) if order.size else 0.0

    ranked: list[XAIFeature] = []
    for index in order:
        contribution = float(raw[index])
        if contribution <= 0.0 and ranked:
            break
        ranked.append(
            XAIFeature(
                feature=_feature_name(feature_names, int(index)),
                contribution=round(contribution, 4),
                level=_level(contribution, top_score),
            )
        )
        if len(ranked) >= _N_TOP:
            break

    if not ranked:
        ranked.append(
            XAIFeature(
                feature=_feature_name(feature_names, 49),
                contribution=0.0,
                level="LOW",
            )
        )

    return ranked


def _vote_affected_side(features: np.ndarray) -> str:
    """
    Infer affected side from paired facial signals.

    Lower EAR means a more closed eye; lower brow height means a dropped brow.
    Mouth and aggregate bilateral features only vote when they clearly lean to
    one side. Ties intentionally resolve to BILATERAL.
    """
    left_votes = 0
    right_votes = 0

    ear_left = float(features[40])
    ear_right = float(features[41])
    if abs(ear_left - ear_right) > _SIDE_EPS:
        if ear_left < ear_right:
            left_votes += 1
        else:
            right_votes += 1

    brow_left = float(features[43])
    brow_right = float(features[44])
    if abs(brow_left - brow_right) > _SIDE_EPS:
        if brow_left < brow_right:
            left_votes += 1
        else:
            right_votes += 1

    left_region = float(np.mean(features[0:20]))
    right_region = float(np.mean(features[20:40]))
    if abs(left_region - right_region) > _SIDE_EPS:
        if left_region > right_region:
            left_votes += 1
        else:
            right_votes += 1

    if left_votes > right_votes:
        return "LEFT"
    if right_votes > left_votes:
        return "RIGHT"
    return "BILATERAL"


def explain_prediction(
    features: Iterable[float],
    feature_names: Sequence[str],
    probabilities: Sequence[float],
    class_id: int,
) -> tuple[list[XAIFeature], str]:
    """
    Return ranked feature contributions and likely affected side.

    Parameters match the existing backend pipeline:
    - ``features``: raw 50-D feature vector from ``feature_extractor``
    - ``feature_names``: names aligned to the 50-D vector
    - ``probabilities``: model probabilities ordered [Normal, Mild, Severe]
    - ``class_id``: selected model class
    """
    vector = np.asarray(list(features), dtype=np.float32).reshape(-1)
    if vector.shape[0] != 50:
        raise ValueError(f"explain_prediction expects 50 features, got {vector.shape[0]}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("explain_prediction received non-finite features")

    pathological_prob = _pathological_probability(probabilities, class_id)
    return (
        _rank_contributions(vector, feature_names, pathological_prob),
        _vote_affected_side(vector),
    )
