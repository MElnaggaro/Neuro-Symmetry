"""
Synthetic training data generator for the 50-D symmetry feature vector.

Feature layout (matches feature_extractor.py):
  [0:40]  bilateral distances   ≥ 0  (mirror-pair euclidean distances)
  [40]    EAR_left              ≥ 0
  [41]    EAR_right             ≥ 0
  [42]    EAR_delta             ≥ 0
  [43]    brow_height_left      (signed)
  [44]    brow_height_right     (signed)
  [45]    brow_height_delta     ≥ 0
  [46]    mouth_y_delta         ≥ 0
  [47]    mouth_x_offset        ≥ 0
  [48]    texture_score         ∈ [0, 1]
  [49]    symmetry_error        ≥ 0  (Normal < Mild < Severe)
"""

from __future__ import annotations

import numpy as np


# ── Per-class statistics (tuned to match real-data distribution) ──────────────

_DIST_NORMAL_MEAN  = 0.015
_DIST_NORMAL_STD   = 0.008
_DIST_MILD_MEAN    = 0.055
_DIST_MILD_STD     = 0.018
_DIST_SEVERE_MEAN  = 0.130
_DIST_SEVERE_STD   = 0.035

_EAR_NORMAL_MEAN   = 0.28
_EAR_NORMAL_STD    = 0.04
_EAR_DELTA_NORMAL  = 0.01
_EAR_DELTA_MILD    = 0.045
_EAR_DELTA_SEVERE  = 0.110

_BROW_NORMAL_STD   = 0.02
_BROW_DELTA_NORMAL = 0.008
_BROW_DELTA_MILD   = 0.035
_BROW_DELTA_SEVERE = 0.090

_TEX_NORMAL        = (0.82, 0.08)  # (mean, std) texture_score for Normal
_TEX_MILD          = (0.68, 0.10)
_TEX_SEVERE        = (0.50, 0.12)

# symmetry_error = α · mean(bilateral_dists) + β · (1 - texture_score)
_ALPHA, _BETA = 0.65, 0.35


def _clip_positive(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, None).astype(np.float32)


def _clip_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def generate_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate *n* Normal-class samples. Returns (n, 50) float32 array."""
    X = np.empty((n, 50), dtype=np.float32)

    dists = _clip_positive(rng.normal(_DIST_NORMAL_MEAN, _DIST_NORMAL_STD, (n, 40)))
    X[:, :40] = dists

    ear_l = _clip_positive(rng.normal(_EAR_NORMAL_MEAN, _EAR_NORMAL_STD, n))
    ear_r = _clip_positive(ear_l + rng.normal(0, _EAR_DELTA_NORMAL, n))
    X[:, 40] = ear_l
    X[:, 41] = ear_r
    X[:, 42] = np.abs(ear_l - ear_r)

    bh_l = rng.normal(0.12, _BROW_NORMAL_STD, n).astype(np.float32)
    bh_r = (bh_l + rng.normal(0, _BROW_DELTA_NORMAL, n)).astype(np.float32)
    X[:, 43] = bh_l
    X[:, 44] = bh_r
    X[:, 45] = np.abs(bh_l - bh_r)

    X[:, 46] = _clip_positive(rng.normal(0.008, 0.004, n))
    X[:, 47] = _clip_positive(rng.normal(0.003, 0.003, n))

    tex = _clip_unit(rng.normal(*_TEX_NORMAL, n))
    X[:, 48] = tex
    X[:, 49] = (_ALPHA * dists.mean(axis=1) + _BETA * (1.0 - tex)).astype(np.float32)
    return X


def generate_mild(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate *n* Mild-class samples. Returns (n, 50) float32 array."""
    X = np.empty((n, 50), dtype=np.float32)

    dists = _clip_positive(rng.normal(_DIST_MILD_MEAN, _DIST_MILD_STD, (n, 40)))
    X[:, :40] = dists

    ear_l = _clip_positive(rng.normal(_EAR_NORMAL_MEAN, _EAR_NORMAL_STD, n))
    ear_r = _clip_positive(ear_l + rng.normal(0, _EAR_DELTA_MILD, n))
    X[:, 40] = ear_l
    X[:, 41] = ear_r
    X[:, 42] = np.abs(ear_l - ear_r)

    bh_l = rng.normal(0.12, _BROW_NORMAL_STD, n).astype(np.float32)
    bh_r = (bh_l + rng.normal(0, _BROW_DELTA_MILD, n)).astype(np.float32)
    X[:, 43] = bh_l
    X[:, 44] = bh_r
    X[:, 45] = np.abs(bh_l - bh_r)

    X[:, 46] = _clip_positive(rng.normal(0.035, 0.012, n))
    X[:, 47] = _clip_positive(rng.normal(0.018, 0.010, n))

    tex = _clip_unit(rng.normal(*_TEX_MILD, n))
    X[:, 48] = tex
    X[:, 49] = (_ALPHA * dists.mean(axis=1) + _BETA * (1.0 - tex)).astype(np.float32)
    return X


def generate_severe(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate *n* Severe-class samples. Returns (n, 50) float32 array."""
    X = np.empty((n, 50), dtype=np.float32)

    dists = _clip_positive(rng.normal(_DIST_SEVERE_MEAN, _DIST_SEVERE_STD, (n, 40)))
    X[:, :40] = dists

    ear_l = _clip_positive(rng.normal(_EAR_NORMAL_MEAN, _EAR_NORMAL_STD, n))
    ear_r = _clip_positive(ear_l + rng.normal(0, _EAR_DELTA_SEVERE, n))
    X[:, 40] = ear_l
    X[:, 41] = ear_r
    X[:, 42] = np.abs(ear_l - ear_r)

    bh_l = rng.normal(0.12, _BROW_NORMAL_STD, n).astype(np.float32)
    bh_r = (bh_l + rng.normal(0, _BROW_DELTA_SEVERE, n)).astype(np.float32)
    X[:, 43] = bh_l
    X[:, 44] = bh_r
    X[:, 45] = np.abs(bh_l - bh_r)

    X[:, 46] = _clip_positive(rng.normal(0.090, 0.025, n))
    X[:, 47] = _clip_positive(rng.normal(0.045, 0.020, n))

    tex = _clip_unit(rng.normal(*_TEX_SEVERE, n))
    X[:, 48] = tex
    X[:, 49] = (_ALPHA * dists.mean(axis=1) + _BETA * (1.0 - tex)).astype(np.float32)
    return X


def build_dataset(
    n_per_class: int = 15_000,
    seed:        int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a balanced synthetic dataset.

    Returns
    -------
    X : (3 * n_per_class, 50) float32
    y : (3 * n_per_class,)    int64  (0=Normal, 1=Mild, 2=Severe)
    """
    rng = np.random.default_rng(seed)
    X_n = generate_normal(n_per_class, rng)
    X_m = generate_mild(n_per_class, rng)
    X_s = generate_severe(n_per_class, rng)

    X = np.concatenate([X_n, X_m, X_s], axis=0)
    y = np.array(
        [0] * n_per_class + [1] * n_per_class + [2] * n_per_class,
        dtype=np.int64,
    )

    perm = rng.permutation(len(y))
    return X[perm].astype(np.float32), y[perm]


__all__ = ["generate_normal", "generate_mild", "generate_severe", "build_dataset"]
