"""
Synthetic dataset generator — feature-space droop simulation.

Works entirely in 50-dim feature space (no images needed), so it can
generate unlimited training samples from scratch.

Label convention
----------------
  0 = Normal
  1 = Mild asymmetry   (unilateral 5–15% droop)
  2 = Severe asymmetry (unilateral 25–50% droop)

Feature layout (matches backend/api/feature_extractor.py)
----------------------------------------------------------
  [0:40]  bilateral mirror distances
  [40]    EAR_left
  [41]    EAR_right
  [42]    EAR_delta
  [43]    brow_height_left
  [44]    brow_height_right
  [45]    brow_height_delta
  [46]    mouth_y_delta
  [47]    mouth_x_offset
  [48]    texture_score
  [49]    symmetry_error
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


# ── Bilateral feature group slices ────────────────────────────────────────────
# Indices within the first 40 bilateral features (aligned to MIRROR_PAIRS order)
_EYE_OUTLINE   = np.arange(0, 9)    # 9 pairs
_EYE_APERTURE  = np.arange(9, 13)   # 4 pairs
_EYEBROW       = np.arange(13, 23)  # 10 pairs
_NOSE          = np.arange(23, 28)  # 5 pairs
_MOUTH         = np.arange(28, 32)  # 4 pairs
_JAW           = np.arange(32, 37)  # 5 pairs
_FOREHEAD      = np.arange(37, 40)  # 3 pairs

# Regions most affected by lower-motor-neuron palsy / stroke droop
_DROOP_REGIONS = np.concatenate([_EYE_OUTLINE, _EYE_APERTURE, _MOUTH, _JAW])
_ALPHA = 0.65
_BETA  = 0.35


# ── Internal generators ───────────────────────────────────────────────────────

def _base_features(rng: Generator, n: int) -> np.ndarray:
    """
    Generate n neutral-face feature vectors.
    All bilateral distances are small (near-zero asymmetry).
    """
    f = np.empty((n, 50), dtype=np.float32)

    # Bilateral distances: tiny symmetric noise
    f[:, :40] = rng.normal(0.02, 0.012, (n, 40)).clip(0.0)

    # EAR — typical open eye ≈ 0.28–0.35
    ear = rng.normal(0.30, 0.025, (n, 2)).clip(0.18, 0.48)
    f[:, 40] = ear[:, 0]           # EAR_left
    f[:, 41] = ear[:, 1]           # EAR_right
    f[:, 42] = np.abs(ear[:, 0] - ear[:, 1])  # EAR_delta

    # Brow heights (above nose, in IPD units)
    brow = rng.normal(0.82, 0.05, (n, 2)).clip(0.55, 1.10)
    f[:, 43] = brow[:, 0]
    f[:, 44] = brow[:, 1]
    f[:, 45] = np.abs(brow[:, 0] - brow[:, 1])

    # Mouth
    f[:, 46] = rng.normal(0.02, 0.010, n).clip(0.0)   # mouth_y_delta
    f[:, 47] = rng.normal(0.05, 0.020, n).clip(0.0)   # mouth_x_offset

    # Texture
    f[:, 48] = rng.normal(0.93, 0.035, n).clip(0.75, 1.00)

    # Symmetry error (recomputed from features, consistent with extractor)
    f[:, 49] = (_ALPHA * f[:, :40].mean(axis=1) +
                _BETA  * (1.0 - f[:, 48])).astype(np.float32)
    return f


def _apply_droop(
    f: np.ndarray,
    rng: Generator,
    droop_range: tuple[float, float],
    ear_drop: float,
    texture_drop: float,
) -> np.ndarray:
    """
    Apply unilateral droop to a batch of neutral feature vectors in place.
    Randomly picks left or right side to be affected.
    """
    n = len(f)

    # How much each affected bilateral distance grows
    droop = rng.uniform(droop_range[0], droop_range[1], (n, len(_DROOP_REGIONS)))
    f[:, _DROOP_REGIONS] += droop

    # EAR drops on the affected side
    left_affected = rng.random(n) < 0.5
    f[ left_affected, 40] = (f[ left_affected, 40] - ear_drop).clip(0.05)
    f[~left_affected, 41] = (f[~left_affected, 41] - ear_drop).clip(0.05)
    f[:, 42] = np.abs(f[:, 40] - f[:, 41])

    # Texture degrades
    f[:, 48] = (f[:, 48] - rng.uniform(0.0, texture_drop, n)).clip(0.0, 1.0)

    # Recompute symmetry error
    f[:, 49] = (_ALPHA * f[:, :40].mean(axis=1) +
                _BETA  * (1.0 - f[:, 48])).astype(np.float32)
    return f


# ── Public API ────────────────────────────────────────────────────────────────

def generate_normal(n: int, rng: Generator) -> np.ndarray:
    """Return n normal-class feature vectors, shape (n, 50)."""
    return _base_features(rng, n)


def generate_mild(n: int, rng: Generator) -> np.ndarray:
    """Return n mild-asymmetry feature vectors, shape (n, 50)."""
    f = _base_features(rng, n)
    return _apply_droop(f, rng, droop_range=(0.05, 0.18), ear_drop=0.06, texture_drop=0.12)


def generate_severe(n: int, rng: Generator) -> np.ndarray:
    """Return n severe-asymmetry feature vectors, shape (n, 50)."""
    f = _base_features(rng, n)
    return _apply_droop(f, rng, droop_range=(0.25, 0.52), ear_drop=0.16, texture_drop=0.32)


def build_dataset(
    n_per_class: int = 5_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a balanced 3-class synthetic dataset.

    Returns
    -------
    features : float32 array, shape (3*n_per_class, 50)
    labels   : int64  array, shape (3*n_per_class,)   — 0 / 1 / 2
    """
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        generate_normal(n_per_class, rng),
        generate_mild(n_per_class, rng),
        generate_severe(n_per_class, rng),
    ], axis=0)
    y = np.repeat([0, 1, 2], n_per_class).astype(np.int64)
    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm].astype(np.float32), y[perm]


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = build_dataset(n_per_class=1_000)
    print(f"Dataset  : {X.shape}  labels {np.unique(y, return_counts=True)}")
    for cls, name in enumerate(["Normal", "Mild", "Severe"]):
        mask = y == cls
        err = X[mask, 49]
        print(f"  {name:8s}  symmetry_error: mean={err.mean():.3f}  std={err.std():.3f}")
