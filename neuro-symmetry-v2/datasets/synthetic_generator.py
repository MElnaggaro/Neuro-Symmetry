"""
Synthetic droop augmentation — stub.

Generates unlimited pathological training samples by warping CelebA normal
faces to simulate unilateral facial droop, ptosis, and nasolabial flattening.

Displacement magnitudes:
    Mild    →  5–15% of interpupillary distance
    Severe  → 25–50% of interpupillary distance

Full implementation belongs to Phase 4 (AI Training — synthetic_augment.py).
"""

from __future__ import annotations


def generate_droop_sample(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("Synthetic generator — full implementation in Phase 4.")


if __name__ == "__main__":
    print("Synthetic droop generator — stub only.")
    print("Full implementation in Phase 4.")
