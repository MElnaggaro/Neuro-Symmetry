"""
Texture Engine — bilateral face symmetry via normalised cross-correlation.

Algorithm:
  1. Crop the face bounding box from the frame (using raw MediaPipe coords).
  2. Resize to a fixed square (PATCH_SIZE × PATCH_SIZE).
  3. Split into left/right halves; horizontally flip the right half.
  4. cv2.matchTemplate(left, right_flipped, TM_CCOEFF_NORMED) → scalar in [-1, 1].
  5. Normalise to [0, 1]: S_texture = (raw + 1) / 2.

S_texture = 1.0 → perfect bilateral texture symmetry (neutral face).
S_texture = 0.5 → uncorrelated.
S_texture = 0.0 → perfectly anti-correlated.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.api.landmark_engine import KEY_POINTS, LandmarkResult

# Internal patch size; larger = more detail but slower
_PATCH_SIZE = 64   # pixels — both halves are resized to (PATCH_SIZE × PATCH_SIZE//2)


class TextureEngine:
    """Stateless bilateral texture comparator."""

    def compute(self, frame_bgr: np.ndarray, result: LandmarkResult) -> float:
        """
        Returns S_texture ∈ [0, 1].
        Falls back to 0.5 (neutral) if the crop is degenerate.
        """
        roi = self._crop_face_roi(frame_bgr, result)
        if roi is None:
            return 0.5

        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize to fixed height × even width so halves are identical size
        target_h = _PATCH_SIZE
        target_w = _PATCH_SIZE & ~1   # round down to nearest even
        grey = cv2.resize(grey, (target_w, target_h), interpolation=cv2.INTER_AREA)

        half = target_w // 2
        left_patch  = grey[:, :half].astype(np.float32)
        right_flip  = np.ascontiguousarray(grey[:, half:][:, ::-1].astype(np.float32))

        # Both patches are target_h × half — matchTemplate returns 1×1
        result_mat = cv2.matchTemplate(left_patch, right_flip, cv2.TM_CCOEFF_NORMED)
        raw = float(np.clip(result_mat[0, 0], -1.0, 1.0))
        return (raw + 1.0) / 2.0

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_face_roi(
        frame_bgr: np.ndarray,
        result: LandmarkResult,
    ) -> np.ndarray | None:
        """
        Crop face bounding box from frame using raw MediaPipe fractional coords.
        Returns None if the crop is too small to be useful.
        """
        h, w = result.frame_hw
        raw = result.raw

        if not raw:
            return None

        xs = np.array([lm.x for lm in raw], dtype=np.float32) * w
        ys = np.array([lm.y for lm in raw], dtype=np.float32) * h

        pad_x = int((xs.max() - xs.min()) * 0.08)
        pad_y = int((ys.max() - ys.min()) * 0.08)

        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(w, int(xs.max()) + pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(h, int(ys.max()) + pad_y)

        if (x1 - x0) < 16 or (y1 - y0) < 16:
            return None

        return frame_bgr[y0:y1, x0:x1]
