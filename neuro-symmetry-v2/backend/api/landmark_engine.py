"""
Landmark Engine — MediaPipe Face Landmarker wrapper.

Provides:
  LandmarkEngine   — processes a BGR frame, returns LandmarkResult or None
  LandmarkResult   — raw + normalised landmarks, pose angles, IPD
  NormalizedLandmarks — 478 pts in nose-centred, roll-corrected pixel space (NO IPD scaling)
  PoseAngles       — roll / yaw / pitch in degrees from 4x4 transform matrix

Normalisation coordinate system
─────────────────────────────────
  Origin  : nose tip (landmark 1)
  Scale   : pixels (NO IPD scaling — matches frontend faceMath.ts)
  Roll    : corrected — eye line is horizontal after transform
  Yaw/Pitch: NOT corrected (handled by frame filtering in InputQualityChecker)

  After normalisation:
    points[KEY_POINTS["nose_tip"]]   ≈ (0, 0, *)   — nose at origin
    points[KEY_POINTS["eye_left"]].y ≈ points[KEY_POINTS["eye_right"]].y  — horizontal eyes

Mirror pairs
─────────────
  MIRROR_PAIRS: list of (left_idx, right_idx) for bilateral symmetry features.
  LEFT_INDICES / RIGHT_INDICES: flat arrays for vectorised operations in Phase 3.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import mediapipe as mp  # noqa: E402

from backend.core import get_settings

# ── MediaPipe Tasks imports ───────────────────────────────────────────────────

_BaseOptions          = mp.tasks.BaseOptions
_FaceLandmarker       = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode          = mp.tasks.vision.RunningMode

# ── Anatomical constants ──────────────────────────────────────────────────────

KEY_POINTS: dict[str, int] = {
    "mouth_left":  61,
    "mouth_right": 291,
    "eye_left":    33,
    "eye_right":   263,
    "nose_tip":     1,
    "brow_left":   70,
    "brow_right":  300,
}

# Mirror pairs (left_idx, right_idx) — bilateral symmetry, used by Phase 3
MIRROR_PAIRS: list[tuple[int, int]] = [
    # Eye outline
    (33,  263), (7,   249), (163, 466), (144, 373), (145, 374),
    (153, 380), (154, 381), (155, 382), (133, 362),
    # Upper eye aperture
    (160, 387), (158, 385), (157, 384), (159, 386),
    # Eyebrow
    (70,  300), (63,  293), (105, 334), (66,  296), (107, 336),
    (55,  285), (65,  295), (52,  282), (53,  283), (46,  276),
    # Nose sides
    (49,  279), (48,  278), (115, 344), (220, 440), (45,  275),
    # Mouth corners & lips
    (61,  291), (57,  287), (185, 409), (84,  314),
    # Cheeks / jaw contour
    (234, 454), (227, 447), (132, 361), (58,  288), (172, 397),
    # Forehead
    (21,  251), (54,  284), (103, 332),
]

LEFT_INDICES:  list[int] = [l for l, _ in MIRROR_PAIRS]
RIGHT_INDICES: list[int] = [r for _, r in MIRROR_PAIRS]

# Default model path — sourced from Settings (overridable via NS_LANDMARKER_MODEL_PATH)
def _default_model_path() -> Path:
    return get_settings().landmarker_model_path


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoseAngles:
    roll:  float   # degrees; positive = clockwise tilt (right side down)
    yaw:   float   # degrees; positive = turning right
    pitch: float   # degrees; positive = looking down


@dataclass
class NormalizedLandmarks:
    """
    478 landmarks in nose-centred, roll-corrected pixel space.
    NO IPD scaling (matches frontend faceMath.ts).
    points: float32 array, shape (478, 3).
    """
    points: np.ndarray   # (478, 3)
    ipd:    float        # interpupillary distance in source image pixels


@dataclass
class LandmarkResult:
    """Full result for one frame — raw landmarks, normalised landmarks, pose."""
    raw:        list               # list of NormalizedLandmark (478)
    normalized: NormalizedLandmarks
    pose:       PoseAngles
    frame_hw:   tuple[int, int]    # (height, width) of source frame

    # FaceLandmarker Tasks API does not expose a per-detection confidence score
    # in IMAGE mode. We return 1.0 when a face is found (binary presence).
    score: float = 1.0

    def key_points(self) -> dict[str, tuple[float, float, float]]:
        """Return {name: (x, y, z)} for each KEY_POINT in normalised space."""
        pts = self.normalized.points
        return {
            name: (float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2]))
            for name, idx in KEY_POINTS.items()
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_pose(mat: np.ndarray) -> PoseAngles:
    """
    Extract roll / yaw / pitch from the 4×4 facial transformation matrix
    provided by FaceLandmarker.  Uses ZYX Euler decomposition.
    """
    r = mat[:3, :3]
    roll  = math.atan2( r[2, 1],  r[2, 2]) * (180.0 / math.pi)
    pitch = math.atan2(-r[2, 0],  math.sqrt(r[2, 1]**2 + r[2, 2]**2)) * (180.0 / math.pi)
    yaw   = math.atan2( r[1, 0],  r[0, 0]) * (180.0 / math.pi)
    return PoseAngles(roll=roll, yaw=yaw, pitch=pitch)


def _normalize_landmarks(
    lms: list,
    frame_hw: tuple[int, int],
) -> NormalizedLandmarks:
    """
    Normalise 478 landmarks to nose-centred, IPD-scaled, roll-corrected space.

    Steps:
      1. Convert fractional [0,1] coords → absolute pixels.
      2. Compute roll angle from inter-eye vector.
      3. Translate to nose-tip origin.
      4. Rotate by -roll (counter-clockwise) to level the eyes.
      5. Scale by IPD so 1 unit == 1 IPD.
    """
    h, w = frame_hw

    # (478, 3) in pixel space; z is already metric-ish, scale with w for consistency
    pts = np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in lms],
        dtype=np.float32,
    )

    nose = pts[KEY_POINTS["nose_tip"]]
    leye = pts[KEY_POINTS["eye_left"]]
    reye = pts[KEY_POINTS["eye_right"]]

    ipd = float(np.linalg.norm(reye[:2] - leye[:2]))
    ipd = max(ipd, 1.0)   # guard against degenerate detection

    # Roll angle of the inter-eye line
    roll = math.atan2(reye[1] - leye[1], reye[0] - leye[0])
    cos_r, sin_r = math.cos(roll), math.sin(roll)

    # Translate, rotate by −roll (NO IPD scaling — matches frontend faceMath.ts)
    # The ONNX model’s scaler handles normalisation; dividing by IPD here would
    # compress the feature distribution toward zero.
    dx = pts[:, 0] - nose[0]
    dy = pts[:, 1] - nose[1]
    x_norm =  dx * cos_r + dy * sin_r
    y_norm = -dx * sin_r + dy * cos_r
    # Z-axis centering (nose tip at depth origin)
    z_norm = pts[:, 2] - nose[2]

    normalised = np.stack([x_norm, y_norm, z_norm], axis=1).astype(np.float32)
    return NormalizedLandmarks(points=normalised, ipd=ipd)


# ── LandmarkEngine ────────────────────────────────────────────────────────────

class LandmarkEngine:
    """
    Thin wrapper around MediaPipe FaceLandmarker (Tasks API).

    Usage (context manager — preferred):
        with LandmarkEngine() as engine:
            result = engine.process(frame_bgr)

    Usage (manual):
        engine = LandmarkEngine()
        result = engine.process(frame_bgr)
        engine.close()
    """

    # EMA alpha for landmark smoothing: lower = smoother but more lag.
    # 0.5 is a good balance for 25-30 fps streams.
    _EMA_ALPHA: float = 0.5

    def __init__(
        self,
        model_path: Optional[Path] = None,
        ema_alpha:  Optional[float] = None,
    ) -> None:
        cfg  = get_settings()
        path = model_path or cfg.landmarker_model_path
        if not path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {path}. "
                "Download it with: python -m backend.api.landmark_engine --download"
            )

        # VIDEO mode enables MediaPipe's internal Kalman/temporal smoothing so
        # landmarks are stable across frames even when the camera shakes.
        # IMAGE mode treats each frame independently — no cross-frame smoothing.
        opts = _FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(path)),
            running_mode=_RunningMode.VIDEO,
            num_faces=cfg.mediapipe.num_faces,
            min_face_detection_confidence=cfg.mediapipe.min_face_detection_confidence,
            min_face_presence_confidence=cfg.mediapipe.min_face_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = _FaceLandmarker.create_from_options(opts)
        self._start_ns:      int                     = time.perf_counter_ns()
        self._ema_alpha:     float                   = ema_alpha if ema_alpha is not None else self._EMA_ALPHA
        self._smoothed_pts:  Optional[np.ndarray]    = None   # (478, 3) EMA state

    def _timestamp_ms(self) -> int:
        """Monotonically increasing milliseconds since engine creation."""
        return (time.perf_counter_ns() - self._start_ns) // 1_000_000

    def process(self, frame_bgr: np.ndarray) -> Optional[LandmarkResult]:
        """
        Process a single BGR frame (cv2 convention).
        Returns LandmarkResult or None if no face is found.

        Landmark positions are smoothed with an EMA filter across consecutive
        frames to suppress camera-shake jitter.  Call reset_session() when
        switching subjects so stale smoothing state is cleared.
        """
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_img, self._timestamp_ms())

        if not result.face_landmarks:
            # Clear EMA so the next detection starts fresh without stale history.
            self._smoothed_pts = None
            return None

        lms      = result.face_landmarks[0]
        mat      = result.facial_transformation_matrixes[0]
        frame_hw = (frame_bgr.shape[0], frame_bgr.shape[1])

        raw_norm = _normalize_landmarks(lms, frame_hw)

        # ── EMA smoothing over normalised landmark positions ──────────────
        # Smooths out per-frame jitter from camera shake without introducing
        # significant lag (alpha=0.5 gives a half-life of ~1 frame at 30 fps).
        if self._smoothed_pts is None:
            self._smoothed_pts = raw_norm.points.copy()
        else:
            self._smoothed_pts = (
                self._ema_alpha * raw_norm.points
                + (1.0 - self._ema_alpha) * self._smoothed_pts
            ).astype(np.float32)

        smoothed_norm = NormalizedLandmarks(
            points=self._smoothed_pts.copy(),
            ipd=raw_norm.ipd,
        )

        return LandmarkResult(
            raw=lms,
            normalized=smoothed_norm,
            pose=_extract_pose(mat),
            frame_hw=frame_hw,
        )

    def reset_session(self) -> None:
        """Clear EMA state — call between subjects or when restarting a session."""
        self._smoothed_pts = None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "LandmarkEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        import urllib.request
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )
        dest = _default_model_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Done. Size: {dest.stat().st_size / 1_048_576:.1f} MB")
        sys.exit(0)

    # Deliverable: feed a test image, print QualityStatus + KEY_POINTS
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if img_arg is None:
        # Fall back to first CelebA image
        from datasets.config import CELEBA_IMGS_ALIGNED
        img_arg = str(CELEBA_IMGS_ALIGNED / "000001.jpg")

    from backend.api.input_quality import InputQualityChecker

    frame   = cv2.imread(img_arg)
    if frame is None:
        print(f"ERROR: cannot read {img_arg}")
        sys.exit(1)

    with LandmarkEngine() as engine:
        result = engine.process(frame)

    checker = InputQualityChecker()
    status  = checker.check(frame, result)

    print(f"Image        : {img_arg}")
    print(f"Quality      : {status.code.value}  ({status.reason or '-'})")

    if result is None:
        print("No face detected.")
        sys.exit(0)

    print(f"Pose         : roll={result.pose.roll:+.1f}°  "
          f"yaw={result.pose.yaw:+.1f}°  pitch={result.pose.pitch:+.1f}°")
    print(f"IPD          : {result.normalized.ipd:.1f} px")
    print(f"Landmarks    : {len(result.raw)}")
    print("Key points (normalised coords):")
    for name, (x, y, z) in result.key_points().items():
        print(f"  {name:15s}  x={x:+.4f}  y={y:+.4f}  z={z:+.5f}")
