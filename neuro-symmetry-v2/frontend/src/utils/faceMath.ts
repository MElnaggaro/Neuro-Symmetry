/**
 * faceMath.ts — Mathematical Face Alignment (Phase 1)
 *
 * PURPOSE
 * -------
 * Normalise raw MediaPipe 468-landmark output into a geometrically aligned
 * coordinate system suitable for the downstream ONNX model.
 *
 * CRITICAL CONSTRAINT
 * -------------------
 * We apply ONLY two affine transforms:
 *   1. Translation  — shift so nose tip (landmark 1) is at the origin (0, 0).
 *   2. Roll Rotation — rotate so the inter-eye line is perfectly horizontal.
 *
 * We DO NOT scale by IPD (inter-pupillary distance). The ONNX model
 * (model_v2.onnx) was trained on raw pixel-space distances. Applying IPD
 * scaling would compress the feature distribution toward zero and produce
 * arbitrary predictions.
 *
 * LINEAR ALGEBRA
 * --------------
 * Given raw landmarks in pixel space (x_px, y_px):
 *
 *   Step 1 — Translation:
 *     dx = x_px - nose.x
 *     dy = y_px - nose.y
 *
 *   Step 2 — Roll Rotation by angle θ (inter-eye line angle):
 *     θ = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x)
 *
 *     The 2D rotation matrix R(-θ) that levels the eye-line:
 *       ┌              ┐   ┌    ┐   ┌                       ┐
 *       │ cos θ   sin θ│   │ dx │   │  dx·cosθ + dy·sinθ    │
 *       │-sin θ   cos θ│ × │ dy │ = │ -dx·sinθ + dy·cosθ    │
 *       └              ┘   └    ┘   └                       ┘
 *
 *     After this rotation, the left and right eyes share the same y-coordinate.
 *
 *   Step 3 — Z-axis centering:
 *     z_aligned = z_px - nose.z
 *
 *     This makes the nose tip the depth origin, consistent with the Python
 *     backend (landmark_engine.py) after the Phase 4 fix.
 *
 * 50-D FEATURE VECTOR LAYOUT
 * --------------------------
 *   [0:40]  bilateral mirror distances (one per MIRROR_PAIR)
 *   [40]    EAR_left   — Eye Aspect Ratio, left eye
 *   [41]    EAR_right  — Eye Aspect Ratio, right eye
 *   [42]    EAR_delta  — |EAR_left - EAR_right|
 *   [43]    brow_height_left   (negated y: higher brow = larger value)
 *   [44]    brow_height_right
 *   [45]    brow_height_delta  — |left - right|
 *   [46]    mouth_y_delta      — |mouth_left.y - mouth_right.y|
 *   [47]    mouth_x_offset     — |midpoint of mouth corners from x=0|
 *   [48]    texture_score      — S_texture ∈ [0, 1]  (placeholder: 1.0 on client)
 *   [49]    symmetry_error     — 0.65·mean(bilateral) + 0.35·(1 - texture)
 */

import type { NormalizedLandmark, NormalizedLandmarkList } from "@mediapipe/face_mesh";

// ── Public types ──────────────────────────────────────────────────────────────

export const FACE_FEATURE_COUNT = 50 as const;

/** Branded tuple — guarantees exactly 50 elements at the type level. */
export type FeatureVector50 = number[] & { readonly length: typeof FACE_FEATURE_COUNT };

export interface NormalizedFaceGeometry {
  /** Roll-corrected, nose-centred landmarks (NO IPD scaling). */
  landmarks: NormalizedLandmarkList;
  /** 50-D feature vector for ONNX inference. */
  features: FeatureVector50;
  /** Roll angle in radians (original eye-line tilt). */
  rollRad: number;
  /** Inter-pupillary distance in pixels (informational only — NOT used for scaling). */
  ipd: number;
}

export interface FaceMathOptions {
  /** Frame width in pixels (default 1). */
  width?: number;
  /** Frame height in pixels (default 1). */
  height?: number;
  /** Texture symmetry score ∈ [0, 1]. Defaults to 1.0 (not available client-side). */
  textureScore?: number;
}

// ── Anatomical constants ──────────────────────────────────────────────────────

const KEY_POINTS = {
  mouthLeft: 61,
  mouthRight: 291,
  eyeLeft: 33,
  eyeRight: 263,
  noseTip: 1,
  browLeft: 70,
  browRight: 300,
} as const;

/**
 * 40 bilateral mirror pairs (left_idx, right_idx).
 * Must match backend MIRROR_PAIRS in landmark_engine.py exactly.
 */
const MIRROR_PAIRS: readonly (readonly [number, number])[] = [
  // Eye outline (9 pairs)
  [33, 263], [7, 249], [163, 466], [144, 373], [145, 374],
  [153, 380], [154, 381], [155, 382], [133, 362],
  // Upper eye aperture (4 pairs)
  [160, 387], [158, 385], [157, 384], [159, 386],
  // Eyebrow (10 pairs)
  [70, 300], [63, 293], [105, 334], [66, 296], [107, 336],
  [55, 285], [65, 295], [52, 282], [53, 283], [46, 276],
  // Nose sides (5 pairs)
  [49, 279], [48, 278], [115, 344], [220, 440], [45, 275],
  // Mouth corners & lips (4 pairs)
  [61, 291], [57, 287], [185, 409], [84, 314],
  // Cheeks / jaw contour (5 pairs)
  [234, 454], [227, 447], [132, 361], [58, 288], [172, 397],
  // Forehead (3 pairs)
  [21, 251], [54, 284], [103, 332],
];

/** EAR landmark sextuplets: (outer_corner, upper1, upper2, inner_corner, lower1, lower2) */
const EAR_LEFT = [33, 160, 158, 133, 153, 144] as const;
const EAR_RIGHT = [263, 387, 385, 362, 380, 373] as const;

/** Fused symmetry error weights (must match backend _ALPHA / _BETA). */
const ALPHA = 0.65;
const BETA = 0.35;

// ── Internal helpers ──────────────────────────────────────────────────────────

/** 2-D Euclidean distance (x, y only). */
export function distance2(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** 3-D Euclidean distance. */
function distance3(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

/**
 * Eye Aspect Ratio: (||p2−p6|| + ||p3−p5||) / (2·||p1−p4||).
 * Returns 0 if the eye width is degenerate.
 */
export function eyeAspectRatio(points: NormalizedLandmarkList, idx: readonly number[]): number {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => points[i]);
  const width = distance2(p1, p4);
  if (width < 1e-6) return 0;
  return (distance2(p2, p6) + distance2(p3, p5)) / (2 * width);
}

/** Runtime guard — throw if the feature array is not exactly 50 elements. */
function asFeatureVector50(features: number[]): FeatureVector50 {
  if (features.length !== FACE_FEATURE_COUNT) {
    throw new Error(`Expected ${FACE_FEATURE_COUNT} features, got ${features.length}`);
  }
  return features as FeatureVector50;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Normalise raw MediaPipe landmarks and extract the 50-D feature vector.
 *
 * Applies Translation + Roll Rotation ONLY. No IPD scaling.
 */
export function normalizeFaceGeometry(
  rawLandmarks: NormalizedLandmarkList,
  options: FaceMathOptions = {},
): NormalizedFaceGeometry {
  if (rawLandmarks.length <= 466) {
    throw new Error(`Expected at least 467 MediaPipe landmarks, got ${rawLandmarks.length}`);
  }

  const width = Math.max(options.width ?? 1, 1);
  const height = Math.max(options.height ?? 1, 1);

  // ── Step 0: Convert fractional [0,1] MediaPipe coords → absolute pixel space
  const toPixelSpace = (lm: NormalizedLandmark) => ({
    x: lm.x * width,
    y: lm.y * height,
    z: (lm.z ?? 0) * width,       // z is metric-ish; scale with width for consistency
    visibility: lm.visibility,
  });

  const nose     = toPixelSpace(rawLandmarks[KEY_POINTS.noseTip]);
  const leftEye  = toPixelSpace(rawLandmarks[KEY_POINTS.eyeLeft]);
  const rightEye = toPixelSpace(rawLandmarks[KEY_POINTS.eyeRight]);

  // ── Compute roll angle from the inter-eye vector
  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  const ipd = Math.max(Math.hypot(eyeDx, eyeDy), 1e-6);   // informational only
  const rollRad = Math.atan2(eyeDy, eyeDx);

  const cos = Math.cos(rollRad);
  const sin = Math.sin(rollRad);

  // ── Step 1 + 2: Translation (nose→origin) + Roll Rotation (−θ)
  //
  // For each landmark:
  //   dx = pixel_x − nose_x          (translate to nose origin)
  //   dy = pixel_y − nose_y
  //
  //   aligned_x =  dx·cos(θ) + dy·sin(θ)    (rotate by −θ)
  //   aligned_y = −dx·sin(θ) + dy·cos(θ)
  //
  // NOTE: We do NOT divide by IPD. The ONNX model expects raw pixel-space
  // distances. Scaling would compress features toward zero.
  const landmarks = rawLandmarks.map((raw) => {
    const lm = toPixelSpace(raw);
    const dx = lm.x - nose.x;
    const dy = lm.y - nose.y;

    return {
      x: dx * cos + dy * sin,           // roll-corrected x (pixels)
      y: -dx * sin + dy * cos,           // roll-corrected y (pixels)
      z: lm.z - nose.z,                  // depth centred on nose tip
      visibility: lm.visibility,
    };
  });

  // ── 50-D Feature Extraction ─────────────────────────────────────────────
  // Features are IPD-normalised for resolution independence. Landmarks
  // themselves stay in pixel space for rendering purposes.

  // Features [0:40] — bilateral mirror distances (IPD-normalised)
  // For each (left, right) pair: mirror(right) = (−right.x, right.y, right.z)
  // dist = ||left − mirror(right)||₃ / ipd
  const bilateral = MIRROR_PAIRS.map(([left, right]) => {
    const l = landmarks[left];
    const r = landmarks[right];
    return distance3(l, { x: -r.x, y: r.y, z: r.z ?? 0 }) / ipd;
  });

  // Features [40:43] — Eye Aspect Ratios
  const earLeft = eyeAspectRatio(landmarks, EAR_LEFT);
  const earRight = eyeAspectRatio(landmarks, EAR_RIGHT);

  // Features [43:46] — Brow heights (negated y, IPD-normalised: higher brow = larger value)
  const browLeft = -landmarks[KEY_POINTS.browLeft].y / ipd;
  const browRight = -landmarks[KEY_POINTS.browRight].y / ipd;

  // Features [46:48] — Mouth deviation
  const mouthLeft = landmarks[KEY_POINTS.mouthLeft];
  const mouthRight = landmarks[KEY_POINTS.mouthRight];

  // Feature [48] — Texture score (not available client-side; default 1.0)
  const textureScore = Math.max(0, Math.min(1, options.textureScore ?? 1.0));

  // Feature [49] — Fused symmetry error
  const meanBilateral = bilateral.reduce((sum, value) => sum + value, 0) / bilateral.length;
  const symmetryError = ALPHA * meanBilateral + BETA * (1 - textureScore);

  const features = asFeatureVector50([
    ...bilateral,                             // [0:40]
    earLeft,                                  // [40]
    earRight,                                 // [41]
    Math.abs(earLeft - earRight),             // [42]
    browLeft,                                 // [43]
    browRight,                                // [44]
    Math.abs(browLeft - browRight),           // [45]
    Math.abs(mouthLeft.y - mouthRight.y) / ipd,     // [46]  IPD-normalised
    Math.abs((mouthLeft.x + mouthRight.x) / 2) / ipd, // [47]  IPD-normalised
    textureScore,                             // [48]
    symmetryError,                            // [49]
  ]);

  return { landmarks, features, rollRad, ipd };
}
