// ─── Configuration ────────────────────────────────────────────────────────────

const KEY_POINTS = {
  mouth_left:  61,
  mouth_right: 291,
  eye_left:    33,
  eye_right:   263,
  nose_tip:     1,
  brow_left:   70,
  brow_right:  300,
};

// Used only for pose estimation — not part of symmetry features
const POSE_LM = {
  left_cheek:  234,
  right_cheek: 454,
  chin:        152,
  forehead:     10,
};

const THRESHOLDS = {
  normal: 0.88,
  mild:   0.70,
};

// Frames with any angle beyond these limits (degrees) are discarded
const POSE_LIMITS = {
  roll:  15,
  yaw:   20,
  pitch: 15,
};

// EMA weight for new frame — lower = smoother but slower to respond
const SMOOTHING_ALPHA  = 0.25;
const CALIBRATION_FRAMES = 120;

// ─── DOM ──────────────────────────────────────────────────────────────────────

const video             = document.getElementById("video");
const canvas            = document.getElementById("overlay");
const ctx               = canvas.getContext("2d");
const scoreDisplay      = document.getElementById("score-display");
const statusBadge       = document.getElementById("status-badge");
const calibrationStatus = document.getElementById("calibration-status");
const calibrationButton = document.getElementById("calibration-button");
const logList           = document.getElementById("log-list");
const poseRollEl        = document.getElementById("pose-roll");
const poseYawEl         = document.getElementById("pose-yaw");
const posePitchEl       = document.getElementById("pose-pitch");
const frameQualityEl    = document.getElementById("frame-quality");

// ─── State ────────────────────────────────────────────────────────────────────

let baseline          = null;
let calibrating       = false;
let calibrationFrames = [];
let lastRiskLevel     = null;
let faceWasDetected   = false;
let smoothedFeatures  = null;   // EMA accumulator — null resets on face loss

// ─── MediaPipe Setup ──────────────────────────────────────────────────────────

calibrationButton.addEventListener("click", startCalibration);
logEvent("App initialized.");

const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
  maxNumFaces:            1,
  refineLandmarks:        true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence:  0.7,
});

faceMesh.onResults(onResults);

const camera = new Camera(video, {
  onFrame: async () => { await faceMesh.send({ image: video }); },
  width:  640,
  height: 480,
});

camera.start();
logEvent("Camera stream requested.");

// ─── Head Pose Estimation ─────────────────────────────────────────────────────

// Returns {roll, yaw, pitch} in degrees.
//
// Roll  — rotation around the depth axis (head tilt left/right).
//         Derived from the signed angle of the inter-eye line.
//
// Yaw   — rotation around the vertical axis (turning left/right).
//         Approximated by comparing the projected width on each side of
//         the nose. When the face turns right the right cheek-to-nose
//         distance shrinks; the ratio encodes the turn magnitude.
//
// Pitch — rotation around the horizontal axis (nodding up/down).
//         Approximated by comparing upper (forehead→nose) vs lower
//         (nose→chin) projected heights.
function estimatePose(landmarks) {
  const leftEye    = landmarks[KEY_POINTS.eye_left];
  const rightEye   = landmarks[KEY_POINTS.eye_right];
  const noseTip    = landmarks[KEY_POINTS.nose_tip];
  const leftCheek  = landmarks[POSE_LM.left_cheek];
  const rightCheek = landmarks[POSE_LM.right_cheek];
  const chin       = landmarks[POSE_LM.chin];
  const forehead   = landmarks[POSE_LM.forehead];

  const roll = Math.atan2(
    rightEye.y - leftEye.y,
    rightEye.x - leftEye.x,
  ) * (180 / Math.PI);

  const leftWidth  = noseTip.x - leftCheek.x;
  const rightWidth = rightCheek.x - noseTip.x;
  const totalWidth = leftWidth + rightWidth;
  const yaw = totalWidth > 1e-6
    ? ((rightWidth - leftWidth) / totalWidth) * 90
    : 0;

  const upperHeight = noseTip.y - forehead.y;
  const lowerHeight = chin.y - noseTip.y;
  const totalHeight = upperHeight + lowerHeight;
  const pitch = totalHeight > 1e-6
    ? ((lowerHeight - upperHeight) / totalHeight) * 90
    : 0;

  return { roll, yaw, pitch };
}

function isFrameUsable({ roll, yaw, pitch }) {
  return (
    Math.abs(roll)  < POSE_LIMITS.roll  &&
    Math.abs(yaw)   < POSE_LIMITS.yaw   &&
    Math.abs(pitch) < POSE_LIMITS.pitch
  );
}

// ─── Face Normalization ───────────────────────────────────────────────────────

// Rotates all landmarks in-plane by −roll to make the eye line horizontal,
// then re-centers on the nose tip. Coordinate scale is preserved so the
// existing score formula needs no changes. Original array is not mutated.
function normalizeLandmarks(landmarks) {
  const noseTip  = landmarks[KEY_POINTS.nose_tip];
  const leftEye  = landmarks[KEY_POINTS.eye_left];
  const rightEye = landmarks[KEY_POINTS.eye_right];

  const cx   = noseTip.x;
  const cy   = noseTip.y;

  // roll angle of the eye line; we rotate by its negative to undo it
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const cosR = Math.cos(roll);
  const sinR = Math.sin(roll);

  return landmarks.map(({ x, y, z, visibility }) => {
    const dx = x - cx;
    const dy = y - cy;
    return {
      x:  dx * cosR + dy * sinR + cx,
      y: -dx * sinR + dy * cosR + cy,
      z,
      visibility,
    };
  });
}

// ─── Feature Extraction ───────────────────────────────────────────────────────

function extractFeatures(landmarks) {
  const get = (name) => landmarks[KEY_POINTS[name]];

  const noseX          = get("nose_tip").x;
  const mouthMid       = (get("mouth_left").x + get("mouth_right").x) / 2;
  const mouthDeviation = Math.abs(mouthMid - noseX);

  const eyeLeft        = Math.abs(get("eye_left").y  - get("nose_tip").y);
  const eyeRight       = Math.abs(get("eye_right").y - get("nose_tip").y);
  const eyeAsymmetry   = Math.abs(eyeLeft - eyeRight);

  const browLeft       = Math.abs(get("brow_left").y  - get("nose_tip").y);
  const browRight      = Math.abs(get("brow_right").y - get("nose_tip").y);
  const browAsymmetry  = Math.abs(browLeft - browRight);

  return [mouthDeviation, eyeAsymmetry, browAsymmetry];
}

// ─── Temporal Smoothing (EMA) ─────────────────────────────────────────────────

// Exponential moving average applied per feature dimension.
// Resets automatically when smoothedFeatures is set to null.
function smoothFeatures(features) {
  if (smoothedFeatures === null) {
    smoothedFeatures = [...features];
    return [...smoothedFeatures];
  }
  smoothedFeatures = features.map(
    (f, i) => SMOOTHING_ALPHA * f + (1 - SMOOTHING_ALPHA) * smoothedFeatures[i],
  );
  return [...smoothedFeatures];
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function computeScore(features) {
  const weights = [0.5, 0.3, 0.2];
  const error   = features.reduce((sum, f, i) => sum + weights[i] * f, 0);
  return Math.exp(-error * 10);
}

function getRiskLevel(score) {
  if (score >= THRESHOLDS.normal) return { level: 0, label: "Normal",    color: "#22c55e" };
  if (score >= THRESHOLDS.mild)   return { level: 1, label: "Mild",      color: "#eab308" };
  return                                  { level: 2, label: "High Risk", color: "#ef4444" };
}

// ─── Calibration ──────────────────────────────────────────────────────────────

function startCalibration() {
  calibrating       = true;
  calibrationFrames = [];
  smoothedFeatures  = null;   // fresh EMA for calibration session
  calibrationStatus.textContent = "Hold a neutral face to capture your baseline.";
  logEvent("Calibration started.");
}

function updateCalibration(features) {
  if (!calibrating) return;

  calibrationFrames.push(features);
  const progress = Math.round((calibrationFrames.length / CALIBRATION_FRAMES) * 100);
  calibrationStatus.textContent = `Calibrating... ${progress}%`;

  if (calibrationFrames.length >= CALIBRATION_FRAMES) {
    baseline = calibrationFrames[0].map((_, i) =>
      calibrationFrames.reduce((sum, f) => sum + f[i], 0) / calibrationFrames.length,
    );
    calibrating = false;
    calibrationStatus.textContent = "Baseline captured.";
    logEvent("Calibration complete. Baseline captured.");
  }
}

function applyBaseline(features) {
  if (!baseline) return features;
  return features.map((f, i) => Math.abs(f - baseline[i]));
}

// ─── Drawing ──────────────────────────────────────────────────────────────────

// Draws original (un-normalized) landmarks so the overlay matches the video.
// Dot color shifts green → red as the frame becomes less usable.
function drawMesh(landmarks, usable) {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = usable ? "rgba(0,255,180,0.75)" : "rgba(239,68,68,0.55)";

  Object.values(KEY_POINTS).forEach((index) => {
    const pt = landmarks[index];
    ctx.beginPath();
    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

// ─── Pose Display ─────────────────────────────────────────────────────────────

function updatePoseDisplay(pose, usable) {
  const fmt = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}°`;
  poseRollEl.textContent  = fmt(pose.roll);
  poseYawEl.textContent   = fmt(pose.yaw);
  posePitchEl.textContent = fmt(pose.pitch);
  frameQualityEl.textContent = usable ? "OK" : "FILTERED";
  frameQualityEl.className   = usable ? "ok" : "filtered";
}

// ─── Logging ──────────────────────────────────────────────────────────────────

function logEvent(message) {
  const ts = new Date().toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
  console.log(`[${ts}] ${message}`);

  const li = document.createElement("li");
  li.textContent = `${ts} ${message}`;
  logList.prepend(li);

  while (logList.children.length > 6) {
    logList.removeChild(logList.lastElementChild);
  }
}

// ─── Main Result Handler ──────────────────────────────────────────────────────

function onResults(results) {
  if (!results.multiFaceLandmarks?.length) {
    statusBadge.textContent = "No face detected";
    statusBadge.style.color = "#94a3b8";
    if (faceWasDetected) {
      logEvent("Face lost.");
      faceWasDetected  = false;
      lastRiskLevel    = null;
      smoothedFeatures = null;   // reset EMA so next detection starts clean
    }
    return;
  }

  if (!faceWasDetected) {
    logEvent("Face detected.");
    faceWasDetected = true;
  }

  const landmarks = results.multiFaceLandmarks[0];

  // 1. Pose estimation on raw landmarks
  const pose   = estimatePose(landmarks);
  const usable = isFrameUsable(pose);

  // 2. Always render mesh and pose readout
  drawMesh(landmarks, usable);
  updatePoseDisplay(pose, usable);

  // 3. Discard frames with excessive head movement — avoids false alerts
  if (!usable) {
    statusBadge.textContent = "Reposition head";
    statusBadge.style.color = "#f97316";
    return;
  }

  // 4. Roll-correct → extract → smooth → baseline → score
  const normalized = normalizeLandmarks(landmarks);
  const raw        = extractFeatures(normalized);
  const smooth     = smoothFeatures(raw);

  updateCalibration(smooth);

  const adjusted = applyBaseline(smooth);
  const score    = computeScore(adjusted);
  const risk     = getRiskLevel(score);

  // 5. Update score + status
  scoreDisplay.textContent = `${Math.round(score * 100)}%`;
  statusBadge.textContent  = risk.label;
  statusBadge.style.color  = risk.color;

  if (risk.level !== lastRiskLevel) {
    logEvent(`Risk changed to ${risk.label} (${Math.round(score * 100)}%).`);
    lastRiskLevel = risk.level;
  }
}
