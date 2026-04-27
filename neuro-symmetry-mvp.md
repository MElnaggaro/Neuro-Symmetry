# 🧠 Neuro-Symmetry — MVP
## Minimum Viable Product: What to Build First

> **Goal:** Working demo in hands within 2–4 weeks. Prove the concept. Win the room.

---

## 🎯 MVP Scope (What's In / What's Out)

| Feature | MVP | Later |
|---------|-----|-------|
| Face landmark detection | ✅ | — |
| Geometric asymmetry score | ✅ | — |
| Live webcam feed | ✅ | — |
| Risk level indicator (3 levels) | ✅ | — |
| Real-time score display | ✅ | — |
| Personalized baseline (2-min calibration) | ✅ | — |
| Texture fusion | ❌ | v2 |
| Temperature calibration | ❌ | v2 |
| XAI breakdown | ❌ | v2 |
| FAST protocol logic | ❌ | v2 |
| Speech / arm detection | ❌ | v3 |
| Edge deployment | ❌ | v2 |
| Synthetic training pipeline | ❌ | v2 |

---

## 🏗️ MVP Stack (Minimal, Fast)

```
Browser (no install needed)
    ↓
MediaPipe Face Mesh  (WASM — runs in browser)
    ↓
Vanilla JS + Canvas  (score overlay)
```

> **Simplest path:** Run everything in the browser. No backend needed for MVP.

---

## 📁 MVP File Structure

```
neuro-symmetry-mvp/
├── index.html          ← single entry point
├── app.js              ← all logic here
├── style.css           ← minimal UI
└── README.md
```

That's it. Four files. One working demo.

---

## 💻 Core Code (Copy-Paste Ready)

### index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Neuro-Symmetry MVP</title>
  <link rel="stylesheet" href="style.css">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
</head>
<body>
  <div id="app">
    <h1>🧠 Neuro-Symmetry</h1>
    <div id="video-container">
      <video id="video" autoplay muted playsinline></video>
      <canvas id="overlay"></canvas>
    </div>
    <div id="dashboard">
      <div id="score-display">--</div>
      <div id="status-badge">Initializing...</div>
      <div id="calibration-btn">
        <button onclick="startCalibration()">🎯 Calibrate Baseline</button>
      </div>
      <div id="calibration-status"></div>
    </div>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

---

### app.js

```javascript
// ─── CONFIG ───────────────────────────────────────────────
const KEY_POINTS = {
  mouth_left:  61,
  mouth_right: 291,
  eye_left:    33,
  eye_right:   263,
  nose_tip:    1,
  brow_left:   70,
  brow_right:  300,
};

const THRESHOLDS = { normal: 0.88, mild: 0.70 };

// ─── STATE ────────────────────────────────────────────────
let baseline      = null;
let calibrating   = false;
let calibFrames   = [];
const CALIB_SECS  = 120;   // 2 minutes
const FPS         = 30;
const CALIB_TOTAL = CALIB_SECS * FPS;

// ─── MEDIAPIPE SETUP ──────────────────────────────────────
const video   = document.getElementById("video");
const canvas  = document.getElementById("overlay");
const ctx     = canvas.getContext("2d");

const faceMesh = new FaceMesh({ locateFile: f =>
  `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });

faceMesh.setOptions({
  maxNumFaces:            1,
  refineLandmarks:        true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence:  0.7,
});

faceMesh.onResults(onResults);

const camera = new Camera(video, {
  onFrame: async () => { await faceMesh.send({ image: video }); },
  width: 640, height: 480,
});
camera.start();

// ─── FEATURE EXTRACTION ───────────────────────────────────
function extractFeatures(landmarks) {
  const get = (name) => landmarks[KEY_POINTS[name]];

  // Mouth deviation (x-offset from nose)
  const nosX     = get("nose_tip").x;
  const mouthMid = (get("mouth_left").x + get("mouth_right").x) / 2;
  const mouthDev = Math.abs(mouthMid - nosX);

  // Eye closure asymmetry (y-distance from nose)
  const eyeL    = Math.abs(get("eye_left").y  - get("nose_tip").y);
  const eyeR    = Math.abs(get("eye_right").y - get("nose_tip").y);
  const eyeAsym = Math.abs(eyeL - eyeR);

  // Brow height asymmetry
  const browL    = Math.abs(get("brow_left").y  - get("nose_tip").y);
  const browR    = Math.abs(get("brow_right").y - get("nose_tip").y);
  const browAsym = Math.abs(browL - browR);

  return [mouthDev, eyeAsym, browAsym];
}

// ─── SCORING ──────────────────────────────────────────────
function computeScore(features) {
  const weights = [0.5, 0.3, 0.2];
  const error   = features.reduce((sum, f, i) => sum + weights[i] * f, 0);
  return Math.exp(-error * 10);
}

function getRiskLevel(score) {
  if (score >= THRESHOLDS.normal) return { level: 0, label: "🟢 Normal",    color: "#22c55e" };
  if (score >= THRESHOLDS.mild)   return { level: 1, label: "🟡 Mild",      color: "#eab308" };
  return                                 { level: 2, label: "🔴 High Risk", color: "#ef4444" };
}

// ─── CALIBRATION ──────────────────────────────────────────
function startCalibration() {
  calibrating = true;
  calibFrames = [];
  document.getElementById("calibration-status").textContent =
    "📍 Hold neutral face for 2 minutes...";
}

function updateCalibration(features) {
  if (!calibrating) return;
  calibFrames.push(features);

  const pct = Math.round((calibFrames.length / CALIB_TOTAL) * 100);
  document.getElementById("calibration-status").textContent =
    `Calibrating... ${pct}%`;

  if (calibFrames.length >= CALIB_TOTAL) {
    baseline = calibFrames[0].map((_, i) =>
      calibFrames.reduce((s, f) => s + f[i], 0) / calibFrames.length
    );
    calibrating = false;
    document.getElementById("calibration-status").textContent =
      "✅ Baseline set!";
  }
}

function applyBaseline(features) {
  if (!baseline) return features;
  return features.map((f, i) => Math.abs(f - baseline[i]));
}

// ─── CANVAS OVERLAY ───────────────────────────────────────
function drawMesh(landmarks) {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(0,255,180,0.6)";

  Object.values(KEY_POINTS).forEach(idx => {
    const lm = landmarks[idx];
    ctx.beginPath();
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// ─── MAIN LOOP ────────────────────────────────────────────
function onResults(results) {
  if (!results.multiFaceLandmarks?.length) return;

  const landmarks = results.multiFaceLandmarks[0];
  const raw       = extractFeatures(landmarks);
  const adjusted  = applyBaseline(raw);
  const score     = computeScore(adjusted);
  const risk      = getRiskLevel(score);

  updateCalibration(raw);
  drawMesh(landmarks);

  document.getElementById("score-display").textContent =
    `${Math.round(score * 100)}%`;

  const badge       = document.getElementById("status-badge");
  badge.textContent = risk.level === 2
    ? "🔴 HIGH RISK — Seek medical attention"
    : risk.label;
  badge.style.color = risk.color;
}
```

---

### style.css

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0f172a;
  color: #f1f5f9;
  font-family: 'Segoe UI', sans-serif;
  display: flex;
  justify-content: center;
  padding: 2rem;
}

#app { text-align: center; max-width: 720px; width: 100%; }

h1 { font-size: 2rem; margin-bottom: 1.5rem; letter-spacing: 0.05em; }

#video-container {
  position: relative;
  display: inline-block;
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid #334155;
}

video, canvas {
  display: block;
  width: 640px;
  height: 480px;
}

canvas { position: absolute; top: 0; left: 0; }

#dashboard {
  margin-top: 1.5rem;
  background: #1e293b;
  border-radius: 12px;
  padding: 1.5rem;
}

#score-display {
  font-size: 4rem;
  font-weight: 700;
  color: #38bdf8;
  line-height: 1;
}

#status-badge {
  font-size: 1.4rem;
  font-weight: 600;
  margin: 0.75rem 0;
  transition: color 0.3s;
}

button {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 0.6rem 1.4rem;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  margin-top: 0.5rem;
}

button:hover { background: #2563eb; }

#calibration-status {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #94a3b8;
}
```

---

## 🚀 How to Run (60 seconds)

```bash
# Option A — open directly in Chrome
# Just double-click index.html

# Option B — local server (recommended, fixes camera permissions)
npx serve .
# Open http://localhost:3000
```

> Chrome requires HTTPS or localhost for camera access. Use `npx serve` if the direct open doesn't work.

---

## 📋 Demo Script (2 minutes)

```
00:00  Open browser → camera activates automatically
00:10  Score: 94–98% | Status: 🟢 Normal
00:20  Click "Calibrate Baseline"
       → (for live demo: skip the 2 min, just show the button exists)
00:40  Stand neutral → score holds stable
00:50  Simulate droop (pull left mouth corner down slightly)
       → Score: 94% → 71% → 58%
       → Badge turns RED: "HIGH RISK"
01:05  Relax face → score recovers to green
01:15  "Detected asymmetry within 500ms using only a webcam."
01:30  Show roadmap: texture layer, XAI, FAST protocol coming next
```

---

## 📈 Roadmap

```
MVP  ──────────────►  v2  ──────────────►  v3
(Now)                (Month 2)             (Month 4)

✅ Geometry score    + Texture fusion      + Speech slur
✅ Live webcam       + Temp. calibration   + Arm drift
✅ 3-level risk      + XAI breakdown       + Edge deploy
✅ Calibration       + FAST protocol       + Clinical trial
✅ Browser demo      + Confirmation layer  + Regulatory path
```

---

## ✅ Definition of Done (MVP)

- [ ] Webcam opens without any install
- [ ] Face mesh dots render on screen
- [ ] Score updates live with < 100ms lag
- [ ] Score drops visibly when droop is simulated
- [ ] Risk badge changes color correctly
- [ ] Calibration button exists and works
- [ ] Runs in Chrome / Edge / Firefox

---

> **This MVP proves one thing:**
> *A standard webcam can detect facial asymmetry in real time — no hardware, no hospital, no delay.*

---

*Build it. Show it. Iterate.*
