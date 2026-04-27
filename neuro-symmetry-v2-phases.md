# Neuro-Symmetry v2.0 — Implementation Phases & Planning

---

## Overview

| Phase | Name | Duration | Goal |
|-------|------|----------|------|
| 1 | Foundation & Environment | Week 1 | Repo, tooling, data pipelines working |
| 2 | Vision & Landmark Engine | Week 2 | Face detection, landmarks, normalization |
| 3 | Feature Engineering | Week 3 | Geometric + texture features extracted |
| 4 | AI Model — Training | Week 4 | Classifier trained, temperature-calibrated |
| 5 | Backend API | Week 5 | FastAPI wired end-to-end, all engines live |
| 6 | Intelligence Layers | Week 6 | Trajectory, FAST protocol, XAI, confirmation |
| 7 | Frontend | Week 7 | React UI, real-time display, all panels |
| 8 | Integration & Testing | Week 8 | E2E tests, metric validation, edge cases |
| 9 | Edge Deployment | Week 9 | TFLite export, mobile / Raspberry Pi |
| 10 | Clinical Validation & Polish | Week 10 | Demo, metrics, regulatory checklist |

---

## Phase 1 — Foundation & Environment

**Goal:** Everything compiles, dependencies are pinned, data loads.

### Tasks

- [ ] Create monorepo structure (`frontend/`, `backend/`, `ai/`, `datasets/`)
- [ ] Python environment — `pyproject.toml` or `requirements.txt`
  - `fastapi`, `uvicorn`, `mediapipe`, `opencv-python`, `torch`, `scipy`, `numpy`
- [ ] Node environment — `package.json` in `frontend/`
  - `react`, `@tensorflow/tfjs`, `@mediapipe/face_mesh`, `recharts`
- [ ] Dataset downloaders / loaders
  - `celeba_loader.py` — CelebA (202k images, normal pretraining)
  - `affectnet_loader.py` — AffectNet (450k images, expression filtering)
  - `palsy_loader.py` — Facial Palsy clinical cohorts
  - `synthetic_generator.py` — Synthetic droop augmentation stub
- [ ] CI skeleton — linting (`ruff`, `eslint`), type checks (`mypy`, `tsc`)
- [ ] Git repo initialized, `.gitignore`, `README.md` stub

### Deliverable
`python -m datasets.celeba_loader` runs without error and outputs sample count.

---

## Phase 2 — Vision & Landmark Engine

**Goal:** Given a frame, get 468 landmarks, normalized and mirrored.

### Files
- `backend/api/landmark_engine.py`
- `backend/api/input_quality.py`

### Tasks

#### 2a — Input Quality Gate (`input_quality.py`)
- [ ] `InputQualityChecker.check(frame, landmarks) -> QualityStatus`
- [ ] Face detection confidence check (`landmarks.score < 0.80`)
- [ ] Lighting check — mean luminance < 40 → `UNRELIABLE`
- [ ] Occlusion check — key landmark visibility < 75% → `UNRELIABLE`
- [ ] Motion blur check — Laplacian variance < 80 → `UNRELIABLE`
- [ ] Pose angle check → `DEGRADED`
- [ ] Unit tests for all 5 failure modes

#### 2b — Landmark Engine (`landmark_engine.py`)
- [ ] MediaPipe Face Mesh initialization (468 landmarks)
- [ ] Face rotation normalization (align eyes horizontally)
- [ ] Scale normalization via interpupillary distance (IPD)
- [ ] Mirror point mapping arrays:
  ```
  left_points  = [33, 160, 158, 133, 153, 144, ...]
  right_points = [263, 387, 385, 362, 380, 373, ...]
  ```
- [ ] Key point extraction (`KEY_POINTS` dict — 7 anatomical anchors)
- [ ] Unit test: normalized face has eyes at same Y, nose at center X

### Deliverable
Feed a test image → get `QualityStatus.OK` + 7-point landmark dict back.

---

## Phase 3 — Feature Engineering

**Goal:** 50 engineered features extracted per frame, reproducible.

### Files
- `backend/api/texture_engine.py`
- Feature extraction module (can live in `landmark_engine.py` or separate)

### Tasks

#### 3a — Geometric Features
- [ ] Mouth deviation — angular offset vs. nose axis
- [ ] Eye closure asymmetry — Eye Aspect Ratio (EAR) delta
- [ ] Eyebrow lift difference — Y-offset bilateral
- [ ] Nasolabial fold depth — curvature estimation via curve fitting
- [ ] Cheek symmetry index — mirror distance bilateral
- [ ] Compose into `numpy` feature vector (shape: `[N_geometric]`)

#### 3b — Texture Features (`texture_engine.py`)
- [ ] Crop face ROI from frame
- [ ] Split: `left_patch = face[:, :w//2]`, flip right patch
- [ ] `cv2.matchTemplate(left, right_flipped, TM_CCOEFF_NORMED)` → `texture_score`
- [ ] Captures: muscle tension, skin deformation, periorbital droop

#### 3c — Fused Symmetry Error
- [ ] Implement formula:
  `symmetry_error = α * Σ(wᵢ * ||Lᵢ - Mirror(Rᵢ)||) + β * (1 - S_texture)`
  with α = 0.65, β = 0.35
- [ ] Score function: `score = exp(-symmetry_error) ∈ (0, 1]`
- [ ] Validate weights on held-out samples

#### 3d — Feature Vector Assembly
- [ ] Stack geometric + texture into 50-dim `numpy` array
- [ ] Feature names list (for XAI later)
- [ ] Unit test: neutral face → score > 0.90

### Deliverable
`extract_features(frame) -> np.ndarray[50]` runs in < 20ms on CPU.

---

## Phase 4 — AI Model Training & Calibration

**Goal:** Calibrated 3-class classifier (Normal / Mild / Severe), ECE < 5%.

### Files
- `ai/model.py`
- `ai/train.py`
- `ai/calibrate_temperature.py`
- `ai/calibrate_baseline.py`
- `ai/synthetic_augment.py`
- `ai/export_tflite.py`

### Tasks

#### 4a — Model Architecture (`model.py`)
- [ ] PyTorch `nn.Module`:
  ```
  Input(50) → Dense(128, ReLU) → Dropout(0.3)
            → Dense(64,  ReLU) → Dropout(0.2)
            → Dense(32,  ReLU)
            → Logits(3)
  ```
- [ ] `CalibratedModel` wrapper with learnable temperature `T`

#### 4b — Training (`train.py`)
- [ ] Data pipeline — load CelebA (normal), palsy/stroke (pathological)
- [ ] Synthetic augmentation via `synthetic_augment.py` (droop simulation)
- [ ] Train/val/test split (70/15/15)
- [ ] Training loop — CrossEntropyLoss, Adam, LR scheduler
- [ ] Checkpoint saving (best val loss)
- [ ] Logging — loss, AUC-ROC, F1 per epoch

#### 4c — Temperature Calibration (`calibrate_temperature.py`)
- [ ] After main training, freeze base model weights
- [ ] Optimize `T` on val set only using LBFGS
- [ ] Target: ECE < 5%
- [ ] Reliability diagram plot (pre vs. post calibration)

#### 4d — Personalized Baseline (`calibrate_baseline.py`)
- [ ] 2-minute calibration session per user
- [ ] Compute `baseline_mean`, `baseline_std` over feature vectors
- [ ] Z-score anomaly: `z = (current - mean) / (std + 1e-6)`
- [ ] Persist baseline per user ID

#### 4e — Synthetic Augmentation (`synthetic_augment.py`)
- [ ] Warp mesh to simulate: unilateral droop, ptosis, nasolabial flattening
- [ ] Severity levels: mild (5–15%), severe (25–50%) displacement
- [ ] Generate unlimited samples from CelebA normals

#### 4f — TFLite Export (`export_tflite.py`)
- [ ] Export calibrated model → TFLite (float16 quantization)
- [ ] Target: < 3MB model size
- [ ] Benchmark: < 50ms inference on Raspberry Pi 4

### Target Metrics

| Metric | Target |
|--------|--------|
| AUC-ROC | > 0.95 |
| Sensitivity | > 0.92 |
| Specificity | > 0.90 |
| F1 Score | > 0.91 |
| ECE | < 5% |

### Deliverable
`python ai/train.py` → `model_calibrated.pt` + `model.tflite`. Metrics printed to stdout.

---

## Phase 5 — Backend API

**Goal:** FastAPI server processes frames end-to-end, returns structured JSON.

### Files
- `backend/api/main.py`
- `backend/api/change_detection.py`
- `backend/api/decision_engine.py`
- `backend/api/confirmation_layer.py`

### Tasks

#### 5a — API Structure (`main.py`)
- [ ] `POST /analyze` — accepts base64 frame or multipart image
- [ ] `WebSocket /stream` — continuous frame stream, push results
- [ ] `POST /calibrate` — start user baseline calibration session
- [ ] `GET /health` — quality gate status

#### 5b — Request Pipeline
```
Frame → InputQualityChecker
      → LandmarkEngine
      → FeatureExtractor
      → CalibratedModel (inference)
      → ZScoreBaseline (anomaly score)
      → TemporalEngine (rolling window)
      → ChangeDetection
      → TrajectoryEngine
      → ClinicalRules (FAST protocol)
      → DecisionEngine
      → ConfirmationLayer
      → XAIEngine
      → JSON response
```

#### 5c — Change Detection (`change_detection.py`)
- [ ] Rolling 60-frame deque at 30fps
- [ ] `delta = current_score - previous_score`
- [ ] Trigger if `delta < -0.25` within `time_elapsed < 1.0s`

#### 5d — Decision Engine (`decision_engine.py`)
- [ ] 4-level classification:
  - Level 0: score > 0.90 → Normal
  - Level 1: 0.70–0.90 → Mild
  - Level 2: 0.50–0.70 → High Risk
  - Level 3: < 0.50 OR COLLAPSE trajectory → Critical

#### 5e — Confirmation Layer (`confirmation_layer.py`)
- [ ] On alert trigger: capture next 10 frames
- [ ] Count frames where `score < threshold`
- [ ] Dispatch if `confirmed >= 7` (70% agreement)
- [ ] Else: log false positive, reset alert

#### 5f — Response Schema
```json
{
  "quality": "OK",
  "score": 0.87,
  "anomaly_score": 1.4,
  "risk_level": "MILD",
  "trajectory": "STABLE",
  "affected_side": "LEFT",
  "onset_seconds": null,
  "confidence": 0.92,
  "xai": [
    {"feature": "mouth_deviation", "contribution": 0.38, "level": "HIGH"},
    {"feature": "texture_anomaly", "contribution": 0.27, "level": "MEDIUM"}
  ],
  "alert": false
}
```

### Deliverable
`curl -X POST /analyze` with a test image returns valid JSON in < 100ms.

---

## Phase 6 — Intelligence Layers

**Goal:** Trajectory engine, FAST clinical protocol, and XAI all integrated.

### Files
- `backend/api/temporal_engine.py`
- `backend/api/trajectory_engine.py`
- `backend/api/clinical_rules.py`
- `backend/api/xai_engine.py`

### Tasks

#### 6a — Temporal Engine (`temporal_engine.py`)
- [ ] `deque(maxlen=60)` — 60-frame rolling window (~2s at 30fps)
- [ ] Rolling mean and variance
- [ ] Feed into trajectory engine and change detection

#### 6b — Trajectory Engine (`trajectory_engine.py`)
- [ ] `classify_trajectory(score_window)` using `scipy.stats.linregress`
- [ ] States: `STABLE`, `LINEAR_DECLINE`, `SUDDEN_DROP`, `OSCILLATING`, `COLLAPSE`
- [ ] `detect_sudden_drop(scores, threshold=0.25, window=30)`
  - baseline = mean of scores before last 30 frames
  - drop = baseline - min(recent)
  - return `drop > 0.25`
- [ ] Unit tests per trajectory pattern

#### 6c — Clinical Rules — FAST Protocol (`clinical_rules.py`)
- [ ] `RiskLevel` enum: NORMAL, MILD, HIGH_RISK, CRITICAL
- [ ] `apply_fast_logic(vision_score, speech_slur=None, arm_drift=None)`
- [ ] FAST Signal Matrix:

  | Face | Speech | Arm | Risk |
  |------|--------|-----|------|
  | Normal | — | — | NORMAL |
  | Mild | — | — | MILD |
  | Mild | Slurred | — | HIGH_RISK |
  | Severe | — | — | HIGH_RISK |
  | Any | Slurred | Drift | CRITICAL |
  | Severe | Slurred | — | CRITICAL |

#### 6d — XAI Engine (`xai_engine.py`)
- [ ] `explain_prediction(features, weights, score)` → ranked contributions
- [ ] Per-feature: `contribution = weight * feature_value`
- [ ] Sort by `abs(contribution)` descending
- [ ] Map to `HIGH / MEDIUM / LOW` level labels
- [ ] Output includes affected side (LEFT / RIGHT / BILATERAL)

### Deliverable
Feed simulated droop frame → API returns `trajectory: COLLAPSE`, correct XAI ranked list, `risk_level: CRITICAL`.

---

## Phase 7 — Frontend

**Goal:** React app with real-time camera feed and all 7 display panels live.

### Files
- `frontend/src/components/FaceMeshOverlay.jsx`
- `frontend/src/components/SymmetryGauge.jsx`
- `frontend/src/components/ScoreGraph.jsx`
- `frontend/src/components/HeatmapOverlay.jsx`
- `frontend/src/components/XAIBreakdown.jsx`
- `frontend/src/components/TrajectoryLabel.jsx`
- `frontend/src/components/RiskIndicator.jsx`

### Tasks

#### 7a — Camera & WebSocket Connection
- [ ] `getUserMedia` → video element
- [ ] Frame capture at 30fps → send to backend `/stream` WebSocket
- [ ] Receive structured JSON response per frame

#### 7b — Face Mesh Overlay (`FaceMeshOverlay.jsx`)
- [ ] Canvas overlay on video element
- [ ] Draw 468 MediaPipe landmarks in real time
- [ ] Highlight asymmetric zones in red/yellow based on XAI output

#### 7c — Symmetry Gauge (`SymmetryGauge.jsx`)
- [ ] Animated dial 0–100%
- [ ] Color: green (> 90%), yellow (70–90%), red (50–70%), flashing red (< 50%)

#### 7d — Asymmetry Heatmap (`HeatmapOverlay.jsx`)
- [ ] Per-zone deviation color map overlaid on face
- [ ] Zones: forehead, left/right eye, left/right cheek, mouth, chin
- [ ] Color intensity proportional to deviation magnitude

#### 7e — Score Graph (`ScoreGraph.jsx`)
- [ ] Last 60 frames displayed (using `recharts` LineChart)
- [ ] Trajectory annotation: label `COLLAPSE`, `DECLINE`, etc. on chart
- [ ] Red threshold line at 0.70 and 0.50

#### 7f — XAI Breakdown Panel (`XAIBreakdown.jsx`)
- [ ] Ranked bar chart of per-feature contributions
- [ ] Color-coded: red = HIGH, yellow = MEDIUM, green = LOW
- [ ] Update live per frame

#### 7g — Trajectory Label (`TrajectoryLabel.jsx`)
- [ ] Large badge: `STABLE` / `LINEAR_DECLINE` / `SUDDEN_DROP` / `COLLAPSE`
- [ ] Animate on state change

#### 7h — Risk Indicator (`RiskIndicator.jsx`)
- [ ] Four-state display: `🟢 Normal` / `🟡 Mild` / `🔴 High Risk` / `🆘 Critical`
- [ ] Audio alert on CRITICAL state
- [ ] Show onset time and affected side on HIGH_RISK+

### Deliverable
Browser opens camera, all 7 panels update live. Simulated droop demo runs end-to-end in < 500ms from onset to alert.

---

## Phase 8 — Integration Testing & Metric Validation

**Goal:** All target metrics met. No silent failures. False alarm rate < 0.1/hr.

### Tasks

#### 8a — End-to-End Tests
- [ ] Happy path: neutral face → 30s → no alert
- [ ] Droop simulation → alert fires in < 500ms
- [ ] Low light → `UNRELIABLE` status, no alert
- [ ] Occlusion (mask) → `UNRELIABLE`, no alert
- [ ] Motion blur → `UNRELIABLE`, no alert
- [ ] Single-frame droop (blink) → confirmation layer rejects false positive

#### 8b — Metric Validation
- [ ] Run test set through full pipeline
- [ ] Report: AUC-ROC, Sensitivity, Specificity, F1, ECE, False Alarm Rate
- [ ] Temporal stability: neutral face 60s, score σ < 0.02

#### 8c — Load Testing
- [ ] Backend: 30 frames/sec sustained for 10 minutes
- [ ] WebSocket latency: p99 < 50ms backend processing
- [ ] Memory: no leak over 10-minute session

#### 8d — Failure Mode Audit
- [ ] All `QualityStatus.UNRELIABLE` paths tested
- [ ] Confirm no emergency dispatch from UNRELIABLE frames

### Deliverable
Test report: all 8 evaluation metrics at or above target. False alarm rate < 0.1/hr confirmed.

---

## Phase 9 — Edge Deployment

**Goal:** System runs offline on Raspberry Pi 4 / iOS / Android.

### Tasks

#### 9a — TFLite Model Finalization
- [ ] Float16 quantization → < 3MB
- [ ] Benchmark on Raspberry Pi 4: < 50ms inference
- [ ] INT8 quantization fallback if needed

#### 9b — Raspberry Pi 4 Setup
- [ ] FastAPI backend on Pi 4 (Python 3.11, Raspberry Pi OS)
- [ ] PiCamera2 input pipeline
- [ ] Test full pipeline at 30fps: confirm < 500ms end-to-end

#### 9c — Mobile (iOS / Android)
- [ ] React Native wrapper or TFLite Flutter plugin
- [ ] On-device inference (no network required)
- [ ] Privacy: no PHI transmitted, all processing local

#### 9d — Offline Mode
- [ ] Backend packaged for standalone execution
- [ ] Local alert (audio/visual) without internet
- [ ] Baseline stored locally per device

### Deliverable
Pi 4 runs demo: camera input → emergency alert in < 500ms, fully offline.

---

## Phase 10 — Clinical Validation, Demo & Polish

**Goal:** Demo-ready, paper-ready, regulatory checklist complete.

### Tasks

#### 10a — Demo Choreography
```
Step 1: Neutral face (15s)
        → Score: 97% | Trajectory: STABLE | Status: 🟢
        → XAI: all features GREEN

Step 2: Simulate left-side droop
        → Heatmap activates LEFT
        → Graph: sudden cliff visible
        → Trajectory: COLLAPSE
        → XAI: Mouth deviation RED | Texture YELLOW
        → Output: "Left-side droop | 92% calibrated | onset: 0.8s"
        → Status: 🆘 EMERGENCY (fired < 500ms)
```

#### 10b — Regulatory Checklist
- [ ] FDA SaMD (Class II) — documentation for US commercialization path
- [ ] HIPAA — confirm no PHI transmitted (on-device processing audit)
- [ ] CE Mark (EU MDR) — European clinical use requirements noted
- [ ] IRB Approval — documentation template for clinical validation study

#### 10c — Paper / Report Sections
- [ ] Abstract: problem, method, results
- [ ] System architecture diagram (from spec)
- [ ] Metrics table (ECE, AUC-ROC, false alarm rate, latency)
- [ ] Reliability diagram (calibration curve)
- [ ] FAST protocol integration rationale
- [ ] Limitations and future work (speech slurring integration)

#### 10d — Code Quality
- [ ] All modules have docstrings and type hints
- [ ] `ruff` linting passes
- [ ] `mypy --strict` passes on backend
- [ ] `tsc --noEmit` passes on frontend

#### 10e — Final README
- [ ] Architecture overview
- [ ] Setup instructions (backend + frontend + edge)
- [ ] Dataset download and preprocessing steps
- [ ] Training and calibration commands
- [ ] Demo run instructions
- [ ] Regulatory notice

### Deliverable
Live demo runs flawlessly. Metrics report attached. Regulatory checklist filled.

---

## Dependency Map

```
Phase 1 (Foundation)
    └── Phase 2 (Vision + Quality Gate)
            └── Phase 3 (Feature Engineering)
                    └── Phase 4 (AI Training)     ← parallel with Phase 5 start
                    └── Phase 5 (Backend API)
                            └── Phase 6 (Intelligence Layers)
                                    └── Phase 7 (Frontend)
                                            └── Phase 8 (Integration Tests)
                                                        └── Phase 9 (Edge)
                                                                └── Phase 10 (Polish)
```

Phases 4 and 5 can overlap once Phase 3 is done — model training runs overnight while API skeleton is built.

---

## Critical Path (Minimum Viable Demo)

If timeline is compressed, this sequence produces a working demo fastest:

| Step | What | Why |
|------|------|-----|
| 1 | Phase 2 (landmarks + quality gate) | Nothing works without this |
| 2 | Phase 3 (geometric features only, skip texture) | 80% of signal |
| 3 | Phase 4 (train model, skip TFLite) | Need classifier |
| 4 | Phase 5 (API — `/analyze` endpoint only) | Backend callable |
| 5 | Phase 6 (trajectory + decision engine only) | Core alerting |
| 6 | Phase 7 (score graph + risk indicator only) | Minimal visible UI |
| 7 | Phase 8 (happy path + droop test only) | Demo confidence |

Texture engine, XAI, FAST protocol, mobile deployment, and full test suite can follow after the demo milestone.

---

*Plan generated for Neuro-Symmetry v2.0 — 10-week world-class implementation track.*
