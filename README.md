<div align="center">

# Neuro-Symmetry

### Real-time facial asymmetry intelligence for early neurological risk screening

Neuro-Symmetry is a full-stack research prototype that combines webcam-based facial landmarking, engineered symmetry features, ONNX inference, temporal risk tracking, and an interactive React dashboard for stroke and facial palsy early-warning workflows.

<strong>Research use only. Not a medical diagnostic device.</strong>

<br/>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=111111)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=for-the-badge&logo=vite&logoColor=white)

![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Landmarks-00AEEF?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-not_configured-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-not_specified-lightgrey?style=for-the-badge)

<br/>

<a href="site/MINI%20DEMO.mp4"><strong>Watch Demo Video</strong></a>
<span>&nbsp;|&nbsp;</span>
<a href="site/index.html"><strong>Static Product Site</strong></a>
<span>&nbsp;|&nbsp;</span>
<a href="site/deck.html"><strong>Pitch Deck</strong></a>

</div>

---

## Project Overview

Neuro-Symmetry analyzes facial asymmetry signals from a live camera feed and turns them into structured risk indicators. The V2 system uses a React frontend for real-time visualization and a FastAPI backend for image analysis, landmark processing, feature extraction, model inference, temporal smoothing, and explainability.

The project exists to explore whether low-cost webcam streams can support early warning workflows around facial nerve asymmetry. It focuses on the "Face" component of FAST-style screening and Bell's palsy monitoring, while explicitly avoiding diagnostic claims.

Primary use cases:

- Real-time facial symmetry monitoring from a standard webcam.
- Research-grade experimentation with engineered facial geometry features.
- Demonstration of an end-to-end AI product: frontend, backend, model, tests, Docker, and demo assets.
- Assistive triage UI that highlights risk level, affected side, trajectory, confidence, and feature contributions.

---

## Key Features

| Feature | What It Does | Implementation |
|---|---|---|
| :zap: Real-time analysis | Streams camera frames to the backend at roughly 6 to 7 FPS | React, `getUserMedia`, Vite proxy, WebSocket `/ws/stream` |
| :brain: Facial landmark engine | Detects and normalizes face landmarks for geometry analysis | MediaPipe Face Landmarker, OpenCV, 478 landmarks |
| :bar_chart: 50-D feature vector | Converts landmarks into interpretable symmetry features | 40 bilateral mirror distances plus EAR, brow, mouth, texture, symmetry error |
| :rocket: ONNX inference | Runs calibrated 3-class model inference on CPU | ONNX Runtime, `model_v2.onnx`, scaler arrays, per-class thresholds |
| :shield: Quality gate | Suppresses unreliable predictions under poor input conditions | Low light, blur, occlusion proxy, roll/yaw/pitch checks |
| :hourglass_flowing_sand: Temporal intelligence | Tracks trend, onset, EMA score, sudden drops, collapse, oscillation | `TemporalEngine`, `TrajectoryEngine`, `ChangeDetector` |
| :rotating_light: Confirmation layer | Avoids single-frame false alarms | Multi-frame `NORMAL -> PENDING -> CONFIRMED` state machine |
| :mag: Explainable AI | Surfaces top feature contributors and likely affected side | XAI ranking over feature regions and scalar asymmetry signals |
| :dart: Personal baseline | Builds a per-WebSocket baseline from calibration frames | 30-frame feature-space baseline, z-score anomaly scoring |
| :desktop_computer: Clinical-style dashboard | Shows live gauge, risk badge, heatmap overlay, timeline, XAI panel | React, Tailwind, Framer Motion, Recharts |
| :clipboard: Progress tracker | Persists session summaries and exports reports | Browser `localStorage`, Tracker route, jsPDF |
| :video_game: Rehab modules | Includes Sentinel, AR Mirror, and Face-Joypad experiences | Static HTML modules plus shared MediaPipe utilities |
| :whale: Containerized runtime | Runs frontend and backend together | `docker-compose.yml`, backend `Dockerfile.v2`, frontend Dockerfile |

---

## Tech Stack

| Layer | Technologies | Notes |
|---|---|---|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS | Main live dashboard and tracker SPA |
| UI and Motion | Framer Motion, Recharts, Sonner, cmdk, lucide-react, jsPDF | Animations, charts, command palette, toasts, PDF export |
| Browser Vision | MediaPipe Face Mesh, Canvas, WebRTC camera APIs | Local overlays, face math, Sentinel, Mirror, Face-Joypad; TensorFlow.js is listed as a dependency but MediaPipe drives the active camera paths |
| Backend API | FastAPI, Uvicorn, Pydantic Settings, WebSockets | REST, WebSocket streaming, OpenAPI docs |
| Computer Vision | MediaPipe Face Landmarker, OpenCV | Backend landmarking, image decode, quality checks |
| AI and ML | PyTorch, ONNX Runtime, NumPy, SciPy, joblib | V2 model training and CPU inference |
| Training Utilities | scikit-learn, matplotlib | Used by `ai/v2/train_v2.py`; install separately for retraining if missing |
| Data | Local NPZ feature caches, local dataset loaders | No committed raw datasets; `/Datasets` is ignored |
| Persistence | Browser `localStorage`, in-process ring buffers | No production database is configured |
| Infrastructure | Docker Compose, Python 3.12 slim, Node 20 Alpine | Local full-stack container workflow |
| Quality Tools | Pytest, Ruff, mypy, ESLint, TypeScript compiler | CI workflow is not currently checked in |

---

## Architecture

Neuro-Symmetry V2 is organized as a browser UI plus a Python inference service. The browser owns camera capture, high-frequency UI overlays, and local interaction modules. The backend owns robust image quality checks, server-side landmarking, feature extraction, model inference, temporal state, and risk logic.

```text
User webcam
    |
    v
React + Vite frontend
    |-- MediaPipe Face Mesh for local overlays and modules
    |-- Canvas heatmaps, gauges, tracker, triage UI
    |
    |  JPEG frame over WebSocket /ws/stream
    v
FastAPI backend
    |
    |-- InputQualityChecker
    |-- MediaPipe FaceLandmarker
    |-- 50-D FeatureExtractor
    |-- DecisionEngine
    |     |-- ONNX Runtime model path when available
    |     +-- threshold fallback when model artifact is missing
    |-- TemporalEngine + ChangeDetector + TrajectoryEngine
    |-- FAST clinical rule matrix + ConfirmationLayer
    +-- XAI contribution ranking
    |
    v
JSON analysis response
    |
    v
Dashboard gauge, risk label, timeline, heatmap, XAI panel, tracker history
```

### Component Breakdown

| Component | Responsibility | Key Files |
|---|---|---|
| Live dashboard | Camera view, overlays, risk UI, calibration, tracker persistence | `frontend/src/App.tsx` |
| Face tracking provider | Singleton browser camera and MediaPipe pipeline with ref-based frame data | `frontend/src/providers/FaceTrackingProvider.tsx` |
| WebSocket hook | Connects the UI to `/ws/stream`, sends frames, receives analysis | `frontend/src/hooks/useWebSocket.ts` |
| Backend application | API wiring, lifespan loading, REST, WebSocket, session state | `backend/api/main.py` |
| Landmark engine | MediaPipe Tasks wrapper and landmark normalization | `backend/api/landmark_engine.py` |
| Feature extractor | 50-D geometry and texture vector | `backend/api/feature_extractor.py` |
| Decision engine | ONNX Runtime inference with calibrated threshold priority | `backend/api/decision_engine.py` |
| XAI engine | Ranks top feature contributors and affected side | `backend/api/xai_engine.py` |
| Temporal intelligence | EMA, onset, slope, sudden drop, collapse, confirmation | `backend/api/temporal_engine.py`, `trajectory_engine.py`, `confirmation_layer.py` |
| Prodromal sentinel | Browser timing assessment endpoint and history buffer | `backend/api/prodromal.py` |
| V2 model training | SymmetryNetV2, loss functions, calibration, export | `ai/v2/` |

---

## Repository Structure

```text
Neuro-Symmetry/
|-- neuro-symmetry-v2/
|   |-- backend/
|   |   |-- api/                 # FastAPI routes, inference pipeline, quality gates, XAI, temporal engines
|   |   |-- core/                # Typed settings loaded from NS_* environment variables
|   |   +-- models/              # Local MediaPipe face_landmarker.task model artifact
|   |-- frontend/
|   |   |-- src/                 # React dashboard, providers, hooks, panels, tracker, modules
|   |   |-- public/              # Sentinel, Mirror, Face-Joypad static pages and shared browser utilities
|   |   |-- Dockerfile           # Node 20 Vite dev container
|   |   +-- package.json         # Frontend scripts and dependencies
|   |-- ai/
|   |   |-- v2/                  # SymmetryNetV2 architecture, training, losses, calibration, ONNX export
|   |   |-- cache/               # Local NPZ feature caches, ignored by Git
|   |   +-- checkpoints/         # Local model artifacts and metrics, ignored by Git
|   |-- datasets/               # CelebA, 300-W, YFP, AffectNet loaders and dataset path config
|   |-- tests/                  # Pytest coverage for engines, API, model utilities, failure modes
|   |-- Dockerfile.v2           # Backend inference service image
|   |-- docker-compose.yml      # Full-stack local orchestration
|   +-- pyproject.toml          # Python package metadata and dev dependencies
|-- neuro-symmetry-mvp/         # Legacy browser-first MVP prototype
|-- site/                       # Static marketing site, pitch deck, and tracked demo video
|-- package.json                # Root-level UUID dependency only
+-- .gitignore                 # Datasets, checkpoints, env files, virtualenvs, node modules
```

---

## Installation

### Prerequisites

| Requirement | Version | Why |
|---|---:|---|
| Python | 3.11+ | `pyproject.toml` requires Python 3.11 or newer |
| Node.js | 20 recommended | Frontend Dockerfile uses Node 20 Alpine |
| npm | 10+ recommended | Frontend dependency installation |
| Docker | Recent Docker Desktop or Docker Engine | Optional full-stack runtime |
| Webcam | Any browser-accessible camera | Required for live camera workflows |

### Backend Setup

```bash
cd neuro-symmetry-v2
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m backend.api.landmark_engine --download
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend health check:

```bash
curl http://127.0.0.1:8000/health
```

OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Frontend Setup

```bash
cd neuro-symmetry-v2/frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

### Docker Setup

```bash
cd neuro-symmetry-v2
docker compose up --build
```

Docker Compose starts:

| Service | URL | Notes |
|---|---|---|
| Backend | `http://localhost:8000` | FastAPI service and `/health` probe |
| Frontend | `http://localhost:3000` | Vite dev server with `/api` and `/ws` proxy |

---

## Configuration

The backend uses `pydantic-settings` with the `NS_` environment prefix. A `.env` file can be placed in `neuro-symmetry-v2/`, and Docker Compose can override the same variables.

### Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NS_LOG_LEVEL` | `INFO` | Backend logging level |
| `NS_API_TITLE` | `Neuro-Symmetry API` | FastAPI application title |
| `NS_API_VERSION` | `2.0.0` | FastAPI version string |
| `NS_DEBUG` | `false` | FastAPI debug flag |
| `NS_CORS_ORIGINS` | `*` | Allowed origins, CSV or JSON-array style depending on deployment |
| `NS_LANDMARKER_MODEL_PATH` | `backend/models/face_landmarker.task` | MediaPipe Face Landmarker model path |
| `NS_CHECKPOINTS_DIR` | `ai/checkpoints` | ONNX model, metrics, scaler, checkpoint directory |
| `NS_METRICS_PATH` | `ai/checkpoints/metrics_v2.json` | V2 metric metadata path |

### Tunable Analysis Variables

| Variable | Default | Description |
|---|---:|---|
| `NS_QUALITY__MIN_LUMINANCE` | `40.0` | Minimum mean HSV luminance |
| `NS_QUALITY__MIN_LAPLACIAN` | `40.0` | Blur threshold using Laplacian variance |
| `NS_QUALITY__MIN_Z_RANGE` | `0.01` | Occlusion proxy threshold |
| `NS_QUALITY__MAX_ROLL_DEG` | `15.0` | Maximum usable roll before degraded status |
| `NS_QUALITY__MAX_YAW_DEG` | `25.0` | Maximum usable yaw before degraded status |
| `NS_QUALITY__MAX_PITCH_DEG` | `20.0` | Maximum usable pitch before degraded status |
| `NS_SESSION__MIN_CALIB_FRAMES` | `30` | Frames required for personal baseline calibration |
| `NS_SESSION__ASSUMED_FPS` | `30.0` | FPS assumption used for onset duration math |
| `NS_CONFIRMATION__CONFIRM_FRAMES` | `5` | Consecutive alert frames required for confirmation |
| `NS_CONFIRMATION__CLEAR_FRAMES` | `10` | Consecutive normal frames required to clear confirmation |
| `NS_CHANGE__WARMUP` | `30` | Rolling baseline warmup frames |
| `NS_CHANGE__Z_THRESHOLD` | `2.5` | Z-score threshold for change detection |

### Example `.env`

```bash
NS_LOG_LEVEL=INFO
NS_CORS_ORIGINS=["http://localhost:3000"]
NS_QUALITY__MIN_LAPLACIAN=40
NS_SESSION__MIN_CALIB_FRAMES=30
NS_CONFIRMATION__CONFIRM_FRAMES=5
```

No external API keys are required by the current implementation.

---

## Running The Project

### Development Mode

```bash
# Terminal 1
cd neuro-symmetry-v2
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
cd neuro-symmetry-v2/frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

### Production-Oriented Build

The backend has a Docker image configured for Uvicorn workers. The frontend Dockerfile currently runs the Vite dev server, so a production deployment should build and serve static frontend assets with a reverse proxy.

```bash
cd neuro-symmetry-v2/frontend
npm run build
npm run preview
```

### Useful Frontend Commands

```bash
cd neuro-symmetry-v2/frontend
npm run lint
npm run lint:tokens
npm run typecheck
npm run build
```

### Useful Backend Commands

```bash
cd neuro-symmetry-v2
pytest
ruff check .
mypy backend ai datasets
```

---

## API Documentation

FastAPI exposes interactive OpenAPI docs at `/docs` once the backend is running.

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe and model availability flags |
| `POST` | `/api/v1/predict` | Raw 50-D feature-vector inference through V2 ONNX model |
| `POST` | `/analyze` | Single image upload analysis, multipart file input |
| `POST` | `/session/reset` | Reset REST rolling session state |
| `WS` | `/ws/stream` | Real-time frame analysis over WebSocket |
| `WS` | `/stream` | Alternate WebSocket route supported by backend |
| `POST` | `/prodromal/assess` | Prodromal temporal-asymmetry assessment |
| `GET` | `/prodromal/history` | In-process recent prodromal assessment history |

Vite development proxy maps frontend `/api/*` requests to the backend without the `/api` prefix, except for the backend's native `/api/v1/predict` route.

<details>
<summary><strong>API Examples</strong></summary>

### Health

```bash
curl http://127.0.0.1:8000/health
```

Example response shape, with values dependent on local model artifacts:

```json
{
  "status": "ok",
  "landmark_engine": true,
  "decision_engine": true,
  "v2_predict": {
    "model_path": ".../ai/checkpoints/model_v2.onnx",
    "model_exists": true,
    "metrics_path": ".../ai/checkpoints/metrics_v2.json",
    "metrics_exists": true
  }
}
```

### Raw Feature Prediction

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.28,0.28,0,0.12,0.12,0,0,0,1,0]}'
```

Example response shape, with values dependent on the submitted feature vector:

```json
{
  "class": "Normal",
  "probability": 0.97
}
```

### Single Image Analysis

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@frame.jpg"
```

Example response shape, with values dependent on the submitted frame:

```json
{
  "quality_code": "OK",
  "quality_reason": null,
  "class_id": 0,
  "class_label": "Normal",
  "confidence": 0.9812,
  "probabilities": [0.9812, 0.015, 0.0038],
  "symmetry_score": 0.9315,
  "anomaly_score": null,
  "risk_level": "NORMAL",
  "trajectory": "STABLE",
  "affected_side": "BILATERAL",
  "onset_seconds": null,
  "xai": [],
  "alert": false,
  "change_detected": false,
  "confirmation_state": "NORMAL",
  "ema_score": 0.9315,
  "latency_ms": 42.7
}
```

### WebSocket Stream

```json
{ "image": "<base64-jpeg-without-data-url-prefix>" }
```

Calibration message:

```json
{ "type": "calibrate", "image": "<base64-jpeg>" }
```

Reset message:

```json
{ "type": "reset" }
```

### Prodromal Assessment

```bash
curl -X POST http://127.0.0.1:8000/prodromal/assess \
  -H "Content-Type: application/json" \
  -d '{
    "right_eye": {"blink_close_ms": 1200, "detected": true},
    "left_eye": {"blink_close_ms": 1430, "detected": true},
    "right_smile": {"elevation_ms": 3600, "detected": true},
    "left_smile": {"elevation_ms": 3820, "detected": true},
    "symptom_flags": {"mastoid_pain": true, "hyperacusis": false, "taste_change": false}
  }'
```

</details>

---

## AI / ML Details

### Feature Engineering

The backend converts each usable face frame into a 50-dimensional feature vector.

| Slice | Features | Meaning |
|---|---|---|
| `0:40` | Bilateral mirror distances | One distance per left/right MediaPipe mirror pair |
| `40` | `ear_left` | Left eye aspect ratio |
| `41` | `ear_right` | Right eye aspect ratio |
| `42` | `ear_delta` | Absolute eye openness asymmetry |
| `43:45` | Brow heights | Left, right, and absolute delta |
| `46` | `mouth_y_delta` | Vertical mouth-corner asymmetry |
| `47` | `mouth_x_offset` | Mouth midpoint offset from facial midline |
| `48` | `texture_score` | Bilateral texture similarity using normalized cross-correlation |
| `49` | `symmetry_error` | `0.65 * mean(bilateral) + 0.35 * (1 - texture_score)` |

The fused live symmetry score is:

```text
symmetry_score = exp(-symmetry_error)
```

### Model Architecture

The V2 classifier is `SymmetryNetV2`, a lightweight tabular neural network for 3 classes: `Normal`, `Mild`, and `Severe`.

```text
Input(50)
  -> BatchNorm1d
  -> Linear(128) + LayerNorm + GELU
  -> SqueezeExcitation(128, reduction=4)
  -> TokenAttention(128, embed_dim=8, heads=4)
  -> Dropout(0.30)
  -> Linear(64) + GELU
  -> Dropout(0.15)
  -> Linear(3 logits)
```

Model details from code:

| Detail | Value |
|---|---:|
| Input dimension | 50 |
| Hidden dimension | 128 |
| Bottleneck dimension | 64 |
| Classes | 3 |
| Default dropout | 0.30 |
| Approximate parameter scale | About 42K parameters, per model docstring |
| Export format | ONNX, opset 17, logits-only graph |

### Training Pipeline

The V2 training script loads feature caches, optionally adds synthetic samples, trains the model, calibrates it, searches thresholds, and exports model artifacts.

| Stage | Implementation |
|---|---|
| Data loading | NPZ feature caches from CelebA, 300-W, AffectNet neutral, and YFP palsy data |
| Optional augmentation | Balanced synthetic 50-D samples from `ai/synthetic_augment.py` |
| Split | 70 percent train, 15 percent validation, 15 percent test after seeded shuffle |
| Scaling | `StandardScaler` fit on train split and persisted with checkpoint metadata |
| Sampling | `WeightedRandomSampler` using inverse class weights |
| Optimization | AdamW, learning rate `3e-4`, weight decay `1e-4`, cosine annealing |
| Loss | Combined Focal Loss and ArcFace angular margin loss |
| Default loss weights | Focal `0.60`, ArcFace `0.40` |
| Focal gamma | `2.0` by default |
| ArcFace margin | `0.45` by default |
| Label smoothing | `0.10` |
| Hard-negative mining | Rebuilds sampler every 15 epochs by default |
| Calibration | Temperature scaling with LBFGS and ECE reporting |
| Thresholding | Per-class F1 grid search with severity priority `Severe -> Mild -> Normal` |
| Export | `model_v2_calibrated.pt`, `model_v2.onnx`, `metrics_v2.json`, curves |

Retraining command:

```bash
cd neuro-symmetry-v2
pip install scikit-learn matplotlib
python -m ai.v2.train_v2 --epochs 80 --batch-size 512 --lr 3e-4
```

### Datasets

Raw datasets are intentionally not committed. The dataset loaders expect a local `Datasets/` directory beside `neuro-symmetry-v2`.

| Dataset | Expected Path | Purpose |
|---|---|---|
| CelebA | `Datasets/Celeba/` | Normal-class face data and landmark metadata |
| 300-W AFW | `Datasets/ibug_300W_large_face_landmark_dataset/afw/` | Landmark accuracy and normal data support |
| Yale Facial Palsy | `Datasets/YFP_Dataset/` | Mild and severe facial palsy labels |
| AffectNet YOLO | `Datasets/YOLO_format/` | Expression-labelled faces, neutral class used as normal data |

Dataset loader metadata in the repo documents:

| Dataset | Count Represented In Loader Code |
|---|---:|
| CelebA | 202,599 images |
| 300-W AFW | 200 images described in loader docs |
| YFP | 14,389 cropped facial-region images |
| AffectNet YOLO | 25,262 images |

---

## Performance / Metrics

The training script writes metrics to `ai/checkpoints/metrics_v2.json`. In this workspace, the latest local V2 metrics artifact reports the following internal test-distribution results. These are not clinical validation results.

| Metric | Value |
|---|---:|
| Accuracy | `0.998154` |
| F1 macro | `0.986023` |
| AUC-ROC | `0.997793` |
| Sensitivity | `0.982309` |
| Specificity | `0.999668` |
| Expected Calibration Error | `0.001794` |
| Input dimension | `50` |
| Training samples | `462,668` |
| Training elapsed time | `40,513.2s` |

Runtime guardrails present in tests:

| Guardrail | Test Coverage |
|---|---|
| Single-frame false positives | Confirmation layer prevents immediate alerts |
| Unreliable inputs | Dark, blurred, faceless, and low-quality frames must not alert |
| WebSocket session isolation | Each stream owns its own rolling state |
| Synthetic stability | Repeated neutral frames are tested for stable symmetry score variance |
| Backend latency target | Synthetic patched pipeline tests assert sub-200ms latency and p99 under 100ms |

Important caveat: the latency tests patch the landmark engine and use synthetic frames, so they validate code-path behavior rather than real-camera production throughput.

---

## Screenshots / Demo

| Asset | Location | Status |
|---|---|---|
| Demo video | [`site/MINI DEMO.mp4`](site/MINI%20DEMO.mp4) | Tracked in the repo |
| Static product site | [`site/index.html`](site/index.html) | Tracked in the repo |
| Interactive pitch deck | [`site/deck.html`](site/deck.html) | Tracked in the repo |
| PowerPoint deck | [`site/deck.pptx`](site/deck.pptx) | Tracked in the repo |
| UI screenshots | `docs/screenshots/` | Placeholder, no screenshot files are currently committed |

Suggested screenshot slots for a future docs pass:

| Slot | Recommended Capture |
|---|---|
| Live dashboard | Camera panel, symmetry gauge, risk label, XAI panel |
| Calibration flow | Baseline calibration ring and ready state |
| Tracker | KPI cards, symmetry-over-time chart, PDF export control |
| Sentinel | Blink/smile temporal asymmetry result screen |
| Mirror therapy | AR mirror overlay with therapy mode enabled |
| Face-Joypad | Game canvas and face-control sidebar |

---

## Challenges & Engineering Decisions

| Decision | Why It Matters |
|---|---|
| Ref-based browser frame pipeline | High-frequency landmarks are stored in mutable refs, avoiding React re-renders at camera frame rate |
| Backend-owned quality gate | Poor lighting, blur, occlusion, and pose issues return `UNRELIABLE` or `DEGRADED` instead of unsafe alerts |
| Engineered 50-D feature vector | Features map to anatomical concepts and are small enough for CPU inference |
| ONNX first inference path | Runtime does not require the PyTorch training stack for normal backend inference |
| Calibrated threshold priority | Severe and mild thresholds are checked before normal to avoid argmax hiding borderline abnormal cases |
| Multi-frame confirmation | Reduces alert flicker and protects against one bad frame triggering a critical UI state |
| Temporal trajectory modeling | Distinguishes stable asymmetry from sudden drop, collapse, linear decline, and oscillation |
| XAI aggregation by facial region | Feature explanations can be rendered as readable dashboard zones instead of opaque vector indices |
| No persisted clinical data store | Current prototype avoids pretending to be a production healthcare system; state is browser-local or in-memory |
| Explicit regulatory boundary | The code and docs frame the system as assistive research software, not FDA/CE/HIPAA-ready software |

---

## Future Improvements

| Priority | Improvement | Rationale |
|---|---|---|
| High | External clinical cohort validation | Current metrics are internal and not sufficient for medical claims |
| High | Demographic and device bias audit | Camera, lighting, skin tone, age, and facial morphology shifts can affect landmarks and features |
| High | Liveness and anti-spoofing layer | The current pipeline does not prevent photo/video replay attacks |
| High | Dynamic FPS measurement | Onset timing currently depends on configured FPS assumptions |
| Medium | Persistent backend storage | Replace in-memory prodromal history and browser-only tracker state with a real database |
| Medium | Production frontend container | Serve compiled `dist/` assets behind Nginx or another production web server |
| Medium | CI workflow | Add GitHub Actions for pytest, linting, type checks, and frontend build |
| Medium | Screenshots and docs assets | Add committed UI screenshots for GitHub rendering |
| Long term | EHR or telemedicine integration | Only after validation, privacy, consent, and regulatory gates are addressed |
| Long term | On-device deployment | Explore fully local inference with smaller artifacts and privacy-preserving UX |

---

## Testing

The Python test suite covers dataset loaders, landmark normalization, feature extraction, input quality, decision fallback, temporal logic, XAI, confirmation, API routes, WebSocket behavior, and failure modes.

```bash
cd neuro-symmetry-v2
pytest
```

Frontend checks:

```bash
cd neuro-symmetry-v2/frontend
npm run lint
npm run lint:tokens
npm run typecheck
npm run build
```

---

## Contribution Guide

Contributions should keep the project technically honest and medically conservative.

1. Open an issue or discussion for major changes to model behavior, risk logic, datasets, or clinical wording.
2. Keep frontend camera-loop work off React render state unless the UI genuinely needs to re-render.
3. Add tests for backend engines, API contracts, temporal state, and failure modes.
4. Do not commit raw datasets, private health data, `.env` files, or model checkpoints unless the licensing and release process are explicit.
5. Keep medical language research-focused and avoid diagnostic claims without external validation.

---

## Regulatory And Safety Notice

Neuro-Symmetry is an assistive research prototype. It is not a clinical diagnostic device, not a substitute for emergency care, and not cleared as FDA SaMD, CE-marked software, or HIPAA-compliant infrastructure.

If a person has sudden facial drooping, arm weakness, speech difficulty, confusion, severe headache, or other acute neurological symptoms, treat it as a medical emergency and contact local emergency services immediately.

---

## License

No license file is currently present in the repository. Until a license is added, default copyright restrictions apply and external reuse is not explicitly granted.
