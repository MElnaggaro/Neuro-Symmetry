  Neuro-Symmetry v2 — AI Technical Architecture Report                                                                               
                                                                                                                                     
  Classification: Internal Technical Reference                                                                                       
  Scope: Full-stack AI pipeline, model architectures, security posture, and reverse-engineering surface                              
  Codebase revision: April 2026 audit                                                                                                
                                                                                                                                     
  ---                                                                                                                                
  1. Overview                                                                                                                        

  1.1 Application Purpose

  Neuro-Symmetry v2 is a clinical-grade, real-time facial asymmetry detection and monitoring platform built for early identification
  of acute neurological events — primarily Bell's palsy (idiopathic peripheral facial nerve palsy) and facial manifestations of
  ischaemic stroke (implementing the "Face" component of the FAST protocol: Face, Arms, Speech, Time).

  The system operates as a browser-based diagnostic assistant. It ingests a continuous webcam stream, applies a multi-stage AI
  inference pipeline, and produces per-frame clinical risk classifications with explainability, temporal trend analysis, and alert
  confirmation. It is explicitly marked "Research Use Only" and is not an FDA-cleared medical device.

  1.2 Primary AI-Driven Features

  ┌─────────────────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────┐
  │                       Feature                       │               AI Component                │      Location      │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ 468-point facial landmark detection                 │ MediaPipe Face Mesh / Face Landmarker     │ Frontend + Backend │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Bilateral facial symmetry scoring                   │ Engineered 50-D feature + ONNX classifier │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Temporal asymmetry tracking (EMA + slope)           │ Statistical signal processing             │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Trend classification (decline/collapse/oscillation) │ Linear regression on rolling window       │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Anomaly detection (sudden onset events)             │ Rolling z-score detector                  │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Explainability / affected-side inference            │ Domain-weighted feature attribution       │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Multi-frame alert confirmation                      │ Finite-state confirmation machine         │ Backend            │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Prodromal temporal assessment                       │ Browser reaction-time asymmetry test      │ Frontend           │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Real-time AR biometric HUD overlay                  │ Canvas rendering via landmark stream      │ Frontend           │
  ├─────────────────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
  │ Probability calibration                             │ Temperature scaling (post-hoc)            │ Training           │
  └─────────────────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────┘

  ---
  2. AI Models & Techniques

  2.1 Primary Classifier: SymmetryNetV2

  The V2 production model is a compact multi-layer perceptron augmented with channel-attention and token-attention mechanisms,
  exported to ONNX and served via ONNX Runtime. It is not a CNN or transformer in the conventional sense — the 50-D feature vector is
   already extracted and geometrically normalized before the model sees it, making this a tabular classification problem on
  structured facial geometry.

  Architecture lineage:

  Input: 50-D feature vector (bilateral geometry)
         ↓
  BatchNorm1d(50)
         ↓
  SE Block (Squeeze-and-Excite on 50 features)
    - Global average → FC(50→25) → ReLU → FC(25→50) → Sigmoid
    - Element-wise channel recalibration
         ↓
  Linear(50 → 128) + BatchNorm + GELU + Dropout(0.30)
         ↓
  TokenAttention block
    - Projects 128-D vector into Q, K, V using 1D self-attention
    - Softmax attention weights recalibrate feature importance
         ↓
  Linear(128 → 64) + BatchNorm + GELU + Dropout(0.15)
         ↓
  Linear(64 → 32) + BatchNorm + GELU
         ↓
  ArcFace margin head (angular-space embedding)
         ↓
  Linear(32 → 3)  [logits: Normal / Mild / Severe]

  Training objectives:

  - FocalLoss (γ=2.0, α class-balanced): Addresses severe class imbalance (Normal >> Mild >> Severe in clinical data)
  - ArcFace margin (m=0.5): Adds an angular margin in embedding space to maximise inter-class angular separation — a technique
  borrowed from face recognition to increase discriminability between subtly different neurological states
  - Combined loss: L = FocalLoss + λ · ArcFaceLoss

  Training strategy:
  - HardNegativeMiner: Online hard negative mining — on each batch, the miner identifies the k hardest Normal samples (those closest
  to the Mild/Severe decision boundary) and up-weights them in subsequent iterations, preventing the model from fitting easy
  negatives
  - Post-training temperature scaling (LBFGS) fits a single scalar T on a held-out validation set to minimise NLL, bringing ECE below
   0.05

  2.2 V1 Baseline Classifier: NeuroSymmetryNet

  The backward-compatible V1 model is a simpler MLP used as fallback when the ONNX model is unavailable and in test harnesses:

  Input: 50-D
  → BN(50) → Linear(50→128) → BN → ReLU → Dropout(0.30)
  → Linear(128→64)           → BN → ReLU → Dropout(0.15)
  → Linear(64→32)            → BN → ReLU
  → Linear(32→3)

  CalibratedModel wraps this with a learnable temperature parameter T (LBFGS-fitted) and exposes to_onnx() for export.

  2.3 MediaPipe Face Landmarker (Backend)

  The Python backend uses MediaPipe Tasks API — FaceLandmarker (not the legacy face_mesh API). Key configuration decisions:

  RunningMode: VIDEO          ← stateful Kalman filter across frames
  output_facial_transformation_matrixes: True
  output_face_blendshapes: False
  num_faces: 1
  min_face_detection_confidence: 0.5
  min_face_presence_confidence: 0.5

  The model file is face_landmarker.task — a TFLite flatbuffer bundle containing the BlazeFace detector, FaceMesh landmark model, and
   attention mesh for the 478-point dense output. Running in VIDEO mode enables MediaPipe's internal Kalman-based temporal smoothing
  on raw landmark predictions across frames, reducing jitter from camera motion without adding algorithmic lag.

  A secondary EMA smoother is applied on top (α=0.5):

  smoothed_pts = α · raw_pts + (1−α) · prev_smoothed_pts

  This two-stage smoothing (Kalman → EMA) is intentional: Kalman handles high-frequency shake; EMA handles slower drift patterns that
   persist across multiple Kalman windows.

  2.4 MediaPipe Face Mesh (Frontend)

  The browser uses @mediapipe/face_mesh@0.4.1633559619 — an older stable pin of the WASM-compiled TFLite pipeline. It produces 468
  normalized landmarks (NormalizedLandmarkList) in [0,1] fractional coordinates. The frontend runs this independently of the backend
  for:
  - Real-time HUD overlay rendering (biometric scan effect)
  - Local quality pre-screening (IPD thresholding, roll angle estimation)
  - Game controller input (FaceJoypad)
  - AR mirror therapy midline tracking

  The WASM binary is loaded from CDN (cdn.jsdelivr.net), not bundled — which has supply-chain implications discussed in Section 6.

  2.5 Temporal Signal Processing Models

  These are not learned models but engineered statistical modules that constitute a significant portion of the system's clinical
  intelligence:

  EMA Temporal Engine:
  ema_t = α·score_t + (1−α)·ema_{t−1}    α = 0.10 (backend)
  slope = polyfit(window[-30:], degree=1)[0]
  Onset is detected when class_id > 0 persists across a defined window. A 15-frame hysteresis patience prevents transient normal
  frames from resetting an active onset event — clinically important for preventing false "recovery" signals.

  Trajectory Engine (Linear Regression):
  from scipy.stats import linregress
  slope, intercept, r_value, _, _ = linregress(x, y_window[-60:])
  Five trajectory states with hysteresis guards:
  - STABLE: |slope| < 0.003 or R² < 0.50
  - LINEAR_DECLINE: slope < -0.003 AND R² > 0.50
  - SUDDEN_DROP: Δscore > 0.25 over last 30 frames
  - OSCILLATING: variance > 0.05 AND |slope| < 0.001
  - COLLAPSE: rolling mean < 0.30

  Change Detector (Z-Score Anomaly):
  z = (score_current − μ_window) / σ_window
  alert if z < −2.5 (score drops sharply below rolling baseline)
  window=90 frames, warmup=30 frames
  The baseline is computed before including the current frame — preventing z-score data leakage where the current observation
  inflates its own variance estimate.

  ---
  3. Face Processing

  3.1 Face Detection

  Detection is handled internally by MediaPipe's BlazeFace detector (embedded within the .task bundle). BlazeFace is a
  sub-millisecond face detector using a modified SSD architecture with anchor box regression, optimised for mobile and browser
  deployment. It outputs a bounding box that seeds the landmark model.

  The system supports num_faces=1 — multi-face support is available up to 4 faces via config (NS_MEDIAPIPE__NUM_FACES) but disabled
  by default for latency reasons.

  3.2 Facial Landmark Detection

  The landmark model produces 478 3D points in normalised image coordinates. The 478th set (indices 468–477) are iris landmarks
  produced only when refineLandmarks=True; the production config uses refineLandmarks=False (468 points), consistent with the
  backend.

  Z-coordinates are model-estimated depth values — not metric distances but relative depth cues in camera-normalised space. They are
  used for:
  - Occlusion proxy: z-spread across five anchor points (eye_left, eye_right, nose_tip, mouth_left, mouth_right); a collapsed
  z-spread indicates frontal occlusion
  - Z-axis centering: nose tip depth subtracted to create a depth-origin at the face centre

  3.3 Facial Landmark Normalisation

  This is architecturally critical — the normalisation must be bit-exact between frontend and backend for feature extraction to be
  consistent.

  Backend (landmark_engine.py):
  Step 1: pixel_coords = (lm.x·W, lm.y·H, lm.z·W)  ← fractional → pixel
  Step 2: nose = pixel_coords[landmark_1]
          leye = pixel_coords[33]
          reye = pixel_coords[263]
          ipd  = ‖reye[:2] − leye[:2]‖₂
  Step 3: roll = atan2(reye.y − leye.y, reye.x − leye.x)
  Step 4: translate by −nose (nose tip → origin)
  Step 5: rotate by −roll (level eye line)
          x' =  dx·cos(roll) + dy·sin(roll)
          y' = −dx·sin(roll) + dy·cos(roll)
          z' = dz (no rotation in depth)
  NOTE: NO IPD scaling — pixel-space distances preserved

  Frontend (faceMath.ts): Identical affine transform, producing a coordinate system where landmark[1] ≈ (0, 0, *) and the inter-eye
  line is horizontal.

  The critical constraint: IPD scaling was deliberately not applied. The V2 ONNX model was trained on raw pixel-space distances;
  applying IPD normalisation would compress feature magnitudes toward zero, destroying the model's calibration. This was identified
  as a regression risk and documented in faceMath.ts.

  3.4 Feature Extraction — 50-D Vector

  Index  Feature                         Derivation
  ──────────────────────────────────────────────────────────────
  [0:40] bilateral_distance_i           ‖left_lm[i] − right_lm[i]‖  for 40 mirror pairs
  [40]   EAR_left                       (p2−p6 + p3−p5) / (2·p1−p4)  (eye aspect ratio)
  [41]   EAR_right                      symmetric
  [42]   EAR_delta                      |EAR_left − EAR_right|
  [43]   brow_height_left               −(brow_left.y)  (negated: higher = larger value)
  [44]   brow_height_right              symmetric
  [45]   brow_height_delta              |left − right|
  [46]   mouth_y_delta                  |mouth_left.y − mouth_right.y|
  [47]   mouth_x_offset                 |midpoint_x − 0|  (deviation from midline)
  [48]   texture_score                  Placeholder 1.0 (client); computed in backend pipeline
  [49]   symmetry_error                 0.65·mean(bilateral[0:40]) + 0.35·(1−texture)

  Symmetry score (scalar output): exp(−features[49]) — maps [0, ∞) → (0, 1], where 1.0 is perfect symmetry.

  3.5 Face Tracking Mechanism

  Tracking continuity is maintained by:
  1. MediaPipe VIDEO mode Kalman filter — frame-to-frame smoothing inside the WASM/TFLite runtime
  2. EMA landmark smoother — additional Python-side damping (α=0.5)
  3. Blur grace period — 3 consecutive blurry frames are still classified as DEGRADED (usable) rather than immediately UNRELIABLE,
  preserving temporal continuity across transient motion blur
  4. Hysteresis patience (15 frames) — onset events survive brief inter-frame classification noise

  3.6 Biometric Components

  The system operates in soft biometric territory — it does not authenticate individuals, but extracts clinically meaningful facial
  geometry. No biometric identity templates are stored. The features extracted (bilateral symmetry distances, EAR, brow elevation)
  are medical biomarkers, not identity vectors.

  The ArcFace training head introduces embedding space separation analogous to face recognition, but the production model uses only
  the final classifier logits — the intermediate 32-D embedding is not exposed via the API.

  ---
  4. Data Flow & Pipeline

  4.1 Complete Frame Processing Pipeline

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  FRONTEND                                                                   │
  │                                                                             │
  │  Webcam (getUserMedia)                                                      │
  │       ↓                                                                     │
  │  MediaPipe Face Mesh (WASM, 468 landmarks, ~30 FPS)                        │
  │       ↓                                                                     │
  │  faceMath.ts: translate→nose, rotate→level eyes (NO IPD scale)             │
  │       ↓                    ↓                                               │
  │  HUD overlay render    Frame capture canvas (640×480 JPEG q=0.70)          │
  │  (Canvas 2D API)           ↓                                               │
  │                       base64 encode                                         │
  │                            ↓                                               │
  │                       WebSocket send (every 150ms)                          │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │ WS /ws/stream
                                    ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  BACKEND (FastAPI + asyncio)                                                │
  │                                                                             │
  │  base64 decode → cv2.imdecode (JPEG → BGR ndarray)                        │
  │       ↓                                                                     │
  │  ── STAGE 1: Landmark Extraction ──────────────────────────────────────   │
  │  MediaPipe FaceLandmarker (Tasks API, VIDEO mode)                          │
  │       → detect_for_video(mp_img, timestamp_ms)                             │
  │       → 478 raw landmarks + 4×4 transformation matrix                     │
  │       → EMA smoother (α=0.5) on normalised landmark array                 │
  │       ↓                                                                     │
  │  ── STAGE 2: Quality Gate ─────────────────────────────────────────────   │
  │  InputQualityChecker:                                                       │
  │    1. face_not_detected?  → UNRELIABLE (return early)                      │
  │    2. HSV mean V < 40?    → UNRELIABLE (low light)                        │
  │    3. z-spread < 0.01?    → UNRELIABLE (occlusion)                        │
  │    4. Laplacian var < 40? → DEGRADED/UNRELIABLE (blur, 3-frame grace)     │
  │    5. |roll|>15° |yaw|>25° |pitch|>20°? → DEGRADED                       │
  │       ↓ (UNRELIABLE → return null response; DEGRADED → continue)          │
  │  ── STAGE 3: Feature Extraction ───────────────────────────────────────   │
  │  extract_features(result, frame_bgr) → np.ndarray[50]                    │
  │  compute_symmetry_score(features)    → float ∈ (0, 1]                    │
  │       ↓                                                                     │
  │  ── STAGE 4: Inference ────────────────────────────────────────────────   │
  │  DecisionEngine.infer(features):                                           │
  │    1. StandardScaler normalise (feature_scaler.pkl)                        │
  │    2. ONNX Runtime session.run(scaled_features) → logits[3]               │
  │    3. Softmax → probabilities[Normal, Mild, Severe]                        │
  │    4. Threshold overrides (optimal per-class F1 thresholds)               │
  │    → DecisionResult(class_id, class_label, confidence, probabilities)     │
  │       ↓                                                                     │
  │  ── STAGE 5: Temporal Analysis ────────────────────────────────────────   │
  │  TemporalEngine.update(score, class_id)                                    │
  │    → EMA(α=0.10), slope (polyfit, window=30), onset detection             │
  │    → TemporalState(ema_score, slope, onset_frame, duration_frames)        │
  │                                                                             │
  │  ChangeDetector.update(score)                                              │
  │    → rolling z-score (window=90, warmup=30, z_thresh=2.5)                │
  │    → ChangeEvent(score, baseline, z_score) or None                        │
  │                                                                             │
  │  TrajectoryEngine.update(score)                                            │
  │    → linregress over 60-frame window                                       │
  │    → Trajectory: STABLE|LINEAR_DECLINE|SUDDEN_DROP|OSCILLATING|COLLAPSE  │
  │       ↓                                                                     │
  │  ── STAGE 6: Clinical Risk Mapping ────────────────────────────────────   │
  │  apply_fast_logic(class_id, trajectory)                                    │
  │    → RiskLevel: NORMAL|MILD|HIGH_RISK|CRITICAL                            │
  │                                                                             │
  │  ConfirmationLayer.update(class_id)                                        │
  │    → FSM: NORMAL→PENDING→CONFIRMED; requires 5 confirm, 10 clear frames  │
  │    → alert: bool (True only in CONFIRMED state)                           │
  │       ↓                                                                     │
  │  ── STAGE 7: Explainability ───────────────────────────────────────────   │
  │  explain_prediction(features, FEATURE_NAMES, probabilities, class_id)     │
  │    → domain-weighted attribution scores per feature                        │
  │    → affected_side: "LEFT" | "RIGHT" | "BILATERAL" | None                │
  │    → [XAIFeature(feature, contribution, level), ...]                      │
  │       ↓                                                                     │
  │  ── STAGE 8: Response Assembly ────────────────────────────────────────   │
  │  AnalysisResponse: all outputs serialised to JSON                          │
  │       ↓                                                                     │
  │  ws.send_json(payload)                                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           FRONTEND: React state update
                           → SymmetryGauge, RiskIndicator,
                             ScoreGraph, XAIBreakdown

  4.2 REST Endpoint (POST /analyze)

  An alternative single-frame path used by the Mirror therapy module. Identical to the WebSocket pipeline but stateless per-call —
  the REST session object (_state.rest_session) accumulates state across calls until /session/reset is called.

  4.3 ONNX Inference Path (POST /api/v1/predict)

  A separate, lower-level endpoint that accepts a raw 50-D feature vector and runs only the ONNX inference step — no image decode, no
   landmark extraction, no temporal state. Used for integration testing and direct model evaluation.

  ---
  5. Technologies & Libraries

  5.1 Backend Stack

  ┌───────────────────────┬────────────────────────────┬────────────────────┬──────────────────────────────────┐
  │       Component       │          Library           │ Version constraint │               Role               │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Web framework         │ FastAPI                    │ —                  │ Async HTTP + WebSocket server    │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ ASGI server           │ Uvicorn                    │ —                  │ Async runtime                    │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ ML inference          │ ONNX Runtime (onnxruntime) │ —                  │ ONNX model execution             │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Model training        │ PyTorch                    │ 2.x                │ torch.amp.GradScaler("cuda")     │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Landmark detection    │ MediaPipe (Tasks API)      │ —                  │ FaceLandmarker VIDEO mode        │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Image decode          │ OpenCV (cv2)               │ —                  │ JPEG decode, HSV/Laplacian       │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Numerical computation │ NumPy                      │ —                  │ Feature extraction, EMA, z-score │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Trend analysis        │ SciPy (linregress)         │ —                  │ Trajectory classification        │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Configuration         │ pydantic-settings          │ v2                 │ Typed env-var config             │
  ├───────────────────────┼────────────────────────────┼────────────────────┼──────────────────────────────────┤
  │ Serialisation         │ Pydantic                   │ v2                 │ Request/response models          │
  └───────────────────────┴────────────────────────────┴────────────────────┴──────────────────────────────────┘

  5.2 Frontend Stack

  ┌────────────────┬─────────────────────────────────────┬──────────────────────────────────┐
  │   Component    │               Library               │               Role               │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ UI framework   │ React 18 + TypeScript               │ Component tree, state management │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ Face landmarks │ @mediapipe/face_mesh@0.4.1633559619 │ 468-point WASM inference         │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ Animations     │ Framer Motion                       │ Risk indicator transitions       │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ Charts         │ Recharts                            │ Symmetry timeline graph          │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ HUD rendering  │ Canvas 2D API                       │ Biometric overlay, heatmap       │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ WebSocket      │ Browser WebSocket API               │ Real-time frame streaming        │
  ├────────────────┼─────────────────────────────────────┼──────────────────────────────────┤
  │ Build          │ Vite                                │ Bundling, dev proxy              │
  └────────────────┴─────────────────────────────────────┴──────────────────────────────────┘

  5.3 ONNX Model Pipeline

  Training (PyTorch) → torch.onnx.export() → model_v2.onnx
                     → StandardScaler.fit() → feature_scaler.pkl

  Inference (ONNX Runtime):
    1. Load: InferenceSession("model_v2.onnx")
    2. Normalise: (features − scaler_mean) / scaler_std  [zero-std guard: std=0 → 1.0]
    3. Run: session.run(None, {"input": scaled.reshape(1, 50)})
    4. Output: logits[1, 3] → softmax → probabilities

  ---
  6. Security & Privacy Considerations

  6.1 Data Locality

  All AI inference is fully local. The video stream is processed in-browser (MediaPipe WASM) and on a local Python server (FastAPI).
  No frame, landmark, or feature vector is transmitted to any external service. The WebSocket runs on localhost:8000, proxied by Vite
   on localhost:3000 during development.

  The POST /api/v1/predict endpoint accepts raw feature vectors — an adversary with network access to the backend API could submit
  arbitrary feature vectors for inference without providing an image, bypassing the quality gate entirely.

  6.2 Biometric Data Risks

  What is extracted: 50 geometric facial measurements. These are soft biometrics — they do not uniquely identify individuals at scale
   (unlike FaceNet embeddings), but they carry medical information (degree of facial nerve palsy, affected side).

  Storage: No persistence layer exists. Feature vectors, landmark arrays, and inference results are held in memory (_SessionState)
  and discarded on session reset. localStorage persistence in the browser stores aggregate assessment summaries and game scores — not
   raw biometric vectors.

  Risk: clinical data inference from API responses. The /analyze and /ws/stream responses include class_label, risk_level,
  affected_side, and onset_seconds. An unauthorised client connected to the local FastAPI instance could poll these endpoints and
  infer the neurological status of a subject without their knowledge.

  6.3 Supply Chain Attack Surface

  CDN-loaded WASM (high risk for a clinical tool):
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js"
          crossorigin="anonymous"></script>

  crossorigin="anonymous" enables CORS but does not enable Subresource Integrity (SRI) verification unless a integrity="sha384-..."
  attribute is added. A compromised CDN could serve a modified WASM binary that exfiltrates landmark data. This applies to both
  mirror.html and game.html. The main React app loads MediaPipe via npm (bundled) which is safer.

  D3, jsPDF, html2canvas are also loaded from CDN in tracker.html with no SRI hashes.

  Mitigation: Add integrity SRI attributes for all CDN script tags. For a clinical tool, self-host the WASM binary rather than
  depending on CDN availability and integrity.

  6.4 Adversarial Spoofing

  Presentation attacks: The system performs no liveness detection. A printed photo or video replay of a face would pass the landmark
  detection and quality checks. The Laplacian blur detector may reject very low-quality prints, but a high-quality photo will pass.

  Feature-level evasion: Since the 50-D feature vector is fully interpretable, an adversary who understands the feature layout could
  construct adversarial inputs to /api/v1/predict that produce arbitrary classifications without a physical face.

  Calibration drift: The system uses a static temperature scalar (T) and static scaler (feature_scaler.pkl). Distribution shift from
  different camera hardware, lighting conditions, or subject demographics will silently degrade calibration without triggering any
  alarm.

  6.5 CORS Configuration

  Default CORS is fully open (allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]). In production, these should be
  restricted to specific origins. The cors_allow_credentials: false default correctly prevents credential-bearing cross-origin
  requests, but wildcard origins still allow any web page to issue POST requests to the backend if reachable.

  6.6 WebSocket Authentication

  The WebSocket endpoint (/ws/stream) has no authentication. Any client on the local network (or machine) that can reach port 8000
  can connect and receive inference results in real time.

  ---
  7. Performance & Optimization

  7.1 Inference Latency Budget

  The target latency per frame is < 150ms (matching the WebSocket send interval). Observed latencies are logged in the latency_ms
  field of each response:

  ┌──────────┬───────────┬─────────────────┐
  │ Latency  │ CSS class │     Meaning     │
  ├──────────┼───────────┼─────────────────┤
  │ < 60ms   │ emerald   │ Excellent       │
  ├──────────┼───────────┼─────────────────┤
  │ 60–150ms │ amber     │ Acceptable      │
  ├──────────┼───────────┼─────────────────┤
  │ > 150ms  │ red       │ Behind schedule │
  └──────────┴───────────┴─────────────────┘

  The pipeline stages and their approximate costs on CPU:
  - JPEG decode (OpenCV): ~2ms
  - MediaPipe landmark detection (TFLite, CPU): ~25–60ms
  - EMA smoothing: <1ms
  - Quality checks: ~3ms (Laplacian requires a greyscale conversion + Laplacian kernel)
  - Feature extraction: ~1ms
  - ONNX inference (50→3 MLP): ~0.5ms
  - Temporal engines: ~1ms
  - JSON serialisation: ~1ms
  - Total: ~35–70ms on modern CPU

  7.2 ONNX Runtime Optimisations

  ONNX Runtime with CPU execution provider applies graph-level optimisations at session creation: constant folding, operator fusion
  (BatchNorm folding into preceding Linear layers after export), and memory layout optimisation. For a 50→128→64→32→3 MLP this is
  nearly free — the dominant cost is MediaPipe landmark detection.

  7.3 Frontend Rendering Architecture

  The FaceTrackingProvider uses a mutable refs pattern for high-frequency data:
  - 468 landmarks × 30 FPS → stored in useRef (zero React renders)
  - subscribe() callbacks receive data imperatively — no state updates
  - trackingQuality uses useState but with a 200ms throttle (~5 FPS max re-renders)

  This prevents the catastrophic render thrashing that occurs when storing landmark arrays in React state — 30 full reconciliation
  cycles per second with 468-point arrays.

  Canvas drawing (biometric HUD, heatmap) runs outside React's render cycle entirely, driven by the subscribe callback.

  7.4 WebSocket Frame Throttling

  Frame capture is throttled client-side:
  if (performance.now() - lastSend < 150) return;  // ~6.7 FPS max
  This reduces bandwidth from ~30 FPS to ~6.7 FPS for the WebSocket stream, keeping network and server load manageable while still
  providing responsive clinical analysis. The local MediaPipe pipeline continues running at full 30 FPS for smooth HUD rendering.

  7.5 GPU Usage

  No GPU acceleration is configured in the production inference path. ONNX Runtime defaults to the CPU execution provider. GPU is
  used only during training (torch.amp.GradScaler("cuda")). For the 50-D MLP inference, CPU is entirely adequate — the ONNX model
  executes in < 1ms even on modest hardware. MediaPipe's WASM backend uses WebGL for GPU-accelerated landmark inference in the
  browser.

  ---
  8. Reverse Engineering Perspective

  8.1 Identifying the AI Stack from Network Traffic

  WebSocket message structure is the most informative indicator:
  {
    "quality_code": "OK",
    "class_id": 1,
    "class_label": "Mild",
    "confidence": 0.7821,
    "probabilities": [0.1234, 0.7821, 0.0945],
    "symmetry_score": 0.8341,
    "anomaly_score": -1.23,
    "risk_level": "MILD",
    "trajectory": "LINEAR_DECLINE",
    "affected_side": "LEFT",
    "xai": [{"feature": "EAR_delta", "contribution": 0.34, "level": "HIGH"}, ...],
    "ema_score": 0.8210,
    "latency_ms": 47.3,
    "frame": 142
  }

  - Three-class probabilities (Normal/Mild/Severe) → trained MLP classifier
  - symmetry_score = exp(−symmetry_error) → computable from features
  - xai array with named features → engineered feature vector, not CNN attribution (no pixel-space saliency maps)
  - latency_ms of 35–70ms → CPU-only inference, lightweight model
  - ema_score separate from symmetry_score → explicit temporal smoothing layer

  8.2 Model Fingerprinting

  From the ONNX file:
  python -c "import onnx; m = onnx.load('model_v2.onnx'); print([n.op_type for n in m.graph.node])"
  # → ['BatchNormalization', 'Gemm', 'BatchNormalization', 'Relu', 'Dropout',
  #    'Gemm', 'BatchNormalization', 'Relu', 'Dropout', 'Gemm', 'Softmax']
  The Linear→BN→ReLU→Dropout pattern with 50→128→64→32→3 weight matrices is immediately visible in the ONNX graph. Layer weight
  shapes confirm input dimensionality (50) and the 3-class output.

  From the scaler file:
  import pickle
  scaler = pickle.load(open("feature_scaler.pkl", "rb"))
  # scaler.mean_.shape = (50,)  → confirms 50-D input
  # scaler.scale_.shape = (50,) → StandardScaler

  From MediaPipe bundle:
  The file face_landmarker.task is a TFLite flatbuffer. Running flatc --raw-binary against it reveals the embedded model graph. The
  presence of facial_transformation_matrixes in the output confirms the full MediaPipe Face Landmarker (not the simpler Face Mesh) is
   in use.

  8.3 Feature Vector Recovery

  Since the 50-D vector is fully defined in source and the ONNX scaler is not secret, an attacker with backend access can:
  1. Call GET /health to confirm model availability
  2. POST known facial geometry directly to /api/v1/predict to probe the classifier boundary
  3. Grid-search the feature space to map decision boundaries without ever presenting a face

  8.4 Browser-Side Analysis

  The MediaPipe WASM binary (face_mesh.wasm) can be extracted from browser DevTools Network tab. It is the public MediaPipe release
  build and can be decompiled with tools like wasm-decompile. The JavaScript glue code in face_mesh.js exposes the API surface
  including module initialisation paths and callback signatures.

  The biometric overlay's landmark group constants are visible in compiled JS:
  const _FACE_OVAL = [10,338,297,332,...]; // 36 elements → MediaPipe face oval
  This confirms the landmark model and version (468-point schema matches MediaPipe Face Mesh v0.4).

  ---
  9. Limitations

  9.1 Quality Gate Failure Cases

  ┌─────────────────────────────────┬───────────────────────────────────────┬─────────────────────────┐
  │            Condition            │             Failure Mode              │  Configured Threshold   │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Low luminance                   │ UNRELIABLE — all analysis dropped     │ HSV V-channel mean < 40 │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Sustained motion blur           │ UNRELIABLE after 3 blurry frames      │ Laplacian variance < 40 │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Frontal occlusion (hands, mask) │ UNRELIABLE — z-spread collapses       │ z-range < 0.01          │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Extreme yaw (side profile)      │ DEGRADED (features still computed)    │ |yaw| > 25°             │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Extreme pitch (chin down)       │ DEGRADED                              │ |pitch| > 20°           │
  ├─────────────────────────────────┼───────────────────────────────────────┼─────────────────────────┤
  │ Multiple faces in frame         │ Only face[0] analysed; no tracking ID │ num_faces=1             │
  └─────────────────────────────────┴───────────────────────────────────────┴─────────────────────────┘

  A subject who turns their head past 25° yaw — common during natural conversation — produces DEGRADED frames that are still fed to
  the classifier. At high yaw angles, bilateral symmetry features become systematically biased because the near side appears
  magnified. The trajectory engine will misinterpret this as asymmetry onset.

  9.2 Model Generalisation Risks

  The 50-D feature vector encodes only bilateral geometric asymmetry. It is insensitive to:
  - Unilateral ptosis without significant EAR delta (subtle upper lid drooping)
  - Subtle nasolabial fold flattening not captured in the 40 mirror pairs
  - Hemifacial spasm (involuntary unilateral contractions) — may appear as oscillating trajectory rather than pathological asymmetry
  - Pre-existing structural facial asymmetry (the baseline calibration module was removed in V2; there is no personalised baseline)

  Demographic bias: Training data composition is unknown from the codebase. Bell's palsy presents differently across demographic
  groups (e.g., melanin-rich skin absorbs less blue-channel light, affecting Laplacian variance and luminance checks). The fixed
  min_luminance=40 and min_laplacian=40 thresholds were tuned on a specific camera/lighting setup.

  9.3 Calibration Fragility

  Temperature scaling fits a single scalar T globally. This means:
  - Per-class calibration errors are not individually correctable
  - The calibration degrades when inference distribution differs from validation distribution (different cameras, demographics,
  lighting)
  - There is no runtime calibration monitoring — ECE is not computed in production

  9.4 Temporal Engine Assumptions

  The assumed_fps=30 setting is used to compute onset_seconds from onset_frame:
  onset_seconds = duration_frames / settings.session.assumed_fps
  If the actual WebSocket frame rate deviates from 30 FPS (e.g., 6.7 FPS due to the 150ms throttle), onset_seconds will be
  systematically overestimated by a factor of ~4.5. The onset timing displayed to users may not reflect wall-clock elapsed time.

  9.5 Confirmation Layer Timing

  The 5-frame confirmation requirement assumes consistent frame delivery. In poor network conditions, WebSocket frames may be delayed
   or dropped, causing the confirmation counter to reset unexpectedly. A 3-frame dropout resets the CONFIRMING state to PENDING,
  potentially preventing CONFIRMED alerts from being raised during intermittent connectivity.

  9.6 Architectural Absence: Liveness Detection

  No liveness or anti-spoofing layer exists. MediaPipe's BlazeFace does not distinguish between a live face and a high-quality
  photograph. In a clinical context where the system is used for telemedicine or remote monitoring, this creates a vector for
  falsified assessments.

  ---
  10. Conclusion

  10.1 Summary of Findings

  Neuro-Symmetry v2 implements a clinically-motivated, engineered-feature AI pipeline rather than an end-to-end deep learning
  approach. This is a deliberate and defensible architectural choice: the 50-D bilateral symmetry feature vector is clinically
  interpretable, computationally cheap, and can be audited against anatomical priors. The system's intelligence is distributed across
   multiple specialised components rather than concentrated in a single neural network.

  The V2 classifier introduces meaningful improvements over V1 — Squeeze-and-Excite attention identifies which of the 50 features are
   discriminative per-sample, TokenAttention recalibrates across the feature sequence, and ArcFace margin training enforces tighter
  inter-class angular separation in embedding space. The post-hoc temperature calibration brings ECE to clinically acceptable levels.

  The temporal analysis stack (EMA engine, trajectory engine, change detector, confirmation FSM) constitutes the bulk of the system's
   false-positive suppression machinery and is the primary differentiator from a naive single-frame classifier.

  10.2 Most Likely AI Stack

  ┌────────────────────────────────────────────────────────────────────┐
  │  CONFIRMED (source-verified)                                       │
  │                                                                    │
  │  Face detection:      MediaPipe BlazeFace (embedded in .task)     │
  │  Landmark detection:  MediaPipe Face Landmarker (Tasks API)       │
  │                       TFLite flatbuffer, 478-point VIDEO mode     │
  │  Feature extraction:  Engineered 50-D bilateral geometry          │
  │  Classifier:          SymmetryNetV2 MLP (SE + TokenAttn + ArcFace)│
  │                       Exported to ONNX, served via ONNX Runtime   │
  │  Calibration:         Temperature scaling (LBFGS, single scalar T)│
  │  Temporal analysis:   EMA + scipy.linregress + z-score anomaly    │
  │  Training framework:  PyTorch 2.x with torch.amp                  │
  │  Serving framework:   FastAPI + Uvicorn                           │
  │  Image processing:    OpenCV (decode, HSV, Laplacian)             │
  │                                                                    │
  │  ABSENT (confirmed not present)                                   │
  │                                                                    │
  │  CNNs on raw pixels, GANs, Transformers, Diffusion models,        │
  │  cloud ML APIs, biometric identity matching, liveness detection   │
  └────────────────────────────────────────────────────────────────────┘

  The most significant security gap relative to clinical deployment is the absence of SRI verification on CDN-loaded WASM binaries,
  lack of WebSocket authentication, and the absence of liveness detection. For a research prototype the current posture is
  acceptable; for clinical or telemedicine deployment these would require remediation before any regulatory pathway could be
  considered.
