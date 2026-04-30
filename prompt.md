# SYSTEM-CRITICAL REFACTORING DIRECTIVE
Act as a Staff/Principal Software Architect and Lead Computer Vision Engineer. Our Neuro-Symmetry V2 application (React + FastAPI + ONNX) requires a surgical, enterprise-grade refactoring. You must resolve 4 critical architectural flaws while injecting advanced UI/UX features. Do not hallucinate; follow these exact constraints.

# ROOT CAUSE ANALYSIS & BUGS TO FIX
1. **React Render Thrashing:** The current global context pushes 468 MediaPipe landmarks into React `useState` at 30 FPS. This causes massive reconciliation blocking, UI lag, and failing calibrations.
2. **Data Distribution Drift:** Previous math normalizations applied IPD scaling. The ONNX model was NOT trained on this scale, causing it to output `0.000` features and random predictions.
3. **Canvas Mapping Mismatch:** Modules like the AR Mirror are incorrectly drawing on the canvas using mathematically aligned coordinates instead of raw screen coordinates, causing disjointed visuals during head tilts.
4. **Sentinel Crash & Legacy Code:** The Sentinel module is throwing `Camera is not defined` because it instantiates its own legacy camera loop instead of consuming the global context.

# YOUR MISSION: 4-PHASE ARCHITECTURE REFACTORING

## PHASE 1: Mathematical Face Alignment (faceMath.ts)
Create `frontend/src/utils/faceMath.ts`.
- **Rule:** We must ONLY apply Translation and Roll Rotation. **DO NOT APPLY SCALING** (to preserve the ONNX model's expected baseline distribution).
- **Translation:** Shift all landmarks so the tip of the nose (Landmark 1) is exactly at `(0, 0)`.
- **Rotation:** Calculate the angle between the outer eye corners (Landmarks 33 and 263). Apply a 2D affine rotation matrix so the eye-line is perfectly horizontal (angle = 0).
- Extract and return the specific 50-D features expected by `model_v2.onnx`.

## PHASE 2: High-Performance Context (FaceTrackingProvider.tsx)
Rewrite the global context to completely eliminate React render thrashing.
- **The Mutable Refs Pattern:** Store `rawScreenLandmarks` and `mathLandmarks` inside a React `useRef`. NEVER store high-frequency frame data in `useState`.
- Initialize `@mediapipe/face_mesh` and the Camera exactly ONCE globally.
- In the `onResults` callback, silently update the refs.
- Use a throttled `useState` (e.g., max 5 FPS) ONLY for UI state like `trackingQuality` ("OK" | "UNRELIABLE").
- Expose the Refs and the UI state to the app.

## PHASE 3: Decoupled Logic & Module Upgrades
Any module doing logic/inference MUST read from `mathLandmarks` (via Refs). Any module doing UI Canvas drawing MUST read from `rawScreenLandmarks` (via Refs).
1. **Sentinel.tsx:** Strip out `camera_utils`. Implement a `requestAnimationFrame` loop inside a `useEffect` that continuously polls the context's `mathLandmarks` ref for the micro-lag (blink/smile) detection without triggering React renders.
2. **Mirror.tsx (Feathering):** Use `rawScreenLandmarks` to calculate the midline. Implement a horizontal gradient mask (Alpha Blending/Edge Feathering) so the mirrored healthy side blends seamlessly into the affected side without a harsh seam.
3. **Joypad.tsx (Combo):** Poll `mathLandmarks`. Add a 3-second "Hold" multiplier. If a sustained eyebrow raise or smile is held for 3 seconds, trigger a 2x multiplier with a canvas visual effect.
4. **Tracker.tsx (PDF):** Upgrade the report generation using `html2canvas` + `jsPDF` to take a high-fidelity snapshot of the D3.js chart and personalized care plan.
5. **DialogManager.tsx:** Replace scattered modals. Build a centralized manager using `ReactDOM.createPortal` and `AnimatePresence` to prevent overlapping popups.

## EXECUTION
Output the complete, strictly-typed, production-ready code sequentially:
1. `faceMath.ts` (Corrected math with detailed linear algebra comments).
2. `FaceTrackingProvider.tsx` (Mutable Refs pattern).
3. `Sentinel.tsx` (Refactored to poll Refs).
4. `Mirror.tsx` (Feathered rendering).
5. `InferenceService.ts` (Ensure it polls `mathLandmarks` and throttles API calls to max 2/sec).

Write extremely clean code. Remove all deadcode and legacy initializations.

# SYSTEM-CRITICAL DIRECTIVE: CORE ALGORITHMIC BUG FIXES
Act as a Principal MLOps and Backend Engineer. Excellent job on the previous UI/UX refactoring. However, a deep-dive architecture review has revealed 4 critical, silent logical/mathematical flaws in the Python backend. These bugs compromise the medical integrity of Neuro-Symmetry V2.

You must surgically fix these 4 Python files. Do not rewrite the entire files; only fix the specific logical flaws described below.

# BUG 1: Broken Threshold Fallback (decision_engine.py)
- **The Bug:** If the ONNX model is missing/fails, the system falls back to a raw threshold calculation (`exp(-features[49])`). However, the system currently applies `StandardScaler` to the features BEFORE the fallback check. The fallback threshold evaluates scaled data (mean 0, var 1), producing arbitrary classifications.
- **The Fix:** Branch the logic. Extract the raw symmetry score BEFORE applying the scaler. Apply the scaler ONLY if routing the features to the ONNX `InferenceSession`. Pass the raw score to `_threshold_decision`.

# BUG 2: Temporal Tracking Instability (temporal_engine.py)
- **The Bug:** The `_onset_frame` tracks symptom duration. Currently, a single false-negative `Normal` frame instantly resets the `_onset_frame` to `None`. In a live camera stream, standard jitter destroys continuous duration tracking.
- **The Fix:** Implement a "Patience / Grace Period" buffer. Introduce a `self.recovery_patience = 15` (frames) variable. When a `Normal` frame is detected, increment a counter. Only reset `_onset_frame` to `None` if the continuous `Normal` frame count exceeds the patience threshold.

# BUG 3: Data Leakage in Baseline Change Detection (change_detection.py)
- **The Bug:** The anomaly detector checks if the current score is `z_threshold` std deviations below the rolling mean. However, it appends the current score to the history buffer BEFORE calculating `mean` and `std`. An anomalous drop drags the mean down instantly and inflates the std, effectively suppressing the Z-score and masking the anomaly (Classic Data Leakage).
- **The Fix:** Calculate the `mean`, `std`, and the final Z-score using the history buffer *excluding* the current score (or calculate it *before* appending the new score to `self.history`).

# BUG 4: Inconsistent Z-Axis Normalization (landmark_engine.py)
- **The Bug:** The frontend (`faceMath.ts`) correctly centers the Z-axis relative to the nose tip `(lm.z - nose.z)`. The Python backend currently leaves the Z-axis absolute. While this cancels out in bilateral distance math, it is geometrically inconsistent.
- **The Fix:** Update `landmark_engine.py`. In the normalization step, explicitly center the Z-axis by subtracting the nose tip's Z coordinate from all landmarks, perfectly mirroring the TypeScript implementation.

# EXECUTION
Output the fully corrected Python code for:
1. `decision_engine.py`
2. `temporal_engine.py`
3. `change_detection.py`
4. `landmark_engine.py`

Write precise, highly-technical inline comments explaining the algorithmic fixes (e.g., "Preventing Z-score data leakage", "Temporal hysteresis applied").
