Act as a Principal Machine Learning Engineer & MLOps Architect specializing in clinical-grade classification systems, imbalanced medical data, and real-time edge deployments.

Context: I am upgrading a facial symmetry classification system (3 classes: Normal [0], Mild [1], Severe [2]) to "Version 2". The current V1 model relies on a 50D MediaPipe Euclidean distance vector, trained on ~660K samples (heavily augmented), but struggles with a specific bottleneck.

📊 V1 Bottleneck (The Problem to Solve):
- F1-Macro is stuck at 0.8689 (Target > 0.92).
- Mild class Precision is severely low (0.59).
- Massive False Positives: 1,958 Normal misclassified as Mild, and 704 Normal misclassified as Severe.
- Current Loss: Weighted CrossEntropy (Failed to separate hard negatives).

🎯 The Objective (V2 Complete Pipeline):
I need a complete architectural overhaul of the training and inference pipelines. We are moving from a simple baseline to a state-of-the-art medical classifier. You must provide clean, modular, and production-ready PyTorch code that implements the following advanced upgrades.

=========================================
🛠️ YOUR MISSION & DELIVERABLES:
=========================================

Please provide step-by-step implementations for the following 5 modules:

### 1. Advanced Feature Engineering & Data Pipeline
- Angles & Ratios: Write a function to extract geometric ratios (e.g., eye area symmetry) and angles (e.g., mouth corner deviation from horizontal) from MediaPipe landmarks, moving beyond raw Euclidean distances.
- Hard Negative Mining: Provide a DataLoader/Sampler strategy to isolate and heavily penalize the exact type of "Hard Normal" samples that cause False Positives.
- Label Smoothing: Implement clinical-grade label smoothing for the targets (e.g., transforming `[1.0, 0.0, 0.0]` into `[0.9, 0.05, 0.05]`) to prevent overconfidence.

### 2. Architecture Upgrade (Lightweight + Attention)
- Upgrade the current ~17K parameter neural network by integrating a Squeeze-and-Excitation (SE) Block or a lightweight Self-Attention layer before the final classification head. 
- Goal: Allow the model to dynamically focus on asymmetrical micro-expressions while maintaining a tiny memory footprint for edge devices.

### 3. Training & Loss Optimization (CRITICAL)
- Ditch standard Weighted CE. Implement `Focal Loss` combined with a `Margin-based penalty` (conceptually similar to ArcFace/CosFace but adapted for standard classification) to force a massive mathematical margin between the "Normal" and "Mild" feature clusters.

### 4. Post-Hoc Optimization & Temporal Smoothing
- Optimal Thresholding: Provide a validation script to compute optimal, class-specific decision thresholds (replacing `argmax`) to maximize Macro F1.
- Confidence Calibration: Implement a Temperature Scaling layer to keep the Expected Calibration Error (ECE) < 0.05.
- Temporal Smoothing: The model will run on live video feeds. Implement a fast `deque-based` state machine. A pathological class (Mild/Severe) MUST persist across 'N' consecutive frames (e.g., N=5) to trigger a positive detection, eliminating flickering.

### 5. The Production Inference Pipeline (STRICT ARCHITECTURAL RULES)
Write the final inference class/script. It MUST strictly adhere to the following 4 architectural principles:
- Rule 1: Strict API Contract. Use `pydantic` or strictly typed dataclasses to define input/output schemas (e.g., probabilities, final prediction, temporal state).
- Rule 2: Environment Variables. ALL configurable parameters (Optimal Thresholds, Temporal Window Size 'N', Temperature value, Model Path) MUST be loaded via `os.environ` or `pydantic-settings`. No hardcoded numbers.
- Rule 3: Standardized Logging. Use Python's `logging` module. Configure proper log levels (INFO, WARNING, ERROR). `print()` is strictly forbidden.
- Rule 4: Docker-Ready. Structure the code so it can be seamlessly encapsulated within a Dockerized microservice (e.g., a FastAPI worker).

Output Format:
- Direct, highly technical, and strictly focused on implementation.
- Clean, thoroughly commented, and type-hinted Python code blocks.
- Ensure the code components can be stitched together into a single cohesive system.