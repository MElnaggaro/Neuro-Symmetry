"""
Phase 6 tests — Trajectory Engine, Clinical Rules (FAST), XAI Engine, Temporal Engine.

All tests run without a landmark model, ONNX model, or real image.

Includes a simulated-droop integration test (TestSimulatedDropIntegration) that
verifies the complete intelligence-layer pipeline against the Phase 6 deliverable:

  40 normal frames (score 0.85) → 30 droop frames (score 0.45)
  → trajectory  = SUDDEN_DROP
  → risk_level  = CRITICAL
  → xai         = ranked list, top contributor = HIGH
  → affected_side = LEFT
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.api.clinical_rules import RiskLevel, apply_fast_logic
from backend.api.feature_extractor import FEATURE_NAMES
from backend.api.temporal_engine import MIN_SLOPE_N, TemporalEngine
from backend.api.trajectory_engine import Trajectory, TrajectoryEngine
from backend.api.xai_engine import XAIFeature, _N_TOP, explain_prediction


# ── Shared synthetic feature helpers ─────────────────────────────────────────

def _make_features(
    ear_l:          float = 0.30,
    ear_r:          float = 0.30,
    brow_l:         float = 0.20,
    brow_r:         float = 0.20,
    texture:        float = 0.90,
    symmetry_error: float = 0.05,
) -> np.ndarray:
    """50-dim feature vector with controlled key values; all others zero."""
    f = np.zeros(50, dtype=np.float32)
    f[40] = ear_l
    f[41] = ear_r
    f[42] = abs(ear_l - ear_r)
    f[43] = brow_l
    f[44] = brow_r
    f[45] = abs(brow_l - brow_r)
    f[48] = texture
    f[49] = symmetry_error
    return f


def _severe_left_droop() -> np.ndarray:
    """Features mimicking a severe left-side facial droop."""
    f = np.zeros(50, dtype=np.float32)
    f[0:9]   = 0.12    # eye outline bilateral distances
    f[9:13]  = 0.08    # eye aperture
    f[13:23] = 0.10    # eyebrow
    f[27:31] = 0.14    # mouth
    f[31:36] = 0.09    # jaw
    f[40]    = 0.10    # ear_left  — drooped, more closed
    f[41]    = 0.32    # ear_right — open
    f[42]    = 0.22    # ear_delta
    f[43]    = 0.08    # brow_height_left  — lower
    f[44]    = 0.22    # brow_height_right — higher
    f[45]    = 0.14    # brow_height_delta
    f[46]    = 0.18    # mouth_y_delta
    f[47]    = 0.12    # mouth_x_offset
    f[48]    = 0.45    # texture_score — reduced (asymmetric texture)
    f[49]    = 0.60    # symmetry_error — high
    return f


# ── TestTrajectoryEngine ──────────────────────────────────────────────────────

class TestTrajectoryEngine:

    def _eng(self) -> TrajectoryEngine:
        return TrajectoryEngine(window=60)

    # --- STABLE ---

    def test_stable_before_min_frames(self):
        eng = self._eng()
        for _ in range(9):
            t = eng.update(0.9)
        assert t == Trajectory.STABLE

    def test_stable_flat_scores(self):
        eng = self._eng()
        t = Trajectory.STABLE
        for _ in range(60):
            t = eng.update(0.90)
        assert t == Trajectory.STABLE

    def test_stable_gently_improving(self):
        eng = self._eng()
        t = Trajectory.STABLE
        for i in range(60):
            t = eng.update(0.5 + i * 0.005)   # slow rise, no collapse/drop
        assert t == Trajectory.STABLE

    # --- COLLAPSE ---

    def test_collapse_sustained_low_mean(self):
        eng = self._eng()
        t = Trajectory.STABLE
        for _ in range(60):
            t = eng.update(0.20)   # mean = 0.20 < collapse_mean 0.30
        assert t == Trajectory.COLLAPSE

    def test_collapse_at_boundary(self):
        eng = self._eng()
        t = Trajectory.STABLE
        for _ in range(60):
            t = eng.update(0.28)   # mean 0.28 < 0.30 → still COLLAPSE
        assert t == Trajectory.COLLAPSE

    def test_no_collapse_above_threshold(self):
        eng = self._eng()
        t = Trajectory.STABLE
        for _ in range(60):
            t = eng.update(0.35)   # mean 0.35 > 0.30 → not COLLAPSE
        assert t != Trajectory.COLLAPSE

    # --- SUDDEN_DROP ---

    def test_sudden_drop_after_normal_baseline(self):
        """40 normal (0.85) + 30 droop (0.45): drop = 0.85 − 0.45 = 0.40 > 0.25."""
        eng = self._eng()
        for _ in range(40):
            eng.update(0.85)
        t = Trajectory.STABLE
        for _ in range(30):
            t = eng.update(0.45)
        assert t == Trajectory.SUDDEN_DROP

    def test_no_sudden_drop_for_small_delta(self):
        """Drop of 0.10 is below threshold 0.25."""
        eng = self._eng()
        for _ in range(40):
            eng.update(0.85)
        t = Trajectory.STABLE
        for _ in range(30):
            t = eng.update(0.75)   # drop = 0.85 − 0.75 = 0.10
        assert t != Trajectory.SUDDEN_DROP

    # --- LINEAR_DECLINE ---

    def test_linear_decline(self):
        """0.65 → 0.45 over 60 frames: slope ≈ −0.0034, R² ≈ 1.0, drop ≈ 0.10 < 0.25."""
        eng = self._eng()
        t = Trajectory.STABLE
        for i in range(60):
            score = 0.65 - i * (0.20 / 59)
            t = eng.update(score)
        assert t == Trajectory.LINEAR_DECLINE

    # --- OSCILLATING ---

    def test_oscillating_alternating_scores(self):
        """Alternating 0.70 / 0.50: std ≈ 0.10 > 0.05, slope ≈ 0."""
        eng = self._eng()
        t = Trajectory.STABLE
        for i in range(60):
            t = eng.update(0.70 if i % 2 == 0 else 0.50)
        assert t == Trajectory.OSCILLATING

    # --- reset ---

    def test_reset_clears_history(self):
        eng = self._eng()
        for _ in range(60):
            eng.update(0.20)   # COLLAPSE territory
        eng.reset()
        t = eng.update(0.20)
        assert t == Trajectory.STABLE   # only 1 frame — too few for non-STABLE

    # --- classify() static method ---

    def test_classify_static_stable(self):
        assert TrajectoryEngine.classify([0.9] * 20) == Trajectory.STABLE

    def test_classify_static_collapse(self):
        assert TrajectoryEngine.classify([0.25] * 60) == Trajectory.COLLAPSE

    def test_classify_static_respects_min_frames(self):
        assert TrajectoryEngine.classify([0.1] * 9) == Trajectory.STABLE

    # --- detect_sudden_drop() static method ---

    def test_detect_sudden_drop_true(self):
        scores = np.array([0.90] * 30 + [0.50] * 30, dtype=np.float64)
        assert TrajectoryEngine.detect_sudden_drop(scores) is True

    def test_detect_sudden_drop_false_small_delta(self):
        scores = np.array([0.90] * 30 + [0.80] * 30, dtype=np.float64)
        assert TrajectoryEngine.detect_sudden_drop(scores) is False

    def test_detect_sudden_drop_false_insufficient_frames(self):
        scores = np.array([0.90] * 10 + [0.40] * 10, dtype=np.float64)
        assert TrajectoryEngine.detect_sudden_drop(scores) is False

    def test_detect_sudden_drop_threshold_respected(self):
        scores = np.array([0.80] * 30 + [0.60] * 30, dtype=np.float64)
        assert TrajectoryEngine.detect_sudden_drop(scores, threshold=0.25) is False
        assert TrajectoryEngine.detect_sudden_drop(scores, threshold=0.15) is True

    def test_detect_sudden_drop_custom_window(self):
        # [0.90]*20 + [0.50]*30 — clear drop in the second half.
        #
        # window=30: earlier = first 20 frames = [0.90], baseline = 0.90
        #            recent_min = 0.50,  drop = 0.40 > 0.25  → True
        #
        # window=10: earlier = first 40 frames = [0.90]*20 + [0.50]*20, baseline = 0.70
        #            recent_min = 0.50,  drop = 0.20 ≤ 0.25  → False
        #            (large window dilutes the baseline, masking the drop)
        scores = np.array([0.90] * 20 + [0.50] * 30, dtype=np.float64)
        assert TrajectoryEngine.detect_sudden_drop(scores, threshold=0.25, window=30) is True
        assert TrajectoryEngine.detect_sudden_drop(scores, threshold=0.25, window=10) is False


# ── TestClinicalRules ─────────────────────────────────────────────────────────

class TestClinicalRules:

    # --- Core FAST matrix ---

    def test_normal_face_stable_is_normal(self):
        assert apply_fast_logic(0, "STABLE") == RiskLevel.NORMAL

    def test_mild_face_is_mild(self):
        assert apply_fast_logic(1, "STABLE") == RiskLevel.MILD

    def test_mild_face_with_speech_slur_is_high_risk(self):
        assert apply_fast_logic(1, "STABLE", speech_slur=True) == RiskLevel.HIGH_RISK

    def test_severe_face_is_high_risk(self):
        assert apply_fast_logic(2, "STABLE") == RiskLevel.HIGH_RISK

    def test_severe_face_with_speech_slur_is_critical(self):
        assert apply_fast_logic(2, "STABLE", speech_slur=True) == RiskLevel.CRITICAL

    def test_speech_and_arm_drift_always_critical(self):
        for face_class in (0, 1, 2):
            assert (
                apply_fast_logic(face_class, "STABLE", speech_slur=True, arm_drift=True)
                == RiskLevel.CRITICAL
            )

    # --- Trajectory escalation ---

    def test_normal_face_collapse_no_escalation(self):
        """face_class_id=0 cannot be escalated by trajectory (not pathological)."""
        assert apply_fast_logic(0, "COLLAPSE") == RiskLevel.NORMAL

    def test_normal_face_sudden_drop_no_escalation(self):
        assert apply_fast_logic(0, "SUDDEN_DROP") == RiskLevel.NORMAL

    def test_mild_face_collapse_is_critical(self):
        assert apply_fast_logic(1, "COLLAPSE") == RiskLevel.CRITICAL

    def test_mild_face_sudden_drop_is_critical(self):
        assert apply_fast_logic(1, "SUDDEN_DROP") == RiskLevel.CRITICAL

    def test_severe_face_collapse_is_critical(self):
        assert apply_fast_logic(2, "COLLAPSE") == RiskLevel.CRITICAL

    def test_severe_face_sudden_drop_is_critical(self):
        assert apply_fast_logic(2, "SUDDEN_DROP") == RiskLevel.CRITICAL

    def test_severe_face_linear_decline_stays_high_risk(self):
        """LINEAR_DECLINE is not in the critical-trajectory set."""
        assert apply_fast_logic(2, "LINEAR_DECLINE") == RiskLevel.HIGH_RISK

    def test_severe_face_oscillating_stays_high_risk(self):
        assert apply_fast_logic(2, "OSCILLATING") == RiskLevel.HIGH_RISK

    # --- None arguments treated as falsy ---

    def test_none_speech_arm_no_effect(self):
        assert apply_fast_logic(0, "STABLE", speech_slur=None, arm_drift=None) == RiskLevel.NORMAL
        assert apply_fast_logic(1, "STABLE", speech_slur=None, arm_drift=None) == RiskLevel.MILD

    # --- Risk level is a string enum ---

    def test_risk_level_values_are_strings(self):
        for level in RiskLevel:
            assert isinstance(level.value, str)


# ── TestXAIEngine ─────────────────────────────────────────────────────────────

class TestXAIEngine:

    _probs_severe = [0.05, 0.10, 0.85]
    _probs_normal = [0.95, 0.03, 0.02]

    # --- Output structure ---

    def test_returns_at_most_n_top_features(self):
        result, _ = explain_prediction(
            _make_features(), FEATURE_NAMES, self._probs_severe, 2
        )
        assert 1 <= len(result) <= _N_TOP

    def test_contributions_are_descending(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, self._probs_severe, 2
        )
        contribs = [f.contribution for f in result]
        assert contribs == sorted(contribs, reverse=True)

    def test_levels_are_valid_strings(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, self._probs_severe, 2
        )
        for feat in result:
            assert feat.level in ("HIGH", "MEDIUM", "LOW")

    def test_top_contributor_is_high_level(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, self._probs_severe, 2
        )
        assert result[0].level == "HIGH"

    def test_result_items_are_xai_feature_instances(self):
        result, _ = explain_prediction(
            _make_features(), FEATURE_NAMES, self._probs_severe, 2
        )
        for feat in result:
            assert isinstance(feat, XAIFeature)
            assert isinstance(feat.feature, str)
            assert isinstance(feat.contribution, float)

    # --- Contribution magnitudes ---

    def test_normal_prediction_has_small_contributions(self):
        """With pathological_prob ≈ 0.05 all contributions should be near zero."""
        result, _ = explain_prediction(
            _make_features(symmetry_error=0.02), FEATURE_NAMES, self._probs_normal, 0
        )
        for feat in result:
            assert feat.contribution < 0.10, (
                f"Expected low contribution for Normal, got {feat.contribution} for {feat.feature}"
            )

    def test_severe_prediction_has_nonzero_contributions(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, self._probs_severe, 2
        )
        assert result[0].contribution > 0

    # --- Affected side detection ---

    def test_affected_side_left_ear_and_brow(self):
        """ear_left < ear_right, brow_left < brow_right → LEFT."""
        f = np.zeros(50, dtype=np.float32)
        f[40] = 0.10   # ear_left  (closed)
        f[41] = 0.32   # ear_right (open)
        f[43] = 0.08   # brow_height_left  (low)
        f[44] = 0.22   # brow_height_right (high)
        f[48] = 0.90
        _, side = explain_prediction(f, FEATURE_NAMES, self._probs_severe, 2)
        assert side == "LEFT"

    def test_affected_side_right_ear_and_brow(self):
        """ear_right < ear_left, brow_right < brow_left → RIGHT."""
        f = np.zeros(50, dtype=np.float32)
        f[40] = 0.32   # ear_left  (open)
        f[41] = 0.10   # ear_right (closed)
        f[43] = 0.22   # brow_height_left  (high)
        f[44] = 0.08   # brow_height_right (low)
        f[48] = 0.90
        _, side = explain_prediction(f, FEATURE_NAMES, self._probs_severe, 2)
        assert side == "RIGHT"

    def test_affected_side_bilateral_symmetric_values(self):
        """Equal EAR and brow on both sides → BILATERAL."""
        f = np.zeros(50, dtype=np.float32)
        f[40] = f[41] = 0.30
        f[43] = f[44] = 0.20
        f[48] = 0.90
        _, side = explain_prediction(f, FEATURE_NAMES, self._probs_severe, 2)
        assert side == "BILATERAL"

    def test_affected_side_bilateral_tiny_delta_below_threshold(self):
        """Delta of 0.005 is below the 0.01 vote threshold → BILATERAL."""
        f = np.zeros(50, dtype=np.float32)
        f[40] = 0.300   # ear_left
        f[41] = 0.305   # ear_right  — Δ = 0.005 < 0.01
        f[43] = f[44] = 0.20
        f[48] = 0.90
        _, side = explain_prediction(f, FEATURE_NAMES, self._probs_severe, 2)
        assert side == "BILATERAL"

    def test_affected_side_tie_resolves_to_bilateral(self):
        """ear says LEFT, brow says RIGHT → tie → BILATERAL."""
        f = np.zeros(50, dtype=np.float32)
        f[40] = 0.10   # ear_left  < ear_right  → LEFT vote
        f[41] = 0.30
        f[43] = 0.30   # brow_left > brow_right → RIGHT vote
        f[44] = 0.10
        f[48] = 0.90
        _, side = explain_prediction(f, FEATURE_NAMES, self._probs_severe, 2)
        assert side == "BILATERAL"

    def test_feature_names_in_output_are_valid(self):
        """All surfaced feature names must come from the real FEATURE_NAMES list."""
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, self._probs_severe, 2
        )
        for feat in result:
            assert feat.feature in FEATURE_NAMES, (
                f"XAI returned unknown feature name: {feat.feature!r}"
            )


# ── TestTemporalEngine ────────────────────────────────────────────────────────

class TestTemporalEngine:

    def _eng(self) -> TemporalEngine:
        return TemporalEngine()

    def test_first_frame_ema_equals_score(self):
        state = self._eng().update(0.80, 0)
        assert state.ema_score == pytest.approx(0.80, abs=1e-4)

    def test_ema_smooths_toward_new_value(self):
        eng = self._eng()
        eng.update(0.80, 0)
        state = eng.update(0.50, 0)
        # α=0.10: ema = 0.10·0.50 + 0.90·0.80 = 0.77
        assert 0.75 < state.ema_score < 0.80

    def test_ema_converges_to_constant_input(self):
        eng = self._eng()
        for _ in range(200):
            state = eng.update(0.75, 0)
        assert state.ema_score == pytest.approx(0.75, abs=0.01)

    def test_no_onset_for_all_normal_frames(self):
        eng = self._eng()
        for _ in range(15):
            state = eng.update(0.90, 0)
        assert state.onset_frame is None
        assert state.duration_frames == 0

    def test_onset_set_on_first_alert_frame(self):
        eng = self._eng()
        eng.update(0.90, 0)            # frame 1 — Normal
        state = eng.update(0.45, 2)    # frame 2 — Severe → onset
        assert state.onset_frame == 2
        assert state.duration_frames == 0

    def test_duration_increments_per_alert_frame(self):
        eng = self._eng()
        eng.update(0.90, 0)
        eng.update(0.45, 2)   # onset at frame 2
        state = eng.update(0.40, 2)   # frame 3 — duration = 3 − 2 = 1
        assert state.duration_frames == 1

    def test_onset_clears_on_return_to_normal(self):
        eng = self._eng()
        eng.update(0.45, 2)   # alert
        state = eng.update(0.90, 0)   # back to Normal
        assert state.onset_frame is None
        assert state.duration_frames == 0

    def test_slope_is_zero_below_min_frames(self):
        eng = self._eng()
        for _ in range(MIN_SLOPE_N - 1):
            state = eng.update(0.50, 0)
        assert state.slope == 0.0

    def test_slope_negative_for_declining_scores(self):
        eng = self._eng()
        for i in range(30):
            eng.update(1.0 - i * 0.02, 0)   # 1.0 → 0.42
        state = eng.update(0.40, 0)
        assert state.slope < 0.0

    def test_slope_positive_for_rising_scores(self):
        eng = self._eng()
        for i in range(30):
            eng.update(0.40 + i * 0.02, 0)  # 0.40 → 0.98
        state = eng.update(1.0, 0)
        assert state.slope > 0.0

    def test_frame_count_increments(self):
        eng = self._eng()
        for _ in range(7):
            state = eng.update(0.90, 0)
        assert state.frame_count == 7

    def test_reset_clears_all_state(self):
        eng = self._eng()
        for _ in range(30):
            eng.update(0.50, 2)
        eng.reset()
        state = eng.update(0.90, 0)
        assert state.frame_count == 1
        assert state.onset_frame is None
        assert state.slope == 0.0


# ── TestSimulatedDropIntegration ──────────────────────────────────────────────

class TestSimulatedDropIntegration:
    """
    Phase 6 deliverable: simulated droop → correct trajectory, risk level, and XAI.

    Scenario A — SUDDEN_DROP
      40 frames @ 0.85, then 30 frames @ 0.45.
      After the 60-frame deque settles: 30 normal + 30 droop.
      baseline(first 30) = 0.85, recent_min = 0.45, drop = 0.40 > 0.25 → SUDDEN_DROP.

    Scenario B — COLLAPSE
      60 frames @ 0.20.  mean = 0.20 < 0.30 → COLLAPSE.

    Both scenarios + Severe face class → CRITICAL via FAST.
    """

    # --- Scenario A: SUDDEN_DROP ---

    def test_sudden_drop_trajectory_after_droop(self):
        eng = TrajectoryEngine()
        for _ in range(40):
            eng.update(0.85)
        final = Trajectory.STABLE
        for _ in range(30):
            final = eng.update(0.45)
        assert final == Trajectory.SUDDEN_DROP

    def test_sudden_drop_plus_severe_face_is_critical(self):
        assert apply_fast_logic(2, "SUDDEN_DROP") == RiskLevel.CRITICAL

    def test_sudden_drop_plus_mild_face_is_critical(self):
        assert apply_fast_logic(1, "SUDDEN_DROP") == RiskLevel.CRITICAL

    # --- Scenario B: COLLAPSE ---

    def test_collapse_trajectory_sustained_low(self):
        eng = TrajectoryEngine()
        final = Trajectory.STABLE
        for _ in range(60):
            final = eng.update(0.20)
        assert final == Trajectory.COLLAPSE

    def test_collapse_plus_severe_face_is_critical(self):
        assert apply_fast_logic(2, "COLLAPSE") == RiskLevel.CRITICAL

    # --- XAI deliverable ---

    def test_xai_list_non_empty_for_droop_frame(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, [0.05, 0.10, 0.85], 2
        )
        assert len(result) > 0

    def test_xai_top_contributor_level_high(self):
        result, _ = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, [0.05, 0.10, 0.85], 2
        )
        assert result[0].level == "HIGH"
        assert result[0].contribution > 0.0

    def test_xai_affected_side_left_for_left_droop(self):
        _, side = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, [0.05, 0.10, 0.85], 2
        )
        assert side == "LEFT"

    # --- Full end-to-end intelligence pipeline (no image required) ---

    def test_full_pipeline_sudden_drop_to_critical_left(self):
        """
        The complete intelligence-layer stack from score stream to CRITICAL alert
        and LEFT-side XAI, matching the Phase 6 deliverable.
        """
        # 1. Score stream → trajectory
        eng = TrajectoryEngine()
        for _ in range(40):
            eng.update(0.85)
        trajectory = Trajectory.STABLE
        for _ in range(30):
            trajectory = eng.update(0.45)

        assert trajectory == Trajectory.SUDDEN_DROP

        # 2. FAST protocol → risk level
        face_class_id = 2   # Severe classifier output
        risk_level = apply_fast_logic(face_class_id, trajectory.value)

        assert risk_level == RiskLevel.CRITICAL

        # 3. XAI → ranked contributions + affected side
        probs = [0.05, 0.10, 0.85]
        xai_features, affected_side = explain_prediction(
            _severe_left_droop(), FEATURE_NAMES, probs, face_class_id
        )

        assert len(xai_features) > 0, "XAI list must not be empty"
        assert xai_features[0].level == "HIGH", (
            f"Top XAI contributor should be HIGH, got {xai_features[0].level}"
        )
        assert affected_side == "LEFT", (
            f"Expected LEFT-side droop, got {affected_side}"
        )

    def test_full_pipeline_collapse_to_critical(self):
        """Sustained very low scores → COLLAPSE → CRITICAL."""
        eng = TrajectoryEngine()
        final = Trajectory.STABLE
        for _ in range(60):
            final = eng.update(0.20)

        assert final == Trajectory.COLLAPSE
        assert apply_fast_logic(2, final.value) == RiskLevel.CRITICAL
