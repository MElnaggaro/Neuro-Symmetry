export type RiskLevel        = "NORMAL" | "MILD" | "HIGH_RISK" | "CRITICAL";
export type TrajectoryState  = "STABLE" | "LINEAR_DECLINE" | "SUDDEN_DROP" | "OSCILLATING" | "COLLAPSE";
export type QualityCode      = "OK" | "DEGRADED" | "UNRELIABLE";
export type AffectedSide     = "LEFT" | "RIGHT" | "BILATERAL";
export type ConfirmationState = "NORMAL" | "PENDING" | "CONFIRMED";
export type XAILevel         = "HIGH" | "MEDIUM" | "LOW";

export interface XAIFeature {
  feature:      string;
  contribution: number;
  level:        XAILevel;
}

export interface AnalysisResponse {
  quality_code:       QualityCode;
  quality_reason:     string | null;
  class_id:           number | null;
  class_label:        string | null;
  confidence:         number | null;
  probabilities:      number[] | null;
  symmetry_score:     number | null;
  anomaly_score:      number | null;
  risk_level:         RiskLevel | null;
  trajectory:         TrajectoryState | null;
  affected_side:      AffectedSide | null;
  onset_seconds:      number | null;
  xai:                XAIFeature[];
  alert:              boolean;
  change_detected:    boolean;
  confirmation_state: ConfirmationState;
  ema_score:          number | null;
  latency_ms:         number;
}

export interface CalibrateResponse {
  type?:            "calibrate";
  frames_recorded:  number;
  ready:            boolean;
  min_frames:       number;
  message:          string;
}

export interface HistoryPoint {
  frame: number;
  score: number;
}
