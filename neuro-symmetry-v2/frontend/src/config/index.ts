// ── Application configuration ─────────────────────────────────────────────────
// Re-export everything from palette.ts to avoid duplication.
// palette.ts is the single source of truth for all config + palette constants.

export {
  VIDEO_CONFIG,
  MEDIAPIPE_CONFIG,
  CANVAS_CONFIG,
  HEATMAP_CONFIG,
  APP_CONFIG,
  RISK_PALETTE,
  TRAJECTORY_PALETTE,
  TRAJECTORY_GRAPH_COLOR,
  XAI_PALETTE,
  TYPE_SCALE,
  LETTER_SPACING,
  GLASS,
  MOTION,
} from "./palette";