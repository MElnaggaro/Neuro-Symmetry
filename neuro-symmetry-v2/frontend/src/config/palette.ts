// ── Application configuration ─────────────────────────────────────────────────
// Single source of truth for all tuneable constants.

export const VIDEO_CONFIG = {
  WIDTH:  640,
  HEIGHT: 480,
} as const;

export const MEDIAPIPE_CONFIG = {
  maxNumFaces:             1,
  refineLandmarks:         true,
  minDetectionConfidence:  0.5,
  minTrackingConfidence:   0.5,
} as const;

export const CANVAS_CONFIG = {
  LANDMARK_COLOR:  "rgba(6, 182, 212, 0.62)",
  SHADOW_COLOR:    "#06b6d4",
  SHADOW_BLUR:     2.5,
  LANDMARK_RADIUS: 1.3,
} as const;

export const RISK_PALETTE = {
  NORMAL:     { border: "#10b981", glow: "rgba(16,185,129,0.2)",  bg: "rgba(16,185,129,0.05)", badge: "rgba(16,185,129,0.15)", fg: "#34d399", track: "#34d399", text: "#34d399" },
  MILD:       { border: "#f59e0b", glow: "rgba(245,158,11,0.25)", bg: "rgba(245,158,11,0.05)", badge: "rgba(245,158,11,0.15)", fg: "#fbbf24", track: "#fbbf24", text: "#fbbf24" },
  HIGH_RISK:  { border: "#f97316", glow: "rgba(249,115,22,0.3)",  bg: "rgba(249,115,22,0.05)", badge: "rgba(249,115,22,0.15)", fg: "#fb923c", track: "#fb923c", text: "#fb923c" },
  CRITICAL:   { border: "#ef4444", glow: "rgba(239,68,68,0.42)",  bg: "rgba(239,68,68,0.05)", badge: "rgba(239,68,68,0.15)", fg: "#f87171", track: "#f87171", text: "#f87171" },
} as const;

export const TRAJECTORY_PALETTE = {
  STABLE:         { color: "#10b981", dark: "rgba(16,185,129,0.1)", label: "STABLE", icon: "●", pulse: false },
  LINEAR_DECLINE: { color: "#f59e0b", dark: "rgba(245,158,11,0.1)", label: "DECLINE", icon: "▼", pulse: false },
  SUDDEN_DROP:    { color: "#f97316", dark: "rgba(249,115,22,0.1)", label: "DROP", icon: "▼▼", pulse: true },
  OSCILLATING:    { color: "#8b5cf6", dark: "rgba(139,92,246,0.1)", label: "UNSTABLE", icon: "〰", pulse: false },
  COLLAPSE:       { color: "#ef4444", dark: "rgba(239,68,68,0.1)", label: "CRITICAL", icon: "X", pulse: true },
} as const;

export const TRAJECTORY_GRAPH_COLOR = {
  STABLE:         "#10b981",
  LINEAR_DECLINE: "#f59e0b",
  SUDDEN_DROP:    "#f97316",
  OSCILLATING:    "#8b5cf6",
  COLLAPSE:       "#ef4444",
} as const;

export const XAI_PALETTE = {
  HIGH:   { dot: "#ef4444", bar: "from-red-500 to-red-400",     label: "#fca5a5", badge: "bg-red-900/20 border-red-500/40 text-red-400"      },
  MEDIUM: { dot: "#f59e0b", bar: "from-amber-500 to-amber-400", label: "#fcd34d", badge: "bg-amber-900/20 border-amber-500/40 text-amber-400" },
  LOW:    { dot: "#64748b", bar: "from-slate-500 to-slate-400", label: "#94a3b8", badge: "bg-slate-800/50 border-slate-600/40 text-slate-500"  },
} as const;

export const HEATMAP_CONFIG = {
  /**
   * Normalised [x, y, w, h] viewport zones per feature name.
   *
   * `symmetry_error` is intentionally NOT mapped — it is a holistic, whole-face
   * measure with no specific facial region. Including it would draw a
   * full-frame red rectangle that obscures every other zone underneath. The
   * XAI list panel still shows it; the heatmap stays clean.
   */
  ZONE_MAP: {
    // Named scalar features (indices 40-48)
    ear_right:         [0.27, 0.36, 0.17, 0.15],
    ear_left:          [0.56, 0.36, 0.17, 0.15],
    ear_delta:         [0.24, 0.34, 0.52, 0.18],
    brow_height_right: [0.26, 0.22, 0.18, 0.12],
    brow_height_left:  [0.56, 0.22, 0.18, 0.12],
    brow_height_delta: [0.23, 0.19, 0.54, 0.17],
    mouth_y_delta:     [0.35, 0.60, 0.30, 0.13],
    mouth_x_offset:    [0.32, 0.57, 0.36, 0.16],
    texture_score:     [0.18, 0.17, 0.64, 0.66],
    // Bilateral distance region groups (indices 0-39, aggregated by region)
    eye_outline:       [0.20, 0.25, 0.60, 0.22],
    eye_aperture:      [0.25, 0.27, 0.50, 0.17],
    eyebrow:           [0.22, 0.14, 0.56, 0.16],
    nose:              [0.35, 0.38, 0.30, 0.24],
    mouth:             [0.28, 0.57, 0.44, 0.20],
    jaw:               [0.22, 0.72, 0.56, 0.18],
    forehead:          [0.25, 0.05, 0.50, 0.12],
  } as Record<string, [number, number, number, number]>,

  FILL: {
    HIGH:    "rgba(239,68,68,0.20)",
    MEDIUM:  "rgba(249,115,22,0.14)",
    LOW:     "rgba(16,185,129,0.09)",
    DEFAULT: "rgba(148,163,184,0.10)",
  } as const,

  STROKE: {
    HIGH:    "#ef4444cc",
    MEDIUM:  "#f97316aa",
    LOW:     "#10b98177",
    DEFAULT: "#94a3b8aa",
  } as const,

  LABEL: {
    HIGH:    "#f87171",
    MEDIUM:  "#fb923c",
    LOW:     "#34d399",
    DEFAULT: "#94a3b8",
  } as const,
} as const;

export const APP_CONFIG = {
  MAX_HISTORY:            60,
  MIN_CALIBRATION_FRAMES: 30,
  /** Throttle to ~6–7 fps toward the backend. */
  SEND_INTERVAL_MS:       150,
} as const;