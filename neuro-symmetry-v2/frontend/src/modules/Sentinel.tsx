/**
 * Sentinel — Micro-Lag Asymmetry Detection Module (Phase 3)
 *
 * Polls `mathLandmarksRef` via requestAnimationFrame to detect
 * blink and smile timing asymmetries without triggering React renders.
 *
 * Architecture:
 *   - All per-frame logic reads from mutable refs (zero renders)
 *   - React state is used ONLY for low-frequency UI updates (~2 FPS)
 *
 * BUG-6 fix: Blink asymmetry now decays between blinks and only measures
 *   timing asymmetry within a 500ms temporal window.
 * BUG-7 fix: Uses proper 6-point Eye Aspect Ratio formula matching faceMath.ts.
 */

import { useEffect, useRef, useState } from "react";
import { useFaceTracking } from "@/providers/FaceTrackingProvider";
import logger from "@/utils/logger";

// ── Landmark indices for micro-lag detection ──────────────────────────────────

const LM = {
  // Proper 6-point EAR indices matching faceMath.ts
  R_EYE: [33, 160, 158, 133, 153, 144] as const,  // outer, upper1, upper2, inner, lower1, lower2
  L_EYE: [263, 387, 385, 362, 380, 373] as const,
  MOUTH_L: 61,
  MOUTH_R: 291,
} as const;

// ── Configurable thresholds ───────────────────────────────────────────────────

const BLINK_THRESHOLD = 0.20;     // EAR below which the eye is "closed" (ratio, not raw y-delta)
const SMILE_THRESHOLD = 0.008;    // mouth corner y-delta indicating smile asymmetry
const REPORT_INTERVAL_MS = 500;   // how often to update the React UI state
const BLINK_WINDOW_MS = 500;      // max time between L/R blink closures to consider them "paired"
const ASYM_DECAY_RATE = 0.95;     // exponential decay per frame when no fresh blink pair

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SentinelMetrics {
  blinkAsymMs: number;      // blink timing asymmetry (ms)
  smileAsymDelta: number;   // smile corner y-delta
  status: "NORMAL" | "ASYMMETRY_DETECTED";
  lastUpdated: number;
}

interface EyeState {
  closed: boolean;
  closedAt: number;   // timestamp of most recent open→closed transition
  openedAt: number;
}

// ── Proper 6-point EAR (matches faceMath.ts) ──────────────────────────────────

function distance2(ax: number, ay: number, bx: number, by: number): number {
  return Math.hypot(ax - bx, ay - by);
}

interface Landmark { x: number; y: number; z?: number }

function eyeAspectRatio(lm: Landmark[], idx: readonly number[]): number {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => lm[i]);
  const width = distance2(p1.x, p1.y, p4.x, p4.y);
  if (width < 1e-6) return 0;
  return (distance2(p2.x, p2.y, p6.x, p6.y) + distance2(p3.x, p3.y, p5.x, p5.y)) / (2 * width);
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Sentinel() {
  const { mathLandmarksRef, status: trackingStatus, start } = useFaceTracking();
  const [metrics, setMetrics] = useState<SentinelMetrics>({
    blinkAsymMs: 0,
    smileAsymDelta: 0,
    status: "NORMAL",
    lastUpdated: 0,
  });

  // ── Mutable state for per-frame tracking (no renders) ─────────────────
  const leftEyeRef = useRef<EyeState>({ closed: false, closedAt: 0, openedAt: 0 });
  const rightEyeRef = useRef<EyeState>({ closed: false, closedAt: 0, openedAt: 0 });
  const lastReportRef = useRef(0);
  const blinkAsymRef = useRef(0);
  const smileAsymRef = useRef(0);

  // ── Start tracking on mount ───────────────────────────────────────────
  useEffect(() => { start(); }, [start]);

  // ── RAF loop — polls mathLandmarks ref without React renders ──────────
  useEffect(() => {
    if (trackingStatus !== "ready") return;

    let running = true;
    const tick = () => {
      if (!running) return;

      const lm = mathLandmarksRef.current;
      if (lm && lm.length > 466) {
        const now = performance.now();

        // ── Blink detection using proper 6-point EAR ──────────────────
        const rEAR = eyeAspectRatio(lm, LM.R_EYE);
        const lEAR = eyeAspectRatio(lm, LM.L_EYE);

        // Right eye state machine
        const rClosed = rEAR < BLINK_THRESHOLD;
        if (rClosed && !rightEyeRef.current.closed) {
          rightEyeRef.current = { closed: true, closedAt: now, openedAt: rightEyeRef.current.openedAt };
        } else if (!rClosed && rightEyeRef.current.closed) {
          rightEyeRef.current = { closed: false, closedAt: rightEyeRef.current.closedAt, openedAt: now };
        }

        // Left eye state machine
        const lClosed = lEAR < BLINK_THRESHOLD;
        if (lClosed && !leftEyeRef.current.closed) {
          leftEyeRef.current = { closed: true, closedAt: now, openedAt: leftEyeRef.current.openedAt };
        } else if (!lClosed && leftEyeRef.current.closed) {
          leftEyeRef.current = { closed: false, closedAt: leftEyeRef.current.closedAt, openedAt: now };
        }

        // ── Blink asymmetry — only valid for paired blinks within a time window ──
        const rClosedAt = rightEyeRef.current.closedAt;
        const lClosedAt = leftEyeRef.current.closedAt;
        if (rClosedAt > 0 && lClosedAt > 0) {
          const delta = Math.abs(rClosedAt - lClosedAt);
          if (delta < BLINK_WINDOW_MS) {
            // Fresh paired blink — update asymmetry
            blinkAsymRef.current = delta;
          } else {
            // Stale — decay the reading toward zero
            blinkAsymRef.current *= ASYM_DECAY_RATE;
            if (blinkAsymRef.current < 0.5) blinkAsymRef.current = 0;
          }
        }

        // ── Smile asymmetry ───────────────────────────────────────────
        smileAsymRef.current = Math.abs(lm[LM.MOUTH_L].y - lm[LM.MOUTH_R].y);

        // ── Throttled UI update (~2 FPS) ──────────────────────────────
        if (now - lastReportRef.current > REPORT_INTERVAL_MS) {
          lastReportRef.current = now;
          const isAsym =
            blinkAsymRef.current > 50 || smileAsymRef.current > SMILE_THRESHOLD;

          setMetrics({
            blinkAsymMs: Math.round(blinkAsymRef.current),
            smileAsymDelta: Math.round(smileAsymRef.current * 1000) / 1000,
            status: isAsym ? "ASYMMETRY_DETECTED" : "NORMAL",
            lastUpdated: now,
          });
        }
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
    logger.info("Sentinel", "RAF loop started");

    return () => {
      running = false;
      logger.info("Sentinel", "RAF loop stopped");
    };
  }, [trackingStatus, mathLandmarksRef]);

  // ── Render ────────────────────────────────────────────────────────────
  const isAsym = metrics.status === "ASYMMETRY_DETECTED";

  return (
    <div className="flex flex-col gap-3 p-4 rounded-2xl bg-[#1a1a2e] border border-[#2a2a4a]">
      <div className="flex items-center gap-2">
        <div
          className="w-3 h-3 rounded-full"
          style={{ background: isAsym ? "#ef4444" : "#10b981" }}
        />
        <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
          Sentinel — Micro-Lag Detection
        </h2>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-xl bg-[#1e1e35] border border-[#2a2a4a]">
          <div className="text-[9px] font-semibold tracking-[0.2em] text-slate-500 uppercase mb-1">
            Blink Asymmetry
          </div>
          <div className="text-xl font-mono font-bold tabular-nums"
               style={{ color: metrics.blinkAsymMs > 50 ? "#f87171" : "#34d399" }}>
            {metrics.blinkAsymMs} ms
          </div>
        </div>

        <div className="p-3 rounded-xl bg-[#1e1e35] border border-[#2a2a4a]">
          <div className="text-[9px] font-semibold tracking-[0.2em] text-slate-500 uppercase mb-1">
            Smile Δ
          </div>
          <div className="text-xl font-mono font-bold tabular-nums"
               style={{ color: metrics.smileAsymDelta > SMILE_THRESHOLD ? "#f87171" : "#34d399" }}>
            {metrics.smileAsymDelta.toFixed(3)}
          </div>
        </div>
      </div>

      {isAsym && (
        <div className="p-2 rounded-lg bg-red-900/20 border border-red-500/30 text-[11px] text-red-300 font-semibold text-center">
          ⚠ Micro-lag asymmetry detected — monitor closely
        </div>
      )}
    </div>
  );
}
