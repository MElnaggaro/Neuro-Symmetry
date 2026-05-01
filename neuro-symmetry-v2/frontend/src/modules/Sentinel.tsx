/**
 * Sentinel — Micro-Lag Asymmetry Detection Module (Phase 3)
 *
 * Polls `mathLandmarksRef` via requestAnimationFrame to detect
 * blink and smile timing asymmetries without triggering React renders.
 *
 * Architecture:
 *   - All per-frame logic reads from mutable refs (zero renders)
 *   - React state is used ONLY for low-frequency UI updates (~2 FPS)
 *   - `useRafLoop` handles the RAF lifecycle and logging
 *   - `eyeAspectRatio` / `distance2` imported from faceMath (no duplication)
 */

import { useRef, useState, useEffect } from "react";
import { useFaceTracking } from "@/providers/FaceTrackingProvider";
import { eyeAspectRatio } from "@/utils/faceMath";
import { useRafLoop } from "@/hooks/useRafLoop";

// ── Landmark indices for micro-lag detection ──────────────────────────────────

const LM = {
  R_EYE: [33, 160, 158, 133, 153, 144] as const,
  L_EYE: [263, 387, 385, 362, 380, 373] as const,
  MOUTH_L: 61,
  MOUTH_R: 291,
} as const;

// ── Configurable thresholds ───────────────────────────────────────────────────

const BLINK_THRESHOLD    = 0.20;
const SMILE_THRESHOLD    = 0.008;
const REPORT_INTERVAL_MS = 500;
const BLINK_WINDOW_MS    = 500;
const ASYM_DECAY_RATE    = 0.95;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SentinelMetrics {
  blinkAsymMs:    number;
  smileAsymDelta: number;
  status:         "NORMAL" | "ASYMMETRY_DETECTED";
  lastUpdated:    number;
}

interface EyeState {
  closed:   boolean;
  closedAt: number;
  openedAt: number;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Sentinel() {
  const { mathLandmarksRef, status: trackingStatus, start } = useFaceTracking();
  const [metrics, setMetrics] = useState<SentinelMetrics>({
    blinkAsymMs:    0,
    smileAsymDelta: 0,
    status:         "NORMAL",
    lastUpdated:    0,
  });

  // ── Mutable state for per-frame tracking (no renders) ─────────────────
  const leftEyeRef    = useRef<EyeState>({ closed: false, closedAt: 0, openedAt: 0 });
  const rightEyeRef   = useRef<EyeState>({ closed: false, closedAt: 0, openedAt: 0 });
  const lastReportRef = useRef(0);
  const blinkAsymRef  = useRef(0);
  const smileAsymRef  = useRef(0);

  useEffect(() => { start(); }, [start]);

  // ── RAF loop — polls mathLandmarks ref, zero React renders per frame ──
  useRafLoop(() => {
    const lm = mathLandmarksRef.current;
    if (!lm || lm.length <= 466) return;

    const now  = performance.now();
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

    // Blink asymmetry — only valid for paired blinks within a time window
    const rClosedAt = rightEyeRef.current.closedAt;
    const lClosedAt = leftEyeRef.current.closedAt;
    if (rClosedAt > 0 && lClosedAt > 0) {
      const delta = Math.abs(rClosedAt - lClosedAt);
      if (delta < BLINK_WINDOW_MS) {
        blinkAsymRef.current = delta;
      } else {
        blinkAsymRef.current *= ASYM_DECAY_RATE;
        if (blinkAsymRef.current < 0.5) blinkAsymRef.current = 0;
      }
    }

    smileAsymRef.current = Math.abs(lm[LM.MOUTH_L].y - lm[LM.MOUTH_R].y);

    // Throttled UI update (~2 FPS)
    if (now - lastReportRef.current > REPORT_INTERVAL_MS) {
      lastReportRef.current = now;
      const isAsym = blinkAsymRef.current > 50 || smileAsymRef.current > SMILE_THRESHOLD;
      setMetrics({
        blinkAsymMs:    Math.round(blinkAsymRef.current),
        smileAsymDelta: Math.round(smileAsymRef.current * 1000) / 1000,
        status:         isAsym ? "ASYMMETRY_DETECTED" : "NORMAL",
        lastUpdated:    now,
      });
    }
  }, trackingStatus === "ready", "Sentinel");

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
