import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wifi, WifiOff, RefreshCw, Target, Activity } from "lucide-react";

import { useWebSocket } from "@/hooks/useWebSocket";
import { useCamera }    from "@/hooks/useCamera";

import { Card }           from "@/components/ui/Card";
import { SectionLabel }   from "@/components/ui/SectionLabel";
import { StatTile }       from "@/components/ui/StatTile";
import FaceMeshOverlay    from "@/components/video/FaceMeshOverlay";
import HeatmapOverlay     from "@/components/video/HeatmapOverlay";
import RiskIndicator      from "@/components/panels/RiskIndicator";
import TrajectoryLabel    from "@/components/panels/TrajectoryLabel";
import SymmetryGauge      from "@/components/panels/SymmetryGauge";
import ScoreGraph         from "@/components/panels/ScoreGraph";
import XAIBreakdown       from "@/components/panels/XAIBreakdown";

import type { HistoryPoint, RiskLevel, TrajectoryState } from "@/types/analysis";

// ── Constants ─────────────────────────────────────────────────────────────────
const W = 640, H = 480;
const MAX_HISTORY = 60;
const MIN_CALIB   = 30;

// ── Risk → CSS class for video border + glow ──────────────────────────────────
const RISK_VIDEO_CLASS: Record<RiskLevel, string> = {
  NORMAL:    "risk-border-normal",
  MILD:      "risk-border-mild",
  HIGH_RISK: "risk-border-high",
  CRITICAL:  "risk-border-critical",
};

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const meshCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // ── History ring-buffer ───────────────────────────────────────────────────
  const [history,    setHistory]    = useState<HistoryPoint[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  const [calibMode,  setCalibMode]  = useState(false);
  const [calibCount, setCalibCount] = useState(0);
  const [calibReady, setCalibReady] = useState(false);

  // ── WebSocket — destructure for stable callback references ──────────────
  const {
    status:       wsStatus,
    connect,
    sendFrame,
    resetSession,
    lastResult,
    lastCalib,
  } = useWebSocket();

  // Sync calibration state from WS acks
  useEffect(() => {
    if (!lastCalib) return;
    setCalibCount(lastCalib.frames_recorded);
    if (lastCalib.ready) { setCalibReady(true); setCalibMode(false); }
  }, [lastCalib]);

  // Append each new analysis frame to history
  useEffect(() => {
    if (!lastResult?.symmetry_score) return;
    setFrameCount((n) => n + 1);
    setHistory((h) => {
      const next: HistoryPoint[] = [
        ...h,
        { frame: (h[h.length - 1]?.frame ?? 0) + 1, score: lastResult.symmetry_score! },
      ];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
  }, [lastResult]);

  // ── Camera (stable sendFrame ref so useCamera effect never re-runs) ───────
  const calibModeRef = useRef(calibMode);
  useEffect(() => { calibModeRef.current = calibMode; }, [calibMode]);

  const onFrame = useCallback(
    (b64: string) => { sendFrame(b64, calibModeRef.current); },
    [sendFrame],
  );

  const { videoRef, error: camError, isReady: camReady } = useCamera({
    meshCanvasRef,
    onFrame,
    width:  W,
    height: H,
  });

  // Auto-connect WS once camera is ready
  useEffect(() => { if (camReady) connect(); }, [camReady, connect]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    resetSession();
    setHistory([]);
    setFrameCount(0);
    setCalibCount(0);
    setCalibReady(false);
    setCalibMode(false);
  }, [resetSession]);

  // ── Derived display values (memoised) ─────────────────────────────────────
  const r = lastResult;

  const score       = r?.symmetry_score  ?? 0;
  const riskLevel   = (r?.risk_level     ?? "NORMAL") as RiskLevel;
  const trajectory  = (r?.trajectory     ?? "STABLE") as TrajectoryState;
  const xai         = r?.xai             ?? [];
  const alert       = r?.alert           ?? false;
  const affected    = r?.affected_side   ?? null;
  const onsetSec    = r?.onset_seconds   ?? null;
  const confState   = r?.confirmation_state ?? "NORMAL";
  const latency     = r?.latency_ms      ?? null;
  const classLabel  = r?.class_label     ?? null;
  const anomaly     = r?.anomaly_score   ?? null;
  const emaScore    = r?.ema_score       ?? null;
  const quality     = r?.quality_code    ?? null;

  const wsOk          = wsStatus === "connected";
  const videoRiskClass = RISK_VIDEO_CLASS[riskLevel] ?? "risk-border-normal";

  const anomalyColor = useMemo(() => {
    if (anomaly === null) return "#64748b";
    return Math.abs(anomaly) > 2 ? "#f87171" : "#34d399";
  }, [anomaly]);

  return (
    <div className="flex flex-col h-screen bg-bg-base text-slate-200 font-sans overflow-hidden">

      {/* ── Scanline FX ── */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
        <div className="absolute left-0 right-0 h-16 bg-gradient-to-b from-transparent via-accent-blue/[0.025] to-transparent animate-scan" />
      </div>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header
        className="relative z-10 flex items-center justify-between px-5 py-3
                   bg-bg-surface border-b border-neu-border"
        role="banner"
      >
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[8px] flex items-center justify-center
                          bg-gradient-to-br from-blue-700 to-sky-500 shadow-glow-cyan shrink-0"
               aria-hidden="true">
            <Activity size={16} color="#bfdbfe" strokeWidth={2.5} />
          </div>
          <div>
            <div className="text-[15px] font-extrabold tracking-wide text-slate-200 leading-tight">
              Neuro-Symmetry
            </div>
            <div className="text-[9px] font-semibold tracking-[0.3em] text-accent-cyan uppercase">
              Facial Analysis Platform v2.0
            </div>
          </div>
        </div>

        {/* Status pills */}
        <div className="flex items-center gap-2">
          {latency !== null && (
            <span className={`text-[10px] font-mono px-3 py-1 rounded-full border
              ${latency < 60
                ? "bg-emerald-900/30 border-emerald-500/30 text-emerald-400"
                : latency < 150
                  ? "bg-amber-900/30 border-amber-500/30 text-amber-400"
                  : "bg-red-900/30 border-red-500/30 text-red-400"
              }`}>
              {latency.toFixed(0)} ms
            </span>
          )}

          <motion.span
            layout
            className={`flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1 rounded-full border
              ${wsOk
                ? "bg-emerald-900/20 border-emerald-500/30 text-emerald-400"
                : "bg-orange-900/20 border-orange-500/30 text-orange-400"
              }`}
            aria-live="polite"
            aria-label={`WebSocket status: ${wsStatus}`}
          >
            {wsOk
              ? <Wifi size={11} strokeWidth={2.5} aria-hidden="true" />
              : <WifiOff size={11} strokeWidth={2.5} aria-hidden="true" />
            }
            <span className={wsOk ? "animate-blink" : ""}>
              {wsOk ? "CONNECTED" : wsStatus.toUpperCase()}
            </span>
          </motion.span>

          {calibReady && (
            <span className="flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1 rounded-full
                             bg-cyan-900/20 border border-cyan-500/30 text-cyan-400">
              <Target size={11} strokeWidth={2.5} aria-hidden="true" />
              BASELINE READY
            </span>
          )}
        </div>

        {/* Frame counter */}
        <div className="text-[10px] font-mono text-slate-600 tabular-nums" aria-label="Frame count">
          FRAME {String(frameCount).padStart(6, "0")}
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────── */}
      <main className="relative z-1 flex flex-1 gap-4 p-4 min-h-0" role="main">

        {/* Left column — camera + overlays */}
        <div className="flex flex-col gap-3 shrink-0">

          {/* Video card */}
          <div
            className={`relative rounded-2xl overflow-hidden border-2 transition-all duration-500
                        risk-video-border ${videoRiskClass}`}
            style={{ width: W, height: H }}
            role="region"
            aria-label="Camera feed with facial analysis overlays"
          >
            {/* Corner fiducial marks */}
            {(["tl","tr","bl","br"] as const).map((pos) => (
              <div
                key={pos}
                aria-hidden="true"
                className={`absolute w-[18px] h-[18px] z-10 pointer-events-none
                  ${pos === "tl" ? "top-2 left-2 border-t-2 border-l-2" : ""}
                  ${pos === "tr" ? "top-2 right-2 border-t-2 border-r-2" : ""}
                  ${pos === "bl" ? "bottom-2 left-2 border-b-2 border-l-2" : ""}
                  ${pos === "br" ? "bottom-2 right-2 border-b-2 border-r-2" : ""}
                  border-accent-cyan/60`}
              />
            ))}

            {/* Live camera feed */}
            <video
              ref={videoRef}
              className="w-full h-full object-cover block"
              style={{ transform: "scaleX(-1)" }}
              autoPlay
              playsInline
              muted
              aria-label="Live camera feed"
            />

            {/* Canvas overlays */}
            <FaceMeshOverlay ref={meshCanvasRef} width={W} height={H} />
            <HeatmapOverlay  xai={xai} width={W} height={H} />

            {/* Camera / MediaPipe error banner */}
            <AnimatePresence>
              {camError && (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="absolute inset-0 flex items-center justify-center
                             bg-bg-base/85 text-[12px] text-red-400 text-center px-6"
                  role="alert"
                >
                  {camError}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Trajectory badge — bottom-left */}
            <div className="absolute bottom-3 left-3 flex flex-col gap-2 z-10">
              <TrajectoryLabel trajectory={trajectory} />
              {onsetSec !== null && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                  className="text-[10px] text-red-400 font-bold tracking-widest
                             bg-red-900/75 border border-red-500/40 px-2 py-1 rounded-lg glass"
                >
                  ⏱ ONSET {onsetSec.toFixed(1)}s
                </motion.div>
              )}
            </div>

            {/* Affected side — top-right */}
            {affected && (
              <div className="absolute top-3 right-3 z-10 text-[9px] font-semibold tracking-[0.2em]
                              text-accent-cyan bg-cyan-900/15 border border-accent-cyan/25
                              px-2.5 py-1.5 rounded-lg glass">
                SIDE: {affected}
              </div>
            )}

            {/* Confirmation state — bottom-right */}
            <AnimatePresence>
              {confState !== "NORMAL" && (
                <motion.div
                  key={confState}
                  initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                  className={`absolute bottom-3 right-3 z-10 text-[9px] font-bold tracking-widest
                               px-2.5 py-1.5 rounded-lg glass
                               ${confState === "CONFIRMED"
                                 ? "text-red-400 bg-red-900/70 border border-red-500/40"
                                 : "text-amber-400 bg-amber-900/70 border border-amber-500/40"
                               }`}
                  role="status"
                >
                  {confState}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Stats strip */}
          <div className="flex gap-2 flex-wrap">
            <StatTile label="CLASS"      value={classLabel}                color="#22d3ee" />
            <StatTile label="EMA SCORE"  value={emaScore?.toFixed(3) ?? null} color="#e2e8f0" mono />
            <StatTile label="ANOMALY Δz" value={anomaly?.toFixed(2) ?? null}  color={anomalyColor} mono />
            <StatTile label="QUALITY"    value={quality}
              color={quality === "OK" ? "#34d399" : quality === "DEGRADED" ? "#f59e0b" : "#f87171"} />
          </div>
        </div>

        {/* Right column — analysis panels */}
        <div className="flex flex-col gap-3 flex-1 min-w-0 min-h-0">

          {/* Risk + gauge */}
          <Card>
            <div className="flex items-center gap-4">
              <SymmetryGauge score={score} riskLevel={riskLevel} />
              <div className="flex-1">
                <RiskIndicator riskLevel={riskLevel} alert={alert} />
              </div>
            </div>
          </Card>

          {/* Score timeline */}
          <Card>
            <SectionLabel>Symmetry Timeline — Last {MAX_HISTORY} Frames</SectionLabel>
            <ScoreGraph history={history} trajectory={trajectory} />
          </Card>

          {/* XAI breakdown */}
          <Card flex>
            <SectionLabel>Feature Contributions (XAI)</SectionLabel>
            <div className="overflow-y-auto pr-1">
              <XAIBreakdown xai={xai} />
            </div>
          </Card>

        </div>
      </main>

      {/* ── Footer controls ─────────────────────────────────────────────── */}
      <footer
        className="relative z-10 flex items-center gap-3 px-5 py-2.5
                   bg-bg-surface border-t border-neu-border"
        role="contentinfo"
      >
        {!wsOk && (
          <button
            onClick={connect}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold
                       tracking-wide bg-gradient-to-r from-blue-800 to-sky-700 text-blue-200
                       shadow-glow-cyan hover:brightness-110 transition-all"
            aria-label="Connect to backend WebSocket"
          >
            <Wifi size={13} aria-hidden="true" />
            CONNECT
          </button>
        )}

        <button
          onClick={() => setCalibMode((v) => !v)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold
                      tracking-wide border transition-all
                      ${calibMode
                        ? "bg-amber-900/30 border-amber-500/40 text-amber-400 hover:brightness-110"
                        : "bg-bg-panel border-neu-border text-slate-400 hover:border-neu-borderLight"
                      }`}
          aria-label={calibMode ? "Cancel calibration" : "Start baseline calibration"}
          aria-pressed={calibMode}
        >
          <Target size={13} aria-hidden="true" />
          {calibMode
            ? `CALIBRATING… ${calibCount}/${MIN_CALIB}`
            : calibReady ? "RE-CALIBRATE" : "CALIBRATE BASELINE"}
        </button>

        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold
                     tracking-wide bg-bg-panel border border-neu-border text-red-500
                     hover:border-red-500/40 hover:bg-red-900/15 transition-all"
          aria-label="Reset analysis session"
        >
          <RefreshCw size={13} aria-hidden="true" />
          RESET SESSION
        </button>

        <span className="ml-auto text-[9px] text-slate-700 tracking-widest">
          NEURO-SYMMETRY INTELLIGENCE PLATFORM · RESEARCH USE ONLY
        </span>
      </footer>
    </div>
  );
}
