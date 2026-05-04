import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence }                            from "framer-motion";
import { Wifi, WifiOff, RefreshCw, Target, AlertTriangle } from "lucide-react";

import PredictiveTriage                from "@/modules/PredictiveTriage";
import { useWebSocket }                from "@/hooks/useWebSocket";
import { useFaceTracking }             from "@/providers/FaceTrackingProvider";
import { useDialog }                   from "@/components/ui/DialogManager";
import { Card }                        from "@/components/ui/Card";
import { SectionLabel, StatTile }      from "@/components/ui";
import { FaceMeshOverlay, HeatmapOverlay } from "@/components/video";
import RiskIndicator                   from "@/components/panels/RiskIndicator";
import TrajectoryLabel                 from "@/components/panels/TrajectoryLabel";
import SymmetryGauge                   from "@/components/panels/SymmetryGauge";
import ScoreGraph                      from "@/components/panels/ScoreGraph";
import XAIBreakdown                    from "@/components/panels/XAIBreakdown";
import { EmptyState }                  from "@/components/ui/EmptyState";
import { SkeletonGauge, SkeletonGraph, SkeletonXAI } from "@/components/ui/Skeleton";
import { toastInfo, toastSuccess, toastError, toastCalibration } from "@/utils/toast";
import { AnimatedLogo }                from "@/components/ui/AnimatedLogo";
import { CalibrationRing }             from "@/components/ui/CalibrationRing";
import { useCommands }                 from "@/hooks/useCommands";
import logger                          from "@/utils/logger";
import { clearCanvas, drawBiometricOverlay } from "@/utils/renderer";
import { APP_CONFIG, VIDEO_CONFIG }    from "@/config";
import type { HistoryPoint, RiskLevel, TrajectoryState } from "@/types/analysis";

// ── Constants ─────────────────────────────────────────────────────────────────

const { MAX_HISTORY, MIN_CALIBRATION_FRAMES, SEND_INTERVAL_MS } = APP_CONFIG;
const { WIDTH, HEIGHT }                                          = VIDEO_CONFIG;

const NAV_LINKS = [
  { href: "/sentinel.html", label: "Sentinel"    },
  { href: "/mirror.html",   label: "Mirror"      },
  { href: "/game.html",     label: "Face-Joypad" },
  { href: "#/tracker",      label: "Tracker"     },
] as const;

// Tracker reads `ns_assessments` for session history. Map the live-app risk
// level to the class strings the Tracker badges expect.
const ASSESSMENTS_KEY = "ns_assessments";
const ASSESSMENT_FLUSH_MS = 3000;
const RISK_TO_CLASS: Record<RiskLevel, string> = {
  NORMAL:    "Normal",
  MILD:      "Mild",
  HIGH_RISK: "High Risk",
  CRITICAL:  "Critical",
};

type StoredAssessment = {
  date:             string;
  type:             "live";
  source:           "main_app";
  session_id:       string;
  class:            string;
  probability:      number | null;
  symmetry_score:   number;
  frames:           number;
  game_activations: null;
};

function upsertAssessment(record: StoredAssessment): void {
  try {
    const raw  = localStorage.getItem(ASSESSMENTS_KEY);
    const list = raw ? (JSON.parse(raw) as StoredAssessment[]) : [];
    const idx  = list.findIndex(
      (e) => (e as StoredAssessment).session_id === record.session_id,
    );
    if (idx >= 0) list[idx] = record;
    else          list.push(record);
    if (list.length > 500) list.splice(0, list.length - 500);
    localStorage.setItem(ASSESSMENTS_KEY, JSON.stringify(list));
  } catch {
    // Quota or parse errors: drop silently — the tracker still has older data.
  }
}

const RISK_VIDEO_CLASS: Record<RiskLevel, string> = {
  NORMAL:    "risk-border-normal",
  MILD:      "risk-border-mild",
  HIGH_RISK: "risk-border-high",
  CRITICAL:  "risk-border-critical",
};

const ELEVATED_RISK: ReadonlySet<RiskLevel> = new Set(["MILD", "HIGH_RISK", "CRITICAL"]);

const RISK_RANK: Record<RiskLevel, number> = { NORMAL: 0, MILD: 1, HIGH_RISK: 2, CRITICAL: 3 };

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Providers ──────────────────────────────────────────────────────────────
  const {
    videoRef, meshRef,
    status:          camStatus,
    error:           camError,
    trackingQuality,
    start,
    subscribe,
  } = useFaceTracking();

  const { openDialog, closeDialog } = useDialog();

  const camReady = camStatus === "ready";

  // ── Session state ──────────────────────────────────────────────────────────
  const [history,    setHistory]    = useState<HistoryPoint[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  const [calibMode,      setCalibMode]      = useState(false);
  const [calibCount,     setCalibCount]     = useState(0);
  const [calibReady,     setCalibReady]     = useState(false);
  const [lastGoodScore,  setLastGoodScore]  = useState<number>(0);

  // ── WebSocket ──────────────────────────────────────────────────────────────
  const { status: wsStatus, connect, sendFrame, resetSession, lastResult, lastCalib } = useWebSocket();

  useEffect(() => { logger.info("WebSocket", `status: ${wsStatus}`); }, [wsStatus]);

  // Toast on WebSocket connect/disconnect transitions, but skip the very first
  // "disconnected" mount value to avoid noise on initial load.
  const prevWsStatusRef = useRef<string | null>(null);
  useEffect(() => {
    const prev = prevWsStatusRef.current;
    prevWsStatusRef.current = wsStatus;
    if (prev === null) return;
    if (wsStatus === "connected" && prev !== "connected") {
      toastSuccess("Stream connected");
    } else if (wsStatus === "disconnected" && prev === "connected") {
      toastInfo("Stream disconnected");
    } else if (wsStatus === "error") {
      toastError("Stream error — check the backend");
    }
  }, [wsStatus]);

  useEffect(() => {
    if (!lastCalib) return;
    setCalibCount(lastCalib.frames_recorded);
    if (lastCalib.ready) { setCalibReady(true); setCalibMode(false); }
  }, [lastCalib]);

  // Calibration toasts: loading while in calibMode, success when ready.
  const wasCalibModeRef = useRef(false);
  const wasCalibReadyRef = useRef(false);
  useEffect(() => {
    if (calibMode && !wasCalibModeRef.current) toastCalibration.start();
    if (!calibMode && wasCalibModeRef.current && !calibReady) toastCalibration.cancel();
    wasCalibModeRef.current = calibMode;
  }, [calibMode, calibReady]);
  useEffect(() => {
    if (calibReady && !wasCalibReadyRef.current) toastCalibration.done();
    wasCalibReadyRef.current = calibReady;
  }, [calibReady]);

  // Rolling session aggregate persisted to localStorage so the Tracker page
  // (`#/tracker`) can show data from the main live-analysis app, not just
  // from the Sentinel/Game pages.
  const sessionIdRef    = useRef<string | null>(null);
  const sessionAggRef   = useRef({ sum: 0, count: 0, peakRisk: 0, peakConf: 0, lastClass: "" });
  const lastFlushRef    = useRef<number>(0);

  useEffect(() => {
    const s = lastResult?.symmetry_score;
    if (s == null) return;
    setLastGoodScore(s);
    setFrameCount((n) => n + 1);
    setHistory((h) => {
      const next: HistoryPoint[] = [...h, { frame: (h.length > 0 ? h[h.length - 1].frame : 0) + 1, score: s }];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });

    // ── Tracker persistence (throttled upsert) ──────────────────────────────
    if (sessionIdRef.current === null) {
      sessionIdRef.current = `live-${Date.now()}`;
    }
    const agg     = sessionAggRef.current;
    const risk    = (lastResult?.risk_level ?? "NORMAL") as RiskLevel;
    const riskNum = RISK_RANK[risk];
    agg.sum   += s;
    agg.count += 1;
    if (riskNum >= agg.peakRisk) {
      agg.peakRisk  = riskNum;
      agg.lastClass = RISK_TO_CLASS[risk];
    }
    const conf = lastResult?.confidence ?? 0;
    if (conf > agg.peakConf) agg.peakConf = conf;

    const now = performance.now();
    if (now - lastFlushRef.current >= ASSESSMENT_FLUSH_MS) {
      lastFlushRef.current = now;
      upsertAssessment({
        date:             new Date().toISOString(),
        type:             "live",
        source:           "main_app",
        session_id:       sessionIdRef.current,
        class:            agg.lastClass || "Normal",
        probability:      agg.peakConf || null,
        symmetry_score:   agg.sum / agg.count,
        frames:           agg.count,
        game_activations: null,
      });
    }
  }, [lastResult]);

  // ── Camera initialisation ──────────────────────────────────────────────────
  useEffect(() => { start(); }, [start]);
  useEffect(() => { if (camReady) connect(); }, [camReady, connect]);

  // ── Frame capture → WebSocket ──────────────────────────────────────────────
  // Throttled: capture an offscreen JPEG at most every SEND_INTERVAL_MS.
  const calibModeRef = useRef(calibMode);
  useEffect(() => { calibModeRef.current = calibMode; }, [calibMode]);

  useEffect(() => {
    const cap    = document.createElement("canvas");
    cap.width    = WIDTH;
    cap.height   = HEIGHT;
    const capCtx = cap.getContext("2d")!;
    let lastSend = 0;

    return subscribe(() => {
      const now = performance.now();
      if (now - lastSend < SEND_INTERVAL_MS) return;
      lastSend = now;
      const vid = videoRef.current;
      if (!vid) return;
      capCtx.drawImage(vid, 0, 0, WIDTH, HEIGHT);
      sendFrame(cap.toDataURL("image/jpeg", 0.7), calibModeRef.current);
    });
  }, [subscribe, videoRef, sendFrame]);

  // ── Biometric HUD overlay ──────────────────────────────────────────────────
  // Per-frame draw: called at MediaPipe's rate (~30 FPS) via subscribe().
  useEffect(() => {
    return subscribe((rawLandmarks) => {
      const canvas = meshRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      clearCanvas(ctx, canvas.width, canvas.height);
      drawBiometricOverlay(ctx, rawLandmarks, canvas.width, canvas.height);
    });
  }, [subscribe, meshRef]);

  // Clear the overlay as soon as the face is lost so stale HUD doesn't linger.
  useEffect(() => {
    if (trackingQuality !== "UNRELIABLE") return;
    const canvas = meshRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx) clearCanvas(ctx, canvas.width, canvas.height);
  }, [trackingQuality, meshRef]);

  // Final flush of the in-progress session so it survives a tab close /
  // navigation away even if the throttle window has not elapsed.
  const flushSession = useCallback(() => {
    const id  = sessionIdRef.current;
    const agg = sessionAggRef.current;
    if (id === null || agg.count === 0) return;
    upsertAssessment({
      date:             new Date().toISOString(),
      type:             "live",
      source:           "main_app",
      session_id:       id,
      class:            agg.lastClass || "Normal",
      probability:      agg.peakConf || null,
      symmetry_score:   agg.sum / agg.count,
      frames:           agg.count,
      game_activations: null,
    });
  }, []);

  useEffect(() => {
    const onHide = () => flushSession();
    window.addEventListener("pagehide", onHide);
    return () => {
      window.removeEventListener("pagehide", onHide);
      flushSession();
    };
  }, [flushSession]);

  // ── Reset ──────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    flushSession();
    sessionIdRef.current  = null;
    sessionAggRef.current = { sum: 0, count: 0, peakRisk: 0, peakConf: 0, lastClass: "" };
    lastFlushRef.current  = 0;

    resetSession();
    setHistory([]);
    setFrameCount(0);
    setCalibMode(false);
    setCalibCount(0);
    setCalibReady(false);
    toastInfo("Session reset");
    logger.warn("App", "Session reset");
  }, [resetSession, flushSession]);

  // ── Command palette registration ───────────────────────────────────────────
  const { register: registerCommand } = useCommands();
  useEffect(() => {
    const unsubs = [
      registerCommand({
        id: "nav.live", group: "Navigate", label: "Go to Live Analysis", shortcut: "⌘1",
        run: () => { window.location.hash = "#/"; },
      }),
      registerCommand({
        id: "nav.tracker", group: "Navigate", label: "Go to Progress Tracker", shortcut: "⌘2",
        run: () => { window.location.hash = "#/tracker"; },
      }),
      registerCommand({
        id: "nav.sentinel", group: "Navigate", label: "Open Sentinel (early-warning)",
        run: () => { window.location.href = "/sentinel.html"; },
      }),
      registerCommand({
        id: "nav.mirror", group: "Navigate", label: "Open AR Mirror",
        run: () => { window.location.href = "/mirror.html"; },
      }),
      registerCommand({
        id: "nav.game", group: "Navigate", label: "Open Face-Joypad",
        run: () => { window.location.href = "/game.html"; },
      }),
      registerCommand({
        id: "session.calibrate", group: "Session", label: "Calibrate baseline", shortcut: "C",
        run: () => setCalibMode((v) => !v),
      }),
      registerCommand({
        id: "session.reset", group: "Session", label: "Reset session", shortcut: "R",
        run: handleReset,
      }),
      registerCommand({
        id: "session.connect", group: "Session", label: "Reconnect WebSocket",
        run: () => connect(),
      }),
      registerCommand({
        id: "ui.restartOnboarding", group: "Interface", label: "Restart onboarding",
        run: () => {
          localStorage.removeItem("ns_onboarding_done");
          toastInfo("Onboarding will run on next reload");
        },
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [registerCommand, handleReset, connect]);

  // ── Derived display values ─────────────────────────────────────────────────
  const r = lastResult;

  const riskLevel  = (r?.risk_level   ?? "NORMAL")  as RiskLevel;
  const trajectory = (r?.trajectory   ?? "STABLE")  as TrajectoryState;
  const xai        =  r?.xai            ?? [];
  const alert      =  r?.alert          ?? false;
  const affected   =  r?.affected_side  ?? null;
  const onsetSec   =  r?.onset_seconds  ?? null;
  const confState  =  r?.confirmation_state ?? "NORMAL";
  const latency    =  r?.latency_ms     ?? null;
  const classLabel =  r?.class_label    ?? null;
  const anomaly    =  r?.anomaly_score  ?? null;
  const emaScore   =  r?.ema_score      ?? null;
  const quality    =  r?.quality_code   ?? null;

  const wsOk           = wsStatus === "connected";
  const videoRiskClass = RISK_VIDEO_CLASS[riskLevel];
  const isElevated     = ELEVATED_RISK.has(riskLevel);

  const anomalyColor = useMemo(
    () => anomaly === null ? "#64748b" : Math.abs(anomaly) > 2 ? "#f87171" : "#34d399",
    [anomaly],
  );

  const qualityColor = quality === "OK" ? "#34d399" : quality === "DEGRADED" ? "#f59e0b" : "#f87171";

  const latencyClass = latency === null ? "" :
    latency < 60  ? "bg-emerald-900/30 border-emerald-500/30 text-emerald-400" :
    latency < 150 ? "bg-amber-900/30 border-amber-500/30 text-amber-400"     :
                    "bg-red-900/30 border-red-500/30 text-red-400";

  // ── Triage modal (portal) ──────────────────────────────────────────────────
  const [triageDismissed, setTriageDismissed] = useState(false);
  const prevRiskRef = useRef<RiskLevel>("NORMAL");

  useEffect(() => {
    if (prevRiskRef.current !== riskLevel && isElevated) setTriageDismissed(false);
    prevRiskRef.current = riskLevel;
  }, [riskLevel, isElevated]);

  const showTriage = !triageDismissed && isElevated;

  const dismissTriage = useCallback(() => {
    setTriageDismissed(true);
    closeDialog("triage");
  }, [closeDialog]);

  useEffect(() => {
    if (!showTriage) {
      closeDialog("triage");
      return;
    }

    openDialog(
      "triage",
      <PredictiveTriage
        riskLevel={riskLevel}
        onDismiss={dismissTriage}
      />,
    );
  }, [closeDialog, dismissTriage, openDialog, riskLevel, showTriage]);

  useEffect(() => {
    if (camError) {
      toastError(`Camera: ${camError}`);
      logger.error("Camera", "Init failed", { error: camError });
    }
  }, [camError]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-bg-base text-slate-200 font-sans overflow-hidden">

      {/* Scanline FX */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
        <div className="absolute left-0 right-0 h-16 bg-gradient-to-b from-transparent via-accent-blue/[0.025] to-transparent animate-scan" />
      </div>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="relative z-10 flex items-center justify-between px-5 py-3 bg-bg-surface border-b border-neu-border" role="banner">

        <div className="flex items-center gap-3" data-tour="logo">
          <AnimatedLogo />
          <div>
            <div className="text-[15px] font-extrabold tracking-wide text-slate-200 leading-tight">Neuro-Symmetry</div>
            <div className="text-micro font-semibold tracking-cyber text-accent-cyan uppercase">Facial Analysis Platform v2.0</div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {latency !== null && (
            <span className={`text-[10px] font-mono px-3 py-1 rounded-full border ${latencyClass}`}>
              {latency.toFixed(0)} ms
            </span>
          )}

          <motion.span
            layout
            className={`flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1 rounded-full border ${
              wsOk
                ? "bg-emerald-900/20 border-emerald-500/30 text-emerald-400"
                : "bg-orange-900/20 border-orange-500/30 text-orange-400"
            }`}
            aria-live="polite"
            aria-label={`WebSocket status: ${wsStatus}`}
          >
            {wsOk
              ? <Wifi    size={11} strokeWidth={2.5} aria-hidden="true" />
              : <WifiOff size={11} strokeWidth={2.5} aria-hidden="true" />
            }
            <span className={wsOk ? "animate-blink" : ""}>{wsOk ? "CONNECTED" : wsStatus.toUpperCase()}</span>
          </motion.span>

          {calibReady && (
            <span className="flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1 rounded-full bg-cyan-900/20 border border-cyan-500/30 text-cyan-400">
              <Target size={11} strokeWidth={2.5} aria-hidden="true" />
              BASELINE READY
            </span>
          )}
        </div>

        <nav className="hidden md:flex items-center gap-1" aria-label="Tool navigation">
          {NAV_LINKS.map(({ href, label }) => (
            <a
              key={href}
              href={href}
              data-tour={label === "Tracker" ? "tracker" : undefined}
              className="px-3 py-1 rounded-lg text-[10px] font-semibold tracking-wide text-slate-500 border border-transparent hover:text-accent-cyan hover:border-neu-border transition-all"
            >
              {label}
            </a>
          ))}
        </nav>

        {isElevated && (
          <motion.button
            initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
            onClick={() => setTriageDismissed(false)}
            className="flex items-center gap-1.5 text-[10px] font-bold px-3 py-1 rounded-full border bg-red-900/25 border-red-500/40 text-red-400 hover:bg-red-900/40 transition-all"
            aria-label="Open Predictive Triage"
          >
            <AlertTriangle size={11} aria-hidden="true" />
            TRIAGE
          </motion.button>
        )}

        <div className="text-[10px] font-mono text-slate-600 tabular-nums" aria-label="Frame count">
          FRAME {String(frameCount).padStart(6, "0")}
        </div>
      </header>

      {/* ── Main ──────────────────────────────────────────────────────────── */}
      <main className="relative z-1 flex flex-1 gap-4 p-4 min-h-0" role="main">

        {/* Left — camera + overlays */}
        <div className="flex flex-col gap-3 shrink-0">
          <Card tone="elevated">
            <div
              className={`relative rounded-2xl overflow-hidden border-2 transition-all duration-500 risk-video-border ${videoRiskClass}`}
              style={{ width: WIDTH, height: HEIGHT }}
              role="region"
              aria-label="Camera feed with facial analysis overlays"
            >
              {/* Corner fiducials */}
              {(["tl", "tr", "bl", "br"] as const).map((pos) => (
                <div key={pos} aria-hidden="true"
                  className={[
                    "absolute w-[18px] h-[18px] z-10 pointer-events-none border-accent-cyan/60",
                    pos === "tl" ? "top-2 left-2 border-t-2 border-l-2"    : "",
                    pos === "tr" ? "top-2 right-2 border-t-2 border-r-2"   : "",
                    pos === "bl" ? "bottom-2 left-2 border-b-2 border-l-2" : "",
                    pos === "br" ? "bottom-2 right-2 border-b-2 border-r-2": "",
                  ].join(" ")}
                />
              ))}

              <video ref={videoRef} className="w-full h-full object-cover block"
                style={{ transform: "scaleX(-1)" }} autoPlay playsInline muted aria-label="Live camera feed" />

              <FaceMeshOverlay ref={meshRef} width={WIDTH} height={HEIGHT} />
              <HeatmapOverlay  xai={xai} width={WIDTH} height={HEIGHT} />

              <AnimatePresence>
                {camError && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="absolute inset-0 flex items-center justify-center bg-bg-base/85 text-[12px] text-red-400 text-center px-6"
                    role="alert">
                    {camError}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Trajectory + onset — bottom-left */}
              <div className="absolute bottom-3 left-3 flex flex-col gap-2 z-10">
                <TrajectoryLabel trajectory={trajectory} />
                {onsetSec !== null && (
                  <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                    className="text-[10px] text-red-400 font-bold tracking-widest bg-red-900/75 border border-red-500/40 px-2 py-1 rounded-lg glass">
                    ⏱ ONSET {onsetSec.toFixed(1)}s
                  </motion.div>
                )}
              </div>

              {/* Affected side — top-right */}
              {affected && (
                <div className="absolute top-3 right-3 z-10 text-[9px] font-semibold tracking-[0.2em] text-accent-cyan bg-cyan-900/15 border border-accent-cyan/25 px-2.5 py-1.5 rounded-lg glass">
                  SIDE: {affected}
                </div>
              )}

              {/* Confirmation state — bottom-right */}
              <AnimatePresence>
                {confState !== "NORMAL" && (
                  <motion.div
                    key={confState}
                    initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                    className={`absolute bottom-3 right-3 z-10 text-[9px] font-bold tracking-widest px-2.5 py-1.5 rounded-lg glass ${
                      confState === "CONFIRMED"
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
          </Card>

          {/* Stats strip */}
          <div className="flex gap-2 flex-wrap">
            <StatTile label="CLASS"      value={classLabel}                     color="#22d3ee"    />
            <StatTile label="EMA SCORE"  value={emaScore?.toFixed(3) ?? null}   color="#e2e8f0" mono />
            <StatTile label="ANOMALY Δz" value={anomaly?.toFixed(2) ?? null}    color={anomalyColor} mono />
            <StatTile label="QUALITY"    value={quality}                         color={qualityColor} />
          </div>
        </div>

        {/* Right — analysis panels */}
        <div className="flex flex-col gap-3 flex-1 min-w-0 min-h-0">
          <Card tone="elevated">
            <div data-tour="gauge">
              {wsStatus === "connecting" || (wsOk && lastResult === null) ? (
                <SkeletonGauge />
              ) : (
                <div className="flex items-center gap-4">
                  <SymmetryGauge score={lastGoodScore} riskLevel={riskLevel} />
                  <div className="flex-1">
                    <RiskIndicator riskLevel={riskLevel} alert={alert} />
                  </div>
                </div>
              )}
            </div>
          </Card>

          <Card>
            <SectionLabel>Symmetry Timeline — Last {MAX_HISTORY} Frames</SectionLabel>
            {history.length < 3 ? (
              <SkeletonGraph />
            ) : (
              <ScoreGraph history={history} trajectory={trajectory} />
            )}
          </Card>

          <Card flex>
            <SectionLabel>Feature Contributions (XAI)</SectionLabel>
            <div className="overflow-y-auto pr-1">
              {wsOk && lastResult === null ? (
                <SkeletonXAI />
              ) : trackingQuality === "UNRELIABLE" || xai.length === 0 ? (
                <EmptyState variant="no-face" />
              ) : (
                <XAIBreakdown xai={xai} />
              )}
            </div>
          </Card>
        </div>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="relative z-10 flex items-center gap-3 px-5 py-2.5 bg-bg-surface border-t border-neu-border" role="contentinfo">
        {!wsOk && (
          <button onClick={connect}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold tracking-wide bg-gradient-to-r from-blue-800 to-sky-700 text-blue-200 shadow-glow-cyan hover:brightness-110 transition-all"
            aria-label="Connect to backend WebSocket">
            <Wifi size={13} aria-hidden="true" /> CONNECT
          </button>
        )}

        <div data-tour="calibrate">
          <CalibrationRing count={calibCount} total={MIN_CALIBRATION_FRAMES} active={calibMode}>
            <button
              onClick={() => setCalibMode((v) => !v)}
              aria-label={calibMode ? "Cancel calibration" : "Start baseline calibration"}
              aria-pressed={calibMode}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold tracking-wide border transition-all ring-cyber ${
                calibMode
                  ? "bg-amber-900/30 border-amber-500/40 text-amber-400 hover:brightness-110"
                  : "bg-bg-panel border-neu-border text-slate-400 hover:border-neu-borderLight"
              }`}
            >
              <Target size={13} aria-hidden="true" />
              {calibMode ? `CALIBRATING… ${calibCount}/${MIN_CALIBRATION_FRAMES}` : calibReady ? "RE-CALIBRATE" : "CALIBRATE BASELINE"}
            </button>
          </CalibrationRing>
        </div>

        <button onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-bold tracking-wide bg-bg-panel border border-neu-border text-red-500 hover:border-red-500/40 hover:bg-red-900/15 transition-all"
          aria-label="Reset analysis session">
          <RefreshCw size={13} aria-hidden="true" /> RESET SESSION
        </button>

        <span className="ml-auto text-[9px] text-slate-700 tracking-widest">
          NEURO-SYMMETRY INTELLIGENCE PLATFORM · RESEARCH USE ONLY
        </span>
      </footer>
    </div>
  );
}
