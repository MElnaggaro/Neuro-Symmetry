import { useCallback, useEffect, useRef, useState } from "react";

import FaceMeshOverlay from "./components/FaceMeshOverlay.jsx";
import HeatmapOverlay  from "./components/HeatmapOverlay.jsx";
import RiskIndicator   from "./components/RiskIndicator.jsx";
import ScoreGraph      from "./components/ScoreGraph.jsx";
import SymmetryGauge   from "./components/SymmetryGauge.jsx";
import TrajectoryLabel from "./components/TrajectoryLabel.jsx";
import XAIBreakdown    from "./components/XAIBreakdown.jsx";

// ── Config ────────────────────────────────────────────────────────────────────
const W = 640, H = 480;
const MAX_HISTORY   = 60;
const SEND_INTERVAL = 150;   // ~6-7 fps to backend
const MIN_CALIB     = 30;

const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/stream`;

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:          "#040b16",
  surface:     "#070f1e",
  panel:       "#0a1628",
  card:        "rgba(10,22,44,0.80)",
  border:      "#0f2040",
  borderLight: "#1a3a5c",
  glow:        "rgba(59,130,246,0.15)",
  textPrimary: "#e2e8f0",
  textMuted:   "#475569",
  textDim:     "#1e3a5c",
  accentBlue:  "#3b82f6",
  accentCyan:  "#06b6d4",
};

const RISK_BORDER = {
  NORMAL:    "#10b98155",
  MILD:      "#f59e0b55",
  HIGH_RISK: "#f9731655",
  CRITICAL:  "#ef444480",
};
const RISK_GLOW = {
  NORMAL:    "0 0 30px rgba(16,185,129,0.18)",
  MILD:      "0 0 30px rgba(245,158,11,0.20)",
  HIGH_RISK: "0 0 30px rgba(249,115,22,0.25)",
  CRITICAL:  "0 0 40px rgba(239,68,68,0.35)",
};

// ── Stat tile ─────────────────────────────────────────────────────────────────
function StatTile({ label, value, color = C.textPrimary, mono = false }) {
  return (
    <div style={{
      background: C.panel,
      border: `1px solid ${C.border}`,
      borderRadius: 10,
      padding: "9px 14px",
      minWidth: 90,
    }}>
      <div style={{ fontSize: 8, color: C.textMuted, letterSpacing: 2, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{
        fontSize: 15, fontWeight: 700, color,
        fontFamily: mono ? "'SF Mono','Fira Mono',monospace" : "inherit",
        letterSpacing: mono ? 1 : 0,
      }}>
        {value ?? "—"}
      </div>
    </div>
  );
}

// ── Section label ─────────────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 9, color: C.accentCyan, letterSpacing: 2.5,
      marginBottom: 10, display: "flex", alignItems: "center", gap: 8,
    }}>
      <span style={{ display: "inline-block", width: 16, height: 1, background: C.accentCyan, opacity: 0.5 }} />
      {children}
      <span style={{ flex: 1, height: 1, background: `linear-gradient(90deg,${C.accentCyan}30,transparent)` }} />
    </div>
  );
}

// ── Glass card ────────────────────────────────────────────────────────────────
function Card({ children, style = {} }) {
  return (
    <div style={{
      background: C.card,
      border: `1px solid ${C.border}`,
      borderRadius: 14,
      padding: "14px 16px",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      boxShadow: "0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03)",
      ...style,
    }}>
      {children}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const videoRef     = useRef(null);
  const meshCanvasRef = useRef(null);
  const captureRef   = useRef(null);
  const wsRef        = useRef(null);
  const lastSendRef  = useRef(0);
  const faceMeshRef  = useRef(null);
  const cameraRef    = useRef(null);

  const [result,      setResult]      = useState(null);
  const [history,     setHistory]     = useState([]);
  const [wsStatus,    setWsStatus]    = useState("disconnected");
  const [calibMode,   setCalibMode]   = useState(false);
  const [calibCount,  setCalibCount]  = useState(0);
  const [calibReady,  setCalibReady]  = useState(false);
  const [frameCount,  setFrameCount]  = useState(0);
  const [camError,    setCamError]    = useState(null);

  // ── WebSocket ───────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setWsStatus("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen  = () => setWsStatus("connected");
    ws.onclose = () => setWsStatus("disconnected");
    ws.onerror = () => setWsStatus("error");

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "calibrate") {
          setCalibCount(msg.frames_recorded ?? 0);
          if (msg.ready) { setCalibReady(true); setCalibMode(false); }
          return;
        }
        if (msg.status === "reset") return;

        setResult(msg);
        if (msg.symmetry_score != null) {
          setFrameCount((n) => n + 1);
          setHistory((h) => {
            const next = [...h, {
              frame: (h[h.length - 1]?.frame ?? 0) + 1,
              score: msg.symmetry_score,
            }];
            return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
          });
        }
      } catch { /* malformed JSON */ }
    };
    wsRef.current = ws;
  }, []);

  const sendFrame = useCallback((b64jpeg) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const payload = calibMode
      ? { type: "calibrate", image: b64jpeg }
      : { image: b64jpeg };
    ws.send(JSON.stringify(payload));
  }, [calibMode]);

  const resetSession = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
    setResult(null);
    setHistory([]);
    setFrameCount(0);
    setCalibCount(0);
    setCalibReady(false);
    setCalibMode(false);
  }, []);

  // ── MediaPipe + camera setup ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function setup() {
      const cap = document.createElement("canvas");
      cap.width  = W;
      cap.height = H;
      captureRef.current = cap;

      let FaceMesh, Camera;
      try {
        ({ FaceMesh } = await import("@mediapipe/face_mesh"));
        ({ Camera }   = await import("@mediapipe/camera_utils"));
      } catch {
        setCamError("MediaPipe failed to load. Check network connection.");
        return;
      }
      if (cancelled) return;

      const faceMesh = new FaceMesh({
        locateFile: (f) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`,
      });
      faceMesh.setOptions({
        maxNumFaces:            1,
        refineLandmarks:        true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence:  0.5,
      });

      faceMesh.onResults((res) => {
        if (cancelled) return;

        // Draw mesh on overlay canvas
        const mc = meshCanvasRef.current;
        if (mc) {
          const ctx = mc.getContext("2d");
          ctx.clearRect(0, 0, mc.width, mc.height);
          if (res.multiFaceLandmarks?.length) {
            const lm = res.multiFaceLandmarks[0];
            ctx.shadowColor = "#06b6d4";
            ctx.shadowBlur  = 2;
            ctx.fillStyle   = "rgba(6,182,212,0.65)";
            for (const pt of lm) {
              ctx.beginPath();
              ctx.arc(pt.x * mc.width, pt.y * mc.height, 1.2, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.shadowBlur = 0;
          }
        }

        // Throttled frame → WebSocket
        const now = performance.now();
        if (now - lastSendRef.current >= SEND_INTERVAL && videoRef.current) {
          lastSendRef.current = now;
          const ctx = captureRef.current?.getContext("2d");
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, W, H);
            const b64 = captureRef.current.toDataURL("image/jpeg", 0.6).split(",")[1];
            sendFrame(b64);
          }
        }
      });

      faceMeshRef.current = faceMesh;

      try {
        const camera = new Camera(videoRef.current, {
          onFrame: async () => {
            if (faceMeshRef.current && videoRef.current) {
              await faceMeshRef.current.send({ image: videoRef.current });
            }
          },
          width: W, height: H,
        });
        cameraRef.current = camera;
        camera.start();
        setCamError(null);
      } catch {
        setCamError("Camera access denied. Please allow webcam permission.");
      }

      connectWS();
    }

    setup().catch((e) => setCamError(String(e)));

    return () => {
      cancelled = true;
      cameraRef.current?.stop();
      faceMeshRef.current?.close();
      wsRef.current?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Derived values ──────────────────────────────────────────────────────────
  const score      = result?.symmetry_score  ?? 0;
  const riskLevel  = result?.risk_level      ?? "NORMAL";
  const trajectory = result?.trajectory      ?? "STABLE";
  const xai        = result?.xai             ?? [];
  const alert      = result?.alert           ?? false;
  const affected   = result?.affected_side   ?? null;
  const onsetSec   = result?.onset_seconds   ?? null;
  const confState  = result?.confirmation_state ?? "NORMAL";
  const latency    = result?.latency_ms      ?? null;
  const classLabel = result?.class_label     ?? null;
  const anomaly    = result?.anomaly_score   ?? null;
  const emaScore   = result?.ema_score       ?? null;

  const wsOk      = wsStatus === "connected";
  const vidBorder = RISK_BORDER[riskLevel] ?? RISK_BORDER.NORMAL;
  const vidGlow   = RISK_GLOW[riskLevel]   ?? RISK_GLOW.NORMAL;

  return (
    <div style={{
      fontFamily: "'Inter','Segoe UI',system-ui,sans-serif",
      background: C.bg,
      minHeight: "100vh",
      color: C.textPrimary,
      display: "flex",
      flexDirection: "column",
    }}>

      {/* ── Global CSS ── */}
      <style>{`
        *          { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track  { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb  { background: ${C.border}; border-radius: 2px; }
        @keyframes criticalPulse {
          0%,100% { opacity: 1; box-shadow: 0 0 20px #ef444440; }
          50%     { opacity: 0.7; box-shadow: 0 0 40px #ef444480; }
        }
        @keyframes dot {
          0%,100% { opacity: 1; }  50% { opacity: 0.2; }
        }
        @keyframes scanline {
          0%   { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>

      {/* ── Scanline FX ── */}
      <div style={{
        position: "fixed", top: 0, left: 0, right: 0, height: "60px",
        background: "linear-gradient(transparent, rgba(59,130,246,0.03), transparent)",
        pointerEvents: "none", zIndex: 0,
        animation: "scanline 8s linear infinite",
      }} />

      {/* ── Header ── */}
      <header style={{
        padding: "12px 24px",
        background: `linear-gradient(135deg, ${C.surface}, ${C.panel})`,
        borderBottom: `1px solid ${C.border}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "relative",
        zIndex: 10,
      }}>
        {/* Left: logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg,#1d4ed8,#0ea5e9)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, boxShadow: "0 0 12px #3b82f640",
          }}>⬡</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: 0.5, color: C.textPrimary }}>
              Neuro-Symmetry
            </div>
            <div style={{ fontSize: 9, color: C.accentCyan, letterSpacing: 3 }}>
              FACIAL ANALYSIS v2.0
            </div>
          </div>
        </div>

        {/* Center: status bar */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {latency != null && (
            <span style={{
              fontSize: 10, padding: "3px 10px", borderRadius: 20,
              background: C.panel, border: `1px solid ${C.border}`,
              color: latency < 50 ? "#10b981" : latency < 150 ? "#f59e0b" : "#ef4444",
              fontFamily: "monospace",
            }}>
              {latency.toFixed(0)}ms
            </span>
          )}
          <span style={{
            fontSize: 10, padding: "3px 10px", borderRadius: 20,
            background: wsOk ? "rgba(16,185,129,0.08)" : "rgba(249,115,22,0.08)",
            border: `1px solid ${wsOk ? "#10b98140" : "#f9731640"}`,
            color: wsOk ? "#34d399" : "#fb923c",
            display: "flex", alignItems: "center", gap: 5,
          }}>
            <span style={{ animation: wsOk ? "dot 1.5s infinite" : "none" }}>●</span>
            {wsOk ? "BACKEND CONNECTED" : wsStatus.toUpperCase()}
          </span>
          {calibReady && (
            <span style={{
              fontSize: 10, padding: "3px 10px", borderRadius: 20,
              background: "rgba(6,182,212,0.08)",
              border: "1px solid #06b6d440",
              color: "#22d3ee",
            }}>
              ✓ BASELINE CALIBRATED
            </span>
          )}
        </div>

        {/* Right: frame counter */}
        <div style={{ fontSize: 10, color: C.textMuted, fontFamily: "monospace" }}>
          FRAME {String(frameCount).padStart(5, "0")}
        </div>
      </header>

      {/* ── Body ── */}
      <div style={{
        display: "flex",
        flex: 1,
        gap: 16,
        padding: 16,
        overflow: "hidden",
        position: "relative",
        zIndex: 1,
      }}>

        {/* ── Left column: video + overlays ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Video card */}
          <div style={{
            position: "relative",
            width: W,
            height: H,
            borderRadius: 16,
            overflow: "hidden",
            border: `1.5px solid ${vidBorder}`,
            boxShadow: vidGlow,
            background: "#000",
            flexShrink: 0,
            transition: "border-color 0.5s, box-shadow 0.5s",
          }}>
            {/* Corner markers */}
            {[["top:0,left:0","borderTop,borderLeft"],["top:0,right:0","borderTop,borderRight"],
              ["bottom:0,left:0","borderBottom,borderLeft"],["bottom:0,right:0","borderBottom,borderRight"],
            ].map(([pos], i) => {
              const p = Object.fromEntries(pos.split(",").map((s) => s.split(":")));
              return (
                <div key={i} style={{
                  position: "absolute", width: 18, height: 18, zIndex: 5,
                  ...p,
                  borderTop:    i < 2  ? `2px solid ${C.accentCyan}` : "none",
                  borderBottom: i >= 2 ? `2px solid ${C.accentCyan}` : "none",
                  borderLeft:   i % 2 === 0 ? `2px solid ${C.accentCyan}` : "none",
                  borderRight:  i % 2 === 1 ? `2px solid ${C.accentCyan}` : "none",
                }} />
              );
            })}

            <video
              ref={videoRef}
              style={{ width: W, height: H, display: "block", objectFit: "cover", transform: "scaleX(-1)" }}
              autoPlay playsInline muted
            />
            <FaceMeshOverlay ref={meshCanvasRef} width={W} height={H} />
            <HeatmapOverlay  xai={xai} width={W} height={H} />

            {/* Camera error banner */}
            {camError && (
              <div style={{
                position: "absolute", inset: 0, display: "flex",
                alignItems: "center", justifyContent: "center",
                background: "rgba(4,11,22,0.85)",
                fontSize: 12, color: "#f87171", textAlign: "center", padding: 24,
              }}>
                {camError}
              </div>
            )}

            {/* Trajectory badge - bottom left */}
            <div style={{ position: "absolute", bottom: 14, left: 14, display: "flex", flexDirection: "column", gap: 7, zIndex: 4 }}>
              <TrajectoryLabel trajectory={trajectory} />
              {onsetSec != null && (
                <div style={{
                  fontSize: 10, color: "#f87171",
                  background: "rgba(59,0,0,0.75)",
                  border: "1px solid #ef444440",
                  padding: "3px 10px", borderRadius: 8,
                  backdropFilter: "blur(6px)",
                  letterSpacing: 1,
                }}>
                  ⏱ ONSET {onsetSec.toFixed(1)}s
                </div>
              )}
            </div>

            {/* Affected side - top right */}
            {affected && (
              <div style={{
                position: "absolute", top: 14, right: 14, zIndex: 4,
                fontSize: 9, color: C.accentCyan,
                background: "rgba(6,182,212,0.08)",
                border: "1px solid #06b6d430",
                padding: "4px 10px", borderRadius: 8,
                backdropFilter: "blur(6px)",
                letterSpacing: 2,
              }}>
                SIDE: {affected}
              </div>
            )}

            {/* Confirmation state - bottom right */}
            {confState !== "NORMAL" && (
              <div style={{
                position: "absolute", bottom: 14, right: 14, zIndex: 4,
                fontSize: 9, letterSpacing: 1,
                color: confState === "CONFIRMED" ? "#f87171" : "#fbbf24",
                background: confState === "CONFIRMED" ? "rgba(59,0,0,0.75)" : "rgba(55,33,0,0.75)",
                border: `1px solid ${confState === "CONFIRMED" ? "#ef444440" : "#f59e0b40"}`,
                padding: "4px 10px", borderRadius: 8,
                backdropFilter: "blur(6px)",
              }}>
                {confState}
              </div>
            )}
          </div>

          {/* Quick stats row */}
          <div style={{ display: "flex", gap: 8 }}>
            <StatTile label="CLASS" value={classLabel ?? "—"} color={C.accentCyan} />
            <StatTile label="EMA SCORE" value={emaScore?.toFixed(3) ?? "—"} color={C.textPrimary} mono />
            {anomaly != null && (
              <StatTile label="ANOMALY Δz" value={anomaly.toFixed(2)} color={Math.abs(anomaly) > 2 ? "#f87171" : "#34d399"} mono />
            )}
            <StatTile label="QUALITY" value={result?.quality_code ?? "—"} color={result?.quality_code === "OK" ? "#34d399" : "#f59e0b"} />
          </div>
        </div>

        {/* ── Right column: analysis panels ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>

          {/* Risk + gauge row */}
          <Card style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <SymmetryGauge score={score} riskLevel={riskLevel} />
            <div style={{ flex: 1 }}>
              <RiskIndicator riskLevel={riskLevel} alert={alert} />
            </div>
          </Card>

          {/* Score graph */}
          <Card>
            <SectionLabel>SYMMETRY TIMELINE</SectionLabel>
            <ScoreGraph history={history} trajectory={trajectory} />
          </Card>

          {/* XAI breakdown */}
          <Card style={{ flex: 1, overflowY: "auto" }}>
            <SectionLabel>FEATURE CONTRIBUTIONS  (XAI)</SectionLabel>
            <XAIBreakdown xai={xai} />
          </Card>

        </div>
      </div>

      {/* ── Footer / controls ── */}
      <footer style={{
        padding: "10px 20px",
        background: C.surface,
        borderTop: `1px solid ${C.border}`,
        display: "flex",
        alignItems: "center",
        gap: 10,
        position: "relative",
        zIndex: 10,
      }}>
        {!wsOk && (
          <button
            onClick={connectWS}
            style={{
              padding: "7px 18px", borderRadius: 8, border: "none", cursor: "pointer",
              background: "linear-gradient(135deg,#1d4ed8,#0ea5e9)",
              color: "#dbeafe", fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
              boxShadow: "0 0 12px #3b82f640",
            }}>
            CONNECT
          </button>
        )}

        <button
          onClick={() => setCalibMode((v) => !v)}
          style={{
            padding: "7px 18px", borderRadius: 8, cursor: "pointer",
            background: calibMode
              ? "linear-gradient(135deg,#78350f,#92400e)"
              : `linear-gradient(135deg,${C.panel},${C.card})`,
            border: `1px solid ${calibMode ? "#f59e0b40" : C.border}`,
            color: calibMode ? "#fbbf24" : C.textMuted,
            fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
          }}>
          {calibMode
            ? `CALIBRATING… ${calibCount}/${MIN_CALIB}`
            : calibReady ? "RE-CALIBRATE" : "CALIBRATE BASELINE"}
        </button>

        <button
          onClick={resetSession}
          style={{
            padding: "7px 18px", borderRadius: 8, cursor: "pointer",
            background: `linear-gradient(135deg,${C.panel},${C.card})`,
            border: `1px solid ${C.border}`,
            color: "#f87171", fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
          }}>
          RESET SESSION
        </button>

        <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
          <div style={{ fontSize: 10, color: C.textDim, fontFamily: "monospace" }}>
            NEURO-SYMMETRY INTELLIGENCE PLATFORM  ·  RESEARCH USE ONLY
          </div>
        </div>
      </footer>
    </div>
  );
}
