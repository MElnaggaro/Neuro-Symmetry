import { useEffect, useRef } from "react";

const PALETTE = {
  NORMAL:    { bg: "rgba(16,185,129,0.08)",  border: "#10b981", glow: "#10b98130", fg: "#34d399", badge: "#065f46" },
  MILD:      { bg: "rgba(245,158,11,0.08)",  border: "#f59e0b", glow: "#f59e0b30", fg: "#fbbf24", badge: "#78350f" },
  HIGH_RISK: { bg: "rgba(249,115,22,0.08)",  border: "#f97316", glow: "#f9731640", fg: "#fb923c", badge: "#7c2d12" },
  CRITICAL:  { bg: "rgba(239,68,68,0.12)",   border: "#ef4444", glow: "#ef444460", fg: "#f87171", badge: "#450a0a" },
};

const ICONS = { NORMAL: "✦", MILD: "◈", HIGH_RISK: "▲", CRITICAL: "⚠" };

export default function RiskIndicator({ riskLevel = "NORMAL", alert = false }) {
  const prevRef  = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    if (riskLevel === "CRITICAL" && prevRef.current !== "CRITICAL") beepAlert();
    prevRef.current = riskLevel;
  }, [riskLevel]);

  function beepAlert() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = audioRef.current ?? (audioRef.current = new AudioCtx());
      const t = ctx.currentTime;
      [880, 660, 880].forEach((freq, i) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.22, t + i * 0.18);
        gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.18 + 0.16);
        osc.start(t + i * 0.18);
        osc.stop(t + i * 0.18 + 0.16);
      });
    } catch { /* AudioContext not available */ }
  }

  const c = PALETTE[riskLevel] ?? PALETTE.NORMAL;
  const isCritical = riskLevel === "CRITICAL";

  return (
    <div style={{
      position: "relative",
      padding: "16px 20px",
      borderRadius: 14,
      background: c.bg,
      border: `1.5px solid ${c.border}`,
      boxShadow: `0 0 20px ${c.glow}, inset 0 1px 0 rgba(255,255,255,0.04)`,
      overflow: "hidden",
      animation: isCritical && alert ? "criticalPulse 1.2s ease-in-out infinite" : "none",
      minWidth: 160,
    }}>
      {/* Glow bar */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, transparent, ${c.border}, transparent)`,
      }} />
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: c.badge,
          border: `1px solid ${c.border}40`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16, color: c.fg,
          flexShrink: 0,
        }}>
          {ICONS[riskLevel] ?? "✦"}
        </div>
        <div>
          <div style={{ fontSize: 9, color: c.fg, opacity: 0.6, letterSpacing: 2, marginBottom: 2 }}>
            RISK LEVEL
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, color: c.fg, letterSpacing: 0.5, lineHeight: 1 }}>
            {riskLevel?.replace(/_/g, " ") ?? "—"}
          </div>
        </div>
      </div>
      {alert && (
        <div style={{
          marginTop: 8, fontSize: 9, color: c.fg, letterSpacing: 2,
          display: "flex", alignItems: "center", gap: 5,
        }}>
          <span style={{ animation: "dot 1s infinite" }}>●</span>
          ALERT CONFIRMED
        </div>
      )}
    </div>
  );
}
