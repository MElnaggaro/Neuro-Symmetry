import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, AlertTriangle, Flame, Siren } from "lucide-react";
import type { RiskLevel } from "@/types/analysis";

interface Palette {
  bg:     string;
  border: string;
  glow:   string;
  fg:     string;
  badge:  string;
}

const PALETTE: Record<RiskLevel, Palette> = {
  NORMAL:    { bg: "rgba(16,185,129,0.07)",  border: "#10b981", glow: "#10b98128", fg: "#34d399", badge: "#065f46" },
  MILD:      { bg: "rgba(245,158,11,0.08)",  border: "#f59e0b", glow: "#f59e0b30", fg: "#fbbf24", badge: "#78350f" },
  HIGH_RISK: { bg: "rgba(249,115,22,0.09)",  border: "#f97316", glow: "#f9731640", fg: "#fb923c", badge: "#7c2d12" },
  CRITICAL:  { bg: "rgba(239,68,68,0.12)",   border: "#ef4444", glow: "#ef444458", fg: "#f87171", badge: "#450a0a" },
};

const ICON: Record<RiskLevel, React.ElementType> = {
  NORMAL:    CheckCircle2,
  MILD:      AlertTriangle,
  HIGH_RISK: Flame,
  CRITICAL:  Siren,
};

interface RiskIndicatorProps {
  riskLevel: RiskLevel;
  alert:     boolean;
}

export default function RiskIndicator({ riskLevel, alert }: RiskIndicatorProps) {
  const prevRef  = useRef<RiskLevel | null>(null);
  const audioRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (riskLevel === "CRITICAL" && prevRef.current !== "CRITICAL") playAlert();
    prevRef.current = riskLevel;
  }, [riskLevel]);

  function playAlert() {
    try {
      const AudioCtx = (
        window.AudioContext ?? (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      );
      if (!AudioCtx) return;
      audioRef.current ??= new AudioCtx();
      const ctx = audioRef.current;
      const t   = ctx.currentTime;

      [880, 660, 880].forEach((freq, i) => {
        const osc  = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type           = "sine";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.22, t + i * 0.18);
        gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.18 + 0.15);
        osc.start(t + i * 0.18);
        osc.stop(t + i * 0.18 + 0.15);
      });
    } catch { /* AudioContext unavailable in some contexts */ }
  }

  const c    = PALETTE[riskLevel] ?? PALETTE.NORMAL;
  const Icon = ICON[riskLevel] ?? CheckCircle2;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={riskLevel}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        className="relative rounded-2xl overflow-hidden"
        style={{
          background: c.bg,
          border:     `1.5px solid ${c.border}`,
          boxShadow:  `0 0 22px ${c.glow}, inset 0 1px 0 rgba(255,255,255,0.04)`,
          animation:  riskLevel === "CRITICAL" && alert ? "criticalPulse 1.2s ease-in-out infinite" : "none",
        }}
        role="status"
        aria-live="assertive"
        aria-label={`Risk level: ${riskLevel.replace(/_/g, " ")}`}
      >
        {/* Top gradient line */}
        <div
          className="absolute top-0 left-0 right-0 h-[2px] pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent, ${c.border}, transparent)` }}
        />

        <div className="flex items-center gap-3 px-4 py-3">
          {/* Icon badge */}
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: c.badge, border: `1px solid ${c.border}30` }}
          >
            <Icon size={18} color={c.fg} strokeWidth={2.2} aria-hidden="true" />
          </div>

          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] font-semibold tracking-[0.2em] opacity-60" style={{ color: c.fg }}>
              RISK LEVEL
            </span>
            <span className="text-[22px] font-extrabold tracking-tight leading-none" style={{ color: c.fg }}>
              {riskLevel.replace(/_/g, " ")}
            </span>
          </div>

          {alert && (
            <div className="ml-auto flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full animate-blink" style={{ background: c.fg }} />
              <span className="text-[9px] font-bold tracking-[0.15em] opacity-80" style={{ color: c.fg }}>
                CONFIRMED
              </span>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
