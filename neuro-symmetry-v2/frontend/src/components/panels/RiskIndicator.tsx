import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, AlertTriangle, Flame, Siren } from "lucide-react";
import type { RiskLevel } from "@/types/analysis";
import { RISK_PALETTE } from "@/config/palette";

// ── Icon map ──────────────────────────────────────────────────────────────────

const RISK_ICON: Record<RiskLevel, React.ElementType> = {
  NORMAL:    CheckCircle2,
  MILD:      AlertTriangle,
  HIGH_RISK: Flame,
  CRITICAL:  Siren,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function playCriticalAlert(): void {
  try {
    const AudioCtx =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;

    const ctx = new AudioCtx();
    const t   = ctx.currentTime;

    for (const [i, freq] of ([880, 660, 880] as const).entries()) {
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      const at   = t + i * 0.18;

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type            = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.22, at);
      gain.gain.exponentialRampToValueAtTime(0.001, at + 0.15);
      osc.start(at);
      osc.stop(at + 0.15);
    }
  } catch { /* AudioContext unavailable in certain contexts */ }
}

// ── Component ─────────────────────────────────────────────────────────────────

interface RiskIndicatorProps {
  riskLevel: RiskLevel;
  alert:     boolean;
}

export default function RiskIndicator({ riskLevel, alert }: RiskIndicatorProps) {
  const prevRef = useRef<RiskLevel | null>(null);

  useEffect(() => {
    if (riskLevel === "CRITICAL" && prevRef.current !== "CRITICAL") playCriticalAlert();
    prevRef.current = riskLevel;
  }, [riskLevel]);

  const c    = RISK_PALETTE[riskLevel];
  const Icon = RISK_ICON[riskLevel];

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
        {/* Top gradient accent line */}
        <div
          className="absolute top-0 left-0 right-0 h-[2px] pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent, ${c.border}, transparent)` }}
        />

        <div className="flex items-center gap-3 px-4 py-3">
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