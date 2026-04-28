import { motion, AnimatePresence } from "framer-motion";
import type { TrajectoryState } from "@/types/analysis";

interface Config {
  color: string;
  dark:  string;
  label: string;
  icon:  string;
  pulse: boolean;
}

const CONFIG: Record<TrajectoryState, Config> = {
  STABLE:         { color: "#10b981", dark: "#064e3b", label: "Stable",         icon: "→",  pulse: false },
  LINEAR_DECLINE: { color: "#f59e0b", dark: "#78350f", label: "Linear Decline", icon: "↘",  pulse: false },
  SUDDEN_DROP:    { color: "#ef4444", dark: "#450a0a", label: "Sudden Drop",    icon: "↓↓", pulse: true  },
  COLLAPSE:       { color: "#dc2626", dark: "#3b0000", label: "Collapse",       icon: "⬇",  pulse: true  },
  OSCILLATING:    { color: "#a78bfa", dark: "#2e1065", label: "Oscillating",    icon: "↕",  pulse: false },
};

interface TrajectoryLabelProps {
  trajectory: TrajectoryState;
}

export default function TrajectoryLabel({ trajectory }: TrajectoryLabelProps) {
  const cfg = CONFIG[trajectory] ?? CONFIG.STABLE;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={trajectory}
        initial={{ opacity: 0, scale: 0.88, y: 4 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.88, y: -4 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
        className="inline-flex items-center gap-[7px] px-[14px] py-[6px] rounded-3xl
                   font-bold text-[12px] tracking-[0.07em] uppercase glass
                   whitespace-nowrap"
        style={{
          background: `linear-gradient(135deg, ${cfg.dark}, rgba(0,0,0,0.3))`,
          border:     `1.5px solid ${cfg.color}50`,
          boxShadow:  `0 0 14px ${cfg.color}28`,
          color:       cfg.color,
          animation:  cfg.pulse ? "criticalPulse 1.4s ease-in-out infinite" : "none",
        }}
        role="status"
        aria-label={`Trajectory: ${cfg.label}`}
      >
        <span className="text-[16px] leading-none font-black">{cfg.icon}</span>
        <span>{cfg.label}</span>
      </motion.div>
    </AnimatePresence>
  );
}
