import { motion, AnimatePresence } from "framer-motion";
import type { TrajectoryState } from "@/types/analysis";
import { TRAJECTORY_PALETTE } from "@/config/palette";

interface TrajectoryLabelProps {
  trajectory: TrajectoryState;
}

export default function TrajectoryLabel({ trajectory }: TrajectoryLabelProps) {
  const { color, dark, label, icon, pulse } = TRAJECTORY_PALETTE[trajectory];

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={trajectory}
        initial={{ opacity: 0, scale: 0.88, y: 4 }}
        animate={{ opacity: 1, scale: 1,    y: 0 }}
        exit={{ opacity: 0, scale: 0.88,    y: -4 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
        className="inline-flex items-center gap-[7px] px-[14px] py-[6px] rounded-3xl
                   font-bold text-[12px] tracking-[0.07em] uppercase glass whitespace-nowrap"
        style={{
          background: `linear-gradient(135deg, ${dark}, rgba(0,0,0,0.3))`,
          border:     `1.5px solid ${color}50`,
          boxShadow:  `0 0 14px ${color}28`,
          color,
          animation:  pulse ? "criticalPulse 1.4s ease-in-out infinite" : "none",
        }}
        role="status"
        aria-label={`Trajectory: ${label}`}
      >
        <span className="text-[16px] leading-none font-black">{icon}</span>
        <span>{label}</span>
      </motion.div>
    </AnimatePresence>
  );
}