import { Activity } from "lucide-react";
import { motion } from "framer-motion";

/**
 * Animated brand mark — Activity icon inside a gradient tile with a 4 s
 * breathing glow + a subtle ring. Replaces the static logo block in the header.
 */
export function AnimatedLogo() {
  return (
    <motion.div
      className="relative w-9 h-9 rounded-[10px] flex items-center justify-center
                 bg-gradient-to-br from-blue-700 to-sky-500 shadow-glow-cyan shrink-0"
      aria-hidden="true"
      animate={{ boxShadow: [
        "0 0 8px  rgba(6,182,212,0.30)",
        "0 0 24px rgba(6,182,212,0.55)",
        "0 0 8px  rgba(6,182,212,0.30)",
      ] }}
      transition={{ duration: 4, ease: "easeInOut", repeat: Infinity }}
    >
      <Activity size={17} color="#bfdbfe" strokeWidth={2.5} />
      <span
        aria-hidden="true"
        className="absolute inset-0 rounded-[10px] ring-1 ring-cyan-400/30 pointer-events-none"
      />
    </motion.div>
  );
}
