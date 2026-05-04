import type { ReactNode, CSSProperties } from "react";
import { motion } from "framer-motion";

interface CardProps {
  children:   ReactNode;
  className?: string;
  style?:     CSSProperties;
  /** When true the card expands to fill remaining height. */
  flex?:      boolean;
  /**
   * Visual emphasis. `elevated` adds a 1px conic-gradient inner border
   * (cyan→transparent) to mark hero cards like the gauge or video frame.
   */
  tone?:      "default" | "elevated";
}

/**
 * Glassmorphic dark card — the foundational layout primitive.
 * Uses Framer Motion's layout engine so sibling cards animate
 * smoothly when the XAI breakdown changes height.
 */
export function Card({ children, className = "", style, flex = false, tone = "default" }: CardProps) {
  return (
    <motion.div
      layout
      className={[
        "relative rounded-[14px] overflow-hidden",
        "bg-[rgba(10,22,44,0.82)] glass",
        "border border-neu-border",
        "shadow-card",
        "px-4 py-[14px]",
        tone === "elevated" ? "glass-inset" : "",
        flex ? "flex-1 min-h-0" : "",
        className,
      ].join(" ")}
      style={style}
    >
      {/* Top shimmer line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/[0.10] to-transparent pointer-events-none" />
      {children}
    </motion.div>
  );
}