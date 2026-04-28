import { motion } from "framer-motion";
import type { XAIFeature, XAILevel } from "@/types/analysis";

const LEVEL_STYLE: Record<XAILevel, { dot: string; bar: string; label: string; badge: string }> = {
  HIGH:   { dot: "#ef4444", bar: "from-red-700/60 to-red-500",    label: "#f87171", badge: "bg-red-900/40 border-red-500/30 text-red-400" },
  MEDIUM: { dot: "#f59e0b", bar: "from-amber-700/60 to-amber-400", label: "#fbbf24", badge: "bg-amber-900/40 border-amber-500/30 text-amber-400" },
  LOW:    { dot: "#10b981", bar: "from-emerald-700/60 to-emerald-400", label: "#34d399", badge: "bg-emerald-900/40 border-emerald-500/30 text-emerald-400" },
};

function formatName(raw: string): string {
  return raw
    .replace(/_/g, " ")
    .replace(/\b(left|right)\b/gi, (m) => m[0].toUpperCase() + m.slice(1).toLowerCase())
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, 24);
}

interface XAIBreakdownProps {
  xai: XAIFeature[];
}

const containerVariants = {
  hidden: {},
  show:   { transition: { staggerChildren: 0.045 } },
};

// "as const" preserves the string literal type for ease so it satisfies Framer Motion's Easing union.
const itemVariants = {
  hidden: { opacity: 0, x: -10 },
  show:   { opacity: 1, x: 0, transition: { duration: 0.22, ease: "easeOut" as const } },
};

export default function XAIBreakdown({ xai }: XAIBreakdownProps) {
  if (!xai.length) {
    return (
      <p className="text-[11px] text-slate-600 text-center py-4 tracking-widest">
        — AWAITING ANALYSIS —
      </p>
    );
  }

  const maxVal = Math.max(...xai.map((f) => f.contribution), 1e-9);

  return (
    <motion.ul
      key={xai.map((f) => f.feature).join(",")}
      className="flex flex-col gap-[7px]"
      variants={containerVariants}
      initial="hidden"
      animate="show"
      aria-label="XAI feature contributions"
    >
      {xai.map((f, i) => {
        const st  = LEVEL_STYLE[f.level] ?? LEVEL_STYLE.LOW;
        const pct = (f.contribution / maxVal) * 100;

        return (
          <motion.li
            key={f.feature}
            variants={itemVariants}
            className="flex items-center gap-2"
            aria-label={`${f.feature}: ${f.contribution.toFixed(3)} (${f.level})`}
          >
            {/* Rank */}
            <span className="text-[9px] text-slate-600 font-mono w-3 text-right shrink-0">
              {i + 1}
            </span>

            {/* Level dot */}
            <span
              className="w-[6px] h-[6px] rounded-full shrink-0"
              style={{ background: st.dot, boxShadow: `0 0 5px ${st.dot}` }}
            />

            {/* Feature name */}
            <span
              className="text-[10px] text-slate-400 w-[112px] shrink-0 truncate"
              title={f.feature.replace(/_/g, " ")}
            >
              {formatName(f.feature)}
            </span>

            {/* Animated bar */}
            <div className="flex-1 h-[5px] bg-bg-panel rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full bg-gradient-to-r ${st.bar}`}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.4, ease: "easeOut", delay: i * 0.04 }}
              />
            </div>

            {/* Numeric value */}
            <span
              className="text-[9px] font-mono w-10 text-right shrink-0 tabular-nums"
              style={{ color: st.label }}
            >
              {f.contribution.toFixed(3)}
            </span>

            {/* Level badge */}
            <span className={`text-[8px] font-bold px-[6px] py-[1px] rounded border tracking-[0.08em] shrink-0 ${st.badge}`}>
              {f.level}
            </span>
          </motion.li>
        );
      })}
    </motion.ul>
  );
}
