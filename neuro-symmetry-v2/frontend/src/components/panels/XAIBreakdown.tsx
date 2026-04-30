import { motion } from "framer-motion";
import type { XAIFeature } from "@/types/analysis";
import { XAI_PALETTE } from "@/config/palette";

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatFeatureName(raw: string): string {
  return raw
    .replace(/_/g, " ")
    .replace(/\b(left|right)\b/gi, (m) => m[0].toUpperCase() + m.slice(1).toLowerCase())
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, 24);
}

// ── Animation variants ────────────────────────────────────────────────────────

const container = { hidden: {}, show: { transition: { staggerChildren: 0.045 } } };
const item      = { hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0, transition: { duration: 0.22, ease: "easeOut" as const } } };

// ── Component ─────────────────────────────────────────────────────────────────

interface XAIBreakdownProps {
  xai: XAIFeature[];
}

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
      variants={container}
      initial="hidden"
      animate="show"
      aria-label="XAI feature contributions"
    >
      {xai.map((f, i) => {
        const st  = XAI_PALETTE[f.level];
        const pct = (f.contribution / maxVal) * 100;

        return (
          <motion.li
            key={f.feature}
            variants={item}
            className="flex items-center gap-2"
            aria-label={`${f.feature}: ${f.contribution.toFixed(3)} (${f.level})`}
          >
            <span className="text-[9px] text-slate-600 font-mono w-3 text-right shrink-0">{i + 1}</span>

            <span className="w-[6px] h-[6px] rounded-full shrink-0"
              style={{ background: st.dot, boxShadow: `0 0 5px ${st.dot}` }} />

            <span className="text-[10px] text-slate-400 w-[112px] shrink-0 truncate" title={f.feature.replace(/_/g, " ")}>
              {formatFeatureName(f.feature)}
            </span>

            <div className="flex-1 h-[5px] bg-bg-panel rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full bg-gradient-to-r ${st.bar}`}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.4, ease: "easeOut", delay: i * 0.04 }}
              />
            </div>

            <span className="text-[9px] font-mono w-10 text-right shrink-0 tabular-nums" style={{ color: st.label }}>
              {f.contribution.toFixed(3)}
            </span>

            <span className={`text-[8px] font-bold px-[6px] py-[1px] rounded border tracking-[0.08em] shrink-0 ${st.badge}`}>
              {f.level}
            </span>
          </motion.li>
        );
      })}
    </motion.ul>
  );
}