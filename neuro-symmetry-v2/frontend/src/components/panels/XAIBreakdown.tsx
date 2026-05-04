import { motion } from "framer-motion";
import { useMemo, useState } from "react";
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

type Filter = "all" | "top3" | "high";
const FILTERS: ReadonlyArray<{ id: Filter; label: string }> = [
  { id: "all",  label: "All"     },
  { id: "top3", label: "Top 3"   },
  { id: "high", label: "High"    },
];

// ── Animation variants ────────────────────────────────────────────────────────

const container = { hidden: {}, show: { transition: { staggerChildren: 0.045 } } };
const item      = { hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0, transition: { duration: 0.22, ease: "easeOut" as const } } };

// ── Component ─────────────────────────────────────────────────────────────────

interface XAIBreakdownProps {
  xai: XAIFeature[];
}

export default function XAIBreakdown({ xai }: XAIBreakdownProps) {
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    if (filter === "top3") return xai.slice(0, 3);
    if (filter === "high") return xai.filter((f) => f.level === "HIGH");
    return xai;
  }, [xai, filter]);

  if (!xai.length) {
    return (
      <p className="text-[11px] text-slate-600 text-center py-4 tracking-cyber">
        — AWAITING ANALYSIS —
      </p>
    );
  }

  const maxVal = Math.max(...xai.map((f) => f.contribution), 1e-9);

  return (
    <div className="flex flex-col gap-2">
      {/* Filter chips */}
      <div role="tablist" aria-label="Filter feature contributions" className="flex gap-1">
        {FILTERS.map((f) => {
          const active = filter === f.id;
          return (
            <button
              key={f.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setFilter(f.id)}
              className={`px-2 py-0.5 rounded-full text-micro font-bold tracking-cyber border transition-colors ring-cyber ${
                active
                  ? "bg-cyan-900/25 border-accent-cyan/40 text-accent-cyan"
                  : "bg-transparent border-neu-border text-slate-500 hover:text-slate-300 hover:border-neu-borderLight"
              }`}
            >
              {f.label.toUpperCase()}
            </button>
          );
        })}
      </div>

      {filtered.length === 0 ? (
        <p className="text-[11px] text-slate-600 text-center py-3 tracking-cyber">
          — NO FEATURES MATCH FILTER —
        </p>
      ) : (
        <motion.ul
          key={`${filter}:${filtered.map((f) => f.feature).join(",")}`}
          className="flex flex-col gap-[7px]"
          variants={container}
          initial="hidden"
          animate="show"
          role="list"
          aria-label="XAI feature contributions"
        >
          {filtered.map((f, i) => {
            const st  = XAI_PALETTE[f.level];
            const pct = (f.contribution / maxVal) * 100;

            return (
              <motion.li
                key={f.feature}
                variants={item}
                role="listitem"
                className="flex items-center gap-2"
                aria-label={`${f.feature}: ${f.contribution.toFixed(3)} (${f.level})`}
              >
                <span className="text-[9px] text-slate-600 font-mono w-3 text-right shrink-0">{i + 1}</span>

                <span className="w-[6px] h-[6px] rounded-full shrink-0"
                  style={{ background: st.dot, boxShadow: `0 0 5px ${st.dot}` }} />

                <span className="text-[10px] text-slate-400 w-[112px] shrink-0 truncate" title={f.feature.replace(/_/g, " ")}>
                  {formatFeatureName(f.feature)}
                </span>

                <div
                  className="flex-1 h-[5px] bg-bg-panel rounded-full overflow-hidden"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(pct)}
                >
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
      )}
    </div>
  );
}