import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer, CartesianGrid,
} from "recharts";
import type { TrajectoryState, HistoryPoint } from "@/types/analysis";
import { TRAJECTORY_GRAPH_COLOR } from "@/config/palette";

// ── Tooltip ───────────────────────────────────────────────────────────────────

interface TooltipProps { active?: boolean; payload?: { value: number }[]; label?: string | number }

function riskFromScore(s: number): { label: string; color: string } {
  if (s >= 0.8) return { label: "NORMAL",    color: "#34d399" };
  if (s >= 0.6) return { label: "MILD",      color: "#fbbf24" };
  if (s >= 0.4) return { label: "HIGH RISK", color: "#fb923c" };
  return            { label: "CRITICAL",  color: "#f87171" };
}

function GraphTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const v    = payload[0].value;
  const risk = riskFromScore(v);
  return (
    <div className="glass-card border border-neu-borderLight rounded-lg px-3 py-2 text-[11px] min-w-[120px]">
      <div className="text-micro text-slate-500 tracking-cyber mb-1">FRAME {label}</div>
      <div className="flex items-center justify-between gap-3">
        <span className="text-slate-200 font-bold tabular-nums">{(v * 100).toFixed(1)}%</span>
        <span
          className="text-[8px] font-bold px-1.5 py-[1px] rounded border tracking-cyber"
          style={{
            color:           risk.color,
            borderColor:     `${risk.color}55`,
            backgroundColor: `${risk.color}1a`,
          }}
        >
          {risk.label}
        </span>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

interface ScoreGraphProps {
  history:    HistoryPoint[];
  trajectory: TrajectoryState;
}

export default function ScoreGraph({ history, trajectory }: ScoreGraphProps) {
  const color  = TRAJECTORY_GRAPH_COLOR[trajectory];
  const gradId = `sg_${trajectory}`;

  return (
    <ResponsiveContainer width="100%" height={150}>
      <AreaChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: -22 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.22} />
            <stop offset="95%" stopColor={color} stopOpacity={0.01} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="#0d1f35" strokeDasharray="3 6" vertical={false} />

        {/* Risk-zone shading. */}
        <ReferenceArea y1={0.0} y2={0.4} fill="#ef4444" fillOpacity={0.06} />
        <ReferenceArea y1={0.4} y2={0.6} fill="#f97316" fillOpacity={0.05} />
        <ReferenceArea y1={0.6} y2={0.8} fill="#f59e0b" fillOpacity={0.04} />
        <ReferenceArea y1={0.8} y2={1.0} fill="#10b981" fillOpacity={0.04} />

        <XAxis dataKey="frame" tick={{ fill: "#334155", fontSize: 9 }}
          tickLine={false} axisLine={{ stroke: "#0d1f35" }} interval="preserveStartEnd" />
        <YAxis domain={[0, 1]} ticks={[0, 0.5, 1.0]}
          tick={{ fill: "#334155", fontSize: 9 }} tickLine={false} axisLine={false} />

        <Tooltip content={<GraphTooltip />} cursor={{ stroke: "#22d3ee", strokeOpacity: 0.4, strokeDasharray: "2 4" }} />

        <ReferenceLine y={0.75} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth={1}
          label={{ value: "Mild",   fill: "#f59e0b", fontSize: 8, position: "insideTopRight", offset: 4 }} />
        <ReferenceLine y={0.55} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1}
          label={{ value: "Severe", fill: "#ef4444", fontSize: 8, position: "insideTopRight", offset: 4 }} />

        <Area type="monotone" dataKey="score"
          stroke={color} strokeWidth={2}
          fill={`url(#${gradId})`}
          dot={false}
          activeDot={{ r: 4, fill: color, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}