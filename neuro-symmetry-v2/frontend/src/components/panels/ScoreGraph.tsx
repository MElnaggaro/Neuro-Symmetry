import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, CartesianGrid,
} from "recharts";
import type { TrajectoryState, HistoryPoint } from "@/types/analysis";
import { TRAJECTORY_GRAPH_COLOR } from "@/config/palette";

// ── Tooltip ───────────────────────────────────────────────────────────────────

interface TooltipProps { active?: boolean; payload?: { value: number }[]; label?: string | number }

function GraphTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[rgba(10,22,40,0.95)] border border-neu-borderLight rounded-lg px-3 py-2 text-[11px] glass">
      <div className="text-slate-500 mb-1">Frame {label}</div>
      <div className="text-slate-200 font-bold">{payload[0].value.toFixed(4)}</div>
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

        <XAxis dataKey="frame" tick={{ fill: "#334155", fontSize: 9 }}
          tickLine={false} axisLine={{ stroke: "#0d1f35" }} interval="preserveStartEnd" />
        <YAxis domain={[0, 1]} ticks={[0, 0.5, 1.0]}
          tick={{ fill: "#334155", fontSize: 9 }} tickLine={false} axisLine={false} />

        <Tooltip content={<GraphTooltip />} />

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