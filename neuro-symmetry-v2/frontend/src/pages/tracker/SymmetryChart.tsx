import {
  CartesianGrid, Line, LineChart, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { TrackerEntry } from "./loadHistory";

type Point = { date: string; ts: number; score: number; cls: string };

function toPoints(entries: TrackerEntry[]): Point[] {
  return entries
    .filter((e) => e.symmetry_score != null)
    .map((e) => ({
      date:  e.date,
      ts:    Date.parse(e.date),
      score: e.symmetry_score as number,
      cls:   e.class,
    }));
}

interface TooltipProps {
  active?:  boolean;
  payload?: { payload: Point }[];
}

function ChartTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="glass-card border border-neu-borderLight rounded-lg px-3 py-2 text-[11px] min-w-[140px]">
      <div className="text-micro text-slate-500 tracking-cyber mb-1">{p.date}</div>
      <div className="flex items-center justify-between gap-3">
        <span className="text-slate-200 font-bold tabular-nums">{(p.score * 100).toFixed(1)}%</span>
        <span className="text-label text-slate-400">{p.cls}</span>
      </div>
    </div>
  );
}

const dayFmt = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short" });

export function SymmetryChart({ entries }: { entries: TrackerEntry[] }) {
  const data = toPoints(entries);

  if (data.length === 0) {
    return (
      <div className="text-label text-slate-500 py-8 text-center">
        Run a session to start the symmetry trend.
      </div>
    );
  }

  return (
    <div className="w-full" aria-label="Symmetry over time">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 12, right: 16, bottom: 8, left: -8 }}>
          <CartesianGrid stroke="#0d1f35" strokeDasharray="3 6" vertical={false} />

          <ReferenceArea y1={0.0} y2={0.4} fill="#ef4444" fillOpacity={0.06} />
          <ReferenceArea y1={0.4} y2={0.6} fill="#f97316" fillOpacity={0.05} />
          <ReferenceArea y1={0.6} y2={0.8} fill="#f59e0b" fillOpacity={0.04} />
          <ReferenceArea y1={0.8} y2={1.0} fill="#10b981" fillOpacity={0.04} />

          <XAxis
            dataKey="ts"
            type="number"
            domain={["dataMin", "dataMax"]}
            scale="time"
            tickFormatter={(v: number) => dayFmt.format(new Date(v))}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#0d1f35" }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: "#475569", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />

          <Tooltip
            content={<ChartTooltip />}
            cursor={{ stroke: "#22d3ee", strokeOpacity: 0.4, strokeDasharray: "2 4" }}
          />

          <Line
            type="monotone"
            dataKey="score"
            stroke="#a78bfa"
            strokeWidth={2.4}
            dot={{ r: 4, stroke: "#040b16", strokeWidth: 1.5, fill: "#22d3ee" }}
            activeDot={{ r: 6, fill: "#22d3ee" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
