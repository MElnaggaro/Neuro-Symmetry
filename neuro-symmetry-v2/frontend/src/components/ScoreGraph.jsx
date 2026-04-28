import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, CartesianGrid,
} from "recharts";

const TRAJ_COLOR = {
  STABLE:         "#10b981",
  LINEAR_DECLINE: "#f59e0b",
  SUDDEN_DROP:    "#ef4444",
  COLLAPSE:       "#dc2626",
  OSCILLATING:    "#a78bfa",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(10,22,40,0.95)",
      border: "1px solid #1a3a5c",
      borderRadius: 8,
      padding: "8px 12px",
      fontSize: 11,
      backdropFilter: "blur(8px)",
    }}>
      <div style={{ color: "#475569", marginBottom: 2 }}>Frame {label}</div>
      <div style={{ color: "#e2e8f0", fontWeight: 700 }}>
        {payload[0].value.toFixed(4)}
      </div>
    </div>
  );
};

export default function ScoreGraph({ history = [], trajectory = "STABLE" }) {
  const color  = TRAJ_COLOR[trajectory] ?? TRAJ_COLOR.STABLE;
  const gradId = `scoreGrad_${trajectory}`;

  return (
    <ResponsiveContainer width="100%" height={150}>
      <AreaChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: -22 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0.01} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#0d1f35" strokeDasharray="3 6" vertical={false} />
        <XAxis
          dataKey="frame"
          tick={{ fill: "#334155", fontSize: 9 }}
          tickLine={false}
          axisLine={{ stroke: "#0d1f35" }}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 1]}
          ticks={[0, 0.5, 1.0]}
          tick={{ fill: "#334155", fontSize: 9 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine
          y={0.75} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth={1}
          label={{ value: "Mild", fill: "#f59e0b", fontSize: 8, position: "insideTopRight", offset: 4 }}
        />
        <ReferenceLine
          y={0.55} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1}
          label={{ value: "Severe", fill: "#ef4444", fontSize: 8, position: "insideTopRight", offset: 4 }}
        />
        <Area
          type="monotone"
          dataKey="score"
          stroke={color}
          strokeWidth={2}
          fill={`url(#${gradId})`}
          dot={false}
          activeDot={{ r: 4, fill: color, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
