const LEVEL = {
  HIGH:   { color: "#ef4444", bg: "rgba(239,68,68,0.12)",   dot: "#ef4444" },
  MEDIUM: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)",  dot: "#f59e0b" },
  LOW:    { color: "#10b981", bg: "rgba(16,185,129,0.10)",  dot: "#10b981" },
};

function shortenName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b(left|right)\b/gi, (m) => m[0].toUpperCase())
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, 22);
}

export default function XAIBreakdown({ xai = [] }) {
  if (!xai.length) {
    return (
      <div style={{
        color: "#1e293b", fontSize: 12, textAlign: "center",
        padding: "24px 0", letterSpacing: 1,
      }}>
        — AWAITING ANALYSIS —
      </div>
    );
  }

  const maxVal = Math.max(...xai.map((f) => f.contribution), 1e-9);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {xai.map((f, i) => {
        const pal  = LEVEL[f.level] ?? LEVEL.LOW;
        const pct  = (f.contribution / maxVal) * 100;
        const name = shortenName(f.feature);
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 9 }}>
            {/* Level dot */}
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: pal.dot, flexShrink: 0,
              boxShadow: `0 0 6px ${pal.dot}`,
            }} />
            {/* Feature name */}
            <div style={{
              fontSize: 10, color: "#64748b", width: 118,
              flexShrink: 0, fontVariantNumeric: "tabular-nums",
            }} title={f.feature.replace(/_/g, " ")}>
              {name}
            </div>
            {/* Bar */}
            <div style={{ flex: 1, height: 5, background: "#0d1f35", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`, height: "100%",
                background: `linear-gradient(90deg, ${pal.color}80, ${pal.color})`,
                borderRadius: 3,
                transition: "width 0.3s ease",
              }} />
            </div>
            {/* Value */}
            <div style={{
              fontSize: 9, color: pal.color, width: 44, textAlign: "right",
              fontVariantNumeric: "tabular-nums", letterSpacing: 0.5,
            }}>
              {f.contribution.toFixed(3)}
            </div>
            {/* Level badge */}
            <div style={{
              fontSize: 8, color: pal.color, background: pal.bg,
              padding: "1px 6px", borderRadius: 4, letterSpacing: 1,
              border: `1px solid ${pal.color}30`, flexShrink: 0,
            }}>
              {f.level}
            </div>
          </div>
        );
      })}
    </div>
  );
}
