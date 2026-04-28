const RISK_PALETTE = {
  NORMAL:    { track: "#10b981", glow: "#10b98140", text: "#34d399" },
  MILD:      { track: "#f59e0b", glow: "#f59e0b40", text: "#fbbf24" },
  HIGH_RISK: { track: "#f97316", glow: "#f9731640", text: "#fb923c" },
  CRITICAL:  { track: "#ef4444", glow: "#ef444450", text: "#f87171" },
};

function polarXY(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function svgArc(cx, cy, r, a1, a2) {
  const s = polarXY(cx, cy, r, a1);
  const e = polarXY(cx, cy, r, a2);
  const large = a2 - a1 > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

export default function SymmetryGauge({ score = 0, riskLevel = "NORMAL" }) {
  const pct   = Math.max(0, Math.min(1, score));
  const pal   = RISK_PALETTE[riskLevel] ?? RISK_PALETTE.NORMAL;
  const cx = 90, cy = 88, R = 68, R_INNER = 58;
  const START = -210, SWEEP = 240;
  const scoreAngle = START + pct * SWEEP;
  const needle     = polarXY(cx, cy, 50, scoreAngle);
  const gradId     = `arcGrad_${riskLevel}`;

  return (
    <div style={{ textAlign: "center", userSelect: "none" }}>
      <svg width={180} height={115} viewBox="0 0 180 115" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={pal.track} stopOpacity="0.5" />
            <stop offset="100%" stopColor={pal.track} stopOpacity="1" />
          </linearGradient>
          <filter id="arcGlow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Outer track (background) */}
        <path d={svgArc(cx, cy, R, START, START + SWEEP)}
          fill="none" stroke="#1e293b" strokeWidth={14} strokeLinecap="round" />

        {/* Inner track shadow */}
        <path d={svgArc(cx, cy, R_INNER, START, START + SWEEP)}
          fill="none" stroke="#0f172a" strokeWidth={2} strokeLinecap="round" />

        {/* Danger zone (red below 0.55 threshold) */}
        <path d={svgArc(cx, cy, R, START, START + 0.55 * SWEEP)}
          fill="none" stroke="#ef444418" strokeWidth={14} strokeLinecap="round" />

        {/* Score arc */}
        {pct > 0.005 && (
          <path d={svgArc(cx, cy, R, START, scoreAngle)}
            fill="none" stroke={`url(#${gradId})`} strokeWidth={14} strokeLinecap="round"
            filter="url(#arcGlow)" />
        )}

        {/* Tick marks */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => {
          const a = START + v * SWEEP;
          const p0 = polarXY(cx, cy, R + 9, a);
          const p1 = polarXY(cx, cy, R + 4, a);
          return <line key={v} x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y} stroke="#334155" strokeWidth={1.5} />;
        })}

        {/* Needle */}
        <line x1={cx} y1={cy} x2={needle.x} y2={needle.y}
          stroke={pal.track} strokeWidth={2} strokeLinecap="round"
          filter="url(#arcGlow)" />
        <circle cx={cx} cy={cy} r={5} fill={pal.track} />
        <circle cx={cx} cy={cy} r={2.5} fill="#040b16" />

        {/* Percentage text */}
        <text x={cx} y={cy + 32} textAnchor="middle"
          fill={pal.text} fontSize={28} fontWeight={800} letterSpacing={-1}>
          {Math.round(pct * 100)}%
        </text>
        <text x={cx} y={cy + 44} textAnchor="middle"
          fill="#334155" fontSize={8} letterSpacing={2}>
          SYMMETRY
        </text>
      </svg>
    </div>
  );
}
