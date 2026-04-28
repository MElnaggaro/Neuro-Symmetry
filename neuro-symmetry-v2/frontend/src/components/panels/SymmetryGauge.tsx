import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import type { RiskLevel } from "@/types/analysis";

const RISK_COLOR: Record<RiskLevel, { track: string; text: string; glow: string }> = {
  NORMAL:    { track: "#10b981", text: "#34d399", glow: "#10b98140" },
  MILD:      { track: "#f59e0b", text: "#fbbf24", glow: "#f59e0b40" },
  HIGH_RISK: { track: "#f97316", text: "#fb923c", glow: "#f9731640" },
  CRITICAL:  { track: "#ef4444", text: "#f87171", glow: "#ef444450" },
};

// Polar → Cartesian helper
function polarXY(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

// Build an SVG arc path string
function svgArc(cx: number, cy: number, r: number, a1: number, a2: number): string {
  const s    = polarXY(cx, cy, r, a1);
  const e    = polarXY(cx, cy, r, a2);
  const large = a2 - a1 > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

const CX = 90, CY = 88, R = 68, R_INNER = 56;
const START = -210, SWEEP = 240;
const TICK_POS = [0, 0.25, 0.5, 0.75, 1] as const;

interface SymmetryGaugeProps {
  score:     number;
  riskLevel: RiskLevel;
}

export default function SymmetryGauge({ score, riskLevel }: SymmetryGaugeProps) {
  const clampedScore = Math.max(0, Math.min(1, score));
  const pal  = RISK_COLOR[riskLevel] ?? RISK_COLOR.NORMAL;
  const gradId = `arcGrad_${riskLevel}`;

  // Spring-animated score for smooth needle movement
  const motionScore = useMotionValue(0);
  const springScore = useSpring(motionScore, { stiffness: 90, damping: 22 });

  // Derived needle angle from spring value
  const needleAngle = useTransform(springScore, (v) => START + v * SWEEP);

  // Ref for the arc path — updated imperatively to avoid SVG animation complexity
  const arcRef = useRef<SVGPathElement>(null);

  useEffect(() => { motionScore.set(clampedScore); }, [clampedScore, motionScore]);

  // Subscribe to spring and update arc path directly (keeps React out of the loop)
  useEffect(() => {
    return springScore.on("change", (v) => {
      if (arcRef.current) {
        const angle = START + v * SWEEP;
        arcRef.current.setAttribute(
          "d",
          v > 0.005 ? svgArc(CX, CY, R, START, angle) : "",
        );
      }
    });
  }, [springScore]);

  // Needle tip computed from spring (motion value → motion point)
  const needleX = useTransform(needleAngle, (a) => polarXY(CX, CY, 50, a).x);
  const needleY = useTransform(needleAngle, (a) => polarXY(CX, CY, 50, a).y);

  const pct = Math.round(clampedScore * 100);

  return (
    <div className="flex flex-col items-center select-none" aria-label={`Symmetry score ${pct}%`}>
      <svg width={180} height={115} viewBox="0 0 180 115" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor={pal.track} stopOpacity={0.5} />
            <stop offset="100%" stopColor={pal.track} stopOpacity={1}   />
          </linearGradient>
          <filter id="arcGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Track background */}
        <path d={svgArc(CX, CY, R, START, START + SWEEP)}
          fill="none" stroke="#1e293b" strokeWidth={14} strokeLinecap="round" />

        {/* Danger zone shading below 0.55 */}
        <path d={svgArc(CX, CY, R, START, START + 0.55 * SWEEP)}
          fill="none" stroke="#ef444415" strokeWidth={14} strokeLinecap="round" />

        {/* Inner shadow ring */}
        <path d={svgArc(CX, CY, R_INNER, START, START + SWEEP)}
          fill="none" stroke="#0a1628" strokeWidth={2} strokeLinecap="round" />

        {/* Animated score arc — driven imperatively via arcRef */}
        <path
          ref={arcRef}
          d=""
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth={14}
          strokeLinecap="round"
          filter="url(#arcGlow)"
        />

        {/* Tick marks */}
        {TICK_POS.map((v) => {
          const a  = START + v * SWEEP;
          const p0 = polarXY(CX, CY, R + 9, a);
          const p1 = polarXY(CX, CY, R + 4, a);
          return (
            <line key={v}
              x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y}
              stroke="#334155" strokeWidth={1.5} />
          );
        })}

        {/* Needle — animated via motion values */}
        <motion.line
          x1={CX} y1={CY}
          x2={needleX} y2={needleY}
          stroke={pal.track} strokeWidth={2.2} strokeLinecap="round"
          filter="url(#arcGlow)"
        />
        <circle cx={CX} cy={CY} r={5.5} fill={pal.track} />
        <circle cx={CX} cy={CY} r={2.5} fill="#040b16" />

        {/* Score text */}
        <text x={CX} y={CY + 34} textAnchor="middle"
          fill={pal.text} fontSize={28} fontWeight={800} letterSpacing={-1}>
          {pct}%
        </text>
        <text x={CX} y={CY + 45} textAnchor="middle"
          fill="#334155" fontSize={8} letterSpacing={2} fontWeight={600}>
          SYMMETRY
        </text>
      </svg>
    </div>
  );
}
