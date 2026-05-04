import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import type { RiskLevel } from "@/types/analysis";
import { RISK_PALETTE } from "@/config/palette";

// ── SVG geometry helpers ──────────────────────────────────────────────────────

function polarXY(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function svgArc(cx: number, cy: number, r: number, a1: number, a2: number): string {
  const s    = polarXY(cx, cy, r, a1);
  const e    = polarXY(cx, cy, r, a2);
  const large = a2 - a1 > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

// ── Gauge geometry constants ──────────────────────────────────────────────────

const CX = 90, CY = 88, R = 68, R_INNER = 56;
const START = -210, SWEEP = 240;
const TICKS = [0, 0.25, 0.5, 0.75, 1] as const;

// Zone bands — drawn behind the score arc at low alpha so the player can see
// where the current score sits on the risk spectrum. Lower scores = higher
// risk; the bands sweep from CRITICAL on the left to NORMAL on the right.
const ZONE_BANDS: ReadonlyArray<{ from: number; to: number; color: string }> = [
  { from: 0.0, to: 0.4, color: "#ef4444" }, // critical
  { from: 0.4, to: 0.6, color: "#f97316" }, // high
  { from: 0.6, to: 0.8, color: "#f59e0b" }, // mild
  { from: 0.8, to: 1.0, color: "#10b981" }, // normal
];

const TICK_LABELS: ReadonlyArray<{ v: number; text: string }> = [
  { v: 0.0, text: "0"   },
  { v: 0.5, text: "50"  },
  { v: 1.0, text: "100" },
];

// ── Component ─────────────────────────────────────────────────────────────────

interface SymmetryGaugeProps {
  score:     number;
  riskLevel: RiskLevel;
}

export default function SymmetryGauge({ score, riskLevel }: SymmetryGaugeProps) {
  const clamped = Math.max(0, Math.min(1, score));
  const pal     = RISK_PALETTE[riskLevel];
  const gradId  = `arcGrad_${riskLevel}`;

  const motionScore = useMotionValue(0);
  const springScore = useSpring(motionScore, { stiffness: 90, damping: 22 });
  const needleAngle = useTransform(springScore, (v) => START + v * SWEEP);
  const arcRef      = useRef<SVGPathElement>(null);

  useEffect(() => { motionScore.set(clamped); }, [clamped, motionScore]);

  // Imperatively update the arc path on each spring tick (keeps React out of the loop)
  useEffect(() =>
    springScore.on("change", (v) => {
      arcRef.current?.setAttribute(
        "d",
        v > 0.005 ? svgArc(CX, CY, R, START, START + v * SWEEP) : "",
      );
    }),
  [springScore]);

  const needleX = useTransform(needleAngle, (a) => polarXY(CX, CY, 50, a).x);
  const needleY = useTransform(needleAngle, (a) => polarXY(CX, CY, 50, a).y);

  return (
    <div className="flex flex-col items-center select-none" aria-label={`Symmetry score ${Math.round(clamped * 100)}%`}>
      <svg width={180} height={115} viewBox="0 0 180 115" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor={pal.track} stopOpacity={0.5} />
            <stop offset="100%" stopColor={pal.track} stopOpacity={1}   />
          </linearGradient>
          <filter id="arcGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Track */}
        <path d={svgArc(CX, CY, R, START, START + SWEEP)}    fill="none" stroke="#1e293b"   strokeWidth={14} strokeLinecap="round" />

        {/* Risk-zone bands — semi-transparent under the score arc. */}
        {ZONE_BANDS.map((b) => (
          <path
            key={`${b.from}-${b.to}`}
            d={svgArc(CX, CY, R, START + b.from * SWEEP, START + b.to * SWEEP)}
            fill="none"
            stroke={b.color}
            strokeOpacity={0.18}
            strokeWidth={14}
            strokeLinecap="butt"
          />
        ))}

        <path d={svgArc(CX, CY, R_INNER, START, START + SWEEP)} fill="none" stroke="#0a1628" strokeWidth={2}  strokeLinecap="round" />

        {/* Animated score arc */}
        <path ref={arcRef} d="" fill="none" stroke={`url(#${gradId})`} strokeWidth={14} strokeLinecap="round" filter="url(#arcGlow)" />

        {/* Tick marks */}
        {TICKS.map((v) => {
          const a  = START + v * SWEEP;
          const p0 = polarXY(CX, CY, R + 9, a);
          const p1 = polarXY(CX, CY, R + 4, a);
          return <line key={v} x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y} stroke="#334155" strokeWidth={1.5} />;
        })}

        {/* Tick labels — 0 / 50 / 100 above the major ticks. */}
        {TICK_LABELS.map(({ v, text }) => {
          const a = START + v * SWEEP;
          const p = polarXY(CX, CY, R + 17, a);
          return (
            <text
              key={v}
              x={p.x}
              y={p.y + 3}
              textAnchor="middle"
              fontSize={7.5}
              fontWeight={600}
              fill="#475569"
              letterSpacing="0.18em"
            >
              {text}
            </text>
          );
        })}

        {/* Needle */}
        <motion.line x1={CX} y1={CY} x2={needleX} y2={needleY}
          stroke={pal.track} strokeWidth={2.2} strokeLinecap="round" filter="url(#arcGlow)" />
        <circle cx={CX} cy={CY} r={5.5} fill={pal.track} />
        <circle cx={CX} cy={CY} r={2.5} fill="#040b16" />

        {/* Labels */}
        <text x={CX} y={CY + 34} textAnchor="middle" fill={pal.text} fontSize={28} fontWeight={800} letterSpacing={-1}>
          {Math.round(clamped * 100)}%
        </text>
        <text x={CX} y={CY + 45} textAnchor="middle" fill="#334155" fontSize={8} letterSpacing="0.3em" fontWeight={600}>
          SYMMETRY
        </text>
      </svg>
    </div>
  );
}