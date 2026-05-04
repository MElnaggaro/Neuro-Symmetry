import type { ReactNode } from "react";

type Props = {
  /** Frames captured so far. */
  count: number;
  /** Total frames required. */
  total: number;
  /** True when calibration is actively running. */
  active: boolean;
  children: ReactNode;
};

/**
 * Wraps a button with a conic-gradient progress ring. Inactive → invisible
 * (children render as-is). Active → cyan ring fills from 0° → 360° as
 * `count / total` progresses; finishes amber at full.
 */
export function CalibrationRing({ count, total, active, children }: Props) {
  const pct = Math.min(1, total > 0 ? count / total : 0);
  const deg = Math.round(pct * 360);
  const color = pct >= 1 ? "#10b981" : "#06b6d4";

  if (!active) return <>{children}</>;

  return (
    <div
      className="relative rounded-lg p-[2px]"
      style={{
        background: `conic-gradient(${color} 0deg ${deg}deg, rgba(15,32,64,0.6) ${deg}deg 360deg)`,
        boxShadow:  `0 0 12px ${color}55`,
        transition: "background 120ms linear",
      }}
      aria-hidden={false}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={count}
      aria-label={`Calibration progress ${count} of ${total}`}
    >
      <div className="rounded-[6px] bg-bg-surface">{children}</div>
    </div>
  );
}
