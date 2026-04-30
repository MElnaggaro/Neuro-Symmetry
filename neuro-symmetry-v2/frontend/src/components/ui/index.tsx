import type { ReactNode } from "react";

// ── SectionLabel ──────────────────────────────────────────────────────────────

interface SectionLabelProps { children: ReactNode }

/** Styled section header with decorative accent lines — used inside Card panels. */
export function SectionLabel({ children }: SectionLabelProps) {
  return (
    <div className="flex items-center gap-2 mb-3" role="heading" aria-level={3}>
      <span className="w-4 h-px bg-accent-cyan/50 shrink-0" />
      <span className="text-[9px] font-semibold tracking-[0.2em] text-accent-cyan uppercase whitespace-nowrap">
        {children}
      </span>
      <span className="flex-1 h-px bg-gradient-to-r from-accent-cyan/25 to-transparent" />
    </div>
  );
}

// ── StatTile ──────────────────────────────────────────────────────────────────

interface StatTileProps {
  label:   string;
  value:   string | null | undefined;
  color?:  string;
  /** Render value in monospace font (e.g. scores, z-values). */
  mono?:   boolean;
}

/** Single key-metric tile — rendered in the footer stats strip. */
export function StatTile({ label, value, color = "#94a3b8", mono = false }: StatTileProps) {
  return (
    <div className="flex flex-col gap-1 min-w-[80px] bg-bg-panel border border-neu-border rounded-[10px] px-3 py-2">
      <span className="text-[8px] font-semibold tracking-[0.2em] text-slate-500 uppercase">
        {label}
      </span>
      <span
        className={`text-[15px] font-bold leading-none ${mono ? "font-mono tracking-wide" : ""}`}
        style={{ color }}
      >
        {value ?? "—"}
      </span>
    </div>
  );
}