import { Card } from "@/components/ui/Card";
import type { TrackerKpis } from "./loadHistory";

function fmtPct(v: number | null): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

export function KpiGrid({ kpis }: { kpis: TrackerKpis }) {
  const { todaySym, yestSym, sessions7, sessions30, best, streak } = kpis;

  const symDelta =
    todaySym != null && yestSym != null
      ? ((todaySym - yestSym) * 100).toFixed(1)
      : null;

  const tiles: Array<{ label: string; value: string; delta?: string; deltaClass?: string }> = [
    {
      label: "Symmetry (today)",
      value: fmtPct(todaySym),
      delta: symDelta != null
        ? `${Number(symDelta) >= 0 ? "+" : ""}${symDelta}% vs yesterday`
        : "—",
      deltaClass:
        symDelta != null
          ? Number(symDelta) >= 0 ? "text-risk-normal" : "text-risk-critical"
          : "text-slate-500",
    },
    {
      label: "Sessions (7d)",
      value: String(sessions7),
      delta: `${sessions30} total (30d)`,
    },
    {
      label: "Best Score",
      value: best?.symmetry_score != null ? fmtPct(best.symmetry_score) : "—",
      delta: best?.date ?? "—",
    },
    {
      label: "Streak",
      value: String(streak),
      delta: "consecutive days",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3" role="list" aria-label="Progress KPIs">
      {tiles.map((t) => (
        <Card key={t.label} tone="elevated">
          <div role="listitem" className="flex flex-col items-center text-center gap-1 py-1">
            <div className="text-micro tracking-cyber uppercase text-slate-500">{t.label}</div>
            <div className="text-display font-bold text-slate-200 tabular-nums">{t.value}</div>
            <div className={`text-label font-semibold ${t.deltaClass ?? "text-slate-500"}`}>
              {t.delta}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}
