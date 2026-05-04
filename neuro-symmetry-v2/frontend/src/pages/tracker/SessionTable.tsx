import { useMemo, useState } from "react";
import type { TrackerEntry } from "./loadHistory";

type SortKey  = "date" | "symmetry_score" | "probability" | "game_activations";
type SortDir  = "asc" | "desc";
type Filter   = "all" | "7d" | "30d";

const BADGE: Record<string, string> = {
  "Normal":    "bg-emerald-900/20 text-risk-normal   border-emerald-500/30",
  "Mild":      "bg-amber-900/20   text-risk-mild     border-amber-500/30",
  "High Risk": "bg-orange-900/20  text-risk-high     border-orange-500/30",
  "Critical":  "bg-red-900/20     text-risk-critical border-red-500/30",
};

function withinRange(date: string, days: number): boolean {
  const cutoff = new Date(Date.now() - days * 864e5).toISOString().slice(0, 10);
  return date >= cutoff;
}

export function SessionTable({ entries }: { entries: TrackerEntry[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [filter,  setFilter]  = useState<Filter>("all");

  const rows = useMemo(() => {
    const ranged =
      filter === "all" ? entries
      : filter === "7d"  ? entries.filter((e) => withinRange(e.date, 7))
                         : entries.filter((e) => withinRange(e.date, 30));
    const sorted = [...ranged].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const an = av == null ? -Infinity : typeof av === "number" ? av : Date.parse(av);
      const bn = bv == null ? -Infinity : typeof bv === "number" ? bv : Date.parse(bv);
      return sortDir === "asc" ? an - bn : bn - an;
    });
    return sorted.slice(0, 60);
  }, [entries, filter, sortKey, sortDir]);

  const toggleSort = (k: SortKey) => {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(k); setSortDir("desc"); }
  };

  const indicator = (k: SortKey) => sortKey === k ? (sortDir === "asc" ? " ▲" : " ▼") : "";

  return (
    <div className="flex flex-col gap-3">
      <div role="tablist" aria-label="Filter session range" className="flex gap-1">
        {(["all", "7d", "30d"] as Filter[]).map((f) => {
          const active = filter === f;
          return (
            <button
              key={f}
              role="tab"
              aria-selected={active}
              onClick={() => setFilter(f)}
              className={`px-2.5 py-0.5 rounded-full text-micro font-bold tracking-cyber border transition-colors ring-cyber ${
                active
                  ? "bg-cyan-900/25 border-accent-cyan/40 text-accent-cyan"
                  : "bg-transparent border-neu-border text-slate-500 hover:text-slate-300 hover:border-neu-borderLight"
              }`}
            >
              {f.toUpperCase()}
            </button>
          );
        })}
      </div>

      {rows.length === 0 ? (
        <div className="text-label text-slate-500 py-6 text-center">No sessions in range.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-body">
            <thead>
              <tr className="text-micro tracking-cyber uppercase text-slate-500 border-b border-neu-border">
                <Th onClick={() => toggleSort("date")}            label={`Date${indicator("date")}`}             />
                <Th label="Class" />
                <Th onClick={() => toggleSort("symmetry_score")}  label={`Symmetry${indicator("symmetry_score")}`} />
                <Th onClick={() => toggleSort("probability")}     label={`Confidence${indicator("probability")}`}   />
                <Th onClick={() => toggleSort("game_activations")} label={`Game Acts.${indicator("game_activations")}`} />
                <Th label="Source" />
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => (
                <tr key={`${e.date}-${e.source}-${i}`} className="border-b border-neu-border/60 hover:bg-cyan-900/[0.04]">
                  <td className="px-3 py-2 text-slate-300 tabular-nums">{e.date}</td>
                  <td className="px-3 py-2">
                    <span className={`text-micro font-bold tracking-wide px-2 py-0.5 rounded-full border ${BADGE[e.class] ?? BADGE["Normal"]}`}>
                      {e.class}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-300 tabular-nums">
                    {e.symmetry_score != null ? `${(e.symmetry_score * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-2 text-slate-400 tabular-nums">
                    {e.probability != null ? `${Math.round(e.probability * 100)}%` : "—"}
                  </td>
                  <td className="px-3 py-2 text-slate-400 tabular-nums">
                    {e.game_activations ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-micro text-slate-600 uppercase tracking-wide">
                    {e.source ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Th({ label, onClick }: { label: string; onClick?: () => void }) {
  return (
    <th
      onClick={onClick}
      className={`px-3 py-2 text-left font-semibold ${onClick ? "cursor-pointer select-none hover:text-accent-cyan" : ""}`}
    >
      {label}
    </th>
  );
}
