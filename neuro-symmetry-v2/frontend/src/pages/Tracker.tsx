import { useEffect, useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/Card";
import { SectionLabel } from "@/components/ui";
import { EmptyState } from "@/components/ui/EmptyState";
import { toastError, toastInfo, toastSuccess } from "@/utils/toast";
import { navigate } from "@/router/api";
import { computeKpis, loadHistory, type TrackerEntry } from "./tracker/loadHistory";
import { KpiGrid }       from "./tracker/KpiGrid";
import { SymmetryChart } from "./tracker/SymmetryChart";
import { SessionTable }  from "./tracker/SessionTable";
import { CarePlan }      from "./tracker/CarePlan";
import { exportTrackerPdf } from "./tracker/exportPdf";

export default function Tracker() {
  const [entries, setEntries] = useState<TrackerEntry[]>(() => loadHistory());

  // Re-read on storage events (live data flowing in from /#/ in another tab).
  useEffect(() => {
    const reload = () => setEntries(loadHistory());
    window.addEventListener("storage", reload);
    return () => window.removeEventListener("storage", reload);
  }, []);

  const kpis = useMemo(() => computeKpis(entries), [entries]);

  const chartWrapperRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    if (entries.length === 0) {
      toastInfo("Nothing to export yet — run a session first");
      return;
    }
    setExporting(true);
    try {
      const svg = chartWrapperRef.current?.querySelector("svg") ?? null;
      await exportTrackerPdf({ entries, kpis, chartSvg: svg as SVGSVGElement | null });
      toastSuccess("PDF exported");
    } catch (err) {
      console.error(err);
      toastError("PDF export failed — see console");
    } finally {
      setExporting(false);
    }
  };

  const handleClear = () => {
    if (!confirm("Clear all stored Neuro-Symmetry data? This cannot be undone.")) return;
    ["ns_assessments", "ns_prodromal", "ns_game"].forEach((k) => {
      try { localStorage.removeItem(k); } catch { /* ignore */ }
    });
    setEntries([]);
    toastInfo("All session data cleared");
  };

  return (
    <div className="flex flex-col h-screen bg-bg-base text-slate-200 font-sans overflow-y-auto">
      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-5 py-3 bg-bg-surface border-b border-neu-border">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => navigate("/")}
            className="text-label tracking-cyber uppercase text-slate-500 hover:text-accent-cyan transition-colors ring-cyber rounded px-1"
            aria-label="Back to live analysis"
          >
            ← Live Analysis
          </button>
          <div className="text-h1 font-extrabold tracking-wide text-slate-200">Progress Tracker</div>
          <div className="text-micro font-semibold tracking-cyber text-accent-cyan uppercase">
            Personal Symmetry History
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={exporting}
            className="px-3 py-1.5 rounded-lg text-label font-bold text-blue-100 bg-gradient-to-r from-blue-700 to-sky-600 shadow-glow-cyan hover:brightness-110 transition-all ring-cyber disabled:opacity-60"
          >
            {exporting ? "Rendering…" : "Export PDF"}
          </button>
          <button
            onClick={handleClear}
            className="px-3 py-1.5 rounded-lg text-label font-bold text-red-400 border border-neu-border hover:border-red-500/40 hover:bg-red-900/15 transition-all ring-cyber"
          >
            Clear data
          </button>
        </div>
      </header>

      {entries.length === 0 ? (
        <main className="flex flex-1 items-center justify-center">
          <Card tone="elevated" className="max-w-md">
            <EmptyState
              variant="no-sessions"
              cta={
                <button
                  onClick={() => navigate("/")}
                  className="px-4 py-2 rounded-lg text-label font-bold text-blue-100 bg-gradient-to-r from-blue-700 to-sky-600 shadow-glow-cyan hover:brightness-110 transition-all ring-cyber"
                >
                  Run a live session
                </button>
              }
            />
          </Card>
        </main>
      ) : (
        <main className="relative z-1 flex flex-col gap-4 p-5 max-w-[1100px] w-full mx-auto">
          <KpiGrid kpis={kpis} />

          <Card tone="elevated">
            <SectionLabel>Symmetry Over Time</SectionLabel>
            <div ref={chartWrapperRef}>
              <SymmetryChart entries={entries} />
            </div>
          </Card>

          <Card>
            <SectionLabel>Personalised Care Plan</SectionLabel>
            <CarePlan entries={entries} />
          </Card>

          <Card>
            <SectionLabel>Session History</SectionLabel>
            <SessionTable entries={entries} />
          </Card>

          <p className="text-micro text-slate-600 text-center pb-6">
            This report is from self-monitoring data and is NOT a clinical diagnosis.
          </p>
        </main>
      )}
    </div>
  );
}
