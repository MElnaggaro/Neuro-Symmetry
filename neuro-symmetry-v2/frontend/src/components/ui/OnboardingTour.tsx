import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useLayoutEffect, useState } from "react";
import { useLocalStorage } from "@/hooks/useLocalStorage";

const STEPS: Array<{ tour: string; title: string; body: string }> = [
  { tour: "logo",      title: "Welcome to Neuro-Symmetry",
    body: "Real-time facial symmetry analysis for early Bell's palsy and stroke detection. Press ⌘K any time to open the command palette." },
  { tour: "gauge",     title: "Symmetry gauge",
    body: "The live symmetry score, EMA-smoothed. Color tracks the current risk level — green is normal, red is critical." },
  { tour: "calibrate", title: "Calibrate your baseline",
    body: "Hold still and click CALIBRATE BASELINE for ~5 s so the system learns your personal 'normal'. Decisions become much more accurate." },
  { tour: "tracker",   title: "Track progress over time",
    body: "Every session is saved locally. The Tracker page shows trends, KPIs, and a care-plan you can export as PDF." },
];

type Rect = { top: number; left: number; width: number; height: number };

function getAnchorRect(name: string): Rect | null {
  const el = document.querySelector<HTMLElement>(`[data-tour="${name}"]`);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return { top: r.top, left: r.left, width: r.width, height: r.height };
}

export function OnboardingTour() {
  const [done, setDone] = useLocalStorage<"0" | "1">("ns_onboarding_done", "0");
  const [step, setStep] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const [open, setOpen] = useState(false);

  // Defer first render so anchor elements have mounted.
  useEffect(() => {
    if (done === "1") return;
    const t = window.setTimeout(() => setOpen(true), 600);
    return () => window.clearTimeout(t);
  }, [done]);

  useLayoutEffect(() => {
    if (!open) return;
    const update = () => setRect(getAnchorRect(STEPS[step].tour));
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open, step]);

  if (!open) return null;
  const total = STEPS.length;
  const cur = STEPS[step];

  const finish = () => { setDone("1"); setOpen(false); };
  const next   = () => (step + 1 < total ? setStep(step + 1) : finish());

  // Tooltip placement: below anchor by default, above if near the bottom.
  const tooltipTop  = rect ? Math.min(rect.top + rect.height + 12, window.innerHeight - 200) : window.innerHeight / 2;
  const tooltipLeft = rect ? Math.max(16, Math.min(rect.left, window.innerWidth - 360)) : window.innerWidth / 2 - 160;

  return (
    <AnimatePresence>
      <motion.div
        key="onboarding"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-[90] pointer-events-auto"
        role="dialog"
        aria-modal="true"
        aria-label={`Onboarding step ${step + 1} of ${total}`}
      >
        {/* Dim layer */}
        <div className="absolute inset-0 bg-black/55 backdrop-blur-[2px]" />

        {/* Spotlight cutout */}
        {rect && (
          <motion.div
            layout
            className="absolute rounded-xl ring-2 ring-accent-cyan/70 shadow-[0_0_24px_rgba(6,182,212,0.45)] pointer-events-none"
            style={{
              top:    rect.top - 6,
              left:   rect.left - 6,
              width:  rect.width + 12,
              height: rect.height + 12,
              boxShadow: "0 0 0 9999px rgba(0,0,0,0.45)",
            }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
          />
        )}

        {/* Tooltip card */}
        <motion.div
          key={step}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="absolute glass-card glass-inset rounded-xl border border-neu-border w-[min(340px,90vw)] p-4"
          style={{ top: tooltipTop, left: tooltipLeft }}
        >
          <div className="text-micro tracking-cyber uppercase text-accent-cyan mb-1">
            STEP {step + 1} / {total}
          </div>
          <div className="text-h1 font-bold text-slate-200">{cur.title}</div>
          <div className="text-body text-slate-400 mt-2">{cur.body}</div>

          <div className="mt-4 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={finish}
              className="text-label text-slate-500 hover:text-slate-300 transition-colors ring-cyber rounded px-1"
            >
              Skip tour
            </button>
            <div className="flex items-center gap-2">
              {step > 0 && (
                <button
                  type="button"
                  onClick={() => setStep(step - 1)}
                  className="px-3 py-1.5 rounded-md text-label font-bold text-slate-300 border border-neu-border hover:border-neu-borderLight transition-colors ring-cyber"
                >
                  Back
                </button>
              )}
              <button
                type="button"
                onClick={next}
                className="px-3 py-1.5 rounded-md text-label font-bold text-blue-100 bg-gradient-to-r from-blue-700 to-sky-600 shadow-glow-cyan hover:brightness-110 transition-all ring-cyber"
              >
                {step + 1 === total ? "Got it" : "Next"}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
