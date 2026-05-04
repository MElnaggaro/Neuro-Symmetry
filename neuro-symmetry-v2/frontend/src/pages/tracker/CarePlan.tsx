import { useMemo } from "react";
import type { TrackerEntry } from "./loadHistory";

type Severity = "Normal" | "Mild" | "Severe" | "Critical";

function severityFromEntries(entries: TrackerEntry[]): Severity {
  const recent = entries.slice(-10);
  const counts: Record<string, number> = { Normal: 0, Mild: 0, "High Risk": 0, Critical: 0 };
  for (const e of recent) {
    if (e.class in counts) counts[e.class]++;
  }
  if (counts.Critical > 0) return "Critical";
  if (counts["High Risk"] >= 2) return "Severe";
  if (counts.Mild >= 3) return "Mild";
  return "Normal";
}

function avgSym(entries: TrackerEntry[]): number | null {
  const vals = entries.slice(-10).filter((e) => e.symmetry_score != null).map((e) => e.symmetry_score as number);
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
}

const SECTIONS: Record<Exclude<Severity, never>, ReadonlyArray<{ title: string; items: string[] }>> = {
  Critical: [
    {
      title: "Corticosteroid urgency",
      items: [
        "Contact your neurologist or GP within 24 hours.",
        "Oral prednisolone (1 mg/kg/day for 5 days) is most effective when started within 72 h of onset.",
        "Bring the exported PDF to the appointment.",
      ],
    },
    {
      title: "Eye care protocol",
      items: [
        "Apply lubricating eye drops every 2–3 hours during the day.",
        "Use a moisture chamber or tape the eye shut at night to prevent corneal damage.",
        "Wear sunglasses outdoors to reduce dryness and photophobia.",
      ],
    },
    {
      title: "Rest protocol",
      items: [
        "Limit facial movement exercises to 5 minutes, twice daily — overwork delays recovery.",
        "Maintain adequate sleep (7–9 h); sleep promotes neural repair.",
        "Avoid cold wind directly on the face.",
      ],
    },
  ],
  Severe: [
    {
      title: "Corticosteroid urgency",
      items: [
        "Contact your neurologist or GP within 24 hours.",
        "Oral prednisolone (1 mg/kg/day for 5 days) is most effective when started within 72 h of onset.",
      ],
    },
    {
      title: "Eye care protocol",
      items: [
        "Lubricating eye drops every 2–3 hours.",
        "Tape the eye shut at night; use a moisture chamber.",
      ],
    },
  ],
  Mild: [
    {
      title: "Warm compress",
      items: [
        "Apply a warm (not hot) damp cloth to the affected side for 10 minutes, 3× daily.",
        "Warmth improves microcirculation and eases facial nerve tension.",
      ],
    },
    {
      title: "Draft avoidance",
      items: [
        "Avoid air conditioning vents and fans blowing directly on the face.",
      ],
    },
    {
      title: "Gentle massage",
      items: [
        "With clean, warm fingertips, stroke from chin to cheekbone — 30 strokes per session.",
        "Light circular massage around the eye orbit relieves orbicularis spasm.",
      ],
    },
  ],
  Normal: [
    {
      title: "Daily maintenance exercises",
      items: [
        "Practise slow, controlled blinks 20× every morning.",
        "Hold a gentle smile for 5 seconds, then relax — repeat 10×.",
        "Alternate eyebrow raises: isolate each brow, 5× per side.",
      ],
    },
    {
      title: "Prevention",
      items: [
        "Keep up weekly Sentinel check-ins to catch asymmetry early.",
        "Maintain vitamin B12 and zinc intake.",
        "Use Mirror Therapy 5–10 minutes daily to reinforce motor cortex pathways.",
      ],
    },
  ],
};

const HEADER_CLASS: Record<Severity, string> = {
  Normal:   "text-risk-normal",
  Mild:     "text-risk-mild",
  Severe:   "text-risk-high",
  Critical: "text-risk-critical",
};

const HEADER_LABEL: Record<Severity, string> = {
  Normal:   "Normal — Maintenance & Prevention",
  Mild:     "Mild — Self-Care & Monitoring",
  Severe:   "Severe — Medical Attention Recommended",
  Critical: "Critical — Seek Care Immediately",
};

export function CarePlan({ entries }: { entries: TrackerEntry[] }) {
  const severity = useMemo(() => severityFromEntries(entries), [entries]);
  const sym      = useMemo(() => avgSym(entries), [entries]);

  if (entries.length === 0) {
    return (
      <p className="text-label text-slate-500">
        Complete at least one Sentinel assessment to receive a personalised care plan.
      </p>
    );
  }

  const sections = SECTIONS[severity];

  return (
    <div className="flex flex-col gap-3">
      <div className={`text-h1 font-bold ${HEADER_CLASS[severity]}`}>
        {HEADER_LABEL[severity]}
      </div>
      {sym != null && (
        <div className="text-label text-slate-500">
          Average symmetry (last 10 entries): {(sym * 100).toFixed(1)}%
        </div>
      )}
      {sections.map((sec) => (
        <div key={sec.title}>
          <div className="text-micro tracking-cyber uppercase text-accent-cyan mb-1">
            {sec.title}
          </div>
          <ul className="list-disc list-inside text-body text-slate-300 space-y-1">
            {sec.items.map((it) => <li key={it}>{it}</li>)}
          </ul>
        </div>
      ))}
    </div>
  );
}
