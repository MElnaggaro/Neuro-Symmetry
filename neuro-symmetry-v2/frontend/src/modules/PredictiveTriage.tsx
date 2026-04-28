import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, X, Clock, CheckSquare, Square, ShieldAlert, Activity } from "lucide-react";
import type { RiskLevel } from "@/types/analysis";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StrokeQuestion {
  id: string;
  label: string;
  description: string;
  icon: string;
  strokeIndicator: boolean; // if true, positive answer = possible stroke
}

const STROKE_QUESTIONS: StrokeQuestion[] = [
  {
    id: "arm_weakness",
    label: "Arm or Leg Weakness",
    description: "Can you raise both arms equally? Do you feel sudden weakness or numbness on one side of your body?",
    icon: "💪",
    strokeIndicator: true,
  },
  {
    id: "speech_difficulty",
    label: "Speech Difficulty",
    description: "Is your speech slurred, garbled, or are you struggling to find words? Can you repeat: 'The sky is blue today'?",
    icon: "💬",
    strokeIndicator: true,
  },
  {
    id: "swallowing_difficulty",
    label: "Swallowing Difficulty",
    description: "Do you have trouble swallowing liquids or food? Does food feel like it sticks in your throat?",
    icon: "🫁",
    strokeIndicator: true,
  },
];

const EYE_SAFETY_ITEMS = [
  "Apply lubricating eye drops (preservative-free) every 2–3 hours",
  "At night: tape the eyelid closed or use a moisture chamber goggle",
  "Wear wraparound sunglasses outdoors — UV and wind worsen corneal drying",
  "Do NOT rub the eye — conjunctival abrasion risk is high with reduced blink reflex",
  "Book an urgent ophthalmology appointment if you notice blurred vision or eye pain",
];

const GOLDEN_WINDOW_HOURS = 72;

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatCountdown(msLeft: number): string {
  if (msLeft <= 0) return "EXPIRED";
  const h = Math.floor(msLeft / 3_600_000);
  const m = Math.floor((msLeft % 3_600_000) / 60_000);
  const s = Math.floor((msLeft % 60_000) / 1_000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface PredictiveTriageProps {
  riskLevel: RiskLevel;
  onDismiss: () => void;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PredictiveTriage({ riskLevel, onDismiss }: PredictiveTriageProps) {
  // ── Step management ───────────────────────────────────────────────────────
  // Steps: "stroke" | "golden" | "eye" | "summary"
  const [step, setStep] = useState<"stroke" | "golden" | "eye" | "summary">("stroke");
  const [answers, setAnswers] = useState<Record<string, boolean | null>>({
    arm_weakness: null, speech_difficulty: null, swallowing_difficulty: null,
  });

  // ── Golden Window Timer ───────────────────────────────────────────────────
  const [onsetTime, setOnsetTime] = useState<number | null>(() => {
    const stored = localStorage.getItem("ns_onset_time");
    return stored ? parseInt(stored, 10) : null;
  });
  const [countdown, setCountdown] = useState<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startGoldenTimer = useCallback(() => {
    const now = Date.now();
    setOnsetTime(now);
    localStorage.setItem("ns_onset_time", String(now));
  }, []);

  useEffect(() => {
    if (!onsetTime) return;
    const tick = () => {
      const elapsed = Date.now() - onsetTime;
      setCountdown(Math.max(0, GOLDEN_WINDOW_HOURS * 3_600_000 - elapsed));
    };
    tick();
    timerRef.current = setInterval(tick, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [onsetTime]);

  // ── Eye safety checklist ──────────────────────────────────────────────────
  const [eyeChecks, setEyeChecks] = useState<boolean[]>(
    () => new Array(EYE_SAFETY_ITEMS.length).fill(false)
  );
  const toggleEyeCheck = (i: number) =>
    setEyeChecks(prev => prev.map((v, idx) => idx === i ? !v : v));

  // ── Stroke risk assessment ────────────────────────────────────────────────
  const positiveStrokeFlags = Object.values(answers).filter(v => v === true).length;
  const strokeSuspected = positiveStrokeFlags >= 1;

  const setAnswer = (id: string, val: boolean) =>
    setAnswers(prev => ({ ...prev, [id]: val }));

  const allAnswered = Object.values(answers).every(v => v !== null);

  // ── Urgency color by risk ─────────────────────────────────────────────────
  const urgencyColor =
    riskLevel === "CRITICAL" ? "#ef4444" :
    riskLevel === "HIGH_RISK" ? "#f97316" :
    "#f59e0b";

  const urgencyLabel =
    riskLevel === "CRITICAL" ? "CRITICAL" :
    riskLevel === "HIGH_RISK" ? "HIGH RISK" :
    "MILD RISK";

  // ── Keyboard: Escape to dismiss ───────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onDismiss(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onDismiss]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <AnimatePresence>
      {/* Backdrop */}
      <motion.div
        key="backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4"
        onClick={(e) => { if (e.target === e.currentTarget) onDismiss(); }}
        role="dialog"
        aria-modal="true"
        aria-label="Predictive Triage Assessment"
      >
        <motion.div
          key="modal"
          initial={{ opacity: 0, scale: 0.95, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 12 }}
          transition={{ type: "spring", stiffness: 320, damping: 28 }}
          className="relative w-full max-w-lg bg-[#1a1a2e] border border-[#2a2a4a] rounded-2xl
                     shadow-2xl overflow-hidden"
        >
          {/* Header bar */}
          <div
            className="flex items-center justify-between px-5 py-4 border-b border-[#2a2a4a]"
            style={{ borderTop: `3px solid ${urgencyColor}` }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                style={{ background: `${urgencyColor}22` }}
              >
                <AlertTriangle size={18} color={urgencyColor} strokeWidth={2.5} />
              </div>
              <div>
                <div className="text-[13px] font-extrabold tracking-wide text-slate-200">
                  Predictive Triage
                </div>
                <div className="text-[10px] font-semibold tracking-[0.2em] uppercase"
                     style={{ color: urgencyColor }}>
                  {urgencyLabel} — Facial Asymmetry Detected
                </div>
              </div>
            </div>
            <button
              onClick={onDismiss}
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300
                         hover:bg-white/5 transition-colors"
              aria-label="Dismiss triage"
            >
              <X size={16} />
            </button>
          </div>

          {/* Step indicator */}
          <div className="flex px-5 pt-4 gap-2">
            {(["stroke","golden","eye","summary"] as const).map((s, i) => (
              <div
                key={s}
                className="flex-1 h-1 rounded-full transition-all duration-300"
                style={{
                  background: step === s ? urgencyColor
                    : ["stroke","golden","eye","summary"].indexOf(step) > i ? "#34d399" : "#2a2a4a"
                }}
              />
            ))}
          </div>

          {/* Body */}
          <div className="px-5 py-4 min-h-[260px]">
            <AnimatePresence mode="wait">

              {/* STEP 1: Stroke screening */}
              {step === "stroke" && (
                <motion.div key="stroke"
                  initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} exit={{ opacity:0, x:-20 }}>
                  <div className="flex items-center gap-2 mb-4">
                    <ShieldAlert size={16} color={urgencyColor} />
                    <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
                      Stroke Screen — FAST Protocol
                    </h2>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
                    Facial asymmetry can be an early stroke sign. Answer 3 quick questions to rule it out.
                    If in doubt, <strong className="text-red-400">call emergency services immediately</strong>.
                  </p>

                  <div className="space-y-3">
                    {STROKE_QUESTIONS.map(q => (
                      <div key={q.id}
                           className="rounded-xl border border-[#2a2a4a] p-3 bg-[#1e1e35]">
                        <div className="flex items-start gap-2 mb-2">
                          <span className="text-lg leading-none">{q.icon}</span>
                          <div>
                            <div className="text-[12px] font-semibold text-slate-200">{q.label}</div>
                            <div className="text-[10px] text-slate-400 leading-snug mt-0.5">
                              {q.description}
                            </div>
                          </div>
                        </div>
                        <div className="flex gap-2 mt-2">
                          {[true, false].map(val => (
                            <button
                              key={String(val)}
                              onClick={() => setAnswer(q.id, val)}
                              className={`flex-1 py-1.5 rounded-lg text-[11px] font-bold tracking-wide
                                          transition-all border
                                          ${answers[q.id] === val
                                            ? val
                                              ? "bg-red-900/40 border-red-500/60 text-red-300"
                                              : "bg-emerald-900/30 border-emerald-500/40 text-emerald-400"
                                            : "bg-[#252540] border-[#2a2a4a] text-slate-500 hover:border-slate-500"
                                          }`}
                              aria-pressed={answers[q.id] === val}
                            >
                              {val ? "YES" : "NO"}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>

                  {strokeSuspected && (
                    <motion.div
                      initial={{ opacity:0, y:6 }} animate={{ opacity:1, y:0 }}
                      className="mt-4 p-3 rounded-xl bg-red-900/20 border border-red-500/40 text-red-300
                                 text-[11px] leading-relaxed"
                      role="alert"
                    >
                      <strong>⚠️ Stroke symptoms detected.</strong> Call <strong>emergency services (999 / 112 / 911)</strong> immediately.
                      Do not drive yourself to hospital. Time to treatment is critical.
                    </motion.div>
                  )}

                  <button
                    disabled={!allAnswered}
                    onClick={() => setStep("golden")}
                    className={`w-full mt-4 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                transition-all
                                ${allAnswered
                                  ? "text-white cursor-pointer hover:brightness-110"
                                  : "bg-[#252540] text-slate-600 cursor-not-allowed"
                                }`}
                    style={allAnswered ? { background: urgencyColor } : {}}
                  >
                    Continue →
                  </button>
                </motion.div>
              )}

              {/* STEP 2: Golden Window Timer */}
              {step === "golden" && (
                <motion.div key="golden"
                  initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} exit={{ opacity:0, x:-20 }}>
                  <div className="flex items-center gap-2 mb-4">
                    <Clock size={16} color={urgencyColor} />
                    <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
                      Golden Window Timer
                    </h2>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
                    Bell's Palsy treatment (oral corticosteroids) is most effective within{" "}
                    <strong className="text-amber-400">72 hours of symptom onset</strong>.
                    Mark when your symptoms first appeared to track your treatment window.
                  </p>

                  {onsetTime ? (
                    <div className="rounded-2xl border border-[#2a2a4a] p-5 bg-[#1e1e35] text-center">
                      <div className="text-[10px] font-semibold tracking-[0.2em] uppercase text-slate-500 mb-2">
                        Time Remaining
                      </div>
                      <div
                        className="text-4xl font-mono font-bold tabular-nums"
                        style={{
                          color: countdown > 24*3_600_000 ? urgencyColor
                               : countdown > 6*3_600_000  ? "#f59e0b" : "#ef4444"
                        }}
                      >
                        {formatCountdown(countdown)}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-2">
                        Onset: {new Date(onsetTime).toLocaleString()}
                      </div>
                      <div className="mt-3 w-full bg-[#2a2a4a] rounded-full h-2 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-1000"
                          style={{
                            width: `${Math.max(0, (countdown/(GOLDEN_WINDOW_HOURS*3_600_000))*100).toFixed(1)}%`,
                            background: countdown > 24*3_600_000 ? urgencyColor
                                       : countdown > 6*3_600_000  ? "#f59e0b" : "#ef4444",
                          }}
                        />
                      </div>
                      {countdown <= 0 && (
                        <p className="text-[11px] text-red-400 mt-3 font-semibold">
                          Treatment window has closed. See your doctor regardless — late treatment still helps.
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-dashed border-[#3a3a5a] p-5 text-center bg-[#1e1e35]">
                      <Clock size={32} color="#64748b" className="mx-auto mb-3" />
                      <p className="text-[12px] text-slate-400 mb-4">
                        Tap below when your symptoms first started.
                      </p>
                      <button
                        onClick={startGoldenTimer}
                        className="px-6 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                   text-white transition-all hover:brightness-110"
                        style={{ background: urgencyColor }}
                      >
                        Start Timer Now
                      </button>
                    </div>
                  )}

                  <div className="mt-4 p-3 rounded-xl bg-[#1e1e35] border border-[#2a2a4a]
                                  text-[10px] text-slate-400 leading-relaxed">
                    <strong className="text-slate-300">Next step:</strong> Contact your GP or
                    neurologist today. If out of hours, attend an urgent care or A&amp;E department.
                    Ask specifically about <em>oral prednisolone</em> for Bell's Palsy.
                  </div>

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={() => setStep("stroke")}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 bg-[#252540] border border-[#2a2a4a] text-slate-400
                                 hover:border-slate-500 transition-all"
                    >← Back</button>
                    <button
                      onClick={() => setStep("eye")}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 text-white transition-all hover:brightness-110"
                      style={{ background: urgencyColor }}
                    >Continue →</button>
                  </div>
                </motion.div>
              )}

              {/* STEP 3: Eye Safety Protocol */}
              {step === "eye" && (
                <motion.div key="eye"
                  initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} exit={{ opacity:0, x:-20 }}>
                  <div className="flex items-center gap-2 mb-4">
                    <Activity size={16} color={urgencyColor} />
                    <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
                      Eye Safety Protocol
                    </h2>
                  </div>
                  <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
                    Incomplete eyelid closure (lagophthalmos) puts your cornea at risk.
                    Tick each protective measure as you complete it.
                  </p>

                  <div className="space-y-2.5">
                    {EYE_SAFETY_ITEMS.map((item, i) => (
                      <button
                        key={i}
                        onClick={() => toggleEyeCheck(i)}
                        className={`w-full flex items-start gap-3 text-left p-3 rounded-xl
                                    border transition-all
                                    ${eyeChecks[i]
                                      ? "bg-emerald-900/20 border-emerald-500/30 text-emerald-300"
                                      : "bg-[#1e1e35] border-[#2a2a4a] text-slate-300 hover:border-slate-500"
                                    }`}
                        aria-pressed={eyeChecks[i]}
                      >
                        {eyeChecks[i]
                          ? <CheckSquare size={15} className="shrink-0 mt-0.5 text-emerald-400" />
                          : <Square size={15} className="shrink-0 mt-0.5 text-slate-500" />
                        }
                        <span className="text-[11px] leading-snug">{item}</span>
                      </button>
                    ))}
                  </div>

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={() => setStep("golden")}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 bg-[#252540] border border-[#2a2a4a] text-slate-400
                                 hover:border-slate-500 transition-all"
                    >← Back</button>
                    <button
                      onClick={() => setStep("summary")}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 text-white transition-all hover:brightness-110"
                      style={{ background: urgencyColor }}
                    >Review Summary →</button>
                  </div>
                </motion.div>
              )}

              {/* STEP 4: Summary */}
              {step === "summary" && (
                <motion.div key="summary"
                  initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} exit={{ opacity:0, x:-20 }}>
                  <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide mb-4">
                    Triage Summary
                  </h2>

                  <div className="space-y-3">
                    {/* Stroke */}
                    <div className={`p-3 rounded-xl border text-[11px] leading-snug
                      ${strokeSuspected
                        ? "bg-red-900/20 border-red-500/40 text-red-300"
                        : "bg-emerald-900/15 border-emerald-500/25 text-emerald-400"
                      }`}>
                      <div className="font-semibold mb-1">
                        {strokeSuspected ? "⚠️ Stroke Symptoms Present" : "✅ Stroke Screen: Negative"}
                      </div>
                      {strokeSuspected
                        ? "Seek emergency care immediately. Show this screen to first responders."
                        : "No FAST signs detected. Likely Bell's Palsy (peripheral nerve palsy)."
                      }
                    </div>

                    {/* Golden window */}
                    {onsetTime && (
                      <div className="p-3 rounded-xl border border-[#2a2a4a] bg-[#1e1e35]
                                      text-[11px] text-slate-300">
                        <span className="font-semibold text-amber-400">⏱ Treatment Window: </span>
                        {countdown > 0 ? formatCountdown(countdown) + " remaining" : "Expired"}
                        {countdown > 0 && countdown < 12*3_600_000 && (
                          <span className="text-red-400 font-bold"> — URGENT, seek care now</span>
                        )}
                      </div>
                    )}

                    {/* Eye safety */}
                    <div className={`p-3 rounded-xl border text-[11px] leading-snug
                      ${eyeChecks.filter(Boolean).length === EYE_SAFETY_ITEMS.length
                        ? "bg-emerald-900/15 border-emerald-500/25 text-emerald-400"
                        : "bg-amber-900/15 border-amber-500/30 text-amber-300"
                      }`}>
                      <span className="font-semibold">👁 Eye Protection: </span>
                      {eyeChecks.filter(Boolean).length}/{EYE_SAFETY_ITEMS.length} items completed
                    </div>

                    {/* Recommended action */}
                    <div className="p-3 rounded-xl border border-[#2a2a4a] bg-[#1e1e35]
                                    text-[11px] text-slate-300 leading-relaxed">
                      <div className="font-semibold text-slate-200 mb-1">Recommended Action</div>
                      {strokeSuspected
                        ? "Call emergency services (999/112/911) immediately."
                        : riskLevel === "CRITICAL"
                        ? "Attend A&E or urgent care today. Request prednisolone within 72h of onset."
                        : riskLevel === "HIGH_RISK"
                        ? "Contact your GP today (same-day appointment). Start eye protection immediately."
                        : "See your GP within 48h. Monitor symptoms and complete the Mirror Therapy daily."
                      }
                    </div>
                  </div>

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={() => setStep("eye")}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 bg-[#252540] border border-[#2a2a4a] text-slate-400
                                 hover:border-slate-500 transition-all"
                    >← Back</button>
                    <button
                      onClick={onDismiss}
                      className="flex-1 py-2.5 rounded-xl text-[12px] font-bold tracking-wide
                                 text-white transition-all hover:brightness-110"
                      style={{ background: urgencyColor }}
                    >Close</button>
                  </div>
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
