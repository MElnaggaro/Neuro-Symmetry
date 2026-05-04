import type { ReactNode } from "react";

type Variant = "no-face" | "no-sessions" | "no-data";

const ICONS: Record<Variant, ReactNode> = {
  "no-face": (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <circle cx="28" cy="28" r="20" stroke="rgba(34,211,238,0.4)" strokeWidth="1.5" strokeDasharray="3 4" />
      <circle cx="22" cy="25" r="1.6" fill="rgba(148,163,184,0.6)" />
      <circle cx="34" cy="25" r="1.6" fill="rgba(148,163,184,0.6)" />
      <path d="M22 35 Q28 38 34 35" stroke="rgba(148,163,184,0.6)" strokeWidth="1.4" strokeLinecap="round" fill="none" />
    </svg>
  ),
  "no-sessions": (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <rect x="10" y="14" width="36" height="28" rx="4" stroke="rgba(34,211,238,0.4)" strokeWidth="1.5" />
      <path d="M14 30 L22 24 L30 28 L42 18" stroke="rgba(34,211,238,0.7)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="42" cy="18" r="2.4" fill="rgba(34,211,238,0.7)" />
    </svg>
  ),
  "no-data": (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden="true">
      <circle cx="28" cy="28" r="18" stroke="rgba(148,163,184,0.4)" strokeWidth="1.5" />
      <path d="M28 19 V31" stroke="rgba(148,163,184,0.6)" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="28" cy="36" r="1.6" fill="rgba(148,163,184,0.6)" />
    </svg>
  ),
};

const DEFAULTS: Record<Variant, { title: string; body: string }> = {
  "no-face":     { title: "Face not detected", body: "Center your face in the frame and ensure the room is well lit." },
  "no-sessions":{ title: "No sessions yet",   body: "Run a Live session, Sentinel screening, or Face-Joypad round to start tracking." },
  "no-data":     { title: "Awaiting data",     body: "Streaming will start as soon as the analysis pipeline produces a result." },
};

type Props = {
  variant?: Variant;
  title?:   string;
  body?:    string;
  cta?:     ReactNode;
  className?: string;
};

export function EmptyState({
  variant = "no-data",
  title,
  body,
  cta,
  className = "",
}: Props) {
  const d = DEFAULTS[variant];
  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex flex-col items-center justify-center text-center gap-2 py-8 px-6 ${className}`}
    >
      {ICONS[variant]}
      <div className="text-h2 font-semibold text-slate-300 mt-2">{title ?? d.title}</div>
      <div className="text-label text-slate-500 max-w-[260px]">{body ?? d.body}</div>
      {cta && <div className="mt-3">{cta}</div>}
    </div>
  );
}
