import type { ReactNode } from "react";

interface SectionLabelProps {
  children: ReactNode;
}

/** Styled section header with decorative lines — used inside Card panels. */
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
